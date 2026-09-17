import json
import logging
from typing import Optional, TypedDict

from app.config import settings

logger = logging.getLogger("geosentry.llm")

CAUSES = ["mining", "logging", "ag_expansion", "drought", "artifact"]

CAUSE_DESCRIPTIONS = {
    "mining": "unregulated or illegal mineral extraction",
    "logging": "illegal or unpermitted logging / canopy removal",
    "ag_expansion": "clearing for agricultural expansion",
    "drought": "seasonal or climatic drought stress (not human land-use change)",
    "artifact": "sensor noise, cloud contamination, or another non-physical artifact",
}

REQUIRED_EVENT_KEYS = ("zone_name", "ndvi_delta", "bsi_delta", "viirs_delta", "rainfall_percentile", "area_ha")


def parse_json_response(text: str) -> dict:
    """INPUTS: raw text from an Anthropic response content block. OUTPUTS: the parsed JSON dict.
    Both classify_event()'s and challenge_event()'s prompts say "Respond ONLY with JSON... no
    other text," but Claude doesn't always follow that to the letter - it sometimes wraps the
    answer in a ```json ... ``` markdown fence, which breaks a plain json.loads(). This strips a
    leading/trailing fence if present before parsing; it changes nothing about what was asked of
    the model or how it reasons, only how its answer is unwrapped."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")[1:]  # drop the opening ``` or ```json line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


class Classification(TypedDict):
    cause: str
    confidence: float
    reasoning: str


def _require_event_keys(event: dict) -> None:
    missing = [key for key in REQUIRED_EVENT_KEYS if key not in event]
    if missing:
        raise ValueError(f"event is missing required keys: {missing}")


def format_viirs_for_prompt(viirs_delta: Optional[float]) -> str:
    """INPUTS: a VIIRS z-score or None (gee.detect.current_viirs() found no VIIRS monthly
    composite for this zone/month - a real 1-2 month data lag, not an error). OUTPUTS: the
    value-portion of the prompt's "VIIRS delta: ..." line - an honest "unavailable" note instead
    of a fabricated number. Shared by llm/classify.py's and llm/adversarial.py's real prompts."""
    if viirs_delta is None:
        return "unavailable (monthly composite not yet published)"
    return f"{viirs_delta:+.2f} sigma"


def format_viirs_num(viirs_delta: Optional[float]) -> str:
    """INPUTS: a VIIRS z-score or None. OUTPUTS: a bare signed-number fragment for inline
    interpolation inside a sentence that already supplies its own unit (e.g. "(...sigma)");
    "n/a" when unavailable, never a fabricated number. Used by the synthetic reasoning/rationale
    text (the deterministic fallback used whenever ANTHROPIC_API_KEY isn't set, which is this
    demo's default), shared with llm/adversarial.py."""
    return f"{viirs_delta:+.2f}" if viirs_delta is not None else "n/a"


VIIRS_UNAVAILABLE_INSTRUCTION = (
    "If VIIRS is unavailable, do not treat its absence as evidence for or against any cause. "
    "Weight your classification on the available signals only, and note the missing "
    "corroborator in your reasoning."
)


def ndvi_bsi_summary(ndvi_delta: float, bsi_delta: float) -> str:
    """INPUTS: NDVI and BSI z-scores vs baseline. OUTPUTS: a plain-language sentence describing
    which indicator(s) actually moved - never claims an indicator "dropped" or "rose" when it's
    really just flat (abs(z) < 1). A cause can be flagged off a single indicator (e.g. a pure
    BSI spike with NDVI at 0.00), and a template that always narrates both as if they moved
    together misrepresents the event - see the module docstring / _build_reasoning() below.

      both moved:      "NDVI {direction} {z} sigma and BSI {direction} {z} sigma"
      only NDVI moved: "NDVI {direction} {z} sigma while BSI remained flat"
      only BSI moved:  "BSI {direction} {z} sigma while NDVI remained flat"
      neither moved:   "Neither NDVI nor BSI showed significant change"

    direction is "fell" for negative z, "rose" for positive z; "remained flat" when abs(z) < 1.
    Shared by _build_classify_prompt() (the real Claude prompt) and _build_reasoning() (the
    synthetic fallback used whenever ANTHROPIC_API_KEY isn't set, which is this demo's default),
    so both describe the same event the same honest way."""
    def direction(z: float) -> str:
        return "fell" if z < 0 else "rose"

    ndvi_moved = abs(ndvi_delta) >= 1
    bsi_moved = abs(bsi_delta) >= 1

    if ndvi_moved and bsi_moved:
        return f"NDVI {direction(ndvi_delta)} {ndvi_delta:+.2f} sigma and BSI {direction(bsi_delta)} {bsi_delta:+.2f} sigma"
    if ndvi_moved:
        return f"NDVI {direction(ndvi_delta)} {ndvi_delta:+.2f} sigma while BSI remained flat"
    if bsi_moved:
        return f"BSI {direction(bsi_delta)} {bsi_delta:+.2f} sigma while NDVI remained flat"
    return "Neither NDVI nor BSI showed significant change"


def ordinal(n: float) -> str:
    """INPUTS: a number (e.g. a percentile). OUTPUTS: its ordinal string, e.g. 62 -> '62nd'.
    Shared by llm/classify.py and llm/adversarial.py so rainfall percentiles read naturally in
    both the LLM prompts and the synthetic reasoning/rationale text."""
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def _build_classify_prompt(event: dict) -> str:
    """INPUTS: an event dict (see REQUIRED_EVENT_KEYS). OUTPUTS: the exact prompt string sent to
    Claude to classify the likely cause of the anomaly. viirs_delta may be None
    (format_viirs_for_prompt() renders it as an honest "unavailable" note, never a fabricated
    number) - VIIRS_UNAVAILABLE_INSTRUCTION tells the model explicitly not to read that absence
    as evidence either way. The Summary line (ndvi_bsi_summary()) states plainly which of
    NDVI/BSI actually moved, so the model doesn't infer joint movement from two raw numbers when
    the event was really flagged off a single indicator."""
    cause_lines = "\n".join(f"- {cause}: {desc}" for cause, desc in CAUSE_DESCRIPTIONS.items())
    return (
        "You are a satellite land-change analyst investigating a possible environmental "
        "disturbance.\n\n"
        f"Zone: \"{event['zone_name']}\"\n"
        "Observed anomalies vs the 3-year seasonal baseline for this location:\n"
        f"- NDVI delta (vegetation index change): {event['ndvi_delta']:+.2f} sigma\n"
        f"- BSI delta (bare soil index change): {event['bsi_delta']:+.2f} sigma\n"
        f"- VIIRS delta (night-light radiance change): {format_viirs_for_prompt(event['viirs_delta'])}\n"
        f"- Rainfall percentile (trailing 30 days vs historical): {ordinal(event['rainfall_percentile'])} percentile\n"
        f"- Estimated affected area: {event['area_ha']:.1f} hectares\n\n"
        f"Summary: {ndvi_bsi_summary(event['ndvi_delta'], event['bsi_delta'])}.\n\n"
        f"{VIIRS_UNAVAILABLE_INSTRUCTION}\n\n"
        "Classify the most likely cause of this disturbance as exactly one of:\n"
        f"{cause_lines}\n\n"
        "Respond ONLY with JSON in this exact shape, no other text:\n"
        '{"cause": "<one of ' + "|".join(CAUSES) + '>", "confidence": <float 0-1>, '
        '"reasoning": "<2-3 sentence plain-language paragraph>"}'
    )


def score_causes(event: dict) -> dict[str, float]:
    """INPUTS: an event dict (see REQUIRED_EVENT_KEYS). OUTPUTS: a raw, unnormalized plausibility
    score per cause in CAUSES, built from simple feature heuristics (canopy loss, bare-soil gain,
    night-light gain, dryness, overall signal magnitude). Shared by the synthetic classifier here
    and by llm/adversarial.py's synthetic alternative-ranking, so both pick causes off the same
    feature logic instead of duplicating it.

    event['viirs_delta'] may be None (VIIRS unavailable that month - see format_viirs_for_prompt
    docstring). Treated as 0.0 for this heuristic's arithmetic only - the same "neutral, no
    update either way" convention fusion.bayes.viirs_likelihood_ratio(0.0) == 1.0 already uses -
    a null guard on the formula's input, not a change to the formula or its weights."""
    ndvi_delta = event["ndvi_delta"]
    bsi_delta = event["bsi_delta"]
    viirs_delta = 0.0 if event["viirs_delta"] is None else event["viirs_delta"]
    rainfall_percentile = event["rainfall_percentile"]
    area_ha = event["area_ha"]

    canopy_loss = max(0.0, -ndvi_delta)
    bare_soil_gain = max(0.0, bsi_delta)
    nightlight_gain = max(0.0, viirs_delta)
    dryness = max(0.0, 1 - rainfall_percentile / 100)
    signal_magnitude = (abs(ndvi_delta) + abs(bsi_delta) + abs(viirs_delta)) / 3

    return {
        "mining": bare_soil_gain + nightlight_gain,
        "logging": max(0.0, canopy_loss - 0.5 * nightlight_gain),
        "ag_expansion": 0.5 * canopy_loss + 0.5 * bare_soil_gain + min(area_ha, 100) / 100,
        "drought": canopy_loss * dryness,
        "artifact": max(0.0, 1.5 - signal_magnitude),
    }


def _build_reasoning(cause: str, event: dict) -> str:
    """INPUTS: the chosen cause, the event dict. OUTPUTS: a 2-3 sentence plain-language paragraph
    grounded in the event's actual numbers, standing in for the LLM's reasoning field.
    event['viirs_delta'] may be None (VIIRS unavailable that month) - rendered as an honest
    "unavailable"/"n/a" note (see format_viirs_num), never a fabricated number; the mining and
    logging branches, which make a qualitative claim about VIIRS specifically, say so explicitly
    rather than silently substituting a number. The mining/logging/ag_expansion branches - the
    ones that narrate NDVI and BSI together - use ndvi_bsi_summary() rather than asserting both
    moved: a cause can be flagged off a single indicator (e.g. a pure BSI spike with NDVI at
    0.00), and always claiming "NDVI dropped ... while BSI rose ..." misdescribed that case."""
    ndvi_delta, bsi_delta, viirs_delta = event["ndvi_delta"], event["bsi_delta"], event["viirs_delta"]
    rainfall_percentile, area_ha = event["rainfall_percentile"], event["area_ha"]

    if cause == "mining":
        viirs_clause = (
            f"VIIRS night-light radiance is also elevated ({viirs_delta:+.2f} sigma), suggesting "
            "active night-time extraction equipment"
            if viirs_delta is not None
            else "VIIRS data is unavailable for this window, so night-time activity can't be "
            "confirmed either way"
        )
        return (
            f"{ndvi_bsi_summary(ndvi_delta, bsi_delta)}, a bare-soil signature typical of "
            f"excavation. {viirs_clause} across roughly {area_ha:.1f} ha."
        )
    if cause == "logging":
        viirs_clause = (
            f"no meaningful night-light signal ({viirs_delta:+.2f} sigma)"
            if viirs_delta is not None
            else "no VIIRS data available to corroborate or rule out night-time activity"
        )
        return (
            f"{ndvi_bsi_summary(ndvi_delta, bsi_delta)}, and {viirs_clause}, consistent with "
            f"daytime canopy removal over about {area_ha:.1f} ha rather than extraction or mining."
        )
    if cause == "ag_expansion":
        return (
            f"The disturbance spans {area_ha:.1f} ha. {ndvi_bsi_summary(ndvi_delta, bsi_delta)} - "
            "a footprint and signature more consistent with land being cleared and tilled for "
            "agriculture than a single extraction or logging site."
        )
    if cause == "drought":
        return (
            f"NDVI is down {ndvi_delta:+.2f} sigma, but rainfall is only at the "
            f"{ordinal(rainfall_percentile)} percentile for this zone and season, and BSI/VIIRS show "
            f"little independent change ({bsi_delta:+.2f}/{format_viirs_num(viirs_delta)} sigma). "
            "That pattern points to moisture stress rather than land clearing."
        )
    return (
        f"NDVI, BSI, and VIIRS deltas are all modest "
        f"({ndvi_delta:+.2f}/{bsi_delta:+.2f}/{format_viirs_num(viirs_delta)} sigma) and rainfall is "
        "unremarkable, so this looks like ordinary sensor or cloud-mask noise rather than a real "
        "land-cover change."
    )


def _synthetic_classification(event: dict) -> Classification:
    """INPUTS: an event dict. OUTPUTS: deterministic Classification standing in for the LLM when
    ANTHROPIC_API_KEY is absent or SYNTHETIC_MODE is on. Picks the highest-scoring cause from
    score_causes() and derives a confidence from its share of the total score."""
    scores = score_causes(event)
    cause = max(scores, key=scores.get)
    total = sum(scores.values()) or 1e-6
    confidence = round(min(0.95, max(0.3, scores[cause] / total)), 2)
    return Classification(cause=cause, confidence=confidence, reasoning=_build_reasoning(cause, event))


def classify_event(event: dict) -> Classification:
    """INPUTS: event dict with keys zone_name (str), ndvi_delta/bsi_delta/viirs_delta (float
    z-scores vs baseline), rainfall_percentile (float 0-100), area_ha (float). OUTPUTS:
    Classification with the likely cause (one of CAUSES), confidence (0-1), and a short
    reasoning paragraph. Calls Claude when ANTHROPIC_API_KEY is set and SYNTHETIC_MODE is false;
    otherwise (or on any API failure) returns a deterministic synthetic classification driven by
    the same feature heuristics, so the demo never dies and never silently blocks on a network
    call it can't make."""
    _require_event_keys(event)

    if settings.synthetic_mode or not settings.anthropic_api_key:
        return _synthetic_classification(event)

    prompt = _build_classify_prompt(event)
    logger.debug("classify_event prompt for %s:\n%s", event["zone_name"], prompt)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        data = parse_json_response(response.content[0].text)
        cause = data["cause"]
        if cause not in CAUSES:
            raise ValueError(f"model returned unknown cause: {cause!r}")
        return Classification(cause=cause, confidence=float(data["confidence"]), reasoning=data["reasoning"])
    except Exception as exc:  # noqa: BLE001 - any LLM failure must fall back, not crash the demo
        logger.warning("LLM classification failed (%s); falling back to synthetic classification.", exc)
        return _synthetic_classification(event)
