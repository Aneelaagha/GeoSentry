import logging
from typing import TypedDict

from app.config import settings
from llm.classify import (
    CAUSES,
    VIIRS_UNAVAILABLE_INSTRUCTION,
    Classification,
    _require_event_keys,
    format_viirs_for_prompt,
    format_viirs_num,
    ordinal,
    parse_json_response,
    score_causes,
)

logger = logging.getLogger("geosentry.llm")

MAX_ALTERNATIVES = 3


class Alternative(TypedDict):
    cause: str
    rationale: str


class AdversarialReview(TypedDict):
    alternatives: list[Alternative]
    strength_of_alternative: float


def _rationale_for(cause: str, event: dict) -> str:
    """INPUTS: a candidate alternative cause, the event dict. OUTPUTS: a one-sentence rationale
    for that cause, grounded in the event's numbers - the synthetic stand-in for what the LLM
    would argue in the real path. event['viirs_delta'] may be None (VIIRS unavailable that
    month) - the mining branch makes a qualitative claim about VIIRS specifically and says so
    explicitly rather than fabricating a number; the others use format_viirs_num()'s "n/a"."""
    ndvi_delta, bsi_delta, viirs_delta = event["ndvi_delta"], event["bsi_delta"], event["viirs_delta"]
    rainfall_percentile, area_ha = event["rainfall_percentile"], event["area_ha"]

    if cause == "mining":
        viirs_clause = (
            f"VIIRS is up {viirs_delta:+.2f} sigma"
            if viirs_delta is not None
            else "VIIRS data is unavailable for this window"
        )
        return (
            f"BSI is up {bsi_delta:+.2f} sigma and {viirs_clause}, a bare-soil-plus-nightlight "
            "signature more typical of mineral extraction."
        )
    if cause == "logging":
        return (
            f"NDVI is down {ndvi_delta:+.2f} sigma with little night-light change "
            f"({format_viirs_num(viirs_delta)} sigma), consistent with daytime canopy removal."
        )
    if cause == "ag_expansion":
        return (
            f"The {area_ha:.1f} ha affected area and paired NDVI/BSI shift look more like field "
            "clearing for agriculture than a single extraction or logging site."
        )
    if cause == "drought":
        return (
            f"Rainfall is only at the {ordinal(rainfall_percentile)} percentile, so the NDVI "
            "change could reflect moisture stress rather than land clearing."
        )
    return (
        f"NDVI/BSI/VIIRS deltas ({ndvi_delta:+.2f}/{bsi_delta:+.2f}/{format_viirs_num(viirs_delta)} "
        "sigma) are all modest, within the range of ordinary sensor or cloud-mask noise."
    )


def _build_challenge_prompt(event: dict, primary: Classification) -> str:
    """INPUTS: the event dict, the primary Classification from classify_event. OUTPUTS: the
    exact prompt string sent to Claude to argue against the primary classification.
    event['viirs_delta'] may be None (format_viirs_for_prompt() renders it as an honest
    "unavailable" note, never a fabricated number) - VIIRS_UNAVAILABLE_INSTRUCTION tells the
    model explicitly not to read that absence as evidence either way."""
    return (
        "You are a skeptical reviewer challenging a satellite-based land-change classification "
        "before it triggers a real-world alert. Your job is to argue AGAINST the primary "
        "classification and surface plausible alternative explanations, so the system doesn't "
        "cry wolf.\n\n"
        f"Zone: \"{event['zone_name']}\"\n"
        "Observed anomalies vs the 3-year seasonal baseline:\n"
        f"- NDVI delta: {event['ndvi_delta']:+.2f} sigma\n"
        f"- BSI delta: {event['bsi_delta']:+.2f} sigma\n"
        f"- VIIRS delta: {format_viirs_for_prompt(event['viirs_delta'])}\n"
        f"- Rainfall percentile (trailing 30 days): {ordinal(event['rainfall_percentile'])} percentile\n"
        f"- Estimated affected area: {event['area_ha']:.1f} hectares\n\n"
        f"{VIIRS_UNAVAILABLE_INSTRUCTION}\n\n"
        "Primary classification under review:\n"
        f"- cause: {primary['cause']}\n"
        f"- confidence: {primary['confidence']:.2f}\n"
        f"- reasoning: {primary['reasoning']}\n\n"
        "Argue why this primary cause might NOT be correct. List 1-3 concrete alternative "
        f"explanations (from: {', '.join(CAUSES)} - excluding the primary cause itself), each "
        "with a one-sentence rationale grounded in the numbers above. Then assess how strong "
        "the best alternative case is.\n\n"
        "Respond ONLY with JSON in this exact shape, no other text:\n"
        '{"alternatives": [{"cause": "<cause>", "rationale": "<one sentence>"}, ...], '
        '"strength_of_alternative": <float 0-1, where 0 = the primary classification is clearly '
        'correct and 1 = the primary classification is likely wrong>}'
    )


def _synthetic_challenge(event: dict, primary: Classification) -> AdversarialReview:
    """INPUTS: the event dict, the primary Classification. OUTPUTS: deterministic
    AdversarialReview used when the LLM is unavailable. Ranks every non-primary cause by the
    same score_causes() heuristics used by the synthetic classifier, offers the top few as
    alternatives, and sets strength_of_alternative from how close the best alternative's score
    is to the primary cause's own score (close scores -> a strong alternative case)."""
    scores = score_causes(event)
    primary_score = scores.get(primary["cause"], 0.0)
    ranked = sorted((c for c in CAUSES if c != primary["cause"]), key=lambda c: scores[c], reverse=True)
    top = ranked[:MAX_ALTERNATIVES]

    best_alt_score = scores[top[0]] if top else 0.0
    denom = primary_score + best_alt_score
    raw_strength = 0.5 if denom == 0 else best_alt_score / denom
    strength = round(min(0.95, max(0.05, raw_strength)), 2)

    alternatives = [Alternative(cause=cause, rationale=_rationale_for(cause, event)) for cause in top]
    return AdversarialReview(alternatives=alternatives, strength_of_alternative=strength)


def challenge_event(event: dict, primary_classification: Classification) -> AdversarialReview:
    """INPUTS: event dict (same shape as classify_event's, see REQUIRED_EVENT_KEYS), the primary
    Classification returned by classify_event. OUTPUTS: AdversarialReview - up to 3 alternative
    causes with one-sentence rationales, and strength_of_alternative (0-1: how strong the best
    alternative case is, so a single LLM call is never the sole gate on an alert). Calls Claude
    when ANTHROPIC_API_KEY is set and SYNTHETIC_MODE is false; otherwise (or on any API failure)
    falls back to a deterministic synthetic review."""
    _require_event_keys(event)

    if settings.synthetic_mode or not settings.anthropic_api_key:
        return _synthetic_challenge(event, primary_classification)

    prompt = _build_challenge_prompt(event, primary_classification)
    logger.debug("challenge_event prompt for %s:\n%s", event["zone_name"], prompt)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        data = parse_json_response(response.content[0].text)
        alternatives = [
            Alternative(cause=alt["cause"], rationale=alt["rationale"]) for alt in data["alternatives"]
        ]
        return AdversarialReview(
            alternatives=alternatives, strength_of_alternative=float(data["strength_of_alternative"])
        )
    except Exception as exc:  # noqa: BLE001 - any LLM failure must fall back, not crash the demo
        logger.warning("Adversarial LLM check failed (%s); falling back to synthetic review.", exc)
        return _synthetic_challenge(event, primary_classification)
