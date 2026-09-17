EPSILON = 1e-9  # keeps prior/posterior odds finite at the 0/1 boundary


def _to_odds(p: float) -> float:
    p = min(max(p, EPSILON), 1 - EPSILON)
    return p / (1 - p)


def _to_prob(odds: float) -> float:
    return odds / (1 + odds)


def llm_likelihood_ratio(conf: float) -> float:
    """Magnitude-scaled. Below 0.5 confidence is evidence AGAINST the cause.
    INPUTS: conf - the LLM classifier's confidence in its top cause (float 0-1).
    OUTPUTS: float likelihood ratio. Below 0.5, scales down toward 0.1 (a hedged classification
    should drag the posterior down, not push it up); at/above 0.5, scales up toward 8.5 at
    conf=1.0. Replaces the old step function (>0.7 -> x8, else x2), which treated a 0.32-
    confidence call the same as a 0.69-confidence one and let both round up to "supporting"."""
    if conf < 0.5:
        return max(0.1, conf / 0.5)
    return 1.0 + (conf - 0.5) * 15


def viirs_likelihood_ratio(z: float) -> float:
    """Magnitude-scaled, capped. z=0 neutral, z=2 mid, z=5+ strong.
    INPUTS: z - VIIRS night-light z-score vs baseline (float).
    OUTPUTS: float likelihood ratio in [0.5, 5.0]. Replaces the old step function (>2 -> x5, else
    x1), which made z=2.01 a cliff-edge x5 and z=1.99 a no-op."""
    return max(0.5, min(5.0, 1.0 + z))


def area_likelihood_ratio(area_ha: float) -> float:
    """INPUTS: area_ha - estimated affected area in hectares (float).
    OUTPUTS: float likelihood ratio: 2.0 above 10 ha, 1.2 between 2 and 10 ha, 0.8 at/below 2 ha
    (a sub-minimum area is mild evidence against a real disturbance, not neutral)."""
    if area_ha > 10:
        return 2.0
    if area_ha > 2:
        return 1.2
    return 0.8


def rain_likelihood_ratio(rain_pct: float) -> float:
    """Rain above the 30th percentile rules OUT drought, supporting a non-drought cause. Below
    the 30th percentile, drought is a live alternative.
    INPUTS: rain_pct - trailing rainfall percentile (float 0-100).
    OUTPUTS: float likelihood ratio: 2.0 if rain_pct > 30, else 0.5."""
    return 2.0 if rain_pct > 30 else 0.5


def fuse_confidence(prior: float, evidence: dict[str, float]) -> float:
    """INPUTS: prior (float 0-1) - the base-rate probability that a monitored zone has a real,
    actionable disturbance before any corroborating evidence is weighed. evidence - a dict of
    raw signal values, any subset of:
      - llm_conf: the LLM classifier's confidence (0-1) -> llm_likelihood_ratio()
      - viirs_z: VIIRS night-light z-score vs baseline -> viirs_likelihood_ratio(). May be None
        (gee.detect.current_viirs() found no VIIRS monthly composite for this zone/month - a
        real 1-2 month data lag) as well as simply absent from the dict; both mean the same
        thing here and are treated identically.
      - rain_percentile: trailing rainfall percentile (0-100) -> rain_likelihood_ratio()
      - area_ha: estimated affected area in hectares -> area_likelihood_ratio()
    A key absent from evidence, or present as None, contributes no update (a neutral x1) - VIIRS
    is a corroborator, not a required signal, so a missing composite shrinks the evidence set
    rather than blocking detection. Deliberately NOT compensated for by boosting the other
    ratios: one fewer corroborator means a lower, more honest posterior, not the same posterior
    reached a different way. OUTPUTS: posterior probability (float 0-1) - prior odds multiplied
    by each available signal's likelihood ratio (naive Bayes fusion), converted back to a
    probability and clamped to [0, 1]. All four ratios are magnitude-scaled (not step
    functions), so a hedged LLM call or a barely-over-threshold signal no longer swings the
    posterior as hard as a decisive one."""
    odds = _to_odds(prior)

    if "llm_conf" in evidence:
        odds *= llm_likelihood_ratio(evidence["llm_conf"])
    if evidence.get("viirs_z") is not None:
        odds *= viirs_likelihood_ratio(evidence["viirs_z"])
    if "rain_percentile" in evidence:
        odds *= rain_likelihood_ratio(evidence["rain_percentile"])
    if "area_ha" in evidence:
        odds *= area_likelihood_ratio(evidence["area_ha"])

    return min(max(_to_prob(odds), 0.0), 1.0)


def classify_alert_tier(llm_conf: float, fused_posterior: float, threshold: float) -> str:
    """
    Returns one of:
      'silent'          -- below gate
      'uncertain'       -- above gate but LLM conf < 0.5
      'classified'      -- above gate with confident cause
    INPUTS: llm_conf - the LLM classifier's raw confidence in its top cause (float 0-1, before
    any adversarial tempering); fused_posterior - fuse_confidence()'s output (float 0-1);
    threshold - the alert gate (float 0-1, typically settings.alert_confidence_threshold).
    OUTPUTS: str, one of 'silent' | 'uncertain' | 'classified'."""
    if fused_posterior < threshold:
        return "silent"
    if llm_conf < 0.5:
        return "uncertain"
    return "classified"


def explain_silence(detection: dict, threshold: float = 0.72) -> str:
    """INPUTS: detection - a dict with keys llm_confidence, viirs_delta, rainfall_pct, area_ha,
    fused_confidence (the same fields the /silent-log route assembles per detection row);
    threshold - the alert confidence gate the detection fell below (default 0.72, matching
    settings.alert_confidence_threshold). OUTPUTS: a one-line, human-readable reason this
    detection stayed silent, picked by checking which piece of evidence was weakest, in priority
    order: an ambiguous LLM call, a missing/flat VIIRS night-light corroboration, a rainfall
    signal that argues for drought instead, a sub-minimum affected area, or - if none of those
    stand out - the fused confidence simply falling short of the gate. viirs_delta may be None
    (gee.detect.current_viirs() found no VIIRS composite for that zone/month) - reported as its
    own reason, distinct from a real, measured flat reading."""
    llm_confidence = detection["llm_confidence"]
    viirs_delta = detection["viirs_delta"]
    rainfall_pct = detection["rainfall_pct"]
    area_ha = detection["area_ha"]
    fused_confidence = detection["fused_confidence"]

    if llm_confidence < 0.5:
        return f"LLM confidence only {round(llm_confidence * 100)}% - cause ambiguous"
    if viirs_delta is None:
        return "VIIRS unavailable this window - no night-light corroboration possible"
    if viirs_delta < 1.0:
        return f"VIIRS flat ({viirs_delta:+.2f}z) - no extraction-like night activity to corroborate"
    if rainfall_pct < 30:
        return f"Rainfall {round(rainfall_pct)}pct - drought signal conflicts with extraction hypothesis"
    if area_ha < 2:
        return f"Affected area {area_ha:.1f} ha below minimum"
    return f"Fused confidence {round(fused_confidence * 100)}% below {round(threshold * 100)}% gate"
