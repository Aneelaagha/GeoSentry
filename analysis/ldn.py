"""Land Degradation Neutrality (SDG 15.3.1) scoring and IPCC-2006 carbon-loss estimation.

Additive analysis layer only: reads existing ZoneBaseline/Detection rows and derives new
numbers from them. Does not modify detection logic, fusion math, GEE code, or LLM prompts.

Two spec inputs this module was asked to use don't actually exist in the pipeline, confirmed
by grep across the whole codebase before writing this file:
  - ZoneBaseline.trend_slope: no such column, and nothing anywhere computes a trend slope.
    gee/baselines.py's pooled baseline collapses each zone/month/indicator straight to a single
    median+MAD across sampled years - the year-by-year series a slope needs is never retained.
  - NDBI: never computed. ZoneBaseline.indicator only ever takes 'ndvi' | 'bsi' | 'viirs' | 'rain'.
Computing either for real means adding a new GEE reduction, which is out of scope for this
additive feature. Per explicit direction: productivity's sub-score falls back to the neutral
midpoint (50) until trend_slope exists, and the land-cover sub-score uses BSI alone (read from
the zone's own already-computed, already-persisted Detection.bsi_z - not recomputed here).
"""

from datetime import datetime, timedelta
from math import isnan
from typing import Optional

from sqlmodel import Session, select

from app.db import Detection, ZoneBaseline

# IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 4 (AFOLU), Chapter 4:
# biome-level default above-ground biomass (AGB) values, tonnes dry matter / ha.
AGB_DENSE_FOREST_T_HA = 200.0  # NDVI > 0.70
AGB_WOODLAND_T_HA = 100.0  # 0.50 <= NDVI <= 0.70
AGB_SHRUB_SAVANNA_T_HA = 40.0  # 0.30 <= NDVI < 0.50
AGB_GRASSLAND_T_HA = 10.0  # NDVI < 0.30

FOREST_CONTEXT_MULTIPLIER = 1.15  # closed-canopy bias
SAVANNA_CONTEXT_MULTIPLIER = 0.85

CARBON_FRACTION = 0.47  # IPCC 2006 default carbon fraction of dry matter
CO2_TO_CARBON_RATIO = 44 / 12  # molar mass ratio, CO2:C
FULL_LOSS_EMISSION_FACTOR = 1.0  # see estimate_carbon_loss_tco2e() docstring

PRODUCTIVITY_WEIGHT = 0.40
LAND_COVER_WEIGHT = 0.40
DETECTION_PENALTY_WEIGHT = 0.20

RECENT_DETECTION_WINDOW_DAYS = 365
LAND_COVER_SIGNAL_WINDOW_DAYS = 90


def estimate_agb_t_per_ha(ndvi_baseline_median: float, zone_context: str = "") -> float:
    """Estimate above-ground biomass (AGB) density from a zone's baseline NDVI median, using the
    IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 4, Chapter 4,
    biome-level default AGB values (tonnes dry matter / ha):
        NDVI > 0.70           -> 200.0  (dense tropical forest)
        0.50 <= NDVI <= 0.70  -> 100.0  (woodland / open forest)
        0.30 <= NDVI < 0.50   ->  40.0  (shrubland / savanna)
        NDVI < 0.30           ->  10.0  (grassland / bare ground)
    zone_context is matched case-insensitively: "forest"/"rainforest" applies a 1.15x
    closed-canopy multiplier, "savanna"/"shrub" applies 0.85x. Empty or non-matching context
    applies no adjustment.

    INPUTS: baseline NDVI median (float), zone context string (zone name/description; optional)
    OUTPUTS: AGB density in t/ha
    """
    if ndvi_baseline_median > 0.70:
        agb = AGB_DENSE_FOREST_T_HA
    elif ndvi_baseline_median >= 0.50:
        agb = AGB_WOODLAND_T_HA
    elif ndvi_baseline_median >= 0.30:
        agb = AGB_SHRUB_SAVANNA_T_HA
    else:
        agb = AGB_GRASSLAND_T_HA

    context = zone_context.lower()
    if "forest" in context or "rainforest" in context:
        agb *= FOREST_CONTEXT_MULTIPLIER
    elif "savanna" in context or "shrub" in context:
        agb *= SAVANNA_CONTEXT_MULTIPLIER

    return round(agb, 2)


def estimate_carbon_loss_tco2e(agb_t_per_ha: float, area_ha: float) -> float:
    """Carbon loss from vegetation removal, expressed as CO2-equivalent, using IPCC 2006
    Guidelines for National Greenhouse Gas Inventories, Volume 4, Chapter 4 defaults:
        carbon_stock    = AGB * 0.47             (IPCC default carbon fraction of dry matter)
        co2_equivalent  = carbon_stock * 44/12   (molar mass ratio CO2:C)
        total           = co2_equivalent * area_ha

    Assumes full AGB loss across the affected area (emission factor = 1.0). This overstates loss
    for selective logging, where ~0.5 would be more accurate, but this pipeline has no defensible
    way to differentiate cause-specific removal intensity at this stage - so the simple full-loss
    assumption is used and stated here rather than guessed at per-cause.

    INPUTS: AGB density (t/ha), affected area (ha)
    OUTPUTS: tCO2e (float)
    """
    carbon_stock_t_per_ha = agb_t_per_ha * CARBON_FRACTION
    co2_equivalent_t_per_ha = carbon_stock_t_per_ha * CO2_TO_CARBON_RATIO
    return round(co2_equivalent_t_per_ha * area_ha * FULL_LOSS_EMISSION_FACTOR, 2)


def _clamp_score(value: float) -> int:
    """INPUTS: a raw composite score. OUTPUTS: int clamped to [0, 100]; NaN maps to 50
    (neutral / insufficient data), per this feature's explicit guardrail."""
    if isnan(value):
        return 50
    return int(round(max(0.0, min(100.0, value))))


def _productivity_subscore(trend_slope: Optional[float]) -> float:
    """Maps an NDVI trend slope to a 0-100 productivity sub-score: slope <= -0.05 -> 0
    (declining), slope >= +0.05 -> 100 (improving), linear between (50 at zero slope).

    trend_slope is not currently computed anywhere in this pipeline (see module docstring) -
    None (or NaN) returns the neutral midpoint, not a fabricated slope."""
    if trend_slope is None or isnan(trend_slope):
        return 50.0
    if trend_slope <= -0.05:
        return 0.0
    if trend_slope >= 0.05:
        return 100.0
    return 50.0 + (trend_slope / 0.05) * 50.0


def _land_cover_subscore(bsi_z: Optional[float]) -> float:
    """Maps a BSI z-score - already computed by gee.detect (current - baseline.median) /
    (1.4826 * baseline.mad), the same convention used throughout the pipeline - to a 0-100
    land-cover-stability sub-score. Only a BSI *increase* (more bare-soil/impervious signal than
    baseline) counts as degradation; a decrease (regreening) is not penalized. An increase of
    >= 2 MAD -> 20, <= 1 MAD -> 100, linear between.

    NDBI is not computed anywhere in this pipeline (see module docstring), so this uses BSI
    alone - the closest already-real signal available without adding new GEE code. None (no
    Detection in the last 90 days) -> 100: no evidence of land-cover change, since the most
    recent real observation simply didn't flag one."""
    if bsi_z is None or isnan(bsi_z):
        return 100.0
    mad_units_increase = max(bsi_z, 0.0) / 1.4826
    if mad_units_increase >= 2.0:
        return 20.0
    if mad_units_increase <= 1.0:
        return 100.0
    return 100.0 - (mad_units_increase - 1.0) * 80.0


def _detection_penalty_subscore(detection_count: int) -> float:
    """0 Detection rows for this zone in the trailing 365 days -> 100, 1 -> 40, 2+ -> 10. Counts
    every Detection row regardless of alert status - even a silent pass means the pipeline
    observed a 3-sigma-flagged indicator, weighted lightly (only 20% of the composite) rather
    than ignored."""
    if detection_count <= 0:
        return 100.0
    if detection_count == 1:
        return 40.0
    return 10.0


def compute_ldn_score(zone_id: int, db: Session) -> dict:
    """Composite 0-100 Land Degradation Neutrality score, where 100 = perfect LDN, aligned to
    SDG Indicator 15.3.1's three sub-indicators (productivity, land cover, carbon stock proxy):
        40% productivity trend    <- NDVI trend slope (neutral 50: see module docstring)
        40% land cover stability  <- BSI z-score vs baseline, from the zone's most recent
                                      Detection within the last 90 days
        20% recent detection penalty <- count of Detection rows for this zone in the last 365d

    INPUTS: zone id (int), DB session
    OUTPUTS: {"ldn_score": int, "productivity": int, "land_cover": int,
              "detections_penalty": int, "computed_at": iso8601}
    """
    now = datetime.utcnow()

    trend_slope = None  # see module docstring: not computed anywhere in this pipeline
    productivity_score = _productivity_subscore(trend_slope)

    recent_detection = db.exec(
        select(Detection)
        .where(Detection.zone_id == zone_id)
        .order_by(Detection.detected_at.desc())
        .limit(1)
    ).first()
    bsi_z = None
    if recent_detection is not None and recent_detection.detected_at >= now - timedelta(
        days=LAND_COVER_SIGNAL_WINDOW_DAYS
    ):
        bsi_z = recent_detection.bsi_z
    land_cover_score = _land_cover_subscore(bsi_z)

    cutoff = now - timedelta(days=RECENT_DETECTION_WINDOW_DAYS)
    detection_count = len(
        db.exec(
            select(Detection.id).where(Detection.zone_id == zone_id, Detection.detected_at >= cutoff)
        ).all()
    )
    detections_penalty_score = _detection_penalty_subscore(detection_count)

    composite = (
        PRODUCTIVITY_WEIGHT * productivity_score
        + LAND_COVER_WEIGHT * land_cover_score
        + DETECTION_PENALTY_WEIGHT * detections_penalty_score
    )

    return {
        "ldn_score": _clamp_score(composite),
        "productivity": int(round(productivity_score)),
        "land_cover": int(round(land_cover_score)),
        "detections_penalty": int(round(detections_penalty_score)),
        "computed_at": now.isoformat() + "Z",
    }


def estimate_zone_carbon_loss(zone_id: int, db: Session) -> dict:
    """Sum estimated CO2 loss across every Detection row for a zone that has both an area_ha and
    a resolvable baseline NDVI median (the zone's ZoneBaseline row for that detection's calendar
    month). Detections without a usable area or baseline are skipped, not zero-filled - this is
    a sum over what's actually computable, not an estimate for the zone's whole history.

    INPUTS: zone id (int), DB session
    OUTPUTS: {"total_tco2e": float, "event_count": int}
    """
    detections = db.exec(
        select(Detection).where(Detection.zone_id == zone_id, Detection.estimated_area_ha > 0)
    ).all()

    baseline_median_by_month: dict[int, Optional[float]] = {}
    total_tco2e = 0.0
    event_count = 0
    for detection in detections:
        month = detection.detected_at.month
        if month not in baseline_median_by_month:
            baseline_row = db.exec(
                select(ZoneBaseline).where(
                    ZoneBaseline.zone_id == zone_id,
                    ZoneBaseline.calendar_month == month,
                    ZoneBaseline.indicator == "ndvi",
                )
            ).first()
            baseline_median_by_month[month] = baseline_row.median if baseline_row is not None else None

        ndvi_median = baseline_median_by_month[month]
        if ndvi_median is None:
            continue

        agb_t_per_ha = estimate_agb_t_per_ha(ndvi_median)
        total_tco2e += estimate_carbon_loss_tco2e(agb_t_per_ha, detection.estimated_area_ha)
        event_count += 1

    return {"total_tco2e": round(total_tco2e, 2), "event_count": event_count}
