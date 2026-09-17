import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import NamedTuple, Optional

from sqlmodel import Session

from app.db import Zone
from gee.baselines import BaselineStats, INDICATOR_REDUCE_SCALE_M, get_or_compute_baseline
from gee.indices import add_indices, mask_s2_clouds, prepare_s2
from gee.init import ee_available

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger("geosentry.gee")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
S2_SCALE_M = 10
VIIRS_SCALE_M = 500
LOOKBACK_DAYS = 30  # window for "current" S2 conditions, wide enough to beat cloud gaps
Z_THRESHOLD = 3.0
MIN_CANDIDATE_AREA_HA = 1.0
MAX_ABS_Z = 50.0  # last-resort backstop: even a well-chosen MAD floor (gee.baselines.MAD_FLOORS)
# can't rule out an absurd z-score for every zone/indicator combination, so every z-score gets
# clamped to this range right before it's stored or used downstream.


def _clamp_z(z: float, context: str) -> float:
    """INPUTS: a raw z-score, a short context string identifying what's being clamped (for the
    log line). OUTPUTS: z clamped to [-MAX_ABS_Z, MAX_ABS_Z]. Logs a warning when clamping
    actually changes the value - this is a backstop against a near-zero baseline MAD amplifying
    a tiny real deviation into an absurd z-score (observed: -4.19e12 for a VIIRS baseline whose
    MAD had collapsed to floating-point noise), not a substitute for a correctly floored MAD."""
    clamped = max(-MAX_ABS_Z, min(MAX_ABS_Z, z))
    if clamped != z:
        logger.warning("z-score clamped for %s: raw=%.4f -> %.1f", context, z, clamped)
    return clamped


class ChangeIndicators(NamedTuple):
    ndvi_z: float
    bsi_z: float
    viirs_z: float
    estimated_area_ha: float


def _synthetic_indicators(zone_name: str, zone_type: str) -> ChangeIndicators:
    """INPUTS: zone_name, zone_type (mining/logging/control). OUTPUTS: deterministic
    ChangeIndicators tuned so 'mining' and 'logging' zones trip the alert pipeline and
    'control' stays quiet - lets the demo reliably show both a silent and an alerted path."""
    seed = int(hashlib.sha256(zone_name.encode()).hexdigest()[:8], 16)
    jitter = (seed % 20) / 100  # 0.0-0.19
    if zone_type == "mining":
        return ChangeIndicators(
            ndvi_z=-3.4 - jitter,
            bsi_z=3.8 + jitter,
            viirs_z=2.6 + jitter,
            estimated_area_ha=42.5 + seed % 30,
        )
    if zone_type == "logging":
        return ChangeIndicators(
            ndvi_z=-4.1 - jitter,
            bsi_z=2.1 + jitter,
            viirs_z=0.4 + jitter,
            estimated_area_ha=68.0 + seed % 50,
        )
    return ChangeIndicators(
        ndvi_z=-0.3 + jitter,
        bsi_z=0.2 + jitter,
        viirs_z=0.1 + jitter,
        estimated_area_ha=0.0,
    )


def _z_score(value: Optional[float], median: float, mad: float, context: str = "") -> float:
    """INPUTS: an observed value (or None if Earth Engine had no valid pixels), the baseline
    median/MAD, an optional context string for _clamp_z()'s log line. OUTPUTS: robust z-score
    (value-median)/(1.4826*MAD), clamped to [-MAX_ABS_Z, MAX_ABS_Z]; 0.0 if value is missing."""
    if value is None:
        return 0.0
    return _clamp_z((value - median) / (1.4826 * mad), context)


def _current_composite(geometry, collection_id: str, band_select, prep=None):
    """INPUTS: ee.Geometry, source collection id, band(s) to select, optional per-image prep
    function. OUTPUTS: server-side ee.Image - the median composite of that collection over the
    last LOOKBACK_DAYS days within the zone."""
    now = datetime.utcnow()
    start = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    collection = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(geometry)
    if prep is not None:
        collection = collection.map(prep)
    return collection.select(band_select).median()


def detect_change(
    zone_name: str, zone_type: str, baseline: BaselineStats, aoi_geojson: Optional[str] = None
) -> ChangeIndicators:
    """INPUTS: zone_name, zone_type, the zone's current BaselineStats, and the zone's AOI GeoJSON
    (required for the real Earth Engine path). OUTPUTS: ChangeIndicators - z-scores for
    NDVI/BSI/VIIRS vs baseline, plus the affected area in hectares among pixels flagged as a
    change candidate (|z_NDVI| > 3 OR |z_BSI| > 3), zeroed out if that area is under the 1 ha
    minimum (noise floor). Falls back to deterministic synthetic values when Earth Engine is
    unavailable."""
    if not ee_available():
        return _synthetic_indicators(zone_name, zone_type)

    if not aoi_geojson:
        raise ValueError("aoi_geojson is required to run real Earth Engine change detection.")

    geometry = ee.Geometry(json.loads(aoi_geojson))

    s2_composite = _current_composite(geometry, S2_COLLECTION, ["NDVI", "BSI"], prepare_s2)
    viirs_composite = _current_composite(geometry, VIIRS_COLLECTION, "avg_rad")

    ndvi_z_image = s2_composite.select("NDVI").subtract(baseline.ndvi_median).divide(1.4826 * baseline.ndvi_mad)
    bsi_z_image = s2_composite.select("BSI").subtract(baseline.bsi_median).divide(1.4826 * baseline.bsi_mad)
    candidate_mask = ndvi_z_image.abs().gt(Z_THRESHOLD).Or(bsi_z_image.abs().gt(Z_THRESHOLD))

    area_image = ee.Image.pixelArea().updateMask(candidate_mask).clip(geometry)
    area_stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=S2_SCALE_M, maxPixels=1e9, bestEffort=True
    )

    zone_means = s2_composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=S2_SCALE_M, maxPixels=1e9, bestEffort=True
    )
    viirs_mean = viirs_composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=VIIRS_SCALE_M, maxPixels=1e9, bestEffort=True
    )

    resolved = ee.Dictionary(
        {
            "ndvi_mean": zone_means.get("NDVI"),
            "bsi_mean": zone_means.get("BSI"),
            "viirs_mean": viirs_mean.get("avg_rad"),
            "area_m2": area_stats.get("area"),
        }
    ).getInfo()

    ndvi_z = _z_score(resolved.get("ndvi_mean"), baseline.ndvi_median, baseline.ndvi_mad, f"{zone_name}/ndvi")
    bsi_z = _z_score(resolved.get("bsi_mean"), baseline.bsi_median, baseline.bsi_mad, f"{zone_name}/bsi")
    viirs_z = _z_score(resolved.get("viirs_mean"), baseline.viirs_median, baseline.viirs_mad, f"{zone_name}/viirs")

    area_ha = (resolved.get("area_m2") or 0.0) / 10000
    if area_ha < MIN_CANDIDATE_AREA_HA:
        area_ha = 0.0

    return ChangeIndicators(
        ndvi_z=round(ndvi_z, 4),
        bsi_z=round(bsi_z, 4),
        viirs_z=round(viirs_z, 4),
        estimated_area_ha=round(area_ha, 2),
    )


# =============================================================================================
# Stage 3: real, no-synthetic-fallback detection. Everything above this line (ChangeIndicators,
# detect_change, ee_available()-gated) is untouched and still drives scripts/run_once.py's
# default demo path. Everything below is new, always-real, and raises instead of ever
# synthesizing - used only when a caller explicitly opts in (scripts/detect_once.py, or
# scripts/run_once.py --as-of).
# =============================================================================================


class NoValidPixels(Exception):
    pass


def _s2_prepared_window(zone_geom, days: int, as_of: Optional[datetime]):
    """INPUTS: ee.Geometry, window length in days, optional as-of end date (defaults to now).
    OUTPUTS: server-side ee.ImageCollection - cloud-masked S2 scenes with 'ndvi'/'bsi' bands
    added, over [end-days, end) intersecting the zone. Not yet reduced; shared by
    current_observation() and the candidate-area pixel mask below."""
    end = as_of or datetime.utcnow()
    start = end - timedelta(days=days)
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filterBounds(zone_geom)
        .map(mask_s2_clouds)
        .map(add_indices)
    )


def current_observation(zone_geom, days: int = 14, as_of: Optional[datetime] = None) -> dict:
    """Median NDVI and BSI over the last `days` of cloud-masked S2 imagery.
    Raise NoValidPixels if the collection is empty after cloud filtering.
    INPUTS: ee.Geometry, int days
    OUTPUTS: dict {ndvi: float, bsi: float, image_count: int}"""
    end = as_of or datetime.utcnow()
    start = end - timedelta(days=days)
    collection = _s2_prepared_window(zone_geom, days, as_of)

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise NoValidPixels(
            f"No S2 images intersecting the zone for [{start.date()}, {end.date()})."
        )

    composite = collection.select(["ndvi", "bsi"]).median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=zone_geom,
        scale=INDICATOR_REDUCE_SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    ndvi, bsi = stats.get("ndvi"), stats.get("bsi")
    if ndvi is None or bsi is None:
        raise NoValidPixels(
            f"{image_count} S2 image(s) found for [{start.date()}, {end.date()}) but every "
            "pixel over the zone was cloud-masked out."
        )

    return {"ndvi": ndvi, "bsi": bsi, "image_count": image_count}


def current_viirs(zone_geom, as_of: Optional[datetime] = None) -> Optional[float]:
    """Current month avg_rad over the zone, or None if no VIIRS DNB monthly composite exists for
    that month yet.

    Returns None rather than raising because VIIRS is a corroboration signal, not a primary
    detection signal - the detection pipeline should proceed with other evidence (NDVI/BSI/
    rainfall) when VIIRS is unavailable, with reduced confidence because one corroborator is
    missing (see fusion.bayes.fuse_confidence's viirs_z=None handling). VIIRS DNB monthly
    composites lag real time by 1-2 months, so any --as-of date in the current or previous
    calendar month routinely hits this - it's an expected data gap, not an exceptional error.

    Root cause this replaces: reduceRegion() over a composite with zero source images returns a
    dict that's missing the 'avg_rad' key entirely (not present-with-null) - ee.Dictionary.get()
    on a truly absent key raises server-side (EEException: "Dictionary does not contain key").
    Evaluating the whole stats dict with .getInfo() FIRST, then reading it with a plain Python
    dict.get() - the same pattern current_observation() above already uses - sidesteps that: a
    missing key just returns None, the ordinary Python way, no exception.

    INPUTS: ee.Geometry, optional as_of date
    OUTPUTS: float (avg_rad) or None
    """
    from gee.baselines import _month_date_range  # local: avoids a module-level cycle, see below

    reference = as_of or datetime.utcnow()
    start, end = _month_date_range(reference.year, reference.month)
    collection = ee.ImageCollection(VIIRS_COLLECTION).filterDate(start, end).filterBounds(zone_geom)
    composite = collection.select("avg_rad").median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.median(), geometry=zone_geom, scale=VIIRS_SCALE_M, maxPixels=1e9, bestEffort=True
    ).getInfo()

    return stats.get("avg_rad") if stats else None


def _candidate_area_ha(zone_geom, indicator: str, baseline, days: int, as_of: Optional[datetime]) -> float:
    """INPUTS: ee.Geometry, the flagged indicator ('ndvi'|'bsi'), its ZoneBaseline row, the
    current_observation window length, as-of date. OUTPUTS: float hectares - ee.Image.pixelArea()
    summed over pixels where this indicator's current composite deviates from baseline.median by
    more than Z_THRESHOLD * 1.4826 * baseline.mad."""
    collection = _s2_prepared_window(zone_geom, days, as_of)
    image = collection.select(indicator).median()
    z_image = image.subtract(baseline.median).divide(1.4826 * baseline.mad)
    mask = z_image.abs().gt(Z_THRESHOLD)

    area_image = ee.Image.pixelArea().updateMask(mask).clip(zone_geom)
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=zone_geom, scale=INDICATOR_REDUCE_SCALE_M, maxPixels=1e9, bestEffort=True
    )
    area_m2 = stats.get("area").getInfo()
    return round((area_m2 or 0.0) / 10000, 4)


def detect_candidates(zone: Zone, db: Session, as_of: Optional[datetime] = None) -> list[dict]:
    """For ndvi and bsi:
       - current = current_observation(zone.geom)
       - baseline = get_or_compute_baseline(zone, current_month, indicator, db)
       - z = (current - baseline.median) / (1.4826 * baseline.mad)
       - flag if abs(z) > 3
     If flagged:
       - compute affected area via pixelArea() masked by pixels where the
         current composite exceeds 3-sigma vs baseline
       - compute viirs_z = (current_viirs - viirs_baseline) / (1.4826*viirs_mad), or None if
         current_viirs() found no VIIRS monthly composite for this zone/month (a real 1-2 month
         data lag, not an error - VIIRS is a corroborator, not a required signal)
       - compute rainfall_percentile over last 30 days
     Return list of candidate dicts:
       {indicator, current, baseline_median, z, area_ha, viirs_z, viirs_available, rain_pct,
        carbon_loss_tco2e}
     viirs_z is None (and viirs_available False) exactly when VIIRS was unavailable - never a
     fabricated 0.0, which would falsely read as "VIIRS confirms no night-light activity."
     If nothing flagged, return empty list.
    INPUTS: Zone row, DB session
    OUTPUTS: list[dict]

    Note: raises NoValidPixels straight out of current_observation() if there's no usable
    current-window imagery - never caught here, per the no-synthetic-fallback rule for this
    stage. Callers (scripts/detect_once.py, scripts/run_once.py --as-of) catch it themselves.

    carbon_loss_tco2e (analysis.ldn.estimate_carbon_loss_tco2e, IPCC 2006 AGB defaults) is
    derived from the zone's own NDVI baseline median - a biomass proxy, independent of which
    indicator actually triggered the flag - and this candidate's area_ha. Purely additive: reads
    the ndvi baseline already fetched by this same loop's ndvi pass (or fetches it once if ndvi
    itself never got flagged), never changes the z-score/threshold flagging logic above. None if
    the ndvi baseline is unavailable, never guessed."""
    # Deferred import: gee/corroborate.py imports NoValidPixels + current_viirs from this module
    # at ITS module level, so importing corroborate.py back at THIS module's level would cycle.
    # Doing it here (inside the function body, evaluated at call time not import time) breaks
    # the cycle without duplicating rainfall_percentile's logic in two places.
    from gee.corroborate import rainfall_percentile

    from analysis.ldn import estimate_agb_t_per_ha, estimate_carbon_loss_tco2e

    zone_geom = ee.Geometry(json.loads(zone.aoi_geojson))
    reference_date = as_of or datetime.utcnow()
    calendar_month = reference_date.month

    observation = current_observation(zone_geom, days=14, as_of=as_of)

    candidates = []
    ndvi_baseline_median: Optional[float] = None
    for indicator in ("ndvi", "bsi"):
        current_value = observation[indicator]
        baseline = get_or_compute_baseline(zone, calendar_month, indicator, db)
        if indicator == "ndvi":
            ndvi_baseline_median = baseline.median
        z = (current_value - baseline.median) / (1.4826 * baseline.mad) if baseline.mad else 0.0
        z = _clamp_z(z, f"{zone.name}/{indicator}")

        if abs(z) <= Z_THRESHOLD:
            continue

        area_ha = _candidate_area_ha(zone_geom, indicator, baseline, 14, as_of)

        viirs_value = current_viirs(zone_geom, as_of=as_of)
        viirs_available = viirs_value is not None
        viirs_z = None
        if viirs_available:
            viirs_baseline = get_or_compute_baseline(zone, calendar_month, "viirs", db)
            viirs_z = (
                (viirs_value - viirs_baseline.median) / (1.4826 * viirs_baseline.mad)
                if viirs_baseline.mad
                else 0.0
            )
            viirs_z = round(_clamp_z(viirs_z, f"{zone.name}/viirs"), 4)

        rain_pct = rainfall_percentile(zone_geom, days=30, as_of=as_of)

        carbon_loss_tco2e = None
        if ndvi_baseline_median is not None and area_ha > 0:
            agb_t_per_ha = estimate_agb_t_per_ha(ndvi_baseline_median, zone.name)
            carbon_loss_tco2e = estimate_carbon_loss_tco2e(agb_t_per_ha, area_ha)

        candidates.append(
            {
                "indicator": indicator,
                "current": round(current_value, 4),
                "baseline_median": round(baseline.median, 4),
                "z": round(z, 4),
                "area_ha": area_ha,
                "viirs_z": viirs_z,  # None means "unavailable", never a fabricated 0.0
                "viirs_available": viirs_available,
                "rain_pct": round(rain_pct, 2),
                "carbon_loss_tco2e": carbon_loss_tco2e,
            }
        )

    return candidates
