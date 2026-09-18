import hashlib
import json
import statistics
from datetime import datetime, timedelta
from typing import NamedTuple, Optional

from sqlmodel import Session, select

from app.db import Zone, ZoneBaseline
from gee.indices import add_indices, mask_s2_clouds, prepare_s2
from gee.init import ee_available

try:
    import ee
except ImportError:
    ee = None

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
S2_SCALE_M = 10
VIIRS_SCALE_M = 500
MAD_FLOOR = 1e-6  # avoids a divide-by-zero z-score when a sample happens to be constant - used
# directly by _median_mad() below (the synthetic-aware compute_zone_baseline() path) and as
# _mad()'s default floor when an indicator isn't in MAD_FLOORS.
MAD_FLOORS = {
    "ndvi": 1e-6,
    "bsi": 1e-6,
    "viirs": 0.05,  # nW/cm^2/sr - a shared 1e-6 floor let a near-invariant VIIRS baseline
    # (3 sampled years landing within floating-point noise of each other) blow up a normal
    # deviation into a 4-digit z-score; VIIRS's own units need a floor sized to them, not
    # NDVI/BSI's unitless ~O(1) scale.
    "rain": 1.0,  # mm
}

# Stage 2: real per-indicator baseline cache (ZoneBaseline). Distinct from BaselineStats /
# compute_zone_baseline above, which is the pre-existing synthetic-aware, all-4-indicators-at-once
# path that scripts/run_once.py still drives the live demo with.
INDICATOR_REDUCE_SCALE_M = 20
BASELINE_CACHE_DAYS = 30


class BaselineStats(NamedTuple):
    ndvi_median: float
    ndvi_mad: float
    bsi_median: float
    bsi_mad: float
    viirs_median: float
    viirs_mad: float
    sample_years: str


def _synthetic_seed(zone_name: str, month: int) -> int:
    """INPUTS: zone name, calendar month. OUTPUTS: deterministic int derived from a hash,
    so synthetic baselines are stable across runs for the same zone+month."""
    digest = hashlib.sha256(f"{zone_name}-{month}".encode()).hexdigest()
    return int(digest[:8], 16)


def _synthetic_baseline(zone_name: str, month: int) -> BaselineStats:
    seed = _synthetic_seed(zone_name, month)
    ndvi_median = 0.55 + (seed % 100) / 1000  # healthy-vegetation baseline, ~0.55-0.65
    bsi_median = 0.05 + (seed % 50) / 1000
    viirs_median = 0.2 + (seed % 30) / 100
    return BaselineStats(
        ndvi_median=round(ndvi_median, 4),
        ndvi_mad=0.03,
        bsi_median=round(bsi_median, 4),
        bsi_mad=0.01,
        viirs_median=round(viirs_median, 4),
        viirs_mad=0.05,
        sample_years="synthetic",
    )


def _month_date_range(year: int, month: int) -> tuple[str, str]:
    """INPUTS: calendar year, calendar month (1-12). OUTPUTS: (start, end) ISO date strings
    spanning that whole calendar month, for filterDate()."""
    start = datetime(year, month, 1)
    end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = datetime(end_year, end_month, 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _zone_geometry(aoi_geojson: str):
    """INPUTS: a GeoJSON geometry string (as stored on Zone.aoi_geojson).
    OUTPUTS: the matching ee.Geometry."""
    return ee.Geometry(json.loads(aoi_geojson))


def _median_mad(samples: list[float]) -> tuple[float, float]:
    """INPUTS: list of float samples. OUTPUTS: (median, MAD) with MAD floored above zero so a
    downstream z-score never divides by zero. Empty input returns (0.0, MAD_FLOOR)."""
    if not samples:
        return 0.0, MAD_FLOOR
    median = statistics.median(samples)
    mad = statistics.median([abs(s - median) for s in samples])
    return median, max(mad, MAD_FLOOR)


def _yearly_zone_mean(geometry, year: int, month: int, band: str, collection_id: str, scale: int, prep=None):
    """INPUTS: ee.Geometry, calendar year+month, band to reduce, source collection id, reduction
    scale (m), optional per-image prep function (e.g. cloud-mask + index bands).
    OUTPUTS: server-side ee.Number (or None) - the zone-mean of `band`'s monthly median
    composite for that single year+month. Left un-evaluated so callers can batch multiple of
    these into one getInfo() round-trip."""
    start, end = _month_date_range(year, month)
    collection = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(geometry)
    if prep is not None:
        collection = collection.map(prep)
    composite = collection.select(band).median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=1e9, bestEffort=True
    )
    return stats.get(band)


def compute_zone_baseline(
    zone_name: str, month: int, aoi_geojson: Optional[str] = None, years: Optional[list[int]] = None
) -> BaselineStats:
    """INPUTS: zone_name (str), calendar month (1-12), the zone's AOI GeoJSON (required for the
    real Earth Engine path), years to sample (default: the previous 3 calendar years).
    OUTPUTS: BaselineStats - median/MAD of NDVI, BSI, and VIIRS radiance for that zone+month,
    one sample per sampled year, computed over the sample years. Falls back to deterministic
    synthetic values when Earth Engine is unavailable, so this function always returns and
    never blocks the demo."""
    if not ee_available():
        return _synthetic_baseline(zone_name, month)

    if not aoi_geojson:
        raise ValueError("aoi_geojson is required to compute a real Earth Engine baseline.")

    years = years or [datetime.utcnow().year - offset for offset in (3, 2, 1)]
    geometry = _zone_geometry(aoi_geojson)

    per_year = {}
    for year in years:
        per_year[year] = ee.Dictionary(
            {
                "ndvi": _yearly_zone_mean(geometry, year, month, "NDVI", S2_COLLECTION, S2_SCALE_M, prepare_s2),
                "bsi": _yearly_zone_mean(geometry, year, month, "BSI", S2_COLLECTION, S2_SCALE_M, prepare_s2),
                "viirs": _yearly_zone_mean(geometry, year, month, "avg_rad", VIIRS_COLLECTION, VIIRS_SCALE_M),
            }
        )
    # One getInfo() round-trip for every sampled year, instead of one per band per year.
    resolved = ee.Dictionary(per_year).getInfo()

    ndvi_samples, bsi_samples, viirs_samples = [], [], []
    for year_stats in resolved.values():
        if year_stats.get("ndvi") is not None:
            ndvi_samples.append(year_stats["ndvi"])
        if year_stats.get("bsi") is not None:
            bsi_samples.append(year_stats["bsi"])
        if year_stats.get("viirs") is not None:
            viirs_samples.append(year_stats["viirs"])

    ndvi_median, ndvi_mad = _median_mad(ndvi_samples)
    bsi_median, bsi_mad = _median_mad(bsi_samples)
    viirs_median, viirs_mad = _median_mad(viirs_samples)

    return BaselineStats(
        ndvi_median=round(ndvi_median, 4),
        ndvi_mad=round(ndvi_mad, 4),
        bsi_median=round(bsi_median, 4),
        bsi_mad=round(bsi_mad, 4),
        viirs_median=round(viirs_median, 4),
        viirs_mad=round(viirs_mad, 4),
        sample_years=",".join(str(y) for y in years),
    )


def _mad(values: list[float], indicator: Optional[str] = None) -> float:
    """Median absolute deviation. INPUTS: list[float], optional indicator name ('ndvi' | 'bsi' |
    'viirs' | 'rain' - whichever this MAD is being computed for). OUTPUTS: float, floored at
    MAD_FLOORS[indicator] when indicator is a known key, else the 1e-6 default - always floored,
    so a caller can't accidentally get an unfloored value back by omitting indicator.

    Per-indicator floors matter because indicators have very different natural units/scales:
    a shared 1e-6 floor stopped the divide-by-zero crash, but still let a near-invariant VIIRS
    baseline (3 sampled years landing within floating-point noise of each other, ~1e-15) blow up
    a normal real-world deviation into an absurd z-score (observed: -4.19e12 for Yanomami Mining
    Belt / VIIRS / 2025-09-15) - VIIRS avg_rad and CHIRPS rainfall need floors sized to their own
    units, not NDVI/BSI's unitless ~O(1) scale."""
    if not values:
        return 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    floor = MAD_FLOORS.get(indicator, 1e-6)
    return max(mad, floor)


def _indicator_yearly_sample(zone_geom, year: int, calendar_month: int, indicator: str):
    """INPUTS: ee.Geometry, calendar year+month, indicator name ('viirs'|'rain'). OUTPUTS:
    server-side ee.Number (or None) - that indicator's representative value over the zone for
    that single year+calendar_month: a median VIIRS monthly composite for viirs, or a summed
    CHIRPS month for rain. Left un-evaluated so compute_baseline() can batch every sampled year
    into one getInfo() round-trip. ndvi/bsi are NOT handled here - see _pooled_s2_baseline()."""
    start, end = _month_date_range(year, calendar_month)

    if indicator == "viirs":
        collection = ee.ImageCollection(VIIRS_COLLECTION).filterDate(start, end).filterBounds(zone_geom)
        image = collection.select("avg_rad").median()
        band = "avg_rad"
    elif indicator == "rain":
        collection = ee.ImageCollection(CHIRPS_COLLECTION).filterDate(start, end).filterBounds(zone_geom)
        image = collection.select("precipitation").sum()
        band = "precipitation"
    else:
        raise ValueError(f"unknown indicator: {indicator!r}")

    stats = image.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=zone_geom,
        scale=INDICATOR_REDUCE_SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    )
    return stats.get(band)


def _pooled_s2_baseline(zone_geom, years: list[int], calendar_month: int, indicator: str) -> dict:
    """INPUTS: ee.Geometry, the calendar years to pool, calendar month, indicator ('ndvi' or
    'bsi'). OUTPUTS: {median, mad, sample_count} computed from every cloud-masked S2 pixel
    observation across every image in every sampled year's window for that calendar month -
    NOT from years_back yearly composite scalars.

    Why: collapsing each year to one median-composite scalar first, then taking median+MAD
    across just 3 yearly scalars, is fragile - if two of the three years' composites land close
    together (common in practice), the median-of-deviations collapses toward zero regardless of
    how far the third year sits, understating real inter-annual NDVI/BSI variability by 1-2
    orders of magnitude (see scripts/debug_baseline.py, which caught exactly this on Kambove).
    Pooling every image's pixels first gives dozens of independent observations instead of 3,
    which is stable. sample_count here is the number of S2 images pooled, not years."""
    windows = [_month_date_range(year, calendar_month) for year in years]
    date_filter = ee.Filter.Or(*[ee.Filter.date(start, end) for start, end in windows])

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(zone_geom)
        .filter(date_filter)
        .map(mask_s2_clouds)
        .map(add_indices)
        .select(indicator)
    )
    count = collection.size()

    # Per-pixel temporal median/MAD across every pooled image, then one spatial median over the
    # zone to collapse to the single scalar pair ZoneBaseline stores.
    pixel_median = collection.reduce(ee.Reducer.median())
    pixel_deviation = collection.map(lambda img: img.subtract(pixel_median).abs())
    pixel_mad = pixel_deviation.reduce(ee.Reducer.median())

    combined = pixel_median.rename("median_val").addBands(pixel_mad.rename("mad_val"))
    stats = combined.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=zone_geom,
        scale=INDICATOR_REDUCE_SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    )

    resolved = ee.Dictionary({"stats": stats, "count": count}).getInfo()
    stats_dict = resolved["stats"]
    median = stats_dict.get("median_val")
    mad = stats_dict.get("mad_val")

    return {
        "median": median if median is not None else 0.0,
        "mad": mad if mad is not None else 0.0,
        "sample_count": int(resolved.get("count") or 0),
    }


def compute_baseline(zone_geom, calendar_month: int, indicator: str, years_back: int = 3) -> dict:
    """Compute median + MAD for the same calendar month across the previous
    `years_back` years.
    - ndvi/bsi: S2 SR, cloud-masked, pooled per-pixel across every image in every sampled year
      (not collapsed to one scalar per year first - see _pooled_s2_baseline() for why)
    - viirs: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG, avg_rad, per-year median, then
      median+MAD
    - rain: UCSB-CHG/CHIRPS/DAILY, monthly sum per year, then median+MAD
    Returns {median, mad, sample_count}.
    INPUTS: ee.Geometry, int month 1-12, str indicator, int years_back
    OUTPUTS: dict"""
    current_year = datetime.utcnow().year
    years = [current_year - offset for offset in range(1, years_back + 1)]

    if indicator in ("ndvi", "bsi"):
        return _pooled_s2_baseline(zone_geom, years, calendar_month, indicator)

    per_year = {
        str(year): _indicator_yearly_sample(zone_geom, year, calendar_month, indicator) for year in years
    }
    # One getInfo() round-trip for every sampled year, instead of one per year.
    resolved = ee.Dictionary(per_year).getInfo()

    samples = [v for v in resolved.values() if v is not None]
    median = statistics.median(samples) if samples else 0.0

    return {"median": median, "mad": _mad(samples, indicator=indicator), "sample_count": len(samples)}


def get_or_compute_baseline(zone: Zone, calendar_month: int, indicator: str, db: Session) -> ZoneBaseline:
    """Return cached ZoneBaseline if computed_at within 30 days, else compute
    and upsert.
    INPUTS: Zone row, int month, str indicator, DB session
    OUTPUTS: ZoneBaseline row"""
    existing = db.exec(
        select(ZoneBaseline).where(
            ZoneBaseline.zone_id == zone.id,
            ZoneBaseline.calendar_month == calendar_month,
            ZoneBaseline.indicator == indicator,
        )
    ).first()

    cache_cutoff = datetime.utcnow() - timedelta(days=BASELINE_CACHE_DAYS)
    if existing is not None and existing.computed_at >= cache_cutoff:
        return existing

    zone_geom = ee.Geometry(json.loads(zone.aoi_geojson))
    result = compute_baseline(zone_geom, calendar_month, indicator)

    if existing is not None:
        existing.median = result["median"]
        existing.mad = result["mad"]
        existing.sample_count = result["sample_count"]
        existing.computed_at = datetime.utcnow()
        row = existing
    else:
        row = ZoneBaseline(
            zone_id=zone.id,
            calendar_month=calendar_month,
            indicator=indicator,
            median=result["median"],
            mad=result["mad"],
            sample_count=result["sample_count"],
        )

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def zones_missing_baseline(zones: list[Zone], calendar_month: int, db: Session) -> list[str]:
    """INPUTS: zones an upcoming run would process, the calendar month the run's as_of falls in,
    DB session. OUTPUTS: names of zones that do NOT have a fresh (<=BASELINE_CACHE_DAYS old)
    ZoneBaseline row for both 'ndvi' and 'bsi' - the two indicators gee.detect.detect_candidates()
    always needs before it can flag anything, so a zone missing either one is exactly a zone that
    would fall through to get_or_compute_baseline()'s expensive from-scratch path (pooling 3 years
    of Sentinel-2 imagery per pixel) if the run went ahead. Used by app.main's /run-once to fail
    fast with a clear message instead of quietly eating the whole request timeout budget on a cold
    baseline. Doesn't check 'viirs' - that baseline is only ever fetched once a zone is actually
    flagged, so its absence never costs a full run."""
    cutoff = datetime.utcnow() - timedelta(days=BASELINE_CACHE_DAYS)
    missing = []
    for zone in zones:
        fresh_indicators = {
            row.indicator
            for row in db.exec(
                select(ZoneBaseline).where(
                    ZoneBaseline.zone_id == zone.id,
                    ZoneBaseline.calendar_month == calendar_month,
                    ZoneBaseline.computed_at >= cutoff,
                )
            ).all()
        }
        if not {"ndvi", "bsi"}.issubset(fresh_indicators):
            missing.append(zone.name)
    return missing


def cached_baseline_months(db: Session) -> list[int]:
    """INPUTS: DB session. OUTPUTS: sorted distinct calendar months (1-12) that currently have at
    least one fresh (<=BASELINE_CACHE_DAYS old) ZoneBaseline row - a quick "what's ready" hint for
    a zones_missing_baseline() error response. Not scoped to a particular zone or indicator; it
    only tells you a month has SOME cached baseline, not that every zone is covered for it."""
    cutoff = datetime.utcnow() - timedelta(days=BASELINE_CACHE_DAYS)
    months = db.exec(
        select(ZoneBaseline.calendar_month).where(ZoneBaseline.computed_at >= cutoff).distinct()
    ).all()
    return sorted(set(months))
