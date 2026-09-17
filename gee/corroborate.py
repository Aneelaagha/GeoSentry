import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

from gee.baselines import BaselineStats
from gee.init import ee_available

try:
    import ee
except ImportError:
    ee = None

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
CHIRPS_SCALE_M = 5000  # native CHIRPS resolution is ~5.5km
VIIRS_SCALE_M = 500
RAINFALL_WINDOW_DAYS = 30
RAINFALL_HISTORY_YEARS = 5  # comparison sample for the current 30d total's percentile rank


def _zone_geometry(aoi_geojson: str):
    """INPUTS: a GeoJSON geometry string (as stored on Zone.aoi_geojson).
    OUTPUTS: the matching ee.Geometry."""
    return ee.Geometry(json.loads(aoi_geojson))


def _percentile_rank(value: float, samples: list[float]) -> float:
    """INPUTS: an observed value, a list of historical comparison samples.
    OUTPUTS: value's percentile rank (0-100) within samples - the share of samples at or below
    it. 50.0 (neutral) if there is no history to rank against."""
    if not samples:
        return 50.0
    at_or_below = sum(1 for s in samples if s <= value)
    return round(100 * at_or_below / len(samples), 2)


def _chirps_window_total(geometry, start: str, end: str) -> object:
    """INPUTS: ee.Geometry, ISO start/end dates. OUTPUTS: server-side ee.Number (or None) - the
    zone-mean of total accumulated CHIRPS precipitation (mm) over [start, end)."""
    collection = ee.ImageCollection(CHIRPS_COLLECTION).filterDate(start, end).filterBounds(geometry)
    total = collection.sum()
    stats = total.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=CHIRPS_SCALE_M, maxPixels=1e9, bestEffort=True
    )
    return stats.get("precipitation")


def get_rainfall_percentile(zone_name: str, zone_type: str, aoi_geojson: Optional[str] = None) -> float:
    """INPUTS: zone_name, zone_type, the zone's AOI GeoJSON (required for the real Earth Engine
    path). OUTPUTS: float 0-100, the percentile rank of the trailing-30-day CHIRPS rainfall
    total against the same 30-day window in each of the last RAINFALL_HISTORY_YEARS years. A low
    percentile helps explain an NDVI drop as drought; a normal/high percentile rules drought out
    and corroborates an anthropogenic cause. Falls back to synthetic values offline."""
    if not ee_available():
        seed = int(hashlib.sha256(f"rain-{zone_name}".encode()).hexdigest()[:8], 16)
        if zone_type == "control":
            return 45.0 + seed % 20  # normal rainfall, nothing unusual
        return 50.0 + seed % 25  # normal-ish rainfall -> NOT a drought explanation

    if not aoi_geojson:
        raise ValueError("aoi_geojson is required to look up real CHIRPS rainfall.")

    geometry = _zone_geometry(aoi_geojson)
    now = datetime.utcnow()

    windows = {"current": (now - timedelta(days=RAINFALL_WINDOW_DAYS), now)}
    for years_back in range(1, RAINFALL_HISTORY_YEARS + 1):
        end = now - timedelta(days=365 * years_back)
        windows[f"hist_{years_back}"] = (end - timedelta(days=RAINFALL_WINDOW_DAYS), end)

    totals = {
        key: _chirps_window_total(geometry, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for key, (start, end) in windows.items()
    }
    resolved = ee.Dictionary(totals).getInfo()

    current_total = resolved.pop("current", None)
    historical_totals = [v for v in resolved.values() if v is not None]

    if current_total is None:
        return 50.0  # no valid current-window pixels; stay neutral rather than guessing
    return _percentile_rank(current_total, historical_totals)


# Stage 3: real, no-synthetic-fallback corroboration. Distinct from get_rainfall_percentile()/
# get_viirs_nightlight_zscore() above, which stay synthetic-aware for scripts/run_once.py's
# default demo path. NoValidPixels lives in gee.detect; imported here at module level is safe
# because gee/detect.py never imports gee.corroborate at its own module level (detect_candidates()
# does a deferred, function-local import of rainfall_percentile instead) - see the comment there
# for why.
from gee.detect import NoValidPixels  # noqa: E402

RAINFALL_PERCENTILE_HISTORY_YEARS = 3


def rainfall_percentile(zone_geom, days: int = 30, as_of: Optional[datetime] = None) -> float:
    """CHIRPS daily sum over last `days`, expressed as percentile vs the same
    window in the previous 3 years.
    INPUTS: ee.Geometry, int. OUTPUTS: float 0-100."""
    end = as_of or datetime.utcnow()
    start = end - timedelta(days=days)

    windows = {"current": (start, end)}
    for years_back in range(1, RAINFALL_PERCENTILE_HISTORY_YEARS + 1):
        hist_end = end - timedelta(days=365 * years_back)
        hist_start = hist_end - timedelta(days=days)
        windows[f"hist_{years_back}"] = (hist_start, hist_end)

    totals = {
        key: _chirps_window_total(zone_geom, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
        for key, (s, e) in windows.items()
    }
    resolved = ee.Dictionary(totals).getInfo()

    current_total = resolved.pop("current", None)
    if current_total is None:
        raise NoValidPixels(f"No CHIRPS precipitation pixels for the {days}-day window ending {end.date()}.")

    historical_totals = [v for v in resolved.values() if v is not None]
    return _percentile_rank(current_total, historical_totals)


def get_viirs_nightlight_zscore(
    zone_name: str, zone_type: str, baseline: Optional[BaselineStats] = None, aoi_geojson: Optional[str] = None
) -> float:
    """INPUTS: zone_name, zone_type, the zone's current-month BaselineStats (for VIIRS
    median/MAD), and the zone's AOI GeoJSON - both required for the real Earth Engine path.
    OUTPUTS: float z-score of VIIRS night-light radiance vs baseline. Elevated values
    corroborate active extraction (generators, night operations); near-zero fits logging or
    agricultural expansion, which typically run in daylight. Falls back to synthetic values
    offline."""
    if not ee_available():
        seed = int(hashlib.sha256(f"viirs-{zone_name}".encode()).hexdigest()[:8], 16)
        if zone_type == "mining":
            return 2.2 + seed % 10 / 10
        if zone_type == "logging":
            return 0.3 + seed % 5 / 10
        return 0.1 + seed % 3 / 10

    if not aoi_geojson:
        raise ValueError("aoi_geojson is required to look up real VIIRS night-light radiance.")
    if baseline is None:
        raise ValueError("baseline is required to z-score real VIIRS night-light radiance.")

    geometry = _zone_geometry(aoi_geojson)
    now = datetime.utcnow()
    start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    collection = ee.ImageCollection(VIIRS_COLLECTION).filterDate(start, end).filterBounds(geometry)
    composite = collection.select("avg_rad").median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=VIIRS_SCALE_M, maxPixels=1e9, bestEffort=True
    )
    current = stats.get("avg_rad").getInfo()

    if current is None:
        return 0.0
    return round((current - baseline.viirs_median) / (1.4826 * baseline.viirs_mad), 4)
