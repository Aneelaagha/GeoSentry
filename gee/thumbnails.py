import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

from gee.indices import mask_s2_clouds
from gee.init import ee_available

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger("geosentry.gee")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
THUMB_SIZE_PX = 120
THUMB_VIS_PARAMS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 2500, "gamma": 1.1}
WINDOW_DAYS = 30


def _rgb_composite(geometry, start: str, end: str):
    """INPUTS: ee.Geometry, ISO start/end dates. OUTPUTS: server-side ee.Image - the cloud-masked
    true-color (B4/B3/B2) median composite over that window, clipped to the zone."""
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start, end)
        .filterBounds(geometry)
        .map(mask_s2_clouds)
    )
    return collection.select(["B4", "B3", "B2"]).median().clip(geometry)


def _thumb_data_uri(image, region) -> str:
    """INPUTS: an ee.Image and the region to render. OUTPUTS: a 'data:image/png;base64,...' URI.
    Fetches the actual PNG bytes from ee.Image.getThumbURL() server-side and inlines them, so the
    browser never makes an external request for zone imagery."""
    url = image.getThumbURL(
        {**THUMB_VIS_PARAMS, "dimensions": THUMB_SIZE_PX, "region": region, "format": "png"}
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_before_after_thumbs(aoi_geojson: str, detected_at: datetime) -> Optional[tuple[str, str]]:
    """INPUTS: the zone's AOI GeoJSON, the detection's timestamp. OUTPUTS: (before_data_uri,
    after_data_uri) - real Sentinel-2 true-color thumbnails as inline base64 PNG data URIs: one
    for the WINDOW_DAYS before the detection window, one for the detection's own trailing window.
    None if Earth Engine is unavailable or the export fails, so a thumbnail outage can't break
    the detections list. Only ever called when SYNTHETIC_MODE is false - synthetic mode renders
    its own inline SVG tiles client-side and never reaches this function."""
    if not ee_available():
        return None
    try:
        geometry = ee.Geometry(json.loads(aoi_geojson))
        after_end = detected_at
        after_start = after_end - timedelta(days=WINDOW_DAYS)
        before_end = after_start
        before_start = before_end - timedelta(days=WINDOW_DAYS)

        before_img = _rgb_composite(
            geometry, before_start.strftime("%Y-%m-%d"), before_end.strftime("%Y-%m-%d")
        )
        after_img = _rgb_composite(
            geometry, after_start.strftime("%Y-%m-%d"), after_end.strftime("%Y-%m-%d")
        )
        return _thumb_data_uri(before_img, geometry), _thumb_data_uri(after_img, geometry)
    except Exception as exc:  # noqa: BLE001 - a thumbnail failure must not break the detections list
        logger.warning("GEE thumbnail export failed (%s)", exc)
        return None
