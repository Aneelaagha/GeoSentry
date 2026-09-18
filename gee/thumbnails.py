import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from pathlib import Path
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
THUMB_FETCH_TIMEOUT_S = 15  # hard cap per before/after fetch - covers the whole round trip
# (ee.Image.getThumbURL()'s own synchronous call to Earth Engine's backend, then downloading the
# PNG), since either leg can hang on a flaky connection. One slow/stuck thumbnail must never
# stall the whole batch - see _fetch_thumb_bytes().

THUMBNAILS_DIR = Path(__file__).resolve().parents[1] / "static" / "thumbnails"
THUMBNAIL_URL_PREFIX = "/static/thumbnails"

THUMBNAIL_WORKERS = 5  # bounds total concurrent GEE thumbnail fetches app-wide, shared by every
# trigger (a fresh Detection being persisted, and the startup sweep over old ones) - so a large
# backlog at startup can't fire off dozens of simultaneous EE calls on top of whatever a live
# /run-once pass just queued.
_executor = ThreadPoolExecutor(max_workers=THUMBNAIL_WORKERS, thread_name_prefix="thumbnail")


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


def _fetch_thumb_bytes(image, region) -> Optional[bytes]:
    """INPUTS: an ee.Image and the region to render. OUTPUTS: raw PNG bytes, or None if the
    fetch raised or exceeded THUMB_FETCH_TIMEOUT_S. Runs the whole round trip on its own
    single-use thread so a hung call can be abandoned at the timeout - concurrent.futures can't
    cancel a running thread, so that thread is simply left to die on its own (never joined)
    rather than blocking the caller past THUMB_FETCH_TIMEOUT_S."""

    def _do() -> bytes:
        url = image.getThumbURL(
            {**THUMB_VIS_PARAMS, "dimensions": THUMB_SIZE_PX, "region": region, "format": "png"}
        )
        response = requests.get(url, timeout=THUMB_FETCH_TIMEOUT_S)
        response.raise_for_status()
        return response.content

    with ThreadPoolExecutor(max_workers=1) as one_shot:
        future = one_shot.submit(_do)
        try:
            return future.result(timeout=THUMB_FETCH_TIMEOUT_S)
        except FuturesTimeoutError:
            logger.warning("GEE thumbnail fetch exceeded %ss - abandoning", THUMB_FETCH_TIMEOUT_S)
            return None
        except Exception as exc:  # noqa: BLE001 - thumbnails are decorative, never fatal
            logger.warning("GEE thumbnail fetch failed (%s)", exc)
            return None


def thumbnail_file_path(detection_id: int, side: str) -> Path:
    """INPUTS: a Detection id, side ('before' | 'after'). OUTPUTS: the deterministic on-disk
    path for that cached thumbnail - no DB column needed to track it, since the naming
    convention alone is enough to check existence or generate it."""
    return THUMBNAILS_DIR / f"{detection_id}_{side}.png"


def cached_thumbnail_urls(detection_id: int) -> tuple[Optional[str], Optional[str]]:
    """INPUTS: a Detection id. OUTPUTS: (before_url, after_url) - '/static/thumbnails/...' for
    whichever side already has a cached PNG file on disk, None for whichever doesn't (not yet
    generated, generation still in flight, or a previous attempt failed/timed out). A pure
    file-existence check - no GEE calls, no DB access - so GET /detections stays fast (page-load
    target: under 2s) regardless of thumbnail state."""
    before = thumbnail_file_path(detection_id, "before")
    after = thumbnail_file_path(detection_id, "after")
    return (
        f"{THUMBNAIL_URL_PREFIX}/{before.name}" if before.exists() else None,
        f"{THUMBNAIL_URL_PREFIX}/{after.name}" if after.exists() else None,
    )


def _generate_thumbnail_files(detection_id: int, aoi_geojson: str, detected_at: datetime) -> None:
    """INPUTS: a Detection id, its zone's AOI geojson, its detected_at timestamp. OUTPUTS: none -
    writes {detection_id}_before.png / {detection_id}_after.png under THUMBNAILS_DIR for
    whichever side doesn't already have a cached file and whose GEE fetch succeeds within
    THUMB_FETCH_TIMEOUT_S. Never raises: thumbnails are decorative, so any failure (bad AOI,
    Earth Engine unavailable, a timed-out fetch) is logged and simply leaves that side missing -
    the Detection row itself already persisted and renders fine without it. Runs on a background
    executor thread (see spawn_thumbnail_generation) - never on a request thread."""
    before_path = thumbnail_file_path(detection_id, "before")
    after_path = thumbnail_file_path(detection_id, "after")
    if before_path.exists() and after_path.exists():
        return
    if not ee_available():
        return

    try:
        geometry = ee.Geometry(json.loads(aoi_geojson))
    except Exception as exc:  # noqa: BLE001 - malformed AOI must not crash the background worker
        logger.warning("Thumbnail generation skipped for detection %s: bad AOI geojson (%s)", detection_id, exc)
        return

    after_end = detected_at
    after_start = after_end - timedelta(days=WINDOW_DAYS)
    before_end = after_start
    before_start = before_end - timedelta(days=WINDOW_DAYS)

    windows = {"before": (before_start, before_end), "after": (after_start, after_end)}
    paths = {"before": before_path, "after": after_path}
    for side, (start, end) in windows.items():
        path = paths[side]
        if path.exists():
            continue
        try:
            image = _rgb_composite(geometry, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            content = _fetch_thumb_bytes(image, geometry)
        except Exception as exc:  # noqa: BLE001 - one bad side must not block the other
            logger.warning("GEE thumbnail composite failed for detection %s (%s side): %s", detection_id, side, exc)
            continue
        if content is None:
            continue
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        logger.info("Cached %s thumbnail for detection %s", side, detection_id)


def spawn_thumbnail_generation(detection_id: int, aoi_geojson: str, detected_at: datetime) -> None:
    """INPUTS: a Detection id, its zone's AOI geojson, its detected_at timestamp. OUTPUTS: none -
    submits _generate_thumbnail_files() to the shared background executor and returns
    immediately, so the caller (a just-persisted Detection in _persist_detection(), or
    app.main's startup sweep) is never blocked on real GEE network calls. Safe to call for a
    detection that already has both thumbnails cached - _generate_thumbnail_files() no-ops
    immediately in that case. No-ops entirely (never even submits) when Earth Engine isn't
    available, e.g. SYNTHETIC_MODE - that mode renders its own inline SVG tiles client-side and
    never needs a cached file."""
    if not ee_available():
        return
    _executor.submit(_generate_thumbnail_files, detection_id, aoi_geojson, detected_at)
