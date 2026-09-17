"""Stage 1 smoke test: confirms real Earth Engine auth + a real Sentinel-2 read.
Run with: python -m scripts.gee_smoke_test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import ee  # noqa: E402 - must follow load_dotenv() so GOOGLE_APPLICATION_CREDENTIALS is set

from gee.init import init_ee  # noqa: E402

KAMBOVE_LAT = -10.875
KAMBOVE_LON = 26.595
BUFFER_M = 5000


def main() -> None:
    init_ee()

    geometry = ee.Geometry.Point([KAMBOVE_LON, KAMBOVE_LAT]).buffer(BUFFER_M)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2024-01-01", "2024-12-31")
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    image = collection.first()

    image_id = image.get("system:index").getInfo()
    cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    stats = ndvi.reduceRegion(
        reducer=ee.Reducer.median(), geometry=geometry, scale=20, maxPixels=1e9
    )
    median_ndvi = stats.get("NDVI").getInfo()

    print(f"Image ID: {image_id}")
    print(f"Cloudy pixel percentage: {cloud_pct}")
    print(f"Median NDVI over 5km buffer: {median_ndvi}")


if __name__ == "__main__":
    main()
