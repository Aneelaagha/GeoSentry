"""Stage 2 debug: instrument the NDVI baseline path for zone 1 (Kambove) to find why MAD came
out near-zero. Run with: python -m scripts.debug_baseline
"""
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import ee
from sqlmodel import Session, select

from app.db import Zone, engine
from gee.baselines import S2_COLLECTION, INDICATOR_REDUCE_SCALE_M, _mad, _month_date_range
from gee.indices import add_indices, mask_s2_clouds
from gee.init import init_ee

ZONE_ID = 1
INDICATOR = "ndvi"
YEARS_BACK = 3


def main() -> None:
    init_ee()

    with Session(engine) as session:
        zone = session.exec(select(Zone).where(Zone.id == ZONE_ID)).first()
    if zone is None:
        print(f"No zone with id={ZONE_ID}")
        sys.exit(1)

    zone_geom = ee.Geometry(json.loads(zone.aoi_geojson))
    calendar_month = datetime.utcnow().month
    current_year = datetime.utcnow().year
    years = [current_year - offset for offset in range(1, YEARS_BACK + 1)]

    print(f"Zone: {zone.name} (id={zone.id})")
    print(f"AOI geojson: {zone.aoi_geojson}")
    print(f"Indicator: {INDICATOR}, calendar_month={calendar_month}, years_back={YEARS_BACK}")
    print(f"Years sampled: {years}\n")

    values = []
    for year in years:
        start, end = _month_date_range(year, calendar_month)
        print(f"--- Year {year}: window [{start}, {end}) ---")

        collection = ee.ImageCollection(S2_COLLECTION).filterDate(start, end).filterBounds(zone_geom)
        count = collection.size().getInfo()
        print(f"  images in filtered collection: {count}")

        ids = collection.limit(5).aggregate_array("system:index").getInfo()
        print(f"  image IDs (up to 5): {ids}")

        prepared = collection.map(mask_s2_clouds).map(add_indices)
        image = prepared.select(INDICATOR).median()
        stats = image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=zone_geom,
            scale=INDICATOR_REDUCE_SCALE_M,
            maxPixels=1e9,
            bestEffort=True,
        )
        value = stats.get(INDICATOR).getInfo()
        print(f"  per-year median {INDICATOR}: {value}\n")
        values.append(value)

    print(f"Per-year values: {values}")

    clean = [v for v in values if v is not None]
    median = statistics.median(clean) if clean else None
    mad = _mad(clean, indicator=INDICATOR)
    print(f"Final median: {median}")
    print(f"Final MAD: {mad}")


if __name__ == "__main__":
    main()
