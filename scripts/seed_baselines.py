"""Stage 2: seed real ZoneBaseline rows for the current calendar month, across all zones and
all four indicators (ndvi, bsi, viirs, rain). Run with: python -m scripts.seed_baselines
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from sqlmodel import Session, select

from app.db import Zone, engine, init_db
from gee.baselines import get_or_compute_baseline
from gee.init import init_ee

INDICATORS = ["ndvi", "bsi", "viirs", "rain"]


def main() -> None:
    init_db()
    init_ee()

    calendar_month = datetime.utcnow().month
    start = time.monotonic()
    failures = 0

    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        if not zones:
            print("No zones found. Run scripts/seed_zones.py first.")
            sys.exit(1)

        for zone in zones:
            for indicator in INDICATORS:
                print(f"[{zone.name}] {indicator}...", end=" ", flush=True)
                try:
                    row = get_or_compute_baseline(zone, calendar_month, indicator, session)
                    print(f"median={row.median:.4f} mad={row.mad:.4f} sample_count={row.sample_count}")
                    if row.sample_count == 0:
                        print(f"  WARNING: sample_count=0 for {zone.name}/{indicator} - no valid observations")
                except Exception as exc:  # noqa: BLE001 - report every failure, don't stop early
                    failures += 1
                    print(f"FAILED: {exc}")

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.1f}s")
    if failures:
        print(f"{failures} indicator(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
