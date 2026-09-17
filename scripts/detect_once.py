"""Stage 3: real, no-synthetic-fallback change detection across all zones. Prints one line per
zone: a flagged candidate's full dict, "no change" if nothing flagged, or "no valid current
observations" if Earth Engine had no usable imagery for the window - never falls back to
synthetic data. Run with: python -m scripts.detect_once [--as-of YYYY-MM-DD]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from sqlmodel import Session, select

from app.db import Zone, engine
from gee.detect import NoValidPixels, detect_candidates
from gee.init import init_ee


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD; defaults to now")
    args = parser.parse_args()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d") if args.as_of else None

    init_ee()

    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        if not zones:
            print("No zones found. Run scripts/seed_zones.py first.")
            sys.exit(1)

        for zone in zones:
            try:
                candidates = detect_candidates(zone, session, as_of=as_of)
            except NoValidPixels as exc:
                print(f"{zone.name}: no valid current observations ({exc})")
                continue

            if not candidates:
                print(f"{zone.name}: no change")
                continue

            for candidate in candidates:
                print(f"{zone.name}: CANDIDATE {candidate}")


if __name__ == "__main__":
    main()
