"""Builds calibration.json: a real historical backtest across every zone, run through the
exact same detect -> classify -> adversarial -> fuse -> tier pipeline as scripts/run_once.py
(no detection logic here - this only orchestrates the existing functions), recording every
flagged candidate and why its posterior did or didn't clear the alert gate. Reused if the file
exists and is under CACHE_MAX_AGE old; recomputed (real Earth Engine calls) otherwise.

Run with: python -m scripts.build_calibration
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import ee
from sqlmodel import Session, select

from app.config import settings
from app.db import Zone, engine, init_db
from fusion.bayes import classify_alert_tier, explain_silence, fuse_confidence
from gee.corroborate import rainfall_percentile
from gee.detect import NoValidPixels, detect_candidates
from gee.init import init_ee
from llm.adversarial import challenge_event
from llm.classify import classify_event

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration.json"
CACHE_MAX_AGE = timedelta(hours=24)
BASE_PRIOR = 0.05  # matches scripts/run_once.py's BASE_PRIOR - same prior, same math

# The historical sweep this calibration report is built from: every current zone, over the same
# 5 dry-season dates already validated in this session's backtest (2024/2025 Aug-Oct).
SWEEP_DATES = ["2024-08-15", "2024-09-15", "2024-10-15", "2025-08-15", "2025-09-15"]


def _is_cache_fresh() -> bool:
    """INPUTS: none. OUTPUTS: bool - True if calibration.json exists and was written less than
    CACHE_MAX_AGE ago."""
    if not CALIBRATION_PATH.exists():
        return False
    mtime = datetime.utcfromtimestamp(CALIBRATION_PATH.stat().st_mtime)
    return datetime.utcnow() - mtime < CACHE_MAX_AGE


def _evaluate_zone_date(zone: Zone, as_of: datetime, db: Session) -> tuple[list[dict], bool]:
    """INPUTS: Zone row, as-of datetime, open DB session. OUTPUTS: (window_dicts, had_candidate)
    - window_dicts is one dict per flagged indicator (0, 1, or 2), each carrying the full
    classify/fuse/tier trace for that (zone, date); had_candidate is False if detect_candidates()
    flagged nothing (so callers can still count the window as tested). Raises NoValidPixels
    straight through - callers decide whether to skip or count that as tested."""
    candidates = detect_candidates(zone, db, as_of=as_of)
    if not candidates:
        return [], False

    by_indicator = {c["indicator"]: c for c in candidates}
    ndvi_delta = by_indicator.get("ndvi", {}).get("z", 0.0)
    bsi_delta = by_indicator.get("bsi", {}).get("z", 0.0)
    viirs_delta = next((c["viirs_z"] for c in candidates), 0.0)
    area_ha = max((c["area_ha"] for c in candidates), default=0.0)

    zone_geom = ee.Geometry(json.loads(zone.aoi_geojson))
    rainfall_pct = rainfall_percentile(zone_geom, days=30, as_of=as_of)

    event = {
        "zone_name": zone.name,
        "ndvi_delta": ndvi_delta,
        "bsi_delta": bsi_delta,
        "viirs_delta": viirs_delta,
        "rainfall_percentile": rainfall_pct,
        "area_ha": area_ha,
    }
    classification = classify_event(event)
    review = challenge_event(event, classification)
    effective_llm_conf = classification["confidence"] * (1 - review["strength_of_alternative"])

    posterior = fuse_confidence(
        prior=BASE_PRIOR,
        evidence={
            "llm_conf": effective_llm_conf,
            "viirs_z": viirs_delta,
            "rain_percentile": rainfall_pct,
            "area_ha": area_ha,
        },
    )
    tier = classify_alert_tier(classification["confidence"], posterior, settings.alert_confidence_threshold)

    if tier == "silent":
        suppression_reason = explain_silence(
            {
                "llm_confidence": classification["confidence"],
                "viirs_delta": viirs_delta,
                "rainfall_pct": rainfall_pct,
                "area_ha": area_ha,
                "fused_confidence": posterior,
            },
            settings.alert_confidence_threshold,
        )
    else:
        suppression_reason = f"Alert fired ({tier}) - posterior cleared the gate."

    windows = [
        {
            "as_of": as_of.strftime("%Y-%m-%d"),
            "zone": zone.name,
            "indicator": candidate["indicator"],
            "z": round(candidate["z"], 4),
            "area_ha": round(candidate["area_ha"], 2),
            "llm_cause": classification["cause"],
            "llm_conf": round(classification["confidence"], 4),
            "posterior": round(posterior, 4),
            "tier": tier,
            "suppression_reason": suppression_reason,
        }
        for candidate in candidates
    ]
    return windows, True


def build() -> dict:
    """INPUTS: none. OUTPUTS: the calibration dict (also written to CALIBRATION_PATH) - runs
    detect_candidates() for every zone x SWEEP_DATES combination, using the exact same
    classify/adversarial/fuse/tier pipeline as scripts/run_once.py, without persisting any
    Detection/Alert rows (this is a report generator, not a pipeline run)."""
    init_db()
    init_ee()

    windows: list[dict] = []
    windows_tested = 0
    zone_names: set[str] = set()

    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        for zone in zones:
            zone_names.add(zone.name)
            for date_str in SWEEP_DATES:
                as_of = datetime.strptime(date_str, "%Y-%m-%d")
                try:
                    zone_windows, had_candidate = _evaluate_zone_date(zone, as_of, session)
                except NoValidPixels as exc:
                    print(f"  [SKIPPED] {zone.name} {date_str}: no valid current observations ({exc})")
                    continue
                windows_tested += 1
                if had_candidate:
                    windows.extend(zone_windows)
                    for w in zone_windows:
                        print(f"  CANDIDATE {date_str} {zone.name} {w['indicator']}: tier={w['tier']}")
                else:
                    print(f"  no change: {date_str} {zone.name}")

    calibration = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_windows_tested": windows_tested,
        "total_zones_tested": len(zone_names),
        "total_candidates_flagged": len(windows),
        "total_alerts_fired": sum(1 for w in windows if w["tier"] != "silent"),
        "windows": windows,
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))
    return calibration


def main() -> None:
    if _is_cache_fresh():
        age = datetime.utcnow() - datetime.utcfromtimestamp(CALIBRATION_PATH.stat().st_mtime)
        print(f"calibration.json is {age} old (< {CACHE_MAX_AGE}) - reusing cached file, not recomputing.")
        return

    print("calibration.json missing or stale - recomputing from a fresh historical sweep...")
    calibration = build()
    print(
        f"\nWrote {CALIBRATION_PATH}: {calibration['total_windows_tested']} windows tested across "
        f"{calibration['total_zones_tested']} zones, {calibration['total_candidates_flagged']} candidates "
        f"flagged, {calibration['total_alerts_fired']} alerts fired."
    )


if __name__ == "__main__":
    main()
