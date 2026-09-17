"""End-to-end single pass over every zone: detect -> classify -> corroborate -> fuse ->
alert (or stay silent). Run with: python scripts/run_once.py
Pass --as-of YYYY-MM-DD to use real Earth Engine for a historical date (Stage 3/4) instead of
the synthetic default path - see process_zone_as_of() below.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from sqlmodel import Session, select

from alerts.ntfy import build_alert_message, send_alert
from alerts.twilio_stub import send_sms
from app.config import settings
from app.db import Alert, Detection, Zone, engine, init_db
from fusion.bayes import classify_alert_tier, fuse_confidence
from gee.baselines import compute_zone_baseline
from gee.corroborate import get_rainfall_percentile, rainfall_percentile
from gee.detect import NoValidPixels, detect_candidates, detect_change
from llm.adversarial import challenge_event
from llm.classify import classify_event

NON_ALERTING_CAUSES = {"drought", "artifact"}
BASE_PRIOR = 0.05  # base-rate probability any monitored pass reflects a real disturbance


def _finalize_detection(
    session: Session,
    zone: Zone,
    ndvi_delta: float,
    bsi_delta: float,
    viirs_delta: Optional[float],
    rainfall_pct: float,
    area_ha: float,
    as_of_str: Optional[str] = None,
    carbon_loss_tco2e: Optional[float] = None,
) -> Detection:
    """INPUTS: open DB session, Zone row, the four indicator deltas and rainfall percentile,
    estimated affected area, as_of_str ('YYYY-MM-DD' for a historical pass, None for the
    live synthetic default - stored on Detection.as_of purely as run metadata for
    app.main.get_stats()'s windows_evaluated count; changes no scoring/threshold logic), and
    carbon_loss_tco2e (analysis.ldn.estimate_carbon_loss_tco2e output, already computed by
    gee.detect.detect_candidates for the real --as-of path; None for the synthetic default path,
    which has no real baseline to derive it from - never guessed).

    viirs_delta may be None: gee.detect.current_viirs() found no VIIRS monthly composite for
    this zone/month (a real 1-2 month data lag, not an error - VIIRS is a corroborator, not a
    required signal). The real None is passed straight through everywhere: to the LLM event dict
    (llm/classify.py and llm/adversarial.py render it as an honest "unavailable"/"n/a" note, per
    their own format_viirs_for_prompt/format_viirs_num - never a fabricated number), to
    fuse_confidence (skips the VIIRS likelihood ratio entirely), and to Detection.viirs_z
    (persisted as a real NULL).

    OUTPUTS: the persisted Detection row. Shared tail of the pipeline -
    LLM classify -> adversarial challenge -> Bayesian fuse -> classify_alert_tier() -> persist ->
    alert-or-silent -> fire alert (ntfy + SMS stub) if the tier isn't 'silent' and this zone+cause
    hasn't already alerted within alerts.ntfy.DEDUPE_WINDOW_DAYS. A tier of 'classified' whose
    cause is drought/artifact still stays silent (a confident benign explanation); 'uncertain'
    fires regardless of cause, with the honest cause-uncertain message template. Used by both
    process_zone() (synthetic default path) and process_zone_as_of() (real Earth Engine,
    historical --as-of path) so the two only differ in how they arrive at these five numbers, not
    in what happens after."""
    event = {
        "zone_name": zone.name,
        "ndvi_delta": ndvi_delta,
        "bsi_delta": bsi_delta,
        "viirs_delta": viirs_delta,  # real None passes straight through now
        "rainfall_percentile": rainfall_pct,
        "area_ha": area_ha,
    }
    classification = classify_event(event)
    review = challenge_event(event, classification)

    # A strong adversarial case tempers the LLM's own confidence before it ever reaches
    # fuse_confidence, so a single classification can't be the sole gate on an alert.
    effective_llm_conf = classification["confidence"] * (1 - review["strength_of_alternative"])

    posterior = fuse_confidence(
        prior=BASE_PRIOR,
        evidence={
            "llm_conf": effective_llm_conf,
            "viirs_z": viirs_delta,  # real None passes through - fuse_confidence skips it
            "rain_percentile": rainfall_pct,
            "area_ha": area_ha,
        },
    )

    # tier is computed from the LLM's own RAW confidence (before adversarial tempering) vs the
    # fused posterior - see fusion.bayes.classify_alert_tier. It's a separate axis from
    # NON_ALERTING_CAUSES: a CONFIDENTLY-classified drought/artifact still suppresses the alert
    # (the LLM is sure it's benign), but an UNCERTAIN top cause that happens to land on
    # drought/artifact does not get that same benefit of the doubt - if the fused evidence is
    # strong enough to clear the gate and the LLM itself isn't sure, we fire as 'uncertain'
    # rather than silently trusting a hedged "it's probably nothing."
    tier = classify_alert_tier(classification["confidence"], posterior, settings.alert_confidence_threshold)
    if tier == "silent":
        status = "silent"
    elif tier == "classified" and classification["cause"] in NON_ALERTING_CAUSES:
        status = "silent"
    else:
        status = "alerted"

    adversarial_notes = (
        "; ".join(f"{alt['cause']}: {alt['rationale']}" for alt in review["alternatives"])
        or "No plausible alternative causes identified."
    )

    detection = Detection(
        zone_id=zone.id,
        ndvi_z=ndvi_delta,
        bsi_z=bsi_delta,
        viirs_z=viirs_delta,
        rainfall_percentile=rainfall_pct,
        cause=classification["cause"],
        cause_confidence=classification["confidence"],
        reasoning=classification["reasoning"],
        adversarial_notes=adversarial_notes,
        combined_confidence=posterior,
        estimated_area_ha=area_ha,
        status=status,
        tier=tier,
        as_of=as_of_str,
        carbon_loss_tco2e=carbon_loss_tco2e,
    )
    session.add(detection)
    session.commit()
    session.refresh(detection)

    if status == "alerted":
        sent = send_alert(
            zone_id=zone.id,
            zone_name=zone.name,
            cause=classification["cause"],
            confidence=posterior,
            area_ha=area_ha,
            ndvi_z=ndvi_delta,
            bsi_z=bsi_delta,
            viirs_z=viirs_delta,
            rainfall_percentile=rainfall_pct,
            tier=tier,
            llm_conf=classification["confidence"],
            carbon_loss_tco2e=carbon_loss_tco2e,
        )
        if sent:
            message = build_alert_message(
                zone_name=zone.name,
                cause=classification["cause"],
                confidence=posterior,
                area_ha=area_ha,
                ndvi_z=ndvi_delta,
                bsi_z=bsi_delta,
                viirs_z=viirs_delta,
                rainfall_percentile=rainfall_pct,
                tier=tier,
                llm_conf=classification["confidence"],
                carbon_loss_tco2e=carbon_loss_tco2e,
            )
            send_sms(message)
            session.add(Alert(detection_id=detection.id, zone_id=zone.id, message=message, channel="ntfy+sms_stub"))
            session.commit()

    return detection


def process_zone(session: Session, zone: Zone) -> Detection:
    """INPUTS: open DB session, a Zone row. OUTPUTS: the persisted Detection row for this pass.
    Synthetic-mode-aware default path: baseline -> change detection -> rainfall lookup, then
    _finalize_detection() for classify/fuse/alert. Unchanged behavior from before Stage 3."""
    month = datetime.utcnow().month
    baseline = compute_zone_baseline(zone.name, month, zone.aoi_geojson)
    indicators = detect_change(zone.name, zone.zone_type, baseline, zone.aoi_geojson)
    rainfall_pct = get_rainfall_percentile(zone.name, zone.zone_type, zone.aoi_geojson)
    return _finalize_detection(
        session, zone, indicators.ndvi_z, indicators.bsi_z, indicators.viirs_z, rainfall_pct, indicators.estimated_area_ha
    )


def process_zone_as_of(session: Session, zone: Zone, as_of: datetime) -> Optional[Detection]:
    """INPUTS: open DB session, a Zone row, the historical as-of date to evaluate. OUTPUTS: the
    persisted Detection row, or None if Earth Engine had no valid current-window imagery
    (NoValidPixels) - that zone is skipped for this pass, never synthesized. Real Earth Engine
    equivalent of process_zone(), for Stage 3/4's --as-of flag: detect_candidates() (per-indicator
    3-sigma flagging against the cached ZoneBaseline) in place of the synthetic detect_change(),
    then the same _finalize_detection() tail. Picks the strongest candidate's z per indicator when
    flagged, 0.0 (neutral) when not; area_ha is the max across any flagged indicators (avoids
    double-counting overlapping pixels between an ndvi- and bsi-flagged mask)."""
    try:
        candidates = detect_candidates(zone, session, as_of=as_of)
    except NoValidPixels as exc:
        print(f"  [SKIPPED] {zone.name}: no valid current observations ({exc})")
        return None

    by_indicator = {c["indicator"]: c for c in candidates}
    ndvi_delta = by_indicator.get("ndvi", {}).get("z", 0.0)
    bsi_delta = by_indicator.get("bsi", {}).get("z", 0.0)
    # None (not 0.0) when no candidate was flagged - VIIRS was never even queried that pass, a
    # different situation from "queried but no composite available" (also None, from the
    # candidate dict - see gee.detect.detect_candidates). Both cases are honestly "no VIIRS
    # signal to report" rather than "VIIRS confirms zero activity."
    viirs_delta = next((c["viirs_z"] for c in candidates), None)
    # The candidate with the largest area_ha drives both area_ha and carbon_loss_tco2e - keeps
    # the two numbers consistent with each other (same underlying candidate) rather than area_ha
    # from one flagged indicator and carbon from another's unrelated area.
    strongest_candidate = max(candidates, key=lambda c: c["area_ha"], default=None)
    area_ha = strongest_candidate["area_ha"] if strongest_candidate else 0.0
    carbon_loss_tco2e = strongest_candidate["carbon_loss_tco2e"] if strongest_candidate else None

    zone_geom = ee.Geometry(json.loads(zone.aoi_geojson))
    rainfall_pct = rainfall_percentile(zone_geom, days=30, as_of=as_of)

    return _finalize_detection(
        session,
        zone,
        ndvi_delta,
        bsi_delta,
        viirs_delta,
        rainfall_pct,
        area_ha,
        as_of.strftime("%Y-%m-%d"),
        carbon_loss_tco2e=carbon_loss_tco2e,
    )


def run_once(as_of: Optional[datetime] = None) -> list[Detection]:
    """INPUTS: optional historical as-of date - None (default) uses the synthetic-mode-aware
    process_zone() path; if set, uses the real Earth Engine process_zone_as_of() path instead.
    OUTPUTS: list of Detection rows created, one per zone that produced one (a zone skipped via
    NoValidPixels under --as-of contributes none). Entry point for `python scripts/run_once.py`
    and for the API's POST /run-once trigger, which passes the same as_of through."""
    init_db()
    detections: list[Detection] = []
    with Session(engine, expire_on_commit=False) as session:
        zones = session.exec(select(Zone)).all()
        if not zones:
            print("No zones found. Run scripts/seed_zones.py first.")
            return []
        for zone in zones:
            if as_of is not None:
                detection = process_zone_as_of(session, zone, as_of)
                if detection is None:
                    continue
            else:
                detection = process_zone(session, zone)
            print(
                f"[{detection.status.upper()}] {zone.name}: cause={detection.cause} "
                f"confidence={detection.combined_confidence:.2f}"
            )
            # Only flag this for a pass that actually found something (ndvi/bsi flagged, or a
            # real area) - viirs_z is also None for an ordinary quiet pass where nothing was
            # flagged and VIIRS was never even queried, and logging "VIIRS unavailable" on every
            # routine silent zone would bury the one case operators actually need to notice: a
            # real detection that fired (or nearly did) without its VIIRS corroborator.
            flagged_something = detection.ndvi_z != 0.0 or detection.bsi_z != 0.0 or detection.estimated_area_ha > 0
            if detection.viirs_z is None and flagged_something:
                reference = datetime.strptime(detection.as_of, "%Y-%m-%d") if detection.as_of else datetime.utcnow()
                print(
                    f"  VIIRS unavailable for {reference.strftime('%Y-%m')} - "
                    "detection based on optical + rainfall only"
                )
            detections.append(detection)
    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="YYYY-MM-DD; use real Earth Engine for this historical date instead of the synthetic default path",
    )
    args = parser.parse_args()
    as_of_date = datetime.strptime(args.as_of, "%Y-%m-%d") if args.as_of else None

    if as_of_date is not None:
        # Only touched when --as-of is explicitly passed; the default (no args) path never
        # loads real credentials or calls init_ee(), so SYNTHETIC_MODE=true stays fully offline.
        from dotenv import load_dotenv

        load_dotenv()
        from gee.init import init_ee

        init_ee()

    run_once(as_of=as_of_date)
