"""End-to-end single pass over every zone: detect -> classify -> corroborate -> fuse ->
alert (or stay silent). Run with: python scripts/run_once.py
Pass --as-of YYYY-MM-DD to use real Earth Engine for a historical date (Stage 3/4) instead of
the synthetic default path - see process_zone_as_of() below.
"""
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, wait
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
from gee.thumbnails import spawn_thumbnail_generation
from llm.adversarial import challenge_event
from llm.classify import classify_event

logger = logging.getLogger("geosentry.pipeline")

NON_ALERTING_CAUSES = {"drought", "artifact"}
BASE_PRIOR = 0.05  # base-rate probability any monitored pass reflects a real disturbance

ZONE_WORKER_POOL_SIZE = 5  # bounds concurrency independent of zone count, so a much larger zone
# set wouldn't try to open dozens of simultaneous EE + Anthropic connections at once.
SOFT_ZONE_DEADLINE_S = 110  # leaves ~10s of headroom under app.main's 120s hard request-level
# timeout for collecting whichever zones finished and persisting them on the main thread. A zone
# still running past this deadline is abandoned (concurrent.futures can't cancel a running
# thread) and excluded from this run's results, rather than failing the whole run.


def _classify_and_fuse(
    zone: Zone,
    ndvi_delta: float,
    bsi_delta: float,
    viirs_delta: Optional[float],
    rainfall_pct: float,
    area_ha: float,
    as_of_str: Optional[str] = None,
    carbon_loss_tco2e: Optional[float] = None,
) -> dict:
    """INPUTS: Zone row, the four indicator deltas and rainfall percentile, estimated affected
    area, as_of_str ('YYYY-MM-DD' for a historical pass, None for the live synthetic default -
    stored on Detection.as_of purely as run metadata for app.main.get_stats()'s
    windows_evaluated count; changes no scoring/threshold logic), and carbon_loss_tco2e
    (analysis.ldn.estimate_carbon_loss_tco2e output, already computed by
    gee.detect.detect_candidates for the real --as-of path; None for the synthetic default path,
    which has no real baseline to derive it from - never guessed).

    viirs_delta may be None: gee.detect.current_viirs() found no VIIRS monthly composite for
    this zone/month (a real 1-2 month data lag, not an error - VIIRS is a corroborator, not a
    required signal). The real None is passed straight through everywhere: to the LLM event dict
    (llm/classify.py and llm/adversarial.py render it as an honest "unavailable"/"n/a" note, per
    their own format_viirs_for_prompt/format_viirs_num - never a fabricated number), to
    fuse_confidence (skips the VIIRS likelihood ratio entirely), and eventually to
    Detection.viirs_z (persisted as a real NULL by _persist_detection()).

    OUTPUTS: a dict of everything _persist_detection() needs to build+commit the Detection row
    and decide whether to alert. Deliberately touches no DB session - this is the network-bound
    half of the old single _finalize_detection() (LLM classify -> adversarial challenge ->
    Bayesian fuse -> classify_alert_tier()), split out so run_once() can run it concurrently
    across zones (see its docstring) while every actual DB write still happens serially via
    _persist_detection() on the main thread. Used by both process_zone() (synthetic default
    path) and process_zone_as_of() (real Earth Engine, historical --as-of path) so the two only
    differ in how they arrive at these five numbers, not in what happens after."""
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

    return {
        "ndvi_delta": ndvi_delta,
        "bsi_delta": bsi_delta,
        "viirs_delta": viirs_delta,
        "rainfall_pct": rainfall_pct,
        "area_ha": area_ha,
        "as_of_str": as_of_str,
        "carbon_loss_tco2e": carbon_loss_tco2e,
        "cause": classification["cause"],
        "cause_confidence": classification["confidence"],
        "reasoning": classification["reasoning"],
        "adversarial_notes": adversarial_notes,
        "combined_confidence": posterior,
        "status": status,
        "tier": tier,
    }


def _persist_detection(session: Session, zone: Zone, result: dict) -> Detection:
    """INPUTS: a Session (always the main thread's - see run_once()), the Zone the result
    belongs to, a dict from _classify_and_fuse(). OUTPUTS: the persisted Detection row. Builds +
    commits the Detection row, then fires an alert (ntfy + SMS stub) if status == 'alerted' and
    this zone+cause hasn't already alerted within alerts.ntfy.DEDUPE_WINDOW_DAYS. This is the
    DB-writing half of the old single _finalize_detection() - kept off worker threads so SQLite
    only ever sees one writer for the Detection/Alert tables at a time (see app.db.engine's
    SQLite busy-timeout comment for the one place that still isn't true: detect_candidates()'s
    ZoneBaseline cache read/write, which necessarily happens on each zone's own worker thread).
    Also spawns background Sentinel-2 thumbnail generation for the new row (gee.thumbnails) -
    submitted to a shared executor and returned from immediately, so this never waits on real
    GEE network calls; a no-op under SYNTHETIC_MODE, where the frontend renders its own inline
    SVG tiles instead."""
    detection = Detection(
        zone_id=zone.id,
        ndvi_z=result["ndvi_delta"],
        bsi_z=result["bsi_delta"],
        viirs_z=result["viirs_delta"],
        rainfall_percentile=result["rainfall_pct"],
        cause=result["cause"],
        cause_confidence=result["cause_confidence"],
        reasoning=result["reasoning"],
        adversarial_notes=result["adversarial_notes"],
        combined_confidence=result["combined_confidence"],
        estimated_area_ha=result["area_ha"],
        status=result["status"],
        tier=result["tier"],
        as_of=result["as_of_str"],
        carbon_loss_tco2e=result["carbon_loss_tco2e"],
    )
    session.add(detection)
    session.commit()
    session.refresh(detection)

    spawn_thumbnail_generation(detection.id, zone.aoi_geojson, detection.detected_at)

    if result["status"] == "alerted":
        sent = send_alert(
            zone_id=zone.id,
            zone_name=zone.name,
            cause=result["cause"],
            confidence=result["combined_confidence"],
            area_ha=result["area_ha"],
            ndvi_z=result["ndvi_delta"],
            bsi_z=result["bsi_delta"],
            viirs_z=result["viirs_delta"],
            rainfall_percentile=result["rainfall_pct"],
            tier=result["tier"],
            llm_conf=result["cause_confidence"],
            carbon_loss_tco2e=result["carbon_loss_tco2e"],
        )
        if sent:
            message = build_alert_message(
                zone_name=zone.name,
                cause=result["cause"],
                confidence=result["combined_confidence"],
                area_ha=result["area_ha"],
                ndvi_z=result["ndvi_delta"],
                bsi_z=result["bsi_delta"],
                viirs_z=result["viirs_delta"],
                rainfall_percentile=result["rainfall_pct"],
                tier=result["tier"],
                llm_conf=result["cause_confidence"],
                carbon_loss_tco2e=result["carbon_loss_tco2e"],
            )
            send_sms(message)
            session.add(Alert(detection_id=detection.id, zone_id=zone.id, message=message, channel="ntfy+sms_stub"))
            session.commit()

    return detection


def process_zone(zone: Zone) -> dict:
    """INPUTS: a Zone row. OUTPUTS: a _classify_and_fuse() result dict, ready for
    _persist_detection(). Synthetic-mode-aware default path: baseline -> change detection ->
    rainfall lookup, then _classify_and_fuse(). Takes no DB session - none of these three calls
    touch the database - which is what lets run_once() run it on a worker thread."""
    month = datetime.utcnow().month
    baseline = compute_zone_baseline(zone.name, month, zone.aoi_geojson)
    indicators = detect_change(zone.name, zone.zone_type, baseline, zone.aoi_geojson)
    rainfall_pct = get_rainfall_percentile(zone.name, zone.zone_type, zone.aoi_geojson)
    return _classify_and_fuse(
        zone, indicators.ndvi_z, indicators.bsi_z, indicators.viirs_z, rainfall_pct, indicators.estimated_area_ha
    )


def process_zone_as_of(session: Session, zone: Zone, as_of: datetime) -> Optional[dict]:
    """INPUTS: a Session scoped to the caller (see _zone_worker() - never shared across
    threads), a Zone row, the historical as-of date to evaluate. OUTPUTS: a _classify_and_fuse()
    result dict, or None if Earth Engine had no valid current-window imagery (NoValidPixels) -
    that zone is skipped for this pass, never synthesized. Real Earth Engine equivalent of
    process_zone(), for Stage 3/4's --as-of flag: detect_candidates() (per-indicator 3-sigma
    flagging against the cached ZoneBaseline) in place of the synthetic detect_change(), then
    _classify_and_fuse(). Picks the strongest candidate's z per indicator when flagged, 0.0
    (neutral) when not; area_ha is the max across any flagged indicators (avoids double-counting
    overlapping pixels between an ndvi- and bsi-flagged mask)."""
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

    return _classify_and_fuse(
        zone,
        ndvi_delta,
        bsi_delta,
        viirs_delta,
        rainfall_pct,
        area_ha,
        as_of.strftime("%Y-%m-%d"),
        carbon_loss_tco2e=carbon_loss_tco2e,
    )


def _zone_worker(zone: Zone, as_of: Optional[datetime]) -> dict:
    """INPUTS: a Zone row, the historical as-of date (None for the synthetic default path).
    OUTPUTS: {"zone_id", "zone_name", "result", "error"} - result is a
    process_zone()/process_zone_as_of() result dict, or None for a legitimate NoValidPixels skip
    (never an error); error is str(exc) if this zone's processing raised, else None. Never
    raises itself - any exception is caught here and returned as data instead of propagating to
    run_once()'s future.result(), on the same principle as viirs_z=None elsewhere in this file:
    a zone that errored must not be indistinguishable from a zone that legitimately found
    nothing. Runs on one of run_once()'s worker threads. The --as-of path opens its own
    short-lived Session here (closed before returning) because detect_candidates() reads/writes
    the ZoneBaseline cache and a Session can't be shared across threads - this is never the
    session run_once() later uses to persist the eventual Detection."""
    try:
        if as_of is not None:
            session = Session(engine, expire_on_commit=False)
            try:
                result = process_zone_as_of(session, zone, as_of)
            finally:
                session.close()
        else:
            result = process_zone(zone)
        return {"zone_id": zone.id, "zone_name": zone.name, "result": result, "error": None}
    except Exception as exc:
        logger.exception("zone %s raised during concurrent processing", zone.name)
        return {"zone_id": zone.id, "zone_name": zone.name, "result": None, "error": str(exc)}


def run_once(as_of: Optional[datetime] = None) -> dict:
    """INPUTS: optional historical as-of date - None (default) uses the synthetic-mode-aware
    process_zone() path; if set, uses the real Earth Engine process_zone_as_of() path instead.
    OUTPUTS: {fired, alert_id, tier, zone, posterior, zones_checked, zones_errored, per_zone} -
    the same shape app.main's /run-once endpoint now returns unchanged (see its docstring).
    fired/alert_id/tier/zone/posterior describe the first zone whose Detection.status ==
    'alerted' this pass, or all None/False if none did. zones_checked counts zones that produced
    a real, persisted Detection (silent or alerted) this pass - NOT zones skipped via
    NoValidPixels, NOT zones abandoned past SOFT_ZONE_DEADLINE_S, and NOT zones that errored.
    zones_errored lists the names of zones whose _zone_worker() call raised - kept distinct from
    a silent zone on the same principle as viirs_z=None elsewhere in this file: "errored" and
    "found nothing" must never collapse into the same signal. per_zone has one entry per zone
    actually accounted for this pass (checked or errored - not skipped/abandoned):
    {"zone", "status": "silent"|"alerted"|"error", "posterior", "error"}. A zone still running
    past SOFT_ZONE_DEADLINE_S - typically one paying the cost of a cold, uncached ZoneBaseline
    (see gee.baselines.zones_missing_baseline, which app.main's /run-once checks *before* ever
    calling this) - is simply left out of per_zone entirely rather than failing the whole run.
    Entry point for `python scripts/run_once.py` and for the API's POST /run-once trigger, which
    passes the same as_of through.

    Each zone's detect -> classify -> fuse work (_zone_worker) runs concurrently across up to
    ZONE_WORKER_POOL_SIZE zones, since it's almost entirely spent waiting on Earth Engine and
    Anthropic network calls rather than local compute - sequentially, the same work took roughly
    one zone's latency times the zone count. _zone_worker() never raises - any exception is
    caught there and returned as that zone's "error" field instead, so one bad zone can't kill
    the whole run. Every actual DB write (the Detection row, and its Alert if it fires) still
    happens serially afterward on this function's own main-thread session via
    _persist_detection() - see app.db.engine's SQLite busy-timeout comment for the one exception
    (each worker's own ZoneBaseline read/write)."""
    empty_summary = {
        "fired": False,
        "alert_id": None,
        "tier": None,
        "zone": None,
        "posterior": None,
        "zones_checked": 0,
        "zones_errored": [],
        "per_zone": [],
    }
    init_db()
    with Session(engine, expire_on_commit=False) as session:
        zones = session.exec(select(Zone)).all()
    if not zones:
        print("No zones found. Run scripts/seed_zones.py first.")
        return empty_summary

    worker_outputs_by_zone_id: dict[int, dict] = {}
    executor = ThreadPoolExecutor(max_workers=min(ZONE_WORKER_POOL_SIZE, len(zones)))
    try:
        future_to_zone = {executor.submit(_zone_worker, zone, as_of): zone for zone in zones}
        done, not_done = wait(future_to_zone, timeout=SOFT_ZONE_DEADLINE_S)
        for future in done:
            zone = future_to_zone[future]
            try:
                worker_output = future.result()
            except Exception as exc:
                # _zone_worker() catches its own exceptions and always returns a dict - this is
                # a last-resort net for something failing outside that (e.g. the executor itself
                # breaking), not the normal per-zone error path.
                logger.exception("zone %s raised outside _zone_worker's own error handling", zone.name)
                worker_output = {"zone_id": zone.id, "zone_name": zone.name, "result": None, "error": str(exc)}
            worker_outputs_by_zone_id[zone.id] = worker_output
        if not_done:
            slow_zones = ", ".join(future_to_zone[f].name for f in not_done)
            print(
                f"  [TIMEOUT] {len(not_done)} zone(s) still running past the "
                f"{SOFT_ZONE_DEADLINE_S}s soft deadline ({slow_zones}) - returning results for "
                "the rest instead of failing the whole run."
            )
    finally:
        # Do NOT wait for not_done to finish - concurrent.futures can't cancel a running thread,
        # so shutdown(wait=False) lets this function return now while those zones keep running
        # (and, for --as-of, still populate the ZoneBaseline cache for next time) in the
        # background, same tradeoff app.main.run_once_endpoint already makes at the request level.
        executor.shutdown(wait=False)

    per_zone: list[dict] = []
    zones_errored: list[str] = []
    zones_checked = 0
    summary = dict(empty_summary)

    with Session(engine, expire_on_commit=False) as session:
        for zone in zones:
            worker_output = worker_outputs_by_zone_id.get(zone.id)
            if worker_output is None:
                continue  # abandoned past SOFT_ZONE_DEADLINE_S - not checked, not errored

            if worker_output["error"] is not None:
                zones_errored.append(zone.name)
                per_zone.append(
                    {"zone": zone.name, "status": "error", "posterior": None, "error": worker_output["error"]}
                )
                print(f"[ERROR] {zone.name}: {worker_output['error']}")
                continue

            result = worker_output["result"]
            if result is None:
                continue  # NoValidPixels skip under --as-of - not checked, not errored

            zones_checked += 1
            detection = _persist_detection(session, zone, result)
            per_zone.append(
                {
                    "zone": zone.name,
                    "status": detection.status,
                    "posterior": detection.combined_confidence,
                    "error": None,
                }
            )
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

            if detection.status == "alerted" and not summary["fired"]:
                alert = session.exec(select(Alert).where(Alert.detection_id == detection.id)).first()
                summary.update(
                    fired=True,
                    alert_id=alert.id if alert is not None else None,
                    tier=detection.tier,
                    zone=zone.name,
                    posterior=detection.combined_confidence,
                )

    summary["zones_checked"] = zones_checked
    summary["zones_errored"] = zones_errored
    summary["per_zone"] = per_zone
    return summary


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
