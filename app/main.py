import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from analysis.ldn import (
    compute_ldn_score,
    estimate_agb_t_per_ha,
    estimate_carbon_loss_tco2e,
    estimate_zone_carbon_loss,
)
from app.config import settings
from app.db import Alert, Detection, Zone, ZoneBaseline, engine, init_db
from app.schemas import (
    AlertRead,
    DetectionRead,
    SilentLogEntry,
    ZoneLastDetection,
    ZoneRead,
)
from fusion.bayes import explain_silence
from gee.init import init_ee
from gee.thumbnails import get_before_after_thumbs

RUN_ONCE_TIMEOUT_S = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geosentry")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration.json"
EMPTY_CALIBRATION = {
    "generated_at": None,
    "total_windows_tested": 0,
    "total_zones_tested": 0,
    "total_candidates_flagged": 0,
    "total_alerts_fired": 0,
    "windows": [],
}


def _backfill_carbon_loss() -> None:
    """INPUTS: none. OUTPUTS: none. Populates Detection.carbon_loss_tco2e for pre-existing rows
    (created before the column existed, or before gee.detect started computing it) that have a
    usable area_ha and a resolvable NDVI baseline for their zone+month. Skips - never guesses -
    rows where either is missing. Idempotent: only touches rows where carbon_loss_tco2e IS NULL,
    so it's cheap to run on every startup."""
    with Session(engine) as session:
        candidates = session.exec(
            select(Detection).where(Detection.carbon_loss_tco2e == None, Detection.estimated_area_ha > 0)  # noqa: E711
        ).all()
        if not candidates:
            return

        baseline_median_by_zone_month: dict[tuple[int, int], Optional[float]] = {}
        updated = 0
        for detection in candidates:
            key = (detection.zone_id, detection.detected_at.month)
            if key not in baseline_median_by_zone_month:
                baseline_row = session.exec(
                    select(ZoneBaseline).where(
                        ZoneBaseline.zone_id == detection.zone_id,
                        ZoneBaseline.calendar_month == detection.detected_at.month,
                        ZoneBaseline.indicator == "ndvi",
                    )
                ).first()
                baseline_median_by_zone_month[key] = baseline_row.median if baseline_row is not None else None

            ndvi_median = baseline_median_by_zone_month[key]
            if ndvi_median is None:
                continue

            agb_t_per_ha = estimate_agb_t_per_ha(ndvi_median)
            detection.carbon_loss_tco2e = estimate_carbon_loss_tco2e(agb_t_per_ha, detection.estimated_area_ha)
            session.add(detection)
            updated += 1

        if updated:
            session.commit()
        logger.info(
            "Carbon-loss backfill: updated %d of %d candidate Detection row(s) (rest skipped - no resolvable baseline).",
            updated,
            len(candidates),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not settings.synthetic_mode:
        # init_ee() (Stage 1) reads GOOGLE_APPLICATION_CREDENTIALS/GEE_PROJECT straight from
        # os.environ, which pydantic-settings' own .env parsing never populates - load_dotenv()
        # first, same pattern already used in scripts/run_once.py's --as-of path. Guarded so
        # SYNTHETIC_MODE=true (the default) never touches real credentials at all.
        from dotenv import load_dotenv

        load_dotenv()
        init_ee()
    _backfill_carbon_loss()
    logger.info("GeoSentry started. SYNTHETIC_MODE=%s", settings.synthetic_mode)
    yield


app = FastAPI(title="GeoSentry", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index() -> FileResponse:
    """INPUTS: none. OUTPUTS: the single-page frontend (web/index.html)."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """INPUTS: none. OUTPUTS: dict with service status and current mode, for uptime checks
    and a quick demo sanity check."""
    return {"status": "ok", "synthetic_mode": settings.synthetic_mode}


@app.get("/zones", response_model=list[ZoneRead])
def list_zones() -> list[ZoneRead]:
    """INPUTS: none. OUTPUTS: all zones in the database, each with its most recent Detection (if
    any) embedded as last_detection, and its real AOI polygon (Zone.aoi_geojson, the same
    geometry every real GEE query in this pipeline uses - not a generated fallback, every zone
    already has one) as geometry - the shape the map/zone-list UI needs in one round trip."""
    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        result = []
        for zone in zones:
            latest = session.exec(
                select(Detection)
                .where(Detection.zone_id == zone.id)
                .order_by(Detection.detected_at.desc())
                .limit(1)
            ).first()
            last_detection = (
                ZoneLastDetection(
                    cause=latest.cause,
                    confidence=latest.combined_confidence,
                    area_ha=latest.estimated_area_ha,
                    detected_at=latest.detected_at,
                )
                if latest is not None
                else None
            )
            try:
                geometry = json.loads(zone.aoi_geojson) if zone.aoi_geojson else None
            except (TypeError, ValueError):
                geometry = None
            result.append(
                ZoneRead(
                    id=zone.id,
                    name=zone.name,
                    role=zone.zone_type,
                    lat=float(zone.centroid_lat),
                    lon=float(zone.centroid_lon),
                    created_at=zone.created_at,
                    last_detection=last_detection,
                    geometry=geometry,
                )
            )
        return result


@app.get("/detections", response_model=list[DetectionRead])
def list_detections(zone_id: Optional[int] = None, limit: int = 50) -> list[DetectionRead]:
    """INPUTS: optional zone_id filter, result limit. OUTPUTS: most recent detections, newest
    first, each with before/after thumbnail data URIs when SYNTHETIC_MODE is false (real S2
    imagery via gee.thumbnails); left null in synthetic mode, where the frontend renders its own
    inline SVG tiles instead."""
    with Session(engine) as session:
        query = (
            select(Detection, Zone)
            .join(Zone, Detection.zone_id == Zone.id)
            .order_by(Detection.detected_at.desc())
            .limit(limit)
        )
        if zone_id is not None:
            query = query.where(Detection.zone_id == zone_id)
        rows = session.exec(query).all()

    entries = []
    for detection, zone in rows:
        before_url = after_url = None
        if not settings.synthetic_mode:
            thumbs = get_before_after_thumbs(zone.aoi_geojson, detection.detected_at)
            if thumbs is not None:
                before_url, after_url = thumbs
        entries.append(
            DetectionRead(
                id=detection.id,
                zone_id=detection.zone_id,
                detected_at=detection.detected_at,
                ndvi_z=detection.ndvi_z,
                bsi_z=detection.bsi_z,
                viirs_z=detection.viirs_z,
                rainfall_percentile=detection.rainfall_percentile,
                cause=detection.cause,
                cause_confidence=detection.cause_confidence,
                reasoning=detection.reasoning,
                adversarial_notes=detection.adversarial_notes,
                combined_confidence=detection.combined_confidence,
                estimated_area_ha=detection.estimated_area_ha,
                status=detection.status,
                tier=detection.tier,
                before_thumb_url=before_url,
                after_thumb_url=after_url,
                carbon_loss_tco2e=detection.carbon_loss_tco2e,
            )
        )
    return entries


@app.get("/alerts", response_model=list[AlertRead])
def list_alerts(limit: int = 50) -> list[Alert]:
    """INPUTS: optional result limit. OUTPUTS: most recent alerts, newest first."""
    with Session(engine) as session:
        query = select(Alert).order_by(Alert.sent_at.desc()).limit(limit)
        return session.exec(query).all()


@app.get("/silent-log", response_model=list[SilentLogEntry])
def silent_log(limit: int = 50) -> list[SilentLogEntry]:
    """INPUTS: optional result limit (default 50). OUTPUTS: every Detection whose fused
    posterior fell below settings.alert_confidence_threshold - i.e. it never fired an alert -
    newest first, each with a why_silent line computed at request time by
    fusion.bayes.explain_silence(). Distinct, thinner shape from /detections: built for the
    silent-log panel, not the full detections feed."""
    with Session(engine) as session:
        query = (
            select(Detection, Zone)
            .join(Zone, Detection.zone_id == Zone.id)
            .where(Detection.combined_confidence < settings.alert_confidence_threshold)
            .order_by(Detection.detected_at.desc())
            .limit(limit)
        )
        rows = session.exec(query).all()

    entries = []
    for detection, zone in rows:
        evidence = {
            "llm_confidence": detection.cause_confidence,
            "fused_confidence": detection.combined_confidence,
            "area_ha": detection.estimated_area_ha,
            "viirs_delta": detection.viirs_z,
            "rainfall_pct": detection.rainfall_percentile,
        }
        entries.append(
            SilentLogEntry(
                id=detection.id,
                zone_name=zone.name,
                zone_role=zone.zone_type,
                cause=detection.cause,
                ndvi_delta=detection.ndvi_z,
                bsi_delta=detection.bsi_z,
                reasoning=detection.reasoning,
                created_at=detection.detected_at,
                why_silent=explain_silence(evidence, settings.alert_confidence_threshold),
                tier=detection.tier,
                **evidence,
            )
        )
    return entries


def _combined_suppressed_carbon_tco2e(windows: list[dict]) -> float:
    """INPUTS: calibration.json's 'windows' list. OUTPUTS: summed
    analysis.ldn.estimate_carbon_loss_tco2e() across every window whose verdict stayed 'silent'
    (a suppressed candidate) - i.e. what SDG 15.3.1-relevant loss the confidence gate held back
    an alert on. calibration.json stores zone *names*, not ids, and has no carbon numbers of its
    own, so this resolves each window's zone + calendar month against the live DB's ZoneBaseline
    NDVI median (already real, already seeded) rather than regenerating the whole historical
    sweep. Windows whose zone or baseline can't be resolved are skipped, not zero-filled."""
    total = 0.0
    with Session(engine) as session:
        zones_by_name = {zone.name: zone for zone in session.exec(select(Zone)).all()}
        for window in windows:
            if window.get("tier") != "silent":
                continue
            zone = zones_by_name.get(window.get("zone"))
            if zone is None:
                continue
            try:
                month = datetime.strptime(window["as_of"], "%Y-%m-%d").month
            except (KeyError, ValueError, TypeError):
                continue
            baseline_row = session.exec(
                select(ZoneBaseline).where(
                    ZoneBaseline.zone_id == zone.id,
                    ZoneBaseline.calendar_month == month,
                    ZoneBaseline.indicator == "ndvi",
                )
            ).first()
            if baseline_row is None:
                continue
            agb_t_per_ha = estimate_agb_t_per_ha(baseline_row.median, zone.name)
            total += estimate_carbon_loss_tco2e(agb_t_per_ha, window.get("area_ha", 0.0))
    return round(total, 2)


@app.get("/calibration")
def get_calibration() -> dict:
    """INPUTS: none. OUTPUTS: the contents of calibration.json, built by
    scripts/build_calibration.py - a real historical backtest across every zone showing every
    flagged candidate and why its posterior did or didn't clear the alert gate - plus a derived
    combined_suppressed_carbon_tco2e field (see _combined_suppressed_carbon_tco2e()). Returns an
    empty summary (all-zero counts, no windows) if the file hasn't been generated yet, rather than
    erroring, so the UI panel can render a friendly "not generated yet" state."""
    data = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else dict(EMPTY_CALIBRATION)
    data["combined_suppressed_carbon_tco2e"] = _combined_suppressed_carbon_tco2e(data.get("windows", []))
    return data


@app.get("/stats")
def get_stats() -> dict:
    """INPUTS: none. OUTPUTS: header-strip dashboard stats: zones_monitored (Zone row count),
    windows_evaluated (see below), alerts_fired (Alert row count), false_positives (hardcoded 0 -
    this demo has no ground-truth verification loop to compute a real figure from), synthetic_mode,
    and last_check_at (the most recent Detection.detected_at, or null if none exist yet)."""
    with Session(engine) as session:
        zones_monitored = len(session.exec(select(Zone)).all())
        alerts_fired = len(session.exec(select(Alert)).all())
        latest = session.exec(select(Detection).order_by(Detection.detected_at.desc()).limit(1)).first()

        # windows_evaluated reads from calibration.json when present because it represents the
        # historical backtest sweep (scripts/build_calibration.py) - results that live in that
        # file, not all of which are persisted back into the Detection table. Fall back to
        # counting distinct Detection.as_of values (live --as-of / POST /run-once passes) when
        # calibration.json hasn't been generated yet, then to 0.
        if CALIBRATION_PATH.exists():
            windows_evaluated = json.loads(CALIBRATION_PATH.read_text()).get("total_windows_tested", 0)
        else:
            as_of_values = session.exec(select(Detection.as_of)).all()
            windows_evaluated = len({v for v in as_of_values if v})

    return {
        "zones_monitored": zones_monitored,
        "windows_evaluated": windows_evaluated,
        "alerts_fired": alerts_fired,
        "false_positives": 0,
        "synthetic_mode": settings.synthetic_mode,
        "last_check_at": latest.detected_at.isoformat() if latest is not None else None,
    }


@app.get("/ldn/summary")
def get_ldn_summary() -> dict:
    """INPUTS: none. OUTPUTS: {avg_score, total_tco2e, zones_at_risk} - avg_score is the mean
    analysis.ldn.compute_ldn_score() ldn_score across every zone (rounded), total_tco2e sums
    analysis.ldn.estimate_zone_carbon_loss() across every zone, zones_at_risk counts zones with
    ldn_score < 40 (the same red threshold the frontend badges use). All-zero if there are no
    zones yet."""
    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        if not zones:
            return {"avg_score": 0, "total_tco2e": 0.0, "zones_at_risk": 0}

        scores = []
        total_tco2e = 0.0
        zones_at_risk = 0
        for zone in zones:
            result = compute_ldn_score(zone.id, session)
            scores.append(result["ldn_score"])
            if result["ldn_score"] < 40:
                zones_at_risk += 1
            total_tco2e += estimate_zone_carbon_loss(zone.id, session)["total_tco2e"]

    return {
        "avg_score": round(sum(scores) / len(scores)),
        "total_tco2e": round(total_tco2e, 2),
        "zones_at_risk": zones_at_risk,
    }


@app.get("/ldn/{zone_id}")
def get_ldn_score(zone_id: int) -> dict:
    """INPUTS: zone_id path param. OUTPUTS: analysis.ldn.compute_ldn_score(zone_id, db) - the
    zone's SDG 15.3.1-aligned composite score plus its three sub-scores. 404 if the zone doesn't
    exist."""
    with Session(engine) as session:
        zone = session.get(Zone, zone_id)
        if zone is None:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found.")
        return compute_ldn_score(zone_id, session)


@app.post("/run-once")
def run_once_endpoint(as_of: Optional[str] = None) -> dict:
    """INPUTS: as_of - required query param, 'YYYY-MM-DD'. OUTPUTS: {fired, alert_id, tier, zone,
    posterior} summarizing the run (the first zone whose Detection.status == 'alerted', or all
    None/False if every zone stayed silent); {"error": "timeout", "message": ...} if the pipeline
    takes longer than RUN_ONCE_TIMEOUT_S; or {"error": "gee_unavailable", "message": ..., "detail":
    ...} if Earth Engine itself raised (quota, auth, a malformed server-side call - the routine
    "no imagery for this window" case is already handled per-zone inside run_once() itself via
    gee.detect.NoValidPixels and never reaches here). All three are clean 200s, not 500s - a
    legitimate question deserves a legitimate, honest answer, not a generic unhandled-exception
    page; the full exception is still in the server log either way. 400 if as_of is missing or
    not YYYY-MM-DD. Runs the exact same scripts.run_once.run_once(as_of=...) used by the CLI's
    --as-of flag - real Earth Engine calls across every zone, no detection-logic changes.
    run_once() is synchronous/blocking (the ee SDK has no async API), so it's run in a worker
    thread with a hard timeout via concurrent.futures. Python can't forcibly cancel a running
    thread, so on timeout this request returns immediately without waiting for the executor to
    drain (shutdown(wait=False)) - the in-flight GEE calls keep running in the background until
    they finish or error, and their results are simply not returned to this request."""
    if not as_of:
        raise HTTPException(status_code=400, detail="as_of query parameter is required (YYYY-MM-DD).")
    try:
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be in YYYY-MM-DD format.")

    from scripts.run_once import run_once

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(run_once, as_of_date)
        detections = future.result(timeout=RUN_ONCE_TIMEOUT_S)
    except FuturesTimeoutError:
        # Do NOT wait for the executor to finish. Return immediately.
        return {
            "error": "timeout",
            "message": "Pipeline exceeded 60s. Try a different date.",
        }
    except ee.EEException as exc:
        # A real, unexpected GEE-side error (quota, auth, a malformed server-side call, etc.) -
        # NOT the routine "no imagery for this window" case, which gee.detect.NoValidPixels
        # already catches per-zone inside run_once() itself and simply skips that zone, never
        # reaching here. This is deliberately its own 200 (not a 500): the client asked a
        # legitimate question and got a legitimate, honest answer - "Earth Engine errored" - not
        # a generic unhandled-exception page. It is NOT swallowed silently: the full exception is
        # still in the server log from wherever it was raised.
        return {
            "error": "gee_unavailable",
            "message": "Earth Engine returned an error for this date. Try a different window or check the server log.",
            "detail": str(exc)[:300],
        }
    finally:
        # shutdown(wait=False) so the caller doesn't block on the hung thread
        executor.shutdown(wait=False)

    fired = next((d for d in detections if d.status == "alerted"), None)
    if fired is None:
        return {"fired": False, "alert_id": None, "tier": None, "zone": None, "posterior": None}

    with Session(engine) as session:
        alert = session.exec(select(Alert).where(Alert.detection_id == fired.id)).first()
        zone = session.exec(select(Zone).where(Zone.id == fired.zone_id)).first()

    return {
        "fired": True,
        "alert_id": alert.id if alert is not None else None,
        "tier": fired.tier,
        "zone": zone.name if zone is not None else None,
        "posterior": fired.combined_confidence,
    }
