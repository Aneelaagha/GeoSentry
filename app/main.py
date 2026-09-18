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
from fastapi.staticfiles import StaticFiles
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
from fusion.bayes import classify_alert_tier, explain_silence, fuse_confidence
from gee.init import init_ee
from gee.thumbnails import THUMBNAILS_DIR, cached_thumbnail_urls, spawn_thumbnail_generation

YANOMAMI_DEMO_ZONE_ID = 5  # "Yanomami Mining Belt" - see scripts/seed_zones.py's DEMO_ZONES
YANOMAMI_DEMO_DETECTION = {
    # Documented March 2023 garimpo (illegal artisanal gold mining) surge in Yanomami
    # Indigenous Territory, Roraima - the event that made international headlines over the
    # humanitarian crisis there. Indicator deltas are illustrative of that event's scale, not
    # pulled from a specific satellite pass; every value downstream of this dict is real,
    # production fusion/alerting logic - see _run_yanomami_2023_demo().
    "ndvi_delta": -4.50,
    "bsi_delta": 3.80,
    "viirs_delta": 3.20,
    "rainfall_pct": 45,
    "area_ha": 412.0,
    "llm_cause": "mining",
    "llm_conf": 0.82,
}

RUN_ONCE_TIMEOUT_S = 120  # a backstop, not the real budget: scripts.run_once.run_once() now
# processes zones concurrently and self-limits to its own SOFT_ZONE_DEADLINE_S (110s) before
# returning whatever finished, so a normal run comfortably lands well under this. This request-
# level timeout only fires if something goes wrong beyond that internal budget (e.g. persisting
# results on the main thread runs unexpectedly long). The real "this month is going to be slow"
# case is caught earlier and explicitly - see zones_missing_baseline() below.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geosentry")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration.json"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)  # StaticFiles below requires the dir to exist
# at mount time - created eagerly here (module import time), not deferred to lifespan startup.
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


def _sweep_missing_thumbnails() -> None:
    """INPUTS: none. OUTPUTS: none. Runs once at startup (real-imagery mode only): for every
    Detection row still missing one or both cached thumbnail files - rows persisted before this
    cache existed, or whose earlier generation attempt failed or timed out - spawns the same
    background generation task a live /run-once completion uses
    (gee.thumbnails.spawn_thumbnail_generation). Doesn't block startup: every job lands on the
    shared bounded executor (gee.thumbnails.THUMBNAIL_WORKERS) and this function returns as soon
    as they're all submitted, not when they finish."""
    if settings.synthetic_mode:
        return
    with Session(engine) as session:
        # Newest first - so whatever's actually visible in the detection stream right now (it
        # renders newest-first too) reaches the front of the shared executor's queue before a
        # backlog of older rows.
        rows = session.exec(
            select(Detection, Zone).join(Zone, Detection.zone_id == Zone.id).order_by(Detection.detected_at.desc())
        ).all()
    spawned = 0
    for detection, zone in rows:
        before_url, after_url = cached_thumbnail_urls(detection.id)
        if before_url is None or after_url is None:
            spawn_thumbnail_generation(detection.id, zone.aoi_geojson, detection.detected_at)
            spawned += 1
    if spawned:
        logger.info("Thumbnail sweep: queued background generation for %d detection(s) missing a cached thumbnail.", spawned)


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
    _sweep_missing_thumbnails()
    logger.info("GeoSentry started. SYNTHETIC_MODE=%s", settings.synthetic_mode)
    yield


app = FastAPI(title="GeoSentry", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves cached Sentinel-2 before/after thumbnail PNGs (gee.thumbnails) as plain static files -
# GET /detections just checks whether these exist on disk (cached_thumbnail_urls()) rather than
# fetching from Earth Engine on every request.
app.mount("/static", StaticFiles(directory=THUMBNAILS_DIR.parent), name="static")


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
    first, each with before/after thumbnail URLs ('/static/thumbnails/...') when SYNTHETIC_MODE
    is false AND that detection's real S2 thumbnail is already cached on disk
    (gee.thumbnails.cached_thumbnail_urls) - null otherwise, whether because generation hasn't
    finished yet, failed, or SYNTHETIC_MODE is on (that mode renders its own inline SVG tiles
    client-side and never has cached files). This is a pure file-existence check, never a live
    GEE call - thumbnails are generated out-of-band by a background task spawned when the
    Detection was persisted (scripts.run_once._persist_detection) or by app.main's startup
    sweep, so this endpoint stays fast (page-load target: under 2s) no matter how many
    thumbnails are still missing or in flight. Also spawns generation for any row in this page
    that's still missing one - covers a detection persisted before the app restarted the
    executor, or one whose earlier attempt failed - without blocking this response."""
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
            before_url, after_url = cached_thumbnail_urls(detection.id)
            if before_url is None or after_url is None:
                spawn_thumbnail_generation(detection.id, zone.aoi_geojson, detection.detected_at)
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


def _run_yanomami_2023_demo() -> dict:
    """Demo endpoint: replays a documented 2023 Yanomami garimpo event
    through the full fusion and alerting pipeline. Detection values are
    from a real historical case; every downstream step uses the
    production code path.

    Skips only the LLM classification + adversarial-challenge step (there is no live imagery to
    classify - the cause/confidence below are fixed, historical-record inputs) and goes straight
    into fuse_confidence() and classify_alert_tier() (fusion.bayes), then
    scripts.run_once._persist_detection() - the same function real /run-once passes use to commit
    the Detection row and, if it clears the alert gate, fire the real ntfy push (and SMS stub)
    with the real 7-day zone+cause dedupe."""
    from scripts.run_once import BASE_PRIOR, NON_ALERTING_CAUSES, _persist_detection

    with Session(engine, expire_on_commit=False) as session:
        zone = session.get(Zone, YANOMAMI_DEMO_ZONE_ID)
        if zone is None:
            raise HTTPException(
                status_code=404,
                detail=f"Zone {YANOMAMI_DEMO_ZONE_ID} not found. Run scripts/seed_zones.py first.",
            )

        d = YANOMAMI_DEMO_DETECTION
        posterior = fuse_confidence(
            prior=BASE_PRIOR,
            evidence={
                "llm_conf": d["llm_conf"],
                "viirs_z": d["viirs_delta"],
                "rain_percentile": d["rainfall_pct"],
                "area_ha": d["area_ha"],
            },
        )
        tier = classify_alert_tier(d["llm_conf"], posterior, settings.alert_confidence_threshold)
        if tier == "silent":
            status = "silent"
        elif tier == "classified" and d["llm_cause"] in NON_ALERTING_CAUSES:
            status = "silent"
        else:
            status = "alerted"

        result = {
            "ndvi_delta": d["ndvi_delta"],
            "bsi_delta": d["bsi_delta"],
            "viirs_delta": d["viirs_delta"],
            "rainfall_pct": d["rainfall_pct"],
            "area_ha": d["area_ha"],
            "as_of_str": None,
            "carbon_loss_tco2e": None,
            "cause": d["llm_cause"],
            "cause_confidence": d["llm_conf"],
            "reasoning": "Demo replay of the documented March 2023 Yanomami garimpo surge (not LLM-classified).",
            "adversarial_notes": "Demo replay - adversarial challenge skipped; cause/confidence are fixed historical-case inputs.",
            "combined_confidence": posterior,
            "status": status,
            "tier": tier,
        }
        detection = _persist_detection(session, zone, result)
        alert = session.exec(select(Alert).where(Alert.detection_id == detection.id)).first()

        return {
            "fired": detection.status == "alerted",
            "alert_id": alert.id if alert is not None else None,
            "tier": detection.tier,
            "zone": zone.name,
            "posterior": detection.combined_confidence,
            "demo": "yanomami-2023",
        }


@app.post("/run-once")
def run_once_endpoint(as_of: Optional[str] = None, demo: Optional[str] = None) -> dict:
    """INPUTS: as_of - 'YYYY-MM-DD', required unless demo is set. demo - optional; the only
    recognized value is 'yanomami-2023', which ignores as_of entirely and instead calls
    _run_yanomami_2023_demo() (see its docstring) - a fixed, documented historical detection
    replayed through the real fusion/alerting pipeline, for demoing the alert path without
    waiting on live Earth Engine + LLM calls. OUTPUTS (as_of path): whatever
    scripts.run_once.run_once(as_of=...) returns, passed through unchanged - {fired, alert_id,
    tier, zone, posterior, zones_checked, zones_errored, per_zone} (see that function's
    docstring for the full field-by-field meaning); {"error": "baseline_missing", "message": ...,
    "cached_months": [...], "zones": [...]} immediately, without running anything, if any zone
    lacks a fresh ZoneBaseline for as_of's calendar month (the expensive-if-cold case - see
    gee.baselines.zones_missing_baseline); {"error": "timeout", "message": ...} if the pipeline
    still takes longer than RUN_ONCE_TIMEOUT_S despite that pre-check; or {"error":
    "gee_unavailable", "message": ..., "detail": ...} if Earth Engine itself raised (quota, auth,
    a malformed server-side call - the routine "no imagery for this window" case is already
    handled per-zone inside run_once() itself via gee.detect.NoValidPixels and never reaches
    here, and a per-zone processing exception is now also handled inside run_once() itself via
    _zone_worker() and surfaced as that zone's entry in zones_errored/per_zone, likewise never
    reaching here). All four are clean 200s, not 500s - a legitimate question deserves a
    legitimate, honest answer, not a generic unhandled-exception page; the full exception is
    still in the server log either way. 400 if as_of is missing or not YYYY-MM-DD. Runs the exact
    same scripts.run_once.run_once(as_of=...) used by the CLI's --as-of flag - real Earth Engine
    calls across every zone, no detection-logic changes.
    run_once() is synchronous/blocking (the ee SDK has no async API), so it's run in a worker
    thread with a hard timeout via concurrent.futures. Python can't forcibly cancel a running
    thread, so on timeout this request returns immediately without waiting for the executor to
    drain (shutdown(wait=False)) - the in-flight GEE calls keep running in the background until
    they finish or error, and their results are simply not returned to this request."""
    if demo == "yanomami-2023":
        return _run_yanomami_2023_demo()
    if not as_of:
        raise HTTPException(status_code=400, detail="as_of query parameter is required (YYYY-MM-DD).")
    try:
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be in YYYY-MM-DD format.")

    from gee.baselines import cached_baseline_months, zones_missing_baseline

    with Session(engine) as session:
        zones = session.exec(select(Zone)).all()
        missing = zones_missing_baseline(zones, as_of_date.month, session)
        if missing:
            return {
                "error": "baseline_missing",
                "message": (
                    f"Baselines not yet computed for month {as_of_date.month}. Run "
                    "seed_baselines first, or pick a month that has cached baselines."
                ),
                "cached_months": cached_baseline_months(session),
                "zones": missing,
            }

    from scripts.run_once import run_once

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(run_once, as_of_date)
        return future.result(timeout=RUN_ONCE_TIMEOUT_S)
    except FuturesTimeoutError:
        # Do NOT wait for the executor to finish. Return immediately.
        return {
            "error": "timeout",
            "message": f"Pipeline exceeded {RUN_ONCE_TIMEOUT_S}s. Try a different date.",
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
