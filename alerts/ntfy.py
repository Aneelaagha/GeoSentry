import logging
from datetime import datetime, timedelta

import requests
from sqlmodel import Session, select

from app.config import settings
from app.db import Alert, Detection, engine

logger = logging.getLogger("geosentry.alerts")

DEDUPE_WINDOW_DAYS = 7

ALERT_TEMPLATE = (
    "GeoSentry Alert - Zone {name}. Possible {cause}. Confidence {pct}%. "
    "Estimated affected area {area} ha. NDVI {ndvi:+.2f}, BSI {bsi:+.2f}, "
    "VIIRS {viirs}. Rainfall {rain}pct. Reply ACK to acknowledge."
)

# 'uncertain' tier (fusion.bayes.classify_alert_tier): fused evidence cleared the gate but the
# LLM's own top-cause confidence was below 0.5 - an honest "something's happening, cause unclear"
# alert instead of asserting a hedged cause as if it were confident.
UNCERTAIN_ALERT_TEMPLATE = (
    "GeoSentry Alert - Zone {name}. Significant land disturbance detected. "
    "Cause uncertain (top hypothesis: {cause}, {llm_conf_pct:.0f}%). "
    "Estimated affected area {area} ha. NDVI {ndvi:+.2f}s, BSI {bsi:+.2f}s, "
    "VIIRS {viirs}, rainfall {rain}pct. Recommend ground verification."
)

# Appended to both templates above when a real carbon-loss estimate exists (analysis.ldn,
# IPCC 2006 AGB defaults); omitted entirely when carbon_loss_tco2e is None, never faked.
CARBON_LOSS_LINE = " Estimated carbon loss: {tco2e:,.0f} tCO2e (IPCC 2006 AGB defaults)."


def _format_viirs(viirs_z: float | None, decimals: int, suffix: str) -> str:
    """INPUTS: a VIIRS z-score or None (gee.detect.current_viirs() found no VIIRS monthly
    composite for that zone/month - a real data gap, not an error), decimal places, and the
    template's existing unit suffix ('x' for ALERT_TEMPLATE, 's' for UNCERTAIN_ALERT_TEMPLATE).
    OUTPUTS: the pre-formatted string to drop into the template's plain {viirs} placeholder -
    "n/a" when None (an f-string format spec like {:+.2f} raises on None, and a message that
    fired without VIIRS corroboration should say so plainly, not print a fabricated number)."""
    if viirs_z is None:
        return "n/a"
    return f"{viirs_z:+.{decimals}f}{suffix}"


def build_alert_message(
    zone_name: str,
    cause: str,
    confidence: float,
    area_ha: float,
    ndvi_z: float,
    bsi_z: float,
    viirs_z: float | None,
    rainfall_percentile: float,
    tier: str = "classified",
    llm_conf: float = 0.0,
    carbon_loss_tco2e: float | None = None,
) -> str:
    """INPUTS: the fired Detection's zone name, cause, fused confidence, and indicator values;
    viirs_z may be None (gee.detect.current_viirs() found no VIIRS monthly composite for this
    zone/month - a real 1-2 month data lag, not an error), rendered as "VIIRS n/a" rather than
    a fabricated number; tier ('classified' or 'uncertain', from fusion.bayes.classify_alert_tier
    - 'silent' never reaches this function since no alert fires for it); llm_conf, the LLM's raw
    top-cause confidence (only used when tier='uncertain'); carbon_loss_tco2e, the Detection's
    estimated CO2-equivalent loss (analysis.ldn), appended as an extra sentence when not None,
    skipped entirely otherwise. OUTPUTS: the plain-language alert body - the existing
    ALERT_TEMPLATE for 'classified', or UNCERTAIN_ALERT_TEMPLATE for 'uncertain' - shared by the
    ntfy push, the SMS stub, and the persisted Alert.message row, so every channel says exactly
    the same thing."""
    if tier == "uncertain":
        message = UNCERTAIN_ALERT_TEMPLATE.format(
            name=zone_name,
            cause=cause.replace("_", " "),
            llm_conf_pct=round(llm_conf * 100),
            area=f"{area_ha:.1f}",
            ndvi=ndvi_z,
            bsi=bsi_z,
            viirs=_format_viirs(viirs_z, 2, "s"),
            rain=round(rainfall_percentile),
        )
    else:
        message = ALERT_TEMPLATE.format(
            name=zone_name,
            cause=cause.replace("_", " "),
            pct=round(confidence * 100),
            area=f"{area_ha:.1f}",
            ndvi=ndvi_z,
            bsi=bsi_z,
            viirs=_format_viirs(viirs_z, 1, "x"),
            rain=round(rainfall_percentile),
        )
    if carbon_loss_tco2e is not None:
        message += CARBON_LOSS_LINE.format(tco2e=carbon_loss_tco2e)
    return message


def recently_alerted(zone_id: int, cause: str, window_days: int = DEDUPE_WINDOW_DAYS) -> bool:
    """INPUTS: zone_id, cause, dedupe window in days (default 7). OUTPUTS: bool - True if this
    zone+cause combination already has a fired Alert (joined via its Detection) within the last
    window_days, so a still-ongoing event doesn't re-alert on every pass."""
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    with Session(engine) as session:
        statement = (
            select(Alert)
            .join(Detection, Alert.detection_id == Detection.id)
            .where(Detection.zone_id == zone_id, Detection.cause == cause, Alert.sent_at >= cutoff)
        )
        return session.exec(statement).first() is not None


def send_alert(
    zone_id: int,
    zone_name: str,
    cause: str,
    confidence: float,
    area_ha: float,
    ndvi_z: float,
    bsi_z: float,
    viirs_z: float | None,
    rainfall_percentile: float,
    tier: str = "classified",
    llm_conf: float = 0.0,
    topic: str | None = None,
    carbon_loss_tco2e: float | None = None,
    force: bool = False,
) -> bool:
    """INPUTS: the zone's id/name, the fired Detection's cause, fused confidence, and indicator
    values; tier ('classified' or 'uncertain') and llm_conf (only used when tier='uncertain'),
    carbon_loss_tco2e - all passed straight through to build_alert_message(); optional ntfy topic
    override (defaults to settings.ntfy_topic); force - when True, skips the recently_alerted()
    dedupe check entirely and always POSTs. Only app.main's demo replay endpoint
    (/run-once?demo=yanomami-2023&force=true) ever sets this - a fixed demo replay is expected
    to fire every click, not just once per DEDUPE_WINDOW_DAYS. Every live /run-once and CLI pass
    calls this with force's default (False), so the 7-day zone+cause dedupe is unchanged for
    them. OUTPUTS: bool - True if an alert was actually POSTed to ntfy.sh; False if it was
    skipped as a duplicate (same zone+cause alerted within DEDUPE_WINDOW_DAYS, and force wasn't
    set) or the POST failed. Never raises: a flaky network connection can't crash the demo."""
    if not force and recently_alerted(zone_id, cause):
        logger.info(
            "Skipping duplicate alert: zone '%s' already alerted for cause '%s' within the last %d days.",
            zone_name,
            cause,
            DEDUPE_WINDOW_DAYS,
        )
        return False

    message = build_alert_message(
        zone_name,
        cause,
        confidence,
        area_ha,
        ndvi_z,
        bsi_z,
        viirs_z,
        rainfall_percentile,
        tier,
        llm_conf,
        carbon_loss_tco2e,
    )
    topic = topic or settings.ntfy_topic
    url = f"{settings.ntfy_server.rstrip('/')}/{topic}"
    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers={"Title": "GeoSentry Alert", "Priority": "urgent", "Tags": "warning"},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("ntfy alert sent to topic '%s' for zone '%s'", topic, zone_name)
        return True
    except Exception as exc:  # noqa: BLE001 - network failure must not crash the demo
        logger.warning("ntfy alert failed (%s); message was: %s", exc, message)
        return False
