import logging

from app.config import settings

logger = logging.getLogger("geosentry.alerts")


def send_sms(message: str, to_number: str | None = None) -> bool:
    """INPUTS: message body, optional destination number (defaults to settings.twilio_to_number).
    OUTPUTS: bool True. Stub only: logs what would have been sent via Twilio instead of making a
    real call, so the demo doesn't need live Twilio credentials or incur SMS costs."""
    destination = to_number or settings.twilio_to_number or "+10000000000"
    logger.info("[TWILIO STUB] Would send SMS to %s: %s", destination, message)
    return True
