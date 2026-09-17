import os

import ee
from google.oauth2 import service_account

from app.config import settings

_initialized = False


def init_ee():
    """Initialize Earth Engine with the service account. Idempotent.
    INPUTS: none (reads GOOGLE_APPLICATION_CREDENTIALS and GEE_PROJECT from env)
    OUTPUTS: none (sets up global ee state)"""
    global _initialized
    if _initialized:
        return
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )
    ee.Initialize(credentials=creds, project=os.environ["GEE_PROJECT"])
    _initialized = True


def ee_available() -> bool:
    """INPUTS: none. OUTPUTS: bool, True only when init_ee() has succeeded AND SYNTHETIC_MODE is
    off. Not part of the Stage 1 spec verbatim - kept because gee/baselines.py, gee/detect.py,
    gee/corroborate.py, and gee/thumbnails.py all import it to gate real Earth Engine calls
    behind SYNTHETIC_MODE. Removing it would break their imports (and therefore app/main.py's
    import chain) even when SYNTHETIC_MODE=true."""
    return _initialized and not settings.synthetic_mode
