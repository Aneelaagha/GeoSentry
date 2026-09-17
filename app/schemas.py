from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ZoneLastDetection(BaseModel):
    cause: str
    confidence: float
    area_ha: float
    detected_at: datetime


class ZoneRead(BaseModel):
    id: int
    name: str
    role: str
    lat: float
    lon: float
    created_at: datetime
    last_detection: Optional[ZoneLastDetection] = None
    geometry: Optional[dict[str, Any]] = None  # Zone.aoi_geojson, parsed - the real GeoJSON
    # Polygon every zone already has (used for real GEE queries throughout this pipeline), not a
    # generated fallback. None only if a row somehow has no/invalid aoi_geojson.


class DetectionRead(BaseModel):
    id: int
    zone_id: int
    detected_at: datetime
    ndvi_z: float
    bsi_z: float
    viirs_z: Optional[float] = None  # None when VIIRS had no monthly composite for that zone/month
    rainfall_percentile: float
    cause: str
    cause_confidence: float
    reasoning: str
    adversarial_notes: str
    combined_confidence: float
    estimated_area_ha: float
    status: str
    tier: str
    before_thumb_url: Optional[str] = None
    after_thumb_url: Optional[str] = None
    carbon_loss_tco2e: Optional[float] = None


class AlertRead(BaseModel):
    id: int
    detection_id: int
    zone_id: int
    message: str
    channel: str
    sent_at: datetime


class SilentLogEntry(BaseModel):
    """A Detection whose fused posterior stayed below the alert gate - a deliberately thinner,
    distinct shape from DetectionRead, built for the silent-log panel."""

    id: int
    zone_name: str
    zone_role: str
    cause: str
    llm_confidence: float
    fused_confidence: float
    area_ha: float
    ndvi_delta: float
    bsi_delta: float
    viirs_delta: Optional[float] = None  # None when VIIRS had no monthly composite that month
    rainfall_pct: float
    reasoning: str
    why_silent: str
    tier: str
    created_at: datetime
