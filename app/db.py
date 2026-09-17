from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, UniqueConstraint, create_engine

from app.config import settings


class Zone(SQLModel, table=True):
    """A monitored area of interest (AOI): a suspected mining site, logging corridor, or control forest."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    zone_type: str  # mining | logging | control
    aoi_geojson: str
    centroid_lat: float
    centroid_lon: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Baseline(SQLModel, table=True):
    """Rolling per-zone, per-calendar-month median/MAD baseline for NDVI, BSI, and VIIRS radiance."""

    id: Optional[int] = Field(default=None, primary_key=True)
    zone_id: int = Field(foreign_key="zone.id")
    month: int
    ndvi_median: float
    ndvi_mad: float
    bsi_median: float
    bsi_mad: float
    viirs_median: float
    viirs_mad: float
    sample_years: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class ZoneBaseline(SQLModel, table=True):
    """Real Earth Engine baseline cache: per-zone, per-calendar-month, per-indicator median/MAD,
    refreshed at most every 30 days by gee.baselines.get_or_compute_baseline()."""

    __table_args__ = (
        UniqueConstraint("zone_id", "calendar_month", "indicator", name="uq_zonebaseline_zone_month_indicator"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    zone_id: int = Field(foreign_key="zone.id")
    calendar_month: int
    indicator: str  # 'ndvi' | 'bsi' | 'viirs' | 'rain'
    median: float
    mad: float
    sample_count: int
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class Detection(SQLModel, table=True):
    """One pipeline pass over a zone: indicator z-scores, LLM classification, and fused confidence."""

    id: Optional[int] = Field(default=None, primary_key=True)
    zone_id: int = Field(foreign_key="zone.id")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    ndvi_z: float
    bsi_z: float
    viirs_z: Optional[float] = Field(default=None)  # None = gee.detect.current_viirs() found no
    # VIIRS DNB monthly composite for this zone/month (a real 1-2 month data lag, not an error -
    # VIIRS is a corroborator, not a required signal). Never a fabricated 0.0 - see
    # fusion.bayes.fuse_confidence()'s viirs_z=None handling. Column made nullable via migration
    # in init_db() for pre-existing DBs (was NOT NULL before this).
    rainfall_percentile: float
    cause: str
    cause_confidence: float
    reasoning: str
    adversarial_notes: str = ""
    combined_confidence: float
    estimated_area_ha: float
    status: str = "silent"  # silent | alerted
    tier: str = "silent"  # silent | uncertain | classified - see fusion.bayes.classify_alert_tier
    as_of: Optional[str] = None  # "YYYY-MM-DD" for a historical --as-of / /run-once pass; None
    # for the synthetic default (live demo) path - see app.main.get_stats()'s windows_evaluated.
    carbon_loss_tco2e: Optional[float] = Field(default=None)  # analysis.ldn.estimate_carbon_loss_tco2e()
    # output, IPCC 2006 AGB defaults - null when area_ha or a resolvable baseline NDVI median
    # isn't available (never guessed). Column added via migration in init_db() for pre-existing
    # DBs; see app.main's startup backfill for retroactively populating existing rows.


class Alert(SQLModel, table=True):
    """A fired, plain-language alert - only created when a Detection clears the confidence gate."""

    id: Optional[int] = Field(default=None, primary_key=True)
    detection_id: int = Field(foreign_key="detection.id")
    zone_id: int = Field(foreign_key="zone.id")
    message: str
    channel: str
    sent_at: datetime = Field(default_factory=datetime.utcnow)


engine = create_engine(settings.database_url, echo=False)


def init_db() -> None:
    """INPUTS: none. OUTPUTS: none; creates all tables if they don't already exist, then runs
    the migrations below for DBs created before those columns/constraints existed."""
    SQLModel.metadata.create_all(engine)
    _migrate_carbon_loss_column()
    _migrate_viirs_z_nullable()


def _migrate_carbon_loss_column() -> None:
    """INPUTS: none. OUTPUTS: none. SQLModel.metadata.create_all() only creates missing tables,
    never adds columns to an existing one - so a DB created before carbon_loss_tco2e existed
    needs an explicit ALTER TABLE. Idempotent: checks PRAGMA table_info first, no-ops if the
    column is already there."""
    with engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(detection)").fetchall()}
        if "carbon_loss_tco2e" not in columns:
            conn.exec_driver_sql("ALTER TABLE detection ADD COLUMN carbon_loss_tco2e FLOAT")
            conn.commit()


def _migrate_viirs_z_nullable() -> None:
    """INPUTS: none. OUTPUTS: none. viirs_z was originally NOT NULL; gee.detect.current_viirs()
    can now legitimately return None (a real VIIRS DNB data gap, not an error), and that needs
    to persist as a real NULL, not a fabricated 0.0 that would misrepresent "no VIIRS data" as
    "VIIRS confirmed zero activity." SQLite has no ALTER TABLE ... DROP NOT NULL, so this uses
    the standard SQLite workaround: rename the table, recreate `detection` from the current
    SQLModel schema (viirs_z now Optional above) via create_all(), copy every row across, drop
    the renamed original. Idempotent - checks PRAGMA table_info's notnull flag first and no-ops
    on every later startup once the constraint is gone."""
    with engine.connect() as conn:
        columns = conn.exec_driver_sql("PRAGMA table_info(detection)").fetchall()
        viirs_col = next((c for c in columns if c[1] == "viirs_z"), None)
        if viirs_col is None or viirs_col[3] == 0:  # column missing, or already nullable
            return
        column_list = ", ".join(c[1] for c in columns)
        conn.exec_driver_sql("ALTER TABLE detection RENAME TO detection_pre_viirs_nullable_migration")
        conn.commit()

    SQLModel.metadata.create_all(engine)  # recreates `detection` fresh, viirs_z nullable now
    with engine.connect() as conn:
        conn.exec_driver_sql(
            f"INSERT INTO detection ({column_list}) "
            f"SELECT {column_list} FROM detection_pre_viirs_nullable_migration"
        )
        conn.exec_driver_sql("DROP TABLE detection_pre_viirs_nullable_migration")
        conn.commit()


def get_session():
    """INPUTS: none. OUTPUTS: yields a SQLModel Session for FastAPI dependency injection."""
    with Session(engine) as session:
        yield session
