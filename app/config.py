from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment / .env. All fields have safe
    defaults so the app boots (in SYNTHETIC_MODE) with zero configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    synthetic_mode: bool = True
    database_url: str = "sqlite:///./geosentry.db"

    gee_service_account: str = ""
    gee_service_account_key_path: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    ntfy_topic: str = "geosentry-demo"
    ntfy_server: str = "https://ntfy.sh"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""

    alert_confidence_threshold: float = 0.72


settings = Settings()
