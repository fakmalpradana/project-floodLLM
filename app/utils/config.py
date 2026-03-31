"""Application configuration."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    google_api_key: str = ""
    copernicus_username: str = ""
    copernicus_password: str = ""
    nasa_earthdata_username: str = ""
    nasa_earthdata_password: str = ""

    # GCP
    gcp_project_id: str = ""
    gcp_bucket_name: str = ""
    google_application_credentials: str = ""
    gcp_service_account_email: str = ""

    # App settings
    app_env: str = "development"
    log_level: str = "INFO"

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    output_dir: Path = base_dir / "output"

    # Processing settings
    default_buffer_km: float = 50.0  # Default search radius
    water_threshold_vv: float = -17.0  # dB threshold for water in SAR
    cloud_cover_max: float = 20.0  # Max cloud cover for optical

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # silently drop any unrecognised env vars


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
