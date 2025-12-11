"""
Centralized configuration management using Pydantic Settings.
Replaces scattered os.environ.get() calls throughout the codebase.
Type-safe, validated, and IDE-friendly.
"""

from pydantic_settings import BaseSettings
from typing import Optional


# REMOVED: NangoConfig class - Replaced by Airbyte integration
# Legacy Nango integration IDs are now hardcoded as constants in fastapi_backend_v2.py
# These are used only as fallback defaults for provider names


class ConnectorConfig(BaseSettings):
    """Connector and sync configuration."""
    concurrency: int = 5
    max_concurrency: int = 10
    incremental_sync_enabled: bool = True
    cursor_validity_days: int = 30
    
    # Rate limiting (global defaults)
    rate_limit_per_second: int = 10
    rate_limit_burst: int = 20
    global_max_syncs_per_minute: int = 20
    global_max_queued_syncs_per_provider: int = 100
    global_max_queued_syncs_per_user: int = 10
    sync_lock_expiry_seconds: int = 1800
    
    # Provider-specific rate limits (requests per second)
    gmail_rate_limit: int = 10
    dropbox_rate_limit: int = 5
    google_drive_rate_limit: int = 10
    zoho_mail_rate_limit: int = 5
    quickbooks_rate_limit: int = 3
    xero_rate_limit: int = 3
    zoho_books_rate_limit: int = 5
    stripe_rate_limit: int = 10
    razorpay_rate_limit: int = 4
    
    # Provider-specific concurrency limits
    gmail_concurrency: int = 5
    dropbox_concurrency: int = 3
    google_drive_concurrency: int = 5
    zoho_mail_concurrency: int = 3
    quickbooks_concurrency: int = 3
    xero_concurrency: int = 3
    zoho_books_concurrency: int = 3
    stripe_concurrency: int = 5
    razorpay_concurrency: int = 4
    
    class Config:
        env_prefix = "CONNECTOR_"
        case_sensitive = False


class QueueConfig(BaseSettings):
    """ARQ queue configuration."""
    backend: str = "arq"  # "arq" or "sync"
    redis_url: Optional[str] = None
    
    # Job settings
    job_timeout_seconds: int = 900  # 15 minutes
    job_retry_attempts: int = 3
    job_retry_delay_seconds: int = 60
    
    class Config:
        env_prefix = "QUEUE_"
        case_sensitive = False


class AirbytePythonConfig(BaseSettings):
    """Airbyte Python client configuration."""
    base_url: str = "http://localhost:8000/api/v1"  # Airbyte API endpoint
    api_key: str = ""  # Set via AIRBYTE_API_KEY env var
    workspace_id: str = ""  # Set via AIRBYTE_WORKSPACE_ID env var
    destination_id: str = ""  # Supabase destination ID (set via AIRBYTE_DESTINATION_ID)
    
    # Timeouts (seconds)
    default_timeout: float = 30.0
    oauth_timeout: float = 60.0
    sync_timeout: float = 120.0
    status_timeout: float = 30.0
    
    class Config:
        env_prefix = "AIRBYTE_"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Application-wide configuration."""
    environment: str = "development"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_size_mb: int = 500
    upload_temp_dir: str = "/tmp"
    max_attachment_size_mb: int = 25
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


# Singleton instances
connector_config = ConnectorConfig()
queue_config = QueueConfig()
app_config = AppConfig()
airbyte_config = AirbytePythonConfig()


def get_connector_config() -> ConnectorConfig:
    """Get connector configuration."""
    return connector_config


def get_queue_config() -> QueueConfig:
    """Get queue configuration."""
    return queue_config


def get_app_config() -> AppConfig:
    """Get app configuration."""
    return app_config


def get_airbyte_config() -> AirbytePythonConfig:
    """Get Airbyte configuration."""
    return airbyte_config


# Config singletons are initialized at module-load time and available immediately after import

