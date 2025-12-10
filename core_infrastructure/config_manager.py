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
    concurrency: int = 5  # Max concurrent downloads per sync
    max_concurrency: int = 10  # Hard limit
    
    # Sync settings
    incremental_sync_enabled: bool = True
    cursor_validity_days: int = 30  # Cursors older than this are discarded
    
    # Rate limiting (global defaults)
    rate_limit_per_second: int = 10
    rate_limit_burst: int = 20
    
    # CRITICAL FIX #1: Global rate limiting across all users
    # UPDATED (Dec 1, 2025): Increased from 5 to 20 to support 50 concurrent users
    # Math: 50 users × 100 queries/sync = 5000 concurrent queries
    # With 20 syncs/min: All 50 users can start within 2.5 minutes (vs 10 minutes at 5/min)
    global_max_syncs_per_minute: int = 20  # Max syncs per minute per provider (across ALL users)
    global_max_queued_syncs_per_provider: int = 100  # Max queued syncs per provider
    global_max_queued_syncs_per_user: int = 10  # Max queued syncs per user
    sync_lock_expiry_seconds: int = 1800  # 30 minutes - auto-cleanup for stuck locks
    
    # Provider-specific rate limits (requests per second)
    gmail_rate_limit: int = 10
    dropbox_rate_limit: int = 5
    google_drive_rate_limit: int = 10
    zoho_mail_rate_limit: int = 5
    quickbooks_rate_limit: int = 3  # QB has stricter limits
    xero_rate_limit: int = 3  # Xero has strict API limits
    zoho_books_rate_limit: int = 5
    stripe_rate_limit: int = 10  # Stripe allows higher concurrency
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
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # File upload settings
    max_upload_size_mb: int = 500
    upload_temp_dir: str = "/tmp"
    
    # CRITICAL FIX #7: Attachment size limit to prevent OOM crashes
    # Prevents downloading huge attachments that exhaust worker RAM
    # Math: 5 users × 10 attachments × 50MB = 2.5GB RAM → OOM on 2GB worker
    # Solution: Limit to 25MB per attachment (reasonable for business documents)
    max_attachment_size_mb: int = 25  # Max size for provider attachments (Gmail, Dropbox, etc.)
    
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


# ============================================================================
# PRELOAD PATTERN: Config singletons are initialized at module-load time
# ============================================================================
# The singletons (connector_config, queue_config, app_config, airbyte_config)
# defined above (lines 123-126) are automatically created when module is imported.
# This is the ideal preload pattern - no additional preload code needed!
#
# All config values are immediately available after import:
# - from config_manager import get_connector_config, get_app_config, etc.

