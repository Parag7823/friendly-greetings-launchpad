"""
Centralized configuration management using Pydantic Settings.
Replaces scattered os.environ.get() calls throughout the codebase.
Type-safe, validated, and IDE-friendly.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class NangoConfig(BaseSettings):
    """Nango OAuth connector configuration."""
    base_url: str = "https://api.nango.dev"
    secret_key: str = ""  # Optional - will be set via NANGO_SECRET_KEY env var or default to empty
    
    # Integration IDs
    gmail_integration_id: str = "google-mail"
    dropbox_integration_id: str = "dropbox"
    google_drive_integration_id: str = "google-drive"
    zoho_mail_integration_id: str = "zoho-mail"
    zoho_books_integration_id: str = "zoho-books"
    quickbooks_integration_id: str = "quickbooks-sandbox"
    xero_integration_id: str = "xero"
    stripe_integration_id: str = "stripe"
    razorpay_integration_id: str = "razorpay"
    paypal_integration_id: str = "paypal"
    
    # Timeouts (seconds)
    default_timeout: float = 30.0
    connect_session_timeout: float = 30.0
    gmail_profile_timeout: float = 30.0
    gmail_list_timeout: float = 60.0
    gmail_attachment_timeout: float = 120.0
    gmail_history_timeout: float = 60.0
    
    class Config:
        env_prefix = "NANGO_"
        case_sensitive = False


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
    global_max_syncs_per_minute: int = 5  # Max syncs per minute per provider (across ALL users)
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
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


# Singleton instances
nango_config = NangoConfig()
connector_config = ConnectorConfig()
queue_config = QueueConfig()
app_config = AppConfig()


def get_nango_config() -> NangoConfig:
    """Get Nango configuration."""
    return nango_config


def get_connector_config() -> ConnectorConfig:
    """Get connector configuration."""
    return connector_config


def get_queue_config() -> QueueConfig:
    """Get queue configuration."""
    return queue_config


def get_app_config() -> AppConfig:
    """Get app configuration."""
    return app_config
