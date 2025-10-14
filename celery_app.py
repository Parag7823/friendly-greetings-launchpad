import os
from celery import Celery
from celery.schedules import crontab

# Celery configuration
celery_app = Celery(
    "finley_ingestion",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=60 * 30,  # 30 minutes
    task_soft_time_limit=60 * 25,  # 25 minutes
)

# FIX #6: Implement Celery Beat schedule for periodic connector syncs
celery_app.conf.beat_schedule = {
    # Gmail sync every hour
    'sync-gmail-connections-hourly': {
        'task': 'celery_tasks.sync_gmail_connections',
        'schedule': crontab(minute=0),  # Every hour at :00
        'options': {'queue': 'periodic_syncs'}
    },
    # Dropbox sync every 6 hours
    'sync-dropbox-connections-6h': {
        'task': 'celery_tasks.sync_dropbox_connections',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'options': {'queue': 'periodic_syncs'}
    },
    # Google Drive sync every 6 hours
    'sync-gdrive-connections-6h': {
        'task': 'celery_tasks.sync_gdrive_connections',
        'schedule': crontab(minute=15, hour='*/6'),  # Every 6 hours at :15
        'options': {'queue': 'periodic_syncs'}
    },
    # Zoho Mail sync every 2 hours
    'sync-zohomail-connections-2h': {
        'task': 'celery_tasks.sync_zohomail_connections',
        'schedule': crontab(minute=30, hour='*/2'),  # Every 2 hours at :30
        'options': {'queue': 'periodic_syncs'}
    },
    # QuickBooks sync daily at 2 AM
    'sync-quickbooks-connections-daily': {
        'task': 'celery_tasks.sync_quickbooks_connections',
        'schedule': crontab(minute=0, hour=2),  # Daily at 2:00 AM
        'options': {'queue': 'periodic_syncs'}
    },
    # Xero sync daily at 3 AM
    'sync-xero-connections-daily': {
        'task': 'celery_tasks.sync_xero_connections',
        'schedule': crontab(minute=0, hour=3),  # Daily at 3:00 AM
        'options': {'queue': 'periodic_syncs'}
    },
    # Refresh analytics views every 15 minutes
    'refresh-analytics-views': {
        'task': 'celery_tasks.refresh_materialized_views',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
        'options': {'queue': 'maintenance'}
    },
    # Cleanup old job failures weekly
    'cleanup-job-failures': {
        'task': 'celery_tasks.cleanup_old_failures',
        'schedule': crontab(minute=0, hour=4, day_of_week=0),  # Sunday at 4:00 AM
        'options': {'queue': 'maintenance'}
    },
}

__all__ = ["celery_app"]
