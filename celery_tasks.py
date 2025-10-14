"""
FIX #6: Celery tasks for periodic connector syncs and maintenance operations.

This module implements scheduled background tasks that run via Celery Beat:
- Periodic OAuth connector syncs (Gmail, Dropbox, Google Drive, etc.)
- Analytics materialized view refresh
- Database cleanup and maintenance
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from celery_app import celery_app
from fastapi_backend import (
    supabase,
    NangoClient,
    ConnectorSyncRequest,
    _gmail_sync_run,
    _dropbox_sync_run,
    _gdrive_sync_run,
    _zohomail_sync_run,
    _quickbooks_sync_run,
    _xero_sync_run,
    NANGO_GMAIL_INTEGRATION_ID,
    NANGO_DROPBOX_INTEGRATION_ID,
    NANGO_GOOGLE_DRIVE_INTEGRATION_ID,
    NANGO_ZOHO_MAIL_INTEGRATION_ID,
    NANGO_QUICKBOOKS_INTEGRATION_ID,
    NANGO_XERO_INTEGRATION_ID,
)

logger = logging.getLogger(__name__)

# --------------- Connector Sync Tasks ---------------

async def _sync_active_connections(provider: str, integration_id: str, sync_func) -> Dict[str, Any]:
    """
    Generic function to sync all active connections for a given provider.
    
    Args:
        provider: Provider name for logging
        integration_id: Nango integration ID
        sync_func: Async function to call for each connection
    
    Returns:
        Summary of sync results
    """
    try:
        # Fetch all active connections for this provider
        result = supabase.table('user_connections')\
            .select('id, user_id, nango_connection_id, last_synced_at')\
            .eq('status', 'active')\
            .eq('connector_id', supabase.table('connectors')
                .select('id')
                .eq('integration_id', integration_id)
                .limit(1)
                .execute()
                .data[0]['id'] if supabase.table('connectors')
                .select('id')
                .eq('integration_id', integration_id)
                .limit(1)
                .execute()
                .data else None)\
            .execute()
        
        connections = result.data or []
        
        if not connections:
            logger.info(f"No active {provider} connections found")
            return {"status": "success", "provider": provider, "synced": 0, "skipped": 0, "failed": 0}
        
        logger.info(f"Found {len(connections)} active {provider} connections to sync")
        
        stats = {"synced": 0, "skipped": 0, "failed": 0}
        nango = NangoClient()
        
        for conn in connections:
            try:
                user_id = conn['user_id']
                connection_id = conn['nango_connection_id']
                last_synced = conn.get('last_synced_at')
                
                # Skip if synced recently (within last hour for frequent syncs)
                if last_synced:
                    last_sync_time = datetime.fromisoformat(last_synced.replace('Z', '+00:00'))
                    hours_since_sync = (datetime.utcnow() - last_sync_time).total_seconds() / 3600
                    
                    # Different thresholds for different providers
                    min_hours = 1 if provider in ['Gmail', 'Zoho Mail'] else 6
                    if hours_since_sync < min_hours:
                        logger.info(f"Skipping {provider} sync for {connection_id} (synced {hours_since_sync:.1f}h ago)")
                        stats['skipped'] += 1
                        continue
                
                # Trigger sync
                req = ConnectorSyncRequest(
                    user_id=user_id,
                    connection_id=connection_id,
                    integration_id=integration_id,
                    mode='incremental',  # Use incremental mode for periodic syncs
                    correlation_id=f"celery_beat_{provider.lower()}_{datetime.utcnow().isoformat()}"
                )
                
                await sync_func(nango, req)
                stats['synced'] += 1
                logger.info(f"✅ Successfully synced {provider} connection {connection_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to sync {provider} connection {conn.get('nango_connection_id')}: {e}")
                stats['failed'] += 1
        
        logger.info(f"{provider} periodic sync complete: {stats}")
        return {"status": "success", "provider": provider, **stats}
        
    except Exception as e:
        logger.error(f"❌ {provider} periodic sync task failed: {e}")
        return {"status": "failed", "provider": provider, "error": str(e)}


@celery_app.task(name='celery_tasks.sync_gmail_connections')
def sync_gmail_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active Gmail connections (runs hourly)"""
    import asyncio
    logger.info("🔄 Starting periodic Gmail sync task")
    return asyncio.run(_sync_active_connections('Gmail', NANGO_GMAIL_INTEGRATION_ID, _gmail_sync_run))


@celery_app.task(name='celery_tasks.sync_dropbox_connections')
def sync_dropbox_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active Dropbox connections (runs every 6 hours)"""
    import asyncio
    logger.info("🔄 Starting periodic Dropbox sync task")
    return asyncio.run(_sync_active_connections('Dropbox', NANGO_DROPBOX_INTEGRATION_ID, _dropbox_sync_run))


@celery_app.task(name='celery_tasks.sync_gdrive_connections')
def sync_gdrive_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active Google Drive connections (runs every 6 hours)"""
    import asyncio
    logger.info("🔄 Starting periodic Google Drive sync task")
    return asyncio.run(_sync_active_connections('Google Drive', NANGO_GOOGLE_DRIVE_INTEGRATION_ID, _gdrive_sync_run))


@celery_app.task(name='celery_tasks.sync_zohomail_connections')
def sync_zohomail_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active Zoho Mail connections (runs every 2 hours)"""
    import asyncio
    logger.info("🔄 Starting periodic Zoho Mail sync task")
    return asyncio.run(_sync_active_connections('Zoho Mail', NANGO_ZOHO_MAIL_INTEGRATION_ID, _zohomail_sync_run))


@celery_app.task(name='celery_tasks.sync_quickbooks_connections')
def sync_quickbooks_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active QuickBooks connections (runs daily)"""
    import asyncio
    logger.info("🔄 Starting periodic QuickBooks sync task")
    return asyncio.run(_sync_active_connections('QuickBooks', NANGO_QUICKBOOKS_INTEGRATION_ID, _quickbooks_sync_run))


@celery_app.task(name='celery_tasks.sync_xero_connections')
def sync_xero_connections() -> Dict[str, Any]:
    """Periodic task: Sync all active Xero connections (runs daily)"""
    import asyncio
    logger.info("🔄 Starting periodic Xero sync task")
    return asyncio.run(_sync_active_connections('Xero', NANGO_XERO_INTEGRATION_ID, _xero_sync_run))


# --------------- Maintenance Tasks ---------------

@celery_app.task(name='celery_tasks.refresh_materialized_views')
def refresh_materialized_views() -> Dict[str, Any]:
    """
    Periodic task: Refresh all L3 analytics materialized views (runs every 15 minutes).
    
    This ensures analytics dashboards show up-to-date data.
    """
    try:
        logger.info("🔄 Starting materialized view refresh")
        
        views = [
            'l3_gating_view',
            'l3_core_views',
            'l3_velocity_and_ar_ap_views'
        ]
        
        refreshed = []
        failed = []
        
        for view_name in views:
            try:
                # PostgreSQL REFRESH MATERIALIZED VIEW command
                supabase.rpc('refresh_materialized_view', {'view_name': view_name}).execute()
                refreshed.append(view_name)
                logger.info(f"✅ Refreshed materialized view: {view_name}")
            except Exception as e:
                logger.error(f"❌ Failed to refresh {view_name}: {e}")
                failed.append(view_name)
        
        logger.info(f"Materialized view refresh complete: {len(refreshed)} success, {len(failed)} failed")
        return {
            "status": "success" if not failed else "partial",
            "refreshed": refreshed,
            "failed": failed
        }
        
    except Exception as e:
        logger.error(f"❌ Materialized view refresh task failed: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name='celery_tasks.cleanup_old_failures')
def cleanup_old_failures() -> Dict[str, Any]:
    """
    Periodic task: Clean up old job failures from DLQ (runs weekly).
    
    Removes job_failures records older than 30 days to prevent table bloat.
    """
    try:
        logger.info("🧹 Starting job failures cleanup")
        
        # Delete failures older than 30 days
        cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        result = supabase.table('job_failures')\
            .delete()\
            .lt('created_at', cutoff_date)\
            .execute()
        
        deleted_count = len(result.data) if result.data else 0
        
        logger.info(f"✅ Cleaned up {deleted_count} old job failures")
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date
        }
        
    except Exception as e:
        logger.error(f"❌ Job failures cleanup task failed: {e}")
        return {"status": "failed", "error": str(e)}


# --------------- Task Registration ---------------

__all__ = [
    'sync_gmail_connections',
    'sync_dropbox_connections',
    'sync_gdrive_connections',
    'sync_zohomail_connections',
    'sync_quickbooks_connections',
    'sync_xero_connections',
    'refresh_materialized_views',
    'cleanup_old_failures',
]
