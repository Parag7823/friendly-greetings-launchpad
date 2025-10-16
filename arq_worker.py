import os
from typing import Dict, Any
from arq import Retry

# ARQ imports
from arq.connections import RedisSettings

# Import application code to reuse existing logic end-to-end
from fastapi_backend import (
    NangoClient,
    ConnectorSyncRequest,
    _gmail_sync_run,
    _dropbox_sync_run,
    _gdrive_sync_run,
    _zohomail_sync_run,
    _quickbooks_sync_run,
    _xero_sync_run,
    start_processing_job,
    start_pdf_processing_job,
    supabase,
    logger,
    JOBS_PROCESSED,
    NANGO_GMAIL_INTEGRATION_ID,
    NANGO_DROPBOX_INTEGRATION_ID,
    NANGO_GOOGLE_DRIVE_INTEGRATION_ID,
    NANGO_ZOHO_MAIL_INTEGRATION_ID,
    NANGO_QUICKBOOKS_INTEGRATION_ID,
    NANGO_XERO_INTEGRATION_ID,
)


# --------------- ARQ Task Functions ---------------
# Each task is fully functional and reuses the existing application logic.

async def _retry_or_dlq(ctx, provider: str, req: Dict[str, Any], err: Exception, max_retries: int, base_delay: int) -> None | int:
    """Increment a Redis-backed retry counter; return next delay if should retry, else record to DLQ and return None."""
    try:
        redis = ctx.get('redis') if hasattr(ctx, 'get') else ctx['redis']
        if redis is None:
            raise RuntimeError('redis ctx missing')
        conn_id = req.get('connection_id') or 'unknown'
        corr = req.get('correlation_id') or ''
        key = f"arq:tries:{provider}:{conn_id}:{corr}"
        tries = await redis.incr(key)
        # expire counter after 2h
        if tries == 1:
            await redis.expire(key, 7200)
        if tries <= max_retries:
            # Exponential backoff
            delay = base_delay * (2 ** (tries - 1))
            return delay
        # DLQ record
        try:
            supabase.table('job_failures').insert({
                'provider': provider,
                'user_id': req.get('user_id'),
                'connection_id': conn_id,
                'correlation_id': corr,
                'payload': req,
                'error': str(err)
            }).execute()
        except Exception:
            pass
        try:
            JOBS_PROCESSED.labels(provider=provider, status='failed').inc()
        except Exception:
            pass
        return None
    except Exception:
        # If DLQ/Redis logic fails, don't block retries; default to one short retry
        return base_delay


async def gmail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _gmail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_GMAIL_INTEGRATION_ID, req, e, max_retries=3, base_delay=30)
        if delay is None:
            return {"status": "failed", "provider": NANGO_GMAIL_INTEGRATION_ID}
        raise Retry(defer=delay)


async def dropbox_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _dropbox_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_DROPBOX_INTEGRATION_ID, req, e, max_retries=3, base_delay=30)
        if delay is None:
            return {"status": "failed", "provider": NANGO_DROPBOX_INTEGRATION_ID}
        raise Retry(defer=delay)


async def gdrive_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _gdrive_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_GOOGLE_DRIVE_INTEGRATION_ID, req, e, max_retries=3, base_delay=30)
        if delay is None:
            return {"status": "failed", "provider": NANGO_GOOGLE_DRIVE_INTEGRATION_ID}
        raise Retry(defer=delay)


async def zoho_mail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _zohomail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_ZOHO_MAIL_INTEGRATION_ID, req, e, max_retries=4, base_delay=45)
        if delay is None:
            return {"status": "failed", "provider": NANGO_ZOHO_MAIL_INTEGRATION_ID}
        raise Retry(defer=delay)


async def quickbooks_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _quickbooks_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_QUICKBOOKS_INTEGRATION_ID, req, e, max_retries=4, base_delay=45)
        if delay is None:
            return {"status": "failed", "provider": NANGO_QUICKBOOKS_INTEGRATION_ID}
        raise Retry(defer=delay)


async def xero_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    nango = NangoClient()
    try:
        return await _xero_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        delay = await _retry_or_dlq(ctx, NANGO_XERO_INTEGRATION_ID, req, e, max_retries=4, base_delay=45)
        if delay is None:
            return {"status": "failed", "provider": NANGO_XERO_INTEGRATION_ID}
        raise Retry(defer=delay)


async def process_spreadsheet(ctx, user_id: str, filename: str, storage_path: str, job_id: str) -> Dict[str, Any]:
    """FIX #5: Wrap ARQ spreadsheet processing in transaction for consistency with Phase 1-11"""
    try:
        from transaction_manager import get_transaction_manager
        transaction_manager = get_transaction_manager()
        
        # Wrap entire processing in transaction for atomic operations
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="arq_spreadsheet_processing"
        ) as tx:
            # Reuse the web process' processing logic, including WebSocket updates
            await start_processing_job(user_id, job_id, storage_path, filename)
            logger.info(f"✅ ARQ spreadsheet processing completed for job {job_id}")
            return {"status": "completed", "job_id": job_id}
            
    except Exception as e:
        logger.error(f"❌ ARQ spreadsheet processing failed for job {job_id}: {e}")
        # Transaction auto-rolls back on exception
        # Retry with exponential backoff
        delay = await _retry_or_dlq(ctx, 'spreadsheet_processing', {
            'user_id': user_id,
            'job_id': job_id,
            'filename': filename
        }, e, max_retries=3, base_delay=60)
        if delay is None:
            return {"status": "failed", "job_id": job_id, "error": str(e)}
        raise Retry(defer=delay)


async def process_pdf(ctx, user_id: str, filename: str, storage_path: str, job_id: str) -> Dict[str, Any]:
    """FIX #5: Wrap ARQ PDF processing in transaction for consistency with Phase 1-11"""
    try:
        from transaction_manager import get_transaction_manager
        transaction_manager = get_transaction_manager()
        
        # Wrap entire processing in transaction for atomic operations
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="arq_pdf_processing"
        ) as tx:
            await start_pdf_processing_job(user_id, job_id, storage_path, filename)
            logger.info(f"✅ ARQ PDF processing completed for job {job_id}")
            return {"status": "completed", "job_id": job_id}
            
    except Exception as e:
        logger.error(f"❌ ARQ PDF processing failed for job {job_id}: {e}")
        # Transaction auto-rolls back on exception
        # Retry with exponential backoff
        delay = await _retry_or_dlq(ctx, 'pdf_processing', {
            'user_id': user_id,
            'job_id': job_id,
            'filename': filename
        }, e, max_retries=3, base_delay=60)
        if delay is None:
            return {"status": "failed", "job_id": job_id, "error": str(e)}
        raise Retry(defer=delay)


async def detect_relationships(ctx, user_id: str, file_id: str = None) -> Dict[str, Any]:
    """
    CRITICAL FIX: Background task for relationship detection using optimized database JOINs.
    This decouples heavy analysis from the synchronous ingestion pipeline.
    
    Args:
        ctx: ARQ context
        user_id: User ID to detect relationships for
        file_id: Optional file_id to scope detection to specific file
    """
    try:
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        from openai import AsyncOpenAI
        
        logger.info(f"🔍 Starting background relationship detection for user_id={user_id}, file_id={file_id}")
        
        # Get cache client if available
        try:
            from ai_cache_system import safe_get_ai_cache
            cache_client = safe_get_ai_cache()
        except:
            cache_client = None
        
        # Initialize relationship detector with cache
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        relationship_detector = EnhancedRelationshipDetector(
            openai_client, 
            supabase,
            cache_client=cache_client
        )
        
        # CRITICAL FIX: Use new database-driven detection (no N² loops, no hardcoded filenames)
        relationship_results = await relationship_detector.detect_all_relationships(
            user_id, 
            file_id=file_id
        )
        
        # Relationships are already stored by the detector
        logger.info(f"✅ Background relationship detection completed: {relationship_results.get('total_relationships', 0)} relationships found")
        
        return {
            "status": "success",
            "user_id": user_id,
            "file_id": file_id,
            "total_relationships": relationship_results.get('total_relationships', 0),
            "cross_document_relationships": relationship_results.get('cross_document_relationships', 0),
            "within_file_relationships": relationship_results.get('within_file_relationships', 0),
            "method": "database_joins",
            "complexity": "O(N log N)"
        }
        
    except Exception as e:
        logger.error(f"❌ Background relationship detection failed for user {user_id}: {e}")
        # Retry with exponential backoff
        delay = await _retry_or_dlq(ctx, 'relationship_detection', {'user_id': user_id, 'file_id': file_id}, e, max_retries=3, base_delay=60)
        if delay is None:
            return {"status": "failed", "user_id": user_id, "error": str(e)}
        raise Retry(defer=delay)


# --------------- ARQ Worker Settings ---------------
class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(
        os.environ.get("ARQ_REDIS_URL") or os.environ.get("REDIS_URL") or "redis://localhost:6379"
    )

    functions = [
        gmail_sync,
        dropbox_sync,
        gdrive_sync,
        zoho_mail_sync,
        quickbooks_sync,
        xero_sync,
        process_spreadsheet,
        process_pdf,
        detect_relationships,  # New background task for relationship detection
    ]

    # Keep results in Redis only briefly; we don't depend on ARQ results downstream
    keep_result = 0
    # Reasonable timeout per job (seconds)
    function_timeout = 60 * 15
