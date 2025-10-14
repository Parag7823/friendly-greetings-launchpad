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


async def detect_relationships(ctx, user_id: str) -> Dict[str, Any]:
    """Background task for relationship detection across all user files"""
    try:
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        from openai import AsyncOpenAI
        
        logger.info(f"Starting background relationship detection for user {user_id}")
        
        # Initialize relationship detector
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        relationship_detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Detect all relationships
        relationship_results = await relationship_detector.detect_all_relationships(user_id)
        
        # Store relationships atomically using the same method as main flow
        if relationship_results.get('relationships'):
            from transaction_manager import get_transaction_manager
            transaction_manager = get_transaction_manager()
            
            async with transaction_manager.transaction(
                user_id=user_id,
                operation_type="background_relationship_storage"
            ) as tx:
                # Prepare batch data
                relationships_batch = []
                for relationship in relationship_results['relationships']:
                    rel_data = {
                        'user_id': user_id,
                        'source_event_id': relationship.get('source_event_id'),
                        'target_event_id': relationship.get('target_event_id'),
                        'relationship_type': relationship.get('relationship_type', 'unknown'),
                        'confidence_score': relationship.get('confidence_score', 0.5),
                        'detection_method': relationship.get('detection_method', 'background_task'),
                        'pattern_id': relationship.get('pattern_id'),
                        'reasoning': relationship.get('reasoning', '')
                    }
                    relationships_batch.append(rel_data)
                
                # Batch insert all relationships atomically
                if relationships_batch:
                    result = await tx.insert_batch('relationship_instances', relationships_batch)
                    logger.info(f"✅ Background task stored {len(result)} relationships for user {user_id}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_relationships": relationship_results.get('total_relationships', 0),
            "relationships_stored": len(relationship_results.get('relationships', []))
        }
        
    except Exception as e:
        logger.error(f"❌ Background relationship detection failed for user {user_id}: {e}")
        # Retry with exponential backoff
        delay = await _retry_or_dlq(ctx, 'relationship_detection', {'user_id': user_id}, e, max_retries=3, base_delay=60)
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
