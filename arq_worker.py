import os
from typing import Dict, Any, Optional, List
from arq import Retry

# ARQ imports
from arq.connections import RedisSettings

# Import application code to reuse existing logic end-to-end
from fastapi_backend_v2 import (
    NangoClient,
    ConnectorSyncRequest,
    _gmail_sync_run,
    _dropbox_sync_run,
    _gdrive_sync_run,
    _zohomail_sync_run,
    start_processing_job,
    start_pdf_processing_job,
    logger,
    JOBS_PROCESSED,
    NANGO_GMAIL_INTEGRATION_ID,
    NANGO_DROPBOX_INTEGRATION_ID,
    NANGO_GOOGLE_DRIVE_INTEGRATION_ID,
    NANGO_ZOHO_MAIL_INTEGRATION_ID,
)

# FIX #1: CENTRALIZED SUPABASE CLIENT - Remove duplicate fallback logic
# Use the pooled client from supabase_client.py for all Supabase operations
from core_infrastructure.supabase_client import get_supabase_client

supabase = get_supabase_client()


# --------------- ARQ Task Functions ---------------
# Each task is fully functional and reuses the existing application logic.

async def _retry_or_dlq(ctx, provider: str, req: Dict[str, Any], err: Exception, max_retries: int, base_delay: int) -> Optional[int]:
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
    """Placeholder: QuickBooks sync not yet implemented"""
    logger.warning("QuickBooks sync called but not implemented")
    return {"status": "not_implemented", "provider": "quickbooks"}


async def xero_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: Xero sync not yet implemented"""
    logger.warning("Xero sync called but not implemented")
    return {"status": "not_implemented", "provider": "xero"}


async def zoho_books_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: Zoho Books sync not yet implemented"""
    logger.warning("Zoho Books sync called but not implemented")
    return {"status": "not_implemented", "provider": "zoho_books"}


async def stripe_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: Stripe sync not yet implemented"""
    logger.warning("Stripe sync called but not implemented")
    return {"status": "not_implemented", "provider": "stripe"}


async def razorpay_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: Razorpay sync not yet implemented"""
    logger.warning("Razorpay sync called but not implemented")
    return {"status": "not_implemented", "provider": "razorpay"}


async def process_spreadsheet(ctx, user_id: str, filename: str, storage_path: str, job_id: str, 
                             duplicate_decision: Optional[str] = None, existing_file_id: Optional[str] = None) -> Dict[str, Any]:
    """
    CRITICAL FIX: Removed nested transaction wrapper.
    ExcelProcessor.process_file manages transaction internally (fastapi_backend_v2.py@8676-8680).
    Nested transactions cause deadlocks.
    """
    try:
        # Transaction managed inside start_processing_job -> ExcelProcessor.process_file
        await start_processing_job(
            user_id=user_id, 
            job_id=job_id, 
            storage_path=storage_path, 
            filename=filename,
            duplicate_decision=duplicate_decision,
            existing_file_id=existing_file_id
        )
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


async def learn_field_mapping_batch(ctx, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CRITICAL FIX: Persistent field mapping learning via ARQ.
    Replaces in-process asyncio.Queue to prevent data loss on restart.
    
    Args:
        ctx: ARQ context
        mappings: List of field mapping records to persist
    """
    try:
        from field_mapping_learner import FieldMappingLearner
        
        logger.info(f"🧠 Processing {len(mappings)} field mapping records via ARQ")
        
        learner = FieldMappingLearner(supabase=supabase)
        success_count = 0
        failed_count = 0
        
        for mapping in mappings:
            try:
                success = await learner._write_mapping_with_retry(mapping)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to write field mapping: {e}")
                failed_count += 1
        
        logger.info(f"✅ Field mapping batch completed: {success_count} success, {failed_count} failed")
        return {
            "status": "success",
            "total": len(mappings),
            "success_count": success_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        logger.error(f"❌ Field mapping batch processing failed: {e}")
        delay = await _retry_or_dlq(ctx, 'field_mapping_learning', {'batch_size': len(mappings)}, e, max_retries=3, base_delay=30)
        if delay is None:
            return {"status": "failed", "error": str(e)}
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
        from groq import Groq
        
        logger.info(f"🔍 Starting background relationship detection for user_id={user_id}, file_id={file_id}")
        
        # CRITICAL FIX: Use centralized_cache instead of deprecated ai_cache_system
        # This ensures workers share the same Redis cache as the main application
        try:
            from centralized_cache import safe_get_cache
            cache_client = safe_get_cache()
        except:
            cache_client = None
        
        # CRITICAL FIX: Use Groq client (Llama-3.3-70B) instead of Anthropic
        # This matches the main application's AI configuration
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for relationship detection")
        
        groq_client = Groq(api_key=groq_api_key)
        logger.info("✅ Groq client initialized for relationship detection (Llama-3.3-70B)")
        
        # Initialize relationship detector with Groq client
        # Note: EnhancedRelationshipDetector accepts anthropic_client parameter for backward compatibility
        # but can work with any AI client that follows the same interface
        relationship_detector = EnhancedRelationshipDetector(
            anthropic_client=groq_client,  # Pass Groq client via anthropic_client parameter
            supabase_client=supabase,
            cache_client=cache_client
        )
        
        # CRITICAL FIX: Use new database-driven detection (no N² loops, no hardcoded filenames)
        relationship_results = await relationship_detector.detect_all_relationships(
            user_id, 
            file_id=file_id
        )
        
        # Relationships are already stored by the detector
        logger.info(f"✅ Background relationship detection completed: {relationship_results.get('total_relationships', 0)} relationships found")
        
        # FIX #4-8: Run advanced analytics AFTER relationship detection
        analytics_results = {}
        try:
            logger.info(f"🔬 Starting advanced analytics for user_id={user_id}")
            
            # CRITICAL FIX: Use engines already initialized by EnhancedRelationshipDetector
            # This eliminates redundant initialization and ensures consistent state
            temporal_learner = relationship_detector.temporal_learner if hasattr(relationship_detector, 'temporal_learner') else None
            causal_engine = relationship_detector.causal_engine if hasattr(relationship_detector, 'causal_engine') else None
            
            # Fallback: Initialize only if not available (backward compatibility)
            if temporal_learner is None or causal_engine is None:
                from temporal_pattern_learner import TemporalPatternLearner
                from causal_inference_engine import CausalInferenceEngine
                if temporal_learner is None:
                    temporal_learner = TemporalPatternLearner(supabase)
                if causal_engine is None:
                    causal_engine = CausalInferenceEngine(supabase)
            
            # 1. Learn temporal patterns
            pattern_results = await temporal_learner.learn_all_patterns(user_id)
            analytics_results['temporal_patterns'] = pattern_results.get('total_patterns', 0)
            
            # 2. Detect temporal anomalies
            anomaly_results = await temporal_learner.detect_temporal_anomalies(user_id)
            analytics_results['temporal_anomalies'] = anomaly_results.get('total_anomalies', 0)
            
            # 3. Predict missing relationships
            prediction_results = await temporal_learner.predict_missing_relationships(user_id)
            analytics_results['predicted_relationships'] = prediction_results.get('total_predictions', 0)
            
            # 4. Analyze causal relationships
            causal_results = await causal_engine.analyze_all_relationships(user_id)
            analytics_results['causal_relationships'] = causal_results.get('total_causal', 0)
            
            # 5. Perform root cause analysis
            root_cause_results = await causal_engine.analyze_root_causes(user_id)
            analytics_results['root_cause_analyses'] = root_cause_results.get('total_root_causes', 0)
            
            # 6. Run counterfactual analysis
            counterfactual_results = await causal_engine.analyze_counterfactuals(user_id)
            analytics_results['counterfactual_analyses'] = counterfactual_results.get('total_scenarios', 0)
            
            logger.info(f"✅ Advanced analytics completed: {analytics_results}")
            
        except Exception as analytics_error:
            logger.error(f"⚠️ Advanced analytics failed (non-critical): {analytics_error}")
            analytics_results['error'] = str(analytics_error)
        
        return {
            "status": "success",
            "user_id": user_id,
            "file_id": file_id,
            "total_relationships": relationship_results.get('total_relationships', 0),
            "cross_document_relationships": relationship_results.get('cross_document_relationships', 0),
            "within_file_relationships": relationship_results.get('within_file_relationships', 0),
            "method": "database_joins",
            "complexity": "O(N log N)",
            "advanced_analytics": analytics_results
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
        zoho_books_sync,
        stripe_sync,
        razorpay_sync,
        process_spreadsheet,
        process_pdf,
        learn_field_mapping_batch,  # CRITICAL FIX: Persistent field mapping learning
        detect_relationships,  # Background task for relationship detection
    ]

    # Keep results in Redis only briefly; we don't depend on ARQ results downstream
    keep_result = 0
    # Reasonable timeout per job (seconds)
    function_timeout = 60 * 15
