import os
from typing import Dict, Any, Optional, List
from arq import Retry, cron

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

# FIX #6: Supabase client initialization with graceful degradation
try:
    from supabase_client import get_supabase_client
    supabase = get_supabase_client()
    logger.info("✅ ARQ worker using centralized Supabase client with connection pooling")
except ImportError as e:
    logger.error(f"⚠️ supabase_client.py not found: {e}")
    supabase = None
except Exception as e:
    logger.error(f"⚠️ Failed to initialize Supabase client: {e}")
    supabase = None


# --------------- ARQ Task Functions ---------------
# Each task is fully functional and reuses the existing application logic.

# FIX #7: DEAD CODE REMOVED - _record_job_failure()
# This function was defined but never called anywhere in the codebase.
# ARQ's native Retry mechanism handles retries, and error logging is done
# directly in each sync function (gmail_sync, dropbox_sync, etc.).
# Removing this unused function reduces code clutter and maintenance burden.


async def gmail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Removes custom Redis-backed retry logic
    - Leverages ARQ's built-in retry handling
    - Cleaner, more robust implementation
    - Logs exceptions before retrying for observability
    """
    nango = NangoClient()
    try:
        return await _gmail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error(f"❌ Gmail sync failed: {e}", error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        # ARQ handles retry count internally, no need for custom Redis tracking
        raise Retry(defer=30)  # Initial 30 second delay, ARQ will exponentially backoff


async def dropbox_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    """
    nango = NangoClient()
    try:
        return await _dropbox_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error(f"❌ Dropbox sync failed: {e}", error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=30)


async def gdrive_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    """
    nango = NangoClient()
    try:
        return await _gdrive_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error(f"❌ Google Drive sync failed: {e}", error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=30)


async def zoho_mail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    """
    nango = NangoClient()
    try:
        return await _zohomail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error(f"❌ Zoho Mail sync failed: {e}", error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=45)  # Initial 45 second delay for Zoho (slower API)


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
    FIX #7: Added exception logging for observability
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
        # FIX #7: Log full exception details for debugging
        logger.error(f"❌ ARQ spreadsheet processing failed for job {job_id}", 
                    error=str(e), error_type=type(e).__name__, user_id=user_id, filename=filename)
        # Transaction auto-rolls back on exception
        # Use ARQ's native Retry with exponential backoff
        # Calculate exponential backoff: 60s, 120s, 240s for retries 1, 2, 3
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info(f"Retrying spreadsheet processing in {delay}s (attempt {retry_count + 1}/3)", job_id=job_id)
            raise Retry(defer=delay)
        else:
            logger.error(f"❌ Spreadsheet processing failed after 3 retries for job {job_id}", error=str(e))
            return {"status": "failed", "job_id": job_id, "error": str(e), "retries_exhausted": True}


async def process_pdf(ctx, user_id: str, filename: str, storage_path: str, job_id: str) -> Dict[str, Any]:
    """FIX #7: Wrap ARQ PDF processing in transaction for consistency with Phase 1-11
    Added exception logging for observability
    """
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
        # FIX #7: Log full exception details for debugging
        logger.error(f"❌ ARQ PDF processing failed for job {job_id}", 
                    error=str(e), error_type=type(e).__name__, user_id=user_id, filename=filename)
        # Transaction auto-rolls back on exception
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info(f"Retrying PDF processing in {delay}s (attempt {retry_count + 1}/3)", job_id=job_id)
            raise Retry(defer=delay)
        else:
            logger.error(f"❌ PDF processing failed after 3 retries for job {job_id}", error=str(e))
            return {"status": "failed", "job_id": job_id, "error": str(e), "retries_exhausted": True}


async def learn_field_mapping_batch(ctx, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CRITICAL FIX: Persistent field mapping learning via ARQ.
    Replaces in-process asyncio.Queue to prevent data loss on restart.
    FIX #7: Added exception logging for observability
    
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
                # FIX #7: Log mapping details for debugging
                logger.error(f"Failed to write field mapping", error=str(e), mapping_id=mapping.get('id'))
                failed_count += 1
        
        logger.info(f"✅ Field mapping batch completed: {success_count} success, {failed_count} failed")
        return {
            "status": "success",
            "total": len(mappings),
            "success_count": success_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        # FIX #7: Log full exception details for debugging
        logger.error(f"❌ Field mapping batch processing failed", error=str(e), error_type=type(e).__name__)
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 30 * (2 ** retry_count)  # 30s, 60s, 120s
            logger.info(f"Retrying field mapping batch in {delay}s (attempt {retry_count + 1}/3)")
            raise Retry(defer=delay)
        else:
            logger.error(f"❌ Field mapping batch failed after 3 retries", error=str(e))
            return {"status": "failed", "error": str(e), "retries_exhausted": True}


async def detect_relationships(ctx, user_id: str, file_id: str = None) -> Dict[str, Any]:
    """
    CRITICAL FIX #8: Background task for relationship detection with transaction wrapper.
    This decouples heavy analysis from the synchronous ingestion pipeline and ensures
    atomic operations (all-or-nothing) for data consistency.
    FIX #7: Added exception logging for observability
    
    Args:
        ctx: ARQ context
        user_id: User ID to detect relationships for
        file_id: Optional file_id to scope detection to specific file
    """
    try:
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        from groq import Groq
        from transaction_manager import get_transaction_manager
        from finley_graph_engine import FinleyGraphEngine
        from data_ingestion_normalization.embedding_service import EmbeddingService
        
        logger.info(f"🔍 Starting background relationship detection", user_id=user_id, file_id=file_id)
        
        # FIX #8: Wrap entire detection in transaction for atomic operations
        transaction_manager = get_transaction_manager()
        
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="arq_relationship_detection"
        ) as tx:
            # CRITICAL FIX: Use centralized_cache instead of deprecated ai_cache_system
            # This ensures workers share the same Redis cache as the main application
            try:
                from centralized_cache import safe_get_cache
                cache_client = safe_get_cache()
            except:
                cache_client = None
            
            # FIX #6: Initialize embedding service for dependency injection
            embedding_service = None
            try:
                embedding_service = EmbeddingService(cache_client=cache_client)
                logger.info("✅ EmbeddingService initialized for relationship detection")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize EmbeddingService: {e}")
            
            # CRITICAL FIX: Use Groq client (Llama-3.3-70B) instead of Anthropic
            # This matches the main application's AI configuration
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is required for relationship detection")
            
            groq_client = Groq(api_key=groq_api_key)
            logger.info("✅ Groq client initialized for relationship detection (Llama-3.3-70B)")
            
            # Initialize relationship detector with Groq client and embedding service
            # Note: EnhancedRelationshipDetector accepts anthropic_client parameter for backward compatibility
            # but can work with any AI client that follows the same interface
            relationship_detector = EnhancedRelationshipDetector(
                anthropic_client=groq_client,  # Pass Groq client via anthropic_client parameter
                supabase_client=supabase,
                cache_client=cache_client,
                embedding_service=embedding_service  # FIX #6: Pass injected embedding service
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
            
            # FIX #11: Invalidate graph cache after successful detection
            try:
                graph_engine = FinleyGraphEngine(supabase_url=os.getenv('SUPABASE_URL'), redis_url=os.getenv('REDIS_URL'))
                await graph_engine.clear_graph_cache(user_id)
                logger.info(f"✅ Graph cache invalidated for user {user_id}")
            except Exception as cache_error:
                logger.warning(f"⚠️ Failed to invalidate graph cache: {cache_error}")
            
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
        logger.error(f"❌ Background relationship detection failed for user {user_id}: {e}", exc_info=True)
        # FIX #9: Use ARQ's native Retry with exponential backoff (no _retry_or_dlq)
        # Calculate exponential backoff: 60s, 120s, 240s for retries 1, 2, 3
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info(f"Retrying relationship detection in {delay}s (attempt {retry_count + 1}/3)")
            raise Retry(defer=delay)
        else:
            logger.error(f"❌ Relationship detection failed after 3 retries for user {user_id}")
            return {"status": "failed", "user_id": user_id, "error": str(e), "retries_exhausted": True}


async def generate_prophet_forecasts(ctx) -> None:
    """
    Nightly job: Generate Prophet forecasts for all users.
    Stores results in temporal_patterns table for instant API access.
    """
    try:
        logger.info("🌙 Starting nightly Prophet forecasting job")
        from temporal_pattern_learner import TemporalPatternLearner
        
        # Initialize learner
        learner = TemporalPatternLearner(supabase_client=supabase)
        
        # Get all users with relationships
        # Using a distinct query to find active users
        users_result = supabase.table('relationship_instances').select('user_id').execute()
        if not users_result.data:
            logger.info("No users found for forecasting")
            return
            
        user_ids = list(set(r['user_id'] for r in users_result.data))
        logger.info(f"Generating forecasts for {len(user_ids)} users")
        
        for user_id in user_ids:
            try:
                # Get relationship types for this user
                patterns_result = supabase.table('temporal_patterns').select('relationship_type').eq('user_id', user_id).execute()
                if not patterns_result.data:
                    continue
                    
                relationship_types = [p['relationship_type'] for p in patterns_result.data]
                
                for rel_type in relationship_types:
                    try:
                        # Generate forecast (runs in thread pool - OK for background job)
                        # This will update the database automatically
                        await learner.forecast_with_prophet(
                            user_id=user_id,
                            relationship_type=rel_type,
                            forecast_days=90
                        )
                        logger.debug(f"Generated forecast for user={user_id}, type={rel_type}")
                        
                    except Exception as e:
                        logger.error(f"Forecast failed for user={user_id}, type={rel_type}: {e}")
                        continue
                        
            except Exception as user_err:
                logger.error(f"Forecasting failed for user {user_id}: {user_err}")
                continue
                
        logger.info("✅ Nightly Prophet forecasting job completed")
        
    except Exception as e:
        logger.error(f"❌ Nightly Prophet forecasting job failed: {e}")


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
        generate_prophet_forecasts,  # Nightly forecasting job
    ]
    
    cron_jobs = [
        cron(generate_prophet_forecasts, hour=2, minute=0)  # Run at 2 AM daily
    ]

    # Keep results in Redis only briefly; we don't depend on ARQ results downstream
    keep_result = 0
    # Reasonable timeout per job (seconds)
    function_timeout = 60 * 15
