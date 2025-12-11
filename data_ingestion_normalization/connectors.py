"""Connector models and OAuth logic for API data pipeline processing."""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, field_validator
from datetime import datetime
import os
import tempfile
import uuid
import structlog
import pendulum

try:
    import xxhash
except ImportError:
    xxhash = None

logger = structlog.get_logger(__name__)


class GmailMetadata(BaseModel):
    """Validated metadata for user_connections.metadata field."""
    last_history_id: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[List[str]] = None
    error_count: int = 0
    
    class Config:
        extra = "allow"


class SyncRunStats(BaseModel):
    """Validated stats for sync_runs.stats field."""
    records_fetched: int = 0
    actions_used: int = 0
    attachments_saved: int = 0
    queued_jobs: int = 0
    skipped: int = 0
    
    class Config:
        extra = "allow"


class ZohoMailMetadata(BaseModel):
    """Validated metadata for Zoho Mail user_connections.metadata field."""
    last_sync_token: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[List[str]] = None
    error_count: int = 0
    
    class Config:
        extra = "allow"


class XeroMetadata(BaseModel):
    """Validated metadata for Xero user_connections.metadata field."""
    last_sync_token: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[List[str]] = None
    error_count: int = 0
    tenant_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class StripeMetadata(BaseModel):
    """Validated metadata for Stripe user_connections.metadata field."""
    last_sync_token: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[str] = None
    error_count: int = 0
    account_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class PayPalMetadata(BaseModel):
    """Validated metadata for PayPal user_connections.metadata field."""
    last_sync_token: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[List[str]] = None
    error_count: int = 0
    merchant_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class RazorpayMetadata(BaseModel):
    """Validated metadata for Razorpay user_connections.metadata field."""
    last_sync_token: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_errors: Optional[List[str]] = None
    error_count: int = 0
    merchant_id: Optional[str] = None
    
    class Config:
        extra = "allow"


# ============================================================================
# REQUEST MODELS (Moved from lines 3684-3740)
# ============================================================================

class ConnectorSyncRequest(BaseModel):
    user_id: str
    connection_id: str  # Nango connection id
    integration_id: Optional[str] = None  # defaults to google-mail
    mode: str = "historical"  # historical | incremental
    lookback_days: Optional[int] = 365  # used for historical q filter
    max_results: Optional[int] = 100  # per page
    session_token: Optional[str] = None
    correlation_id: Optional[str] = None

    if field_validator:
        @field_validator('max_results')
        @classmethod
        def _validate_max_results(cls, v):
            if v is None:
                return 100
            try:
                iv = int(v)
            except Exception:
                raise ValueError('max_results must be an integer')
            if iv <= 0:
                raise ValueError('max_results must be positive')
            return min(iv, 1000)

        @field_validator('lookback_days')
        @classmethod
        def _validate_lookback_days(cls, v):
            if v is None:
                return 365
            iv = int(v)
            if iv < 0:
                raise ValueError('lookback_days must be >= 0')
            return iv

        @field_validator('mode')
        @classmethod
        def _validate_mode(cls, v):
            allowed = {'historical', 'incremental'}
            if v not in allowed:
                raise ValueError('mode must be one of historical, incremental')
            return v


class UserConnectionsRequest(BaseModel):
    user_id: str
    session_token: Optional[str] = None


class UpdateFrequencyRequest(BaseModel):
    user_id: str
    connection_id: str  # nango connection id
    minutes: int
    session_token: Optional[str] = None


class ConnectorDisconnectRequest(BaseModel):
    user_id: str
    connection_id: str
    provider: Optional[str] = None
    session_token: Optional[str] = None


# ============================================================================
# STORAGE HELPERS (Moved from line 3826)
# ============================================================================

async def store_external_item_attachment(
    user_id: str, 
    provider: str, 
    message_id: str, 
    filename: str, 
    content: bytes,
    supabase_client
) -> Tuple[str, str]:
    """
    Store attachment bytes to Supabase Storage.
    
    Returns:
        Tuple of (storage_path, file_hash)
    """
    # Import helper function
    from core_infrastructure.utils.helpers import _safe_filename
    
    safe_name = _safe_filename(filename)
    
    # Compute hash for dedupe (xxhash: 5-10x faster for large files)
    file_hash = xxhash.xxh3_128(content).hexdigest() if xxhash else None
    
    # Build storage path
    today = datetime.utcnow().strftime('%Y/%m/%d')
    storage_path = f"external/{provider}/{user_id}/{today}/{message_id}/{safe_name}"
    
    try:
        storage = supabase_client.storage.from_("finely-upload")
        # Render's supabase-py expects a filesystem path, not a BytesIO
        # Write to a secure temporary file and upload by path
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            storage.upload(storage_path, tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp file {tmp_path}: {cleanup_err}")
        return storage_path, file_hash
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        raise


# ============================================================================
# PIPELINE PROCESSING (Moved from line 4141)
# ============================================================================

async def process_api_data_through_pipeline(
    user_id: str,
    data: List[Dict[str, Any]],
    source_platform: str,
    sync_run_id: str,
    user_connection_id: str,
    supabase_client,
    excel_processor_instance
) -> Dict[str, Any]:
    """
    Process API data through the main ExcelProcessor pipeline.
    This ensures consistent duplicate detection, enrichment, and entity resolution.
    
    Args:
        user_id: User ID
        data: List of API records
        source_platform: Platform name
        sync_run_id: Sync run ID for tracking
        user_connection_id: User connection ID
        supabase_client: Supabase client instance
        excel_processor_instance: ExcelProcessor instance
    
    Returns:
        Processing results with stats
    """
    try:
        # Import helper function
        from core_infrastructure.utils.helpers import _convert_api_data_to_csv_format
        
        # Convert API data to CSV format
        csv_bytes, filename = await _convert_api_data_to_csv_format(data, source_platform)
        
        # Calculate file hash using xxh3_128 (standardized)
        file_hash = xxhash.xxh3_128(csv_bytes).hexdigest() if xxhash else str(uuid.uuid4())
        
        # Store CSV in Supabase Storage
        storage_path = f"{user_id}/connector_syncs/{source_platform.lower()}/{filename}"
        try:
            storage = supabase_client.storage.from_("finely-upload")
            storage.upload(storage_path, csv_bytes, {"content-type": "text/csv"})
            logger.info(f"✅ Uploaded {source_platform} CSV to storage: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload {source_platform} CSV to storage: {e}")
            raise
        
        # Create ingestion job for tracking
        job_id = str(uuid.uuid4())
        try:
            supabase_client.table('ingestion_jobs').insert({
                'id': job_id,
                'user_id': user_id,
                'file_name': filename,
                'file_size': len(csv_bytes),
                'file_hash': file_hash,
                'source': f'connector_{source_platform.lower()}',
                'job_type': 'api_sync',
                'status': 'processing',
                'created_at': pendulum.now().to_iso8601_string()
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to create ingestion_job for {source_platform}: {e}")
        
        # Process through main pipeline
        logger.info(f"🔄 Processing {source_platform} data through main ExcelProcessor pipeline...")
        result = await excel_processor_instance.process_file(
            file_content=csv_bytes,
            filename=filename,
            user_id=user_id,
            job_id=job_id,
            file_hash=file_hash,
            source_platform=source_platform
        )
        
        # Update ingestion job status
        try:
            supabase_client.table('ingestion_jobs').update({
                'status': 'completed',
                'updated_at': pendulum.now().to_iso8601_string()
            }).eq('id', job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update ingestion_job status: {e}")
        
        logger.info(f"✅ {source_platform} data processed through main pipeline: {result.get('total_rows', 0)} rows")
        
        return {
            'status': 'success',
            'job_id': job_id,
            'file_hash': file_hash,
            'total_rows': result.get('total_rows', 0),
            'processed_rows': result.get('processed_rows', 0),
            'storage_path': storage_path
        }
        
    except Exception as e:
        logger.error(f"Pipeline processing failed for {source_platform}: {e}")
        raise


# ============================================================================
# OAUTH LOGIC (Moved from line 4250)
# ============================================================================
# Note: The OAuth endpoint logic (connectors_initiate) should STAY in fastapi_backend_v2.py
# as it's a FastAPI endpoint. This is just documentation.
# The endpoint should import from this module:
#
# from core_infrastructure.connectors import ConnectorSyncRequest
# @app.post("/api/connectors/initiate")
# async def connectors_initiate(req: ConnectorSyncRequest):
#     ...endpoint logic stays in backend...
