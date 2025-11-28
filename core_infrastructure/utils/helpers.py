"""
Utility Helper Functions
========================
Consolidated utility functions for JWT handling, Base64 decoding, and data sanitization.

This module centralizes commonly used helper functions to reduce code duplication
and improve maintainability across the application.
"""

import base64
import binascii
import math
import structlog
import os
import hashlib
from typing import Optional, Any, Dict, List
from datetime import datetime

logger = structlog.get_logger(__name__)

# Import polars for NaN detection (using polars exclusively, not pandas)
try:
    import polars as pl
except ImportError:
    pl = None


def clean_jwt_token(token: str) -> str:
    """
    Clean JWT token by removing only harmful whitespace, preserving valid Base64 padding.
    
    Args:
        token: JWT token string that may contain unwanted whitespace
        
    Returns:
        Cleaned JWT token with only harmful whitespace removed
        
    Examples:
        >>> clean_jwt_token("eyJhbGc.\neyJzdWI.")
        "eyJhbGc.eyJzdWI."
    """
    if not token:
        return token
    
    # Only remove newlines, carriage returns, and tabs
    # DO NOT remove spaces as they may be valid Base64 padding
    cleaned = token.strip().replace('\n', '').replace('\r', '').replace('\t', '')
    
    # Ensure it's a valid JWT format (3 parts separated by dots)
    parts = cleaned.split('.')
    if len(parts) == 3:
        # Valid JWT format - return with only harmful whitespace removed
        return cleaned
    else:
        # If not valid JWT format, be more conservative - only remove line breaks
        return token.strip().replace('\n', '').replace('\r', '')


def is_base64(s: str) -> bool:
    """
    Check if string is valid base64.
    
    Args:
        s: String to validate
        
    Returns:
        True if string is valid base64, False otherwise
    """
    try:
        if isinstance(s, str):
            # Check if it looks like base64 (length multiple of 4, valid chars)
            if len(s) % 4 != 0:
                return False
            # Try to decode
            base64.b64decode(s, validate=True)
            return True
    except (ValueError, binascii.Error):
        pass
    return False


def safe_decode_base64(content: str) -> str:
    """
    Safely decode base64 content with fallback to original if decoding fails.
    
    Args:
        content: Base64-encoded string or plain text
        
    Returns:
        Decoded UTF-8 string if valid base64, otherwise original content
        
    Examples:
        >>> safe_decode_base64("aGVsbG8=")
        "hello"
        >>> safe_decode_base64("not-base64")
        "not-base64"
    """
    if not content:
        return content
    
    try:
        if is_base64(content):
            decoded_bytes = base64.b64decode(content)
            # Try to decode as UTF-8 text
            try:
                return decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If not text, return original (might be binary)
                return content
        else:
            return content
    except Exception as e:
        logger.warning(f"Base64 decode failed: {e}")
        return content


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize NaN/Inf values for JSON serialization.
    
    Handles all types of NaN values including:
    - Python float NaN/Inf
    - NumPy NaN/Inf
    - NumPy scalars (np.float64(nan), etc.)
    - Polars null values
    
    Args:
        obj: Object to sanitize (dict, list, float, or any type)
        
    Returns:
        Sanitized object with NaN/Inf replaced by None
        
    Examples:
        >>> sanitize_for_json({'value': float('nan')})
        {'value': None}
        >>> sanitize_for_json([1.0, float('inf'), 3.0])
        [1.0, None, 3.0]
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        # Handle Python float NaN/Inf
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, '__array__') and hasattr(obj, 'dtype'):
        # Handle NumPy scalars (np.float64(nan), np.int32, etc.)
        try:
            import numpy as np
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        except (TypeError, ValueError, ImportError):
            pass  # Not a numeric type or numpy not available
        return obj
    else:
        return obj


# FIX #32: GROQ CLIENT HELPER - Unified initialization pattern
def get_groq_client():
    """
    FIX #32: Unified Groq client initialization
    Provides consistent pattern for all functions that need Groq client.
    
    Attempts to use global groq_client if available, otherwise initializes
    a new client from GROQ_API_KEY environment variable.
    
    Returns:
        Groq client instance
    
    Raises:
        ValueError: If Groq is not available or API key not set
        
    Examples:
        >>> client = get_groq_client()
        >>> response = client.chat.completions.create(...)
    """
    try:
        from groq import Groq
    except ImportError:
        raise ValueError("Groq library not installed. Install with: pip install groq")
    
    # Try to use global client if available (set during app startup)
    try:
        # Import from fastapi_backend_v2 to check global groq_client
        from fastapi_backend_v2 import groq_client as global_groq_client
        if global_groq_client is not None:
            return global_groq_client
    except (ImportError, AttributeError):
        pass  # Global not available, will initialize locally
    
    # Fallback: Initialize from environment
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Create and return local client
    return Groq(api_key=groq_api_key)


# FIX #50: CONVERSATIONAL STATUS MESSAGE GENERATOR (with instructor + groq)
def generate_friendly_status(step: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert technical processing steps into human-readable, conversational status messages.
    
    Uses Groq LLM with instructor library to generate friendly, specific messages.
    Falls back to predefined messages if LLM unavailable.
    
    Args:
        step: Technical step identifier (e.g., 'duplicate_check', 'field_detection')
        context: Optional context dict with details like:
            - 'filename': Name of file being processed
            - 'sheet_count': Number of sheets
            - 'row_count': Number of rows
            - 'similarity_score': Duplicate similarity percentage
            - 'new_rows': New rows found in delta analysis
            - 'existing_rows': Existing rows in delta analysis
    
    Returns:
        Friendly, conversational status message
        
    Examples:
        >>> generate_friendly_status('duplicate_check', {'filename': 'expenses.csv'})
        "Checking if I've seen this file before..."
        
        >>> generate_friendly_status('field_detection', {'columns': ['Date', 'Amount']})
        "Analyzing the structure of your data..."
    """
    context = context or {}
    
    # Predefined friendly messages (fast path - no LLM needed)
    friendly_messages = {
        'initializing_streaming': "Getting ready to read your file...",
        'duplicate_check': "Checking if I've seen this file before...",
        'field_detection': "Analyzing the structure of your data...",
        'platform_detection': "Figuring out where this data came from...",
        'document_classification': "Understanding what type of document this is...",
        'starting_transaction': "Setting up secure storage for your data...",
        'storing': "Saving your file details...",
        'extracting': "Reading through your data...",
        'processing_decision': "Processing your request...",
        'duplicate_found': "Found an exact match - I've processed this before",
        'near_duplicate_found': "Found a similar file - let me compare them",
        'content_duplicate_found': "This data overlaps with something I already have",
        'delta_analysis_complete': "Spotted the differences in your data",
        'entity_resolution': "Matching entities across your data...",
        'classification': "Categorizing your transactions...",
        'enrichment': "Adding context to your data...",
        'complete': "Done! I've processed your file successfully",
        'error': "Oops, something went wrong"
    }
    
    # Return predefined message if available (fast path)
    if step in friendly_messages:
        base_message = friendly_messages[step]
        
        # Enhance with context if available
        if context:
            if step == 'duplicate_check' and context.get('filename'):
                return f"Checking if I've seen {context['filename']} before..."
            elif step == 'field_detection' and context.get('columns'):
                col_count = len(context['columns'])
                return f"Analyzing {col_count} columns in your data..."
            elif step == 'near_duplicate_found' and context.get('similarity_score'):
                score = int(context['similarity_score'] * 100)
                return f"Found a {score}% match with something I processed earlier"
            elif step == 'delta_analysis_complete':
                new = context.get('new_rows', 0)
                existing = context.get('existing_rows', 0)
                return f"Spotted the differences: {new} new rows, {existing} I already know"
            elif step == 'extracting' and context.get('row_count'):
                return f"Reading through {context['row_count']:,} rows of data..."
        
        return base_message
    
    # Fallback: Try to use LLM with instructor for unknown steps
    try:
        import instructor
        
        client = get_groq_client()
        # Wrap Groq client with instructor for structured responses
        client_with_instructor = instructor.from_groq(client)
        
        context_str = ""
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
        
        response = client_with_instructor.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": "Convert technical processing steps into friendly, conversational status updates. Be specific about what you found. Keep messages under 50 characters."
            }, {
                "role": "user",
                "content": f"Step: {step}, Context: {context_str}"
            }],
            max_tokens=50,
            temperature=0.7
        )
        
        message = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated friendly message for '{step}': {message}")
        return message
        
    except ImportError:
        logger.warning("instructor library not installed - using fallback message")
    except Exception as e:
        logger.warning(f"Failed to generate friendly status for '{step}': {e}")
    
    # Final fallback: generic message
    return f"Processing: {step.replace('_', ' ')}..."


async def send_websocket_progress(
    manager: Any,
    job_id: str,
    step: str,
    progress: int,
    context: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send WebSocket progress update with friendly status message.
    
    Combines technical step info with human-readable message and sends via Socket.IO.
    
    Args:
        manager: SocketIOWebSocketManager instance
        job_id: Job ID for routing
        step: Technical step identifier
        progress: Progress percentage (0-100)
        context: Optional context for message generation
        extra_data: Optional additional data to include in update
        
    Examples:
        >>> await send_websocket_progress(
        ...     manager, job_id, 'duplicate_check', 15,
        ...     context={'filename': 'expenses.csv'}
        ... )
    """
    try:
        # Generate friendly message (synchronous function)
        friendly_message = generate_friendly_status(step, context)
        
        # Build payload
        payload = {
            "step": step,
            "message": friendly_message,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add extra data if provided
        if extra_data:
            payload.update(extra_data)
        
        # Send via WebSocket (async)
        await manager.send_update(job_id, payload)
        logger.info(f"✅ WebSocket progress sent: {step} ({progress}%) - {friendly_message}")
        
    except Exception as e:
        logger.error(f"Failed to send WebSocket progress for job {job_id}: {e}")
        # Don't raise - allow processing to continue even if WebSocket fails


# FIX #20: CONSOLIDATED CONNECTOR UPSERT LOGIC
# Replaces 3x duplicate code in _gmail_sync_run, _dropbox_sync_run, _gdrive_sync_run
async def upsert_connector_and_connection(
    supabase_client: Any,
    transaction_manager: Any,
    user_id: str,
    connection_id: str,
    provider_key: str,
    scopes: List[str],
    endpoints_needed: List[str]
) -> tuple:
    """
    FIX #20: Consolidated connector upsert logic.
    
    Replaces duplicate code that appeared 3x in:
    - _gmail_sync_run (lines 9757-9797)
    - _dropbox_sync_run (lines 10254-10281)
    - _gdrive_sync_run (lines 10540-10562)
    
    Args:
        supabase_client: Supabase client instance
        transaction_manager: Transaction manager instance
        user_id: User ID
        connection_id: Nango connection ID
        provider_key: Provider integration ID (NANGO_GMAIL_INTEGRATION_ID, etc.)
        scopes: OAuth scopes required
        endpoints_needed: API endpoints needed
    
    Returns:
        Tuple of (connector_id, user_connection_id) or (None, None) on failure
    """
    try:
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="connector_upsert"
        ) as tx:
            # Upsert connector definition
            try:
                await tx.insert('connectors', {
                    'provider': provider_key,
                    'integration_id': provider_key,
                    'auth_type': 'OAUTH2',
                    'scopes': __import__('orjson').dumps(scopes).decode(),
                    'endpoints_needed': __import__('orjson').dumps(endpoints_needed).decode(),
                    'enabled': True
                })
            except Exception as e:
                # Duplicate key error - connector already exists, which is fine
                logger.debug(f"Connector already exists for {provider_key}: {e}")
            
            # Fetch connector id (still need sync for query)
            connector_row = supabase_client.table('connectors').select('id').eq('provider', provider_key).limit(1).execute()
            connector_id = connector_row.data[0]['id'] if connector_row.data else None
            
            # Upsert user_connection
            try:
                await tx.insert('user_connections', {
                    'user_id': user_id,
                    'connector_id': connector_id,
                    'nango_connection_id': connection_id,
                    'status': 'active',
                    'sync_mode': 'pull'
                })
            except Exception as e:
                # Duplicate key error - user_connection already exists, which is fine
                logger.debug(f"User connection already exists: {e}")
            
            uc_row = supabase_client.table('user_connections').select('id').eq('nango_connection_id', connection_id).limit(1).execute()
            user_connection_id = uc_row.data[0]['id'] if uc_row.data else None
            
            return connector_id, user_connection_id
            
    except Exception as e:
        logger.error(f"Failed to upsert connector records for {provider_key}: {e}")
        return None, None


# FIX #20: CONSOLIDATED SYNC RUN CREATION LOGIC
# Replaces 3x duplicate code in _gmail_sync_run, _dropbox_sync_run, _gdrive_sync_run
async def create_sync_run(
    supabase_client: Any,
    transaction_manager: Any,
    user_id: str,
    user_connection_id: Optional[str],
    mode: str,
    stats: Dict[str, Any]
) -> str:
    """
    FIX #20: Consolidated sync run creation logic.
    
    Replaces duplicate code that appeared 3x in:
    - _gmail_sync_run (lines 9800-9818)
    - _dropbox_sync_run (lines 10283-10300)
    - _gdrive_sync_run (lines 10530-10546)
    
    Args:
        supabase_client: Supabase client instance
        transaction_manager: Transaction manager instance
        user_id: User ID
        user_connection_id: User connection ID (can be None)
        mode: Sync mode ('full', 'incremental', 'historical')
        stats: Initial stats dict
    
    Returns:
        sync_run_id (UUID string)
    """
    import uuid
    import pendulum
    
    sync_run_id = str(uuid.uuid4())
    try:
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="connector_sync_start"
        ) as tx:
            await tx.insert('sync_runs', {
                'id': sync_run_id,
                'user_id': user_id,
                'user_connection_id': user_connection_id,
                'type': mode,
                'status': 'running',
                'started_at': pendulum.now().to_iso8601_string(),
                'stats': __import__('orjson').dumps(stats).decode()
            })
    except Exception as e:
        logger.error(f"Failed to create sync_run: {e}")
        # Still return the ID so processing can continue
    
    return sync_run_id


# FIX #21: SYNC CURSOR MANAGEMENT (Issue #1)
# Read and update sync cursors for incremental syncs
async def get_sync_cursor(
    supabase_client: Any,
    user_connection_id: str,
    resource: str,
    cursor_type: str = 'time'
) -> Optional[str]:
    """Get the last sync cursor for incremental syncs."""
    try:
        cursor_row = supabase_client.table('sync_cursors').select('value').eq(
            'user_connection_id', user_connection_id
        ).eq('resource', resource).eq('cursor_type', cursor_type).limit(1).execute()
        
        if cursor_row.data:
            return cursor_row.data[0]['value']
        return None
    except Exception as e:
        logger.warning(f"Failed to get sync cursor: {e}")
        return None


async def update_sync_cursor(
    transaction_manager: Any,
    user_id: str,
    user_connection_id: str,
    resource: str,
    cursor_value: str,
    cursor_type: str = 'time'
) -> bool:
    """Update the sync cursor after successful sync."""
    try:
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="sync_cursor_update"
        ) as tx:
            # Try insert first (new cursor)
            try:
                await tx.insert('sync_cursors', {
                    'user_id': user_id,
                    'user_connection_id': user_connection_id,
                    'resource': resource,
                    'cursor_type': cursor_type,
                    'value': cursor_value,
                    'updated_at': __import__('pendulum').now().to_iso8601_string()
                })
            except Exception:
                # If insert fails (duplicate), update instead
                await tx.update('sync_cursors', {
                    'value': cursor_value,
                    'updated_at': __import__('pendulum').now().to_iso8601_string()
                }, {
                    'user_connection_id': user_connection_id,
                    'resource': resource,
                    'cursor_type': cursor_type
                })
        return True
    except Exception as e:
        logger.error(f"Failed to update sync cursor: {e}")
        return False


# FIX #22: SYNC COMPLETION TRACKING (Issue #6)
# Track sync completion with status and duration
async def complete_sync_run(
    transaction_manager: Any,
    supabase_client: Any,
    user_id: str,
    sync_run_id: str,
    user_connection_id: str,
    status: str,
    stats: Dict[str, Any],
    errors: List[str],
    start_time: Any
) -> bool:
    """Complete a sync run and update user_connection with completion info."""
    try:
        import pendulum
        end_time = pendulum.now()
        duration_seconds = (end_time - start_time).total_seconds()
        
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="sync_completion"
        ) as tx:
            # Update sync_runs with completion status
            await tx.update('sync_runs', {
                'status': status,
                'finished_at': end_time.to_iso8601_string(),
                'stats': __import__('orjson').dumps({**stats, 'duration_seconds': duration_seconds}).decode(),
                'error': '; '.join(errors[:5])[:500] if errors else None
            }, {'id': sync_run_id})
            
            # Update user_connections with last sync info
            await tx.update('user_connections', {
                'last_synced_at': end_time.to_iso8601_string(),
                'metadata': __import__('orjson').dumps({
                    'last_sync_status': status,
                    'last_sync_duration_seconds': duration_seconds,
                    'last_sync_records': stats.get('records_fetched', 0),
                    'last_sync_errors': len(errors)
                }).decode()
            }, {'id': user_connection_id})
        
        return True
    except Exception as e:
        logger.error(f"Failed to complete sync run: {e}")
        return False


# FIX #23: WEBHOOK RETRY MECHANISM (Issue #5)
# Retry failed webhooks with exponential backoff
async def retry_failed_webhook(
    transaction_manager: Any,
    supabase_client: Any,
    user_id: str,
    webhook_event_id: str,
    retry_count: int = 0,
    max_retries: int = 3
) -> bool:
    """Retry a failed webhook with exponential backoff."""
    try:
        import asyncio
        import pendulum
        
        # Exponential backoff: 2^retry_count seconds (2s, 4s, 8s)
        delay_seconds = 2 ** retry_count
        
        if retry_count >= max_retries:
            logger.warning(f"Webhook {webhook_event_id} exceeded max retries ({max_retries})")
            async with transaction_manager.transaction(
                user_id=user_id,
                operation_type="webhook_final_failure"
            ) as tx:
                await tx.update('webhook_events', {
                    'status': 'failed',
                    'error': f'Max retries ({max_retries}) exceeded'
                }, {'id': webhook_event_id})
            return False
        
        # Schedule retry
        next_retry_at = pendulum.now().add(seconds=delay_seconds)
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="webhook_retry_schedule"
        ) as tx:
            await tx.update('webhook_events', {
                'status': 'queued',
                'metadata': __import__('orjson').dumps({
                    'retry_count': retry_count + 1,
                    'next_retry_at': next_retry_at.to_iso8601_string()
                }).decode()
            }, {'id': webhook_event_id})
        
        logger.info(f"Webhook {webhook_event_id} scheduled for retry in {delay_seconds}s (attempt {retry_count + 1}/{max_retries})")
        return True
    except Exception as e:
        logger.error(f"Failed to schedule webhook retry: {e}")
        return False


# FIX #24: EXTERNAL ITEM ERROR TRACKING (Issue #3)
# Track errors when external items fail to process
async def store_external_item_with_error(
    transaction_manager: Any,
    user_id: str,
    item_data: Dict[str, Any],
    error_message: Optional[str] = None
) -> bool:
    """Store external item with error tracking."""
    try:
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="external_item_store"
        ) as tx:
            if error_message:
                item_data['status'] = 'failed'
                item_data['error'] = error_message[:500]
            
            await tx.insert('external_items', item_data)
        return True
    except Exception as e:
        logger.error(f"Failed to store external item: {e}")
        return False


# FIX #79: SHARED CACHE KEY GENERATION
# Replaces 3x duplicate code in:
# - universal_platform_detector_optimized.py (_generate_detection_id)
# - universal_document_classifier_optimized.py (_generate_classification_id)
# - entity_resolver_optimized.py (_generate_resolution_id)
def generate_cache_key(prefix: str, *args) -> str:
    """
    FIX #79: Unified cache key generation using deterministic hashing.
    
    Generates consistent cache keys for detection, classification, and resolution
    operations. Uses MD5 hashing for deterministic, collision-resistant keys.
    
    Args:
        prefix: Key prefix (e.g., 'detect', 'classify', 'resolve')
        *args: Variable arguments to hash (payload, filename, user_id, entity_name, etc.)
    
    Returns:
        Deterministic cache key in format: "{prefix}_{user_part}_{content_hash}"
        
    Examples:
        >>> generate_cache_key('detect', {'field': 'value'}, 'file.csv', 'user123')
        'detect_user123_a1b2c3d4'
        
        >>> generate_cache_key('classify', {'type': 'invoice'}, 'doc.pdf', 'user456')
        'classify_user456_e5f6g7h8'
        
        >>> generate_cache_key('resolve', 'Company Inc', 'vendor', 'stripe', 'user789')
        'resolve_user789_i9j0k1l2'
    """
    if not args:
        raise ValueError("generate_cache_key requires at least one argument after prefix")
    
    # Convert all arguments to strings and combine
    content_parts = []
    user_part = "anon"
    
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            # For dicts, use sorted items for deterministic hashing
            arg_str = str(sorted(arg.items()))
        else:
            arg_str = str(arg)
        
        content_parts.append(arg_str)
        
        # Extract user_id if it's the last argument and looks like a UUID/user ID
        if i == len(args) - 1 and arg and isinstance(arg, str) and len(arg) > 0:
            user_part = arg[:12]  # Use first 12 chars of user_id
    
    # Hash all content for deterministic key
    content_str = "|".join(content_parts)
    content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
    
    return f"{prefix}_{user_part}_{content_hash}"


# FIX #52: SHARED CACHE INITIALIZATION
# Replaces 3x duplicate code in:
# - universal_platform_detector_optimized.py (lines 101-109)
# - universal_document_classifier_optimized.py (lines 112-120)
# - universal_field_detector.py (lines 108-113)
def initialize_centralized_cache(cache_client=None):
    """
    FIX #52: Unified cache initialization for all detectors.
    
    Initializes centralized Redis cache with fail-fast behavior.
    Prevents cache divergence across workers by enforcing Redis requirement.
    
    Args:
        cache_client: Optional pre-initialized cache client
        
    Returns:
        Initialized cache client
        
    Raises:
        RuntimeError: If Redis cache not available
    """
    from centralized_cache import safe_get_cache
    
    cache = cache_client or safe_get_cache()
    if cache is None:
        raise RuntimeError(
            "Centralized Redis cache not initialized. "
            "Call initialize_cache() at startup or set REDIS_URL environment variable. "
            "MEMORY cache fallback removed to prevent cache divergence across workers."
        )
    
    return cache
