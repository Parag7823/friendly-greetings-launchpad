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
