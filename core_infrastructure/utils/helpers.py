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
