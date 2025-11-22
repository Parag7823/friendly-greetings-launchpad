"""
Utility Helper Functions
========================
Consolidated utility functions for JWT handling and Base64 decoding.

This module centralizes commonly used helper functions to reduce code duplication
and improve maintainability across the application.
"""

import base64
import binascii
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


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
