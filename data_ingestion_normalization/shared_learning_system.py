"""
FIX #81: SHARED LEARNING SYSTEM
================================
Consolidated learning system for platform detection and document classification.
Replaces duplicate in-memory buffer + database persistence logic.

Previously duplicated in:
- universal_platform_detector_optimized.py (lines 895-941)
- universal_document_classifier_optimized.py (lines 1006-1054)

This module provides:
- In-memory buffer for immediate access (last 100 entries)
- Database persistence via production-grade log writers
- Unified interface for both detection and classification
"""

import structlog
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = structlog.get_logger(__name__)


class SharedLearningSystem:
    """
    FIX #81: Unified learning system for detection and classification.
    
    Manages:
    - In-memory history buffer (last 100 entries for fast access)
    - Database persistence via log writers
    - Metrics tracking
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize learning system.
        
        Args:
            buffer_size: Maximum in-memory history entries (default 100)
        """
        self.buffer_size = buffer_size
        self.history = []  # In-memory buffer
        
    async def log_detection(self, result: Dict[str, Any], payload: Dict, filename: str, 
                           user_id: Optional[str], supabase_client=None) -> None:
        """
        Log platform detection result to history and database.
        
        Args:
            result: Detection result dict with detection_id, platform, confidence, method, indicators
            payload: Original payload dict
            filename: Source filename
            user_id: User ID for database logging
            supabase_client: Supabase client for persistence
        """
        # Create learning entry
        learning_entry = {
            'detection_id': result.get('detection_id'),
            'platform': result.get('platform'),
            'confidence': result.get('confidence'),
            'method': result.get('method'),
            'indicators': result.get('indicators', []),
            'payload_keys': list(payload.keys()) if isinstance(payload, dict) else [],
            'filename': filename,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Keep small in-memory buffer for immediate access
        self.history.append(learning_entry)
        if len(self.history) > self.buffer_size:
            self.history = self.history[-self.buffer_size:]
        
        # Persist to database using production-grade log writer
        if user_id:
            try:
                from detection_log_writer import log_platform_detection
                
                await log_platform_detection(
                    user_id=user_id,
                    detection_id=result.get('detection_id', 'unknown'),
                    platform=result.get('platform'),
                    confidence=float(result.get('confidence', 0.0)),
                    method=result.get('method'),
                    indicators=result.get('indicators', []),
                    payload_keys=list(payload.keys()) if isinstance(payload, dict) else [],
                    filename=filename,
                    metadata={
                        'processing_time': result.get('processing_time'),
                        'fallback_used': result.get('fallback_used', False),
                        'category': result.get('category', 'unknown'),
                    },
                    supabase_client=supabase_client,
                )
                logger.debug("Platform detection logged", platform=result.get('platform'))
                
            except Exception as e:
                # Don't fail detection if logging fails
                logger.warning("Failed to log platform detection", error=str(e))
    
    async def log_classification(self, result: Dict[str, Any], payload: Dict, filename: str,
                                user_id: Optional[str], supabase_client=None) -> None:
        """
        Log document classification result to history and database.
        
        Args:
            result: Classification result dict with classification_id, document_type, confidence, method, indicators
            payload: Original payload dict
            filename: Source filename
            user_id: User ID for database logging
            supabase_client: Supabase client for persistence
        """
        # Create learning entry
        learning_entry = {
            'classification_id': result.get('classification_id'),
            'document_type': result.get('document_type'),
            'confidence': result.get('confidence'),
            'method': result.get('method'),
            'indicators': result.get('indicators', []),
            'payload_keys': list(payload.keys()) if isinstance(payload, dict) else [],
            'filename': filename,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Keep small in-memory buffer for immediate access
        self.history.append(learning_entry)
        if len(self.history) > self.buffer_size:
            self.history = self.history[-self.buffer_size:]
        
        # Persist to database using production-grade log writer
        if user_id:
            try:
                from detection_log_writer import log_document_classification
                
                await log_document_classification(
                    user_id=user_id,
                    classification_id=result.get('classification_id', 'unknown'),
                    document_type=result.get('document_type'),
                    confidence=float(result.get('confidence', 0.0)),
                    method=result.get('method'),
                    indicators=result.get('indicators', []),
                    payload_keys=list(payload.keys()) if isinstance(payload, dict) else [],
                    filename=filename,
                    metadata={
                        'processing_time': result.get('processing_time'),
                        'category': result.get('category'),
                        'ocr_used': result.get('ocr_used', False),
                    },
                    supabase_client=supabase_client,
                )
                logger.debug(f"✅ Document classification logged: {result.get('document_type')}")
                
            except Exception as e:
                # Don't fail classification if logging fails
                logger.warning(f"Failed to log document classification: {e}")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get in-memory history.
        
        Args:
            limit: Maximum number of entries to return (None = all)
        
        Returns:
            List of history entries
        """
        if limit:
            return self.history[-limit:]
        return self.history
    
    def clear_history(self) -> None:
        """Clear in-memory history buffer."""
        self.history = []
