
Developer Debug Logger
======================
Captures detailed AI reasoning, confidence scores, and processing details
for developer introspection and debugging.

Author: Finley AI Team
Date: 2025-01-27
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DebugLogEntry:
    """Structured debug log entry"""
    job_id: str
    user_id: str
    stage: str
    component: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'stage': self.stage,
            'component': self.component,
            'data': self.data,
            'metadata': self.metadata or {},
            'created_at': datetime.utcnow().isoformat()
        }


class DebugLogger:
    """
    Centralized debug logger for capturing AI reasoning and processing details.
    
    Features:
    - Stores debug data in Supabase for persistence
    - Streams to WebSocket for real-time monitoring
    - Structured logging with stages and components
    - Automatic metadata capture (timing, errors, warnings)
    """
    
    def __init__(self, supabase_client=None, websocket_manager=None):
        self.supabase = supabase_client
        self.websocket_manager = websocket_manager
        self.enabled = True  # Can be disabled in production
        
    async def log_stage(
        self,
        job_id: str,
        user_id: str,
        stage: str,
        component: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a processing stage with detailed debug data.
        
        Args:
            job_id: Job identifier
            user_id: User identifier
            stage: Processing stage (upload, excel, platform, etc.)
            component: Component name (e.g., "UniversalPlatformDetector")
            data: Debug data (reasoning, confidence, indicators, etc.)
            metadata: Additional metadata (timing, errors, warnings)
        """
        if not self.enabled:
            return
        
        try:
            entry = DebugLogEntry(
                job_id=job_id,
                user_id=user_id,
                stage=stage,
                component=component,
                data=data,
                metadata=metadata
            )
            
            # Store in database
            if self.supabase:
                try:
                    self.supabase.table('debug_logs').insert(entry.to_dict()).execute()
                except Exception as db_err:
                    logger.warning(f"Failed to store debug log in DB: {db_err}")
            
            # Stream to WebSocket for real-time monitoring
            if self.websocket_manager:
                try:
                    await self.websocket_manager.send_debug_update(
                        job_id=job_id,
                        stage=stage,
                        component=component,
                        data=data
                    )
                except Exception as ws_err:
                    logger.warning(f"Failed to send debug update via WebSocket: {ws_err}")
            
            # Also log to console for immediate visibility
            logger.info(f"[DEBUG] {stage}/{component}: {json.dumps(data, indent=2)[:200]}...")
            
        except Exception as e:
            logger.error(f"Debug logging failed: {e}")
    
    async def log_platform_detection(
        self,
        job_id: str,
        user_id: str,
        platform: str,
        confidence: float,
        method: str,
        indicators: List[str],
        reasoning: str,
        ai_prompt: Optional[str] = None,
        ai_response: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Log platform detection details"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage="platform_detection",
            component="UniversalPlatformDetector",
            data={
                "platform": platform,
                "confidence": confidence,
                "method": method,
                "indicators": indicators,
                "reasoning": reasoning,
                "ai_prompt": ai_prompt,
                "ai_response": ai_response
            },
            metadata={
                "processing_time_ms": processing_time_ms
            }
        )
    
    async def log_document_classification(
        self,
        job_id: str,
        user_id: str,
        document_type: str,
        confidence: float,
        method: str,
        indicators: List[str],
        reasoning: str,
        ai_prompt: Optional[str] = None,
        ai_response: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Log document classification details"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage="document_classification",
            component="UniversalDocumentClassifier",
            data={
                "document_type": document_type,
                "confidence": confidence,
                "method": method,
                "indicators": indicators,
                "reasoning": reasoning,
                "ai_prompt": ai_prompt,
                "ai_response": ai_response
            },
            metadata={
                "processing_time_ms": processing_time_ms
            }
        )
    
    async def log_row_processing(
        self,
        job_id: str,
        user_id: str,
        row_number: int,
        vendor_data: Dict[str, Any],
        amount_data: Dict[str, Any],
        date_data: Dict[str, Any],
        category_data: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Log individual row processing details"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage="row_processing",
            component="RowProcessor",
            data={
                "row_number": row_number,
                "vendor": vendor_data,
                "amount": amount_data,
                "date": date_data,
                "category": category_data
            },
            metadata={
                "processing_time_ms": processing_time_ms
            }
        )
    
    async def log_entity_resolution(
        self,
        job_id: str,
        user_id: str,
        raw_name: str,
        candidates: List[Dict[str, Any]],
        winner: str,
        action: str,
        entity_id: str,
        processing_time_ms: Optional[float] = None
    ):
        """Log entity resolution details"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage="entity_resolution",
            component="EntityResolver",
            data={
                "raw_name": raw_name,
                "candidates": candidates,
                "winner": winner,
                "action": action,
                "entity_id": entity_id
            },
            metadata={
                "processing_time_ms": processing_time_ms
            }
        )
    
    async def log_relationship_detection(
        self,
        job_id: str,
        user_id: str,
        relationships: List[Dict[str, Any]],
        total_found: int,
        processing_time_ms: Optional[float] = None
    ):
        """Log relationship detection results"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage="relationship_detection",
            component="EnhancedRelationshipDetector",
            data={
                "relationships": relationships,
                "total_found": total_found
            },
            metadata={
                "processing_time_ms": processing_time_ms
            }
        )
    
    async def log_error(
        self,
        job_id: str,
        user_id: str,
        stage: str,
        component: str,
        error: str,
        stack_trace: Optional[str] = None
    ):
        """Log processing errors"""
        await self.log_stage(
            job_id=job_id,
            user_id=user_id,
            stage=stage,
            component=component,
            data={
                "error": error,
                "stack_trace": stack_trace
            },
            metadata={
                "error": True
            }
        )


# Global debug logger instance
_debug_logger = None

def get_debug_logger(supabase_client=None, websocket_manager=None):
    """Get or create global debug logger instance"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(supabase_client, websocket_manager)
    return _debug_logger
