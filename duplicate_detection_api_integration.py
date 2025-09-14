"""
Duplicate Detection API Integration Layer
========================================

This module provides the integration layer between the production duplicate detection service
and the FastAPI backend, including WebSocket updates and proper error handling.

Features:
- FastAPI endpoint integration
- WebSocket real-time updates
- Structured error responses
- Security validation
- Performance monitoring
- Graceful error handling

Author: Senior Full-Stack Engineer
Version: 2.0.0
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from supabase import Client

from production_duplicate_detection_service import (
    ProductionDuplicateDetectionService,
    FileMetadata,
    DuplicateType,
    DuplicateAction
)

# Configure logging
logger = logging.getLogger(__name__)

class DuplicateDetectionRequest(BaseModel):
    """Request model for duplicate detection API"""
    job_id: str = Field(..., description="Unique job identifier")
    user_id: str = Field(..., description="User identifier")
    file_hash: str = Field(..., description="SHA-256 hash of the file")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")
    enable_near_duplicate: bool = Field(default=True, description="Enable near-duplicate detection")

class DuplicateDetectionResponse(BaseModel):
    """Response model for duplicate detection API"""
    status: str = Field(..., description="Status of the operation")
    is_duplicate: bool = Field(..., description="Whether duplicates were found")
    duplicate_type: str = Field(..., description="Type of duplicate detected")
    similarity_score: float = Field(..., description="Similarity score (0.0-1.0)")
    duplicate_files: List[Dict[str, Any]] = Field(..., description="List of duplicate files")
    recommendation: str = Field(..., description="Recommended action")
    message: str = Field(..., description="Human-readable message")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    requires_user_decision: bool = Field(default=False, description="Whether user decision is required")
    error: Optional[str] = Field(default=None, description="Error message if any")

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept WebSocket connection and store it"""
        await websocket.accept()
        self.active_connections[job_id] = websocket
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, job_id: str):
        """Remove WebSocket connection"""
        if job_id in self.active_connections:
            del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_update(self, job_id: str, data: Dict[str, Any]):
        """Send update to specific WebSocket connection"""
        if job_id in self.active_connections:
            try:
                websocket = self.active_connections[job_id]
                await websocket.send_json(data)
                logger.debug(f"Sent update to job {job_id}: {data.get('step', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to send WebSocket update for job {job_id}: {e}")
                self.disconnect(job_id)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

class DuplicateDetectionAPIIntegration:
    """
    API integration layer for duplicate detection service.
    
    Handles the integration between the production duplicate detection service
    and the FastAPI backend, including WebSocket updates and error handling.
    """
    
    def __init__(self, supabase: Client, redis_client: Optional[Any] = None):
        """
        Initialize the API integration layer.
        
        Args:
            supabase: Supabase client for database operations
            redis_client: Optional Redis client for caching
        """
        self.duplicate_service = ProductionDuplicateDetectionService(supabase, redis_client)
        self.supabase = supabase
        self.redis_client = redis_client
        
        logger.info("Duplicate Detection API Integration initialized")
    
    async def detect_duplicates_with_websocket(
        self, 
        request: DuplicateDetectionRequest,
        file_content: bytes
    ) -> DuplicateDetectionResponse:
        """
        Detect duplicates with WebSocket updates.
        
        Args:
            request: Duplicate detection request
            file_content: Raw file content
            
        Returns:
            DuplicateDetectionResponse with results
        """
        try:
            # Send initial update
            await websocket_manager.send_update(request.job_id, {
                "step": "duplicate_check",
                "message": "🔍 Checking for duplicates...",
                "progress": 10
            })
            
            # Create file metadata
            file_metadata = FileMetadata(
                user_id=request.user_id,
                file_hash=request.file_hash,
                filename=request.filename,
                file_size=request.file_size,
                content_type=request.content_type,
                upload_timestamp=datetime.utcnow()
            )
            
            # Detect duplicates
            result = await self.duplicate_service.detect_duplicates(
                file_content,
                file_metadata,
                enable_near_duplicate=request.enable_near_duplicate
            )
            
            # Send appropriate WebSocket update based on result
            if result.is_duplicate:
                if result.duplicate_type == DuplicateType.EXACT:
                    await websocket_manager.send_update(request.job_id, {
                        "step": "exact_duplicate_found",
                        "message": f"⚠️ Exact duplicate found: {result.message}",
                        "progress": 20,
                        "duplicate_info": {
                            "type": result.duplicate_type.value,
                            "similarity_score": result.similarity_score,
                            "duplicate_files": result.duplicate_files,
                            "recommendation": result.recommendation.value
                        },
                        "requires_user_decision": True
                    })
                elif result.duplicate_type == DuplicateType.NEAR:
                    await websocket_manager.send_update(request.job_id, {
                        "step": "near_duplicate_found",
                        "message": f"🔄 Near-duplicate found: {result.message}",
                        "progress": 20,
                        "duplicate_info": {
                            "type": result.duplicate_type.value,
                            "similarity_score": result.similarity_score,
                            "duplicate_files": result.duplicate_files,
                            "recommendation": result.recommendation.value
                        },
                        "requires_user_decision": True
                    })
            else:
                await websocket_manager.send_update(request.job_id, {
                    "step": "no_duplicates",
                    "message": "✅ No duplicates found, proceeding with processing...",
                    "progress": 20
                })
            
            # Convert result to API response
            response = DuplicateDetectionResponse(
                status="success",
                is_duplicate=result.is_duplicate,
                duplicate_type=result.duplicate_type.value,
                similarity_score=result.similarity_score,
                duplicate_files=result.duplicate_files,
                recommendation=result.recommendation.value,
                message=result.message,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                requires_user_decision=result.is_duplicate,
                error=result.error
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Duplicate detection failed for job {request.job_id}: {e}", exc_info=True)
            
            # Send error update via WebSocket
            await websocket_manager.send_update(request.job_id, {
                "step": "error",
                "message": f"❌ Duplicate detection failed: {str(e)}",
                "progress": 0,
                "error": str(e)
            })
            
            # Return error response
            return DuplicateDetectionResponse(
                status="error",
                is_duplicate=False,
                duplicate_type=DuplicateType.NONE.value,
                similarity_score=0.0,
                duplicate_files=[],
                recommendation=DuplicateAction.REPLACE.value,
                message="Duplicate detection failed",
                confidence=0.0,
                processing_time_ms=0,
                requires_user_decision=False,
                error=str(e)
            )
    
    async def handle_duplicate_decision(
        self,
        job_id: str,
        user_id: str,
        file_hash: str,
        decision: str
    ) -> Dict[str, Any]:
        """
        Handle user's decision about duplicate file.
        
        Args:
            job_id: Job identifier
            user_id: User identifier
            file_hash: File hash
            decision: User's decision (replace, keep_both, skip, merge)
            
        Returns:
            Dictionary with decision result
        """
        try:
            # Send decision update
            await websocket_manager.send_update(job_id, {
                "step": "processing_decision",
                "message": f"Processing decision: {decision}",
                "progress": 30
            })
            
            # Handle the decision
            if decision == "replace":
                # Mark old files as replaced
                await self._mark_files_as_replaced(user_id, file_hash)
                
                await websocket_manager.send_update(job_id, {
                    "step": "replaced_files",
                    "message": "✅ Old files marked as replaced, proceeding with new file...",
                    "progress": 40
                })
                
                return {
                    "status": "success",
                    "action": "proceed_with_new",
                    "message": "Old files replaced successfully"
                }
                
            elif decision == "keep_both":
                await websocket_manager.send_update(job_id, {
                    "step": "keeping_both",
                    "message": "✅ Keeping both files, proceeding with new file...",
                    "progress": 40
                })
                
                return {
                    "status": "success",
                    "action": "proceed_with_new",
                    "message": "Both files will be kept"
                }
                
            elif decision == "skip":
                await websocket_manager.send_update(job_id, {
                    "step": "skipped_upload",
                    "message": "⏭️ Upload skipped due to duplicate",
                    "progress": 100
                })
                
                return {
                    "status": "success",
                    "action": "abort",
                    "message": "Upload skipped"
                }
                
            elif decision == "merge":
                await websocket_manager.send_update(job_id, {
                    "step": "merging_files",
                    "message": "🔄 Merging files, analyzing differences...",
                    "progress": 40
                })
                
                # TODO: Implement merge logic
                return {
                    "status": "success",
                    "action": "proceed_with_merge",
                    "message": "Files will be merged"
                }
                
            else:
                raise ValueError(f"Invalid decision: {decision}")
                
        except Exception as e:
            logger.error(f"Failed to handle duplicate decision: {e}", exc_info=True)
            
            await websocket_manager.send_update(job_id, {
                "step": "decision_error",
                "message": f"❌ Failed to process decision: {str(e)}",
                "progress": 0,
                "error": str(e)
            })
            
            return {
                "status": "error",
                "action": "abort",
                "message": f"Failed to process decision: {str(e)}"
            }
    
    async def _mark_files_as_replaced(self, user_id: str, file_hash: str) -> None:
        """Mark files with matching hash as replaced"""
        try:
            # Update files with matching hash
            result = self.supabase.table('raw_records').update({
                'status': 'replaced',
                'updated_at': datetime.utcnow().isoformat()
            }).eq('user_id', user_id).eq('content->>file_hash', file_hash).execute()
            
            logger.info(f"Marked {len(result.data)} files as replaced for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to mark files as replaced: {e}")
            raise
    
    async def get_duplicate_metrics(self) -> Dict[str, Any]:
        """Get duplicate detection metrics for monitoring"""
        try:
            service_metrics = await self.duplicate_service.get_metrics()
            
            return {
                **service_metrics,
                "active_websocket_connections": len(websocket_manager.active_connections),
                "service_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "service_status": "error",
                "error": str(e)
            }
    
    async def clear_duplicate_cache(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear duplicate detection cache"""
        try:
            await self.duplicate_service.clear_cache(user_id)
            
            return {
                "status": "success",
                "message": f"Cache cleared for {'user ' + user_id if user_id else 'all users'}"
            }
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {
                "status": "error",
                "message": f"Failed to clear cache: {str(e)}"
            }

# FastAPI endpoint functions
async def detect_duplicates_endpoint(
    request: DuplicateDetectionRequest,
    file_content: bytes,
    duplicate_api: DuplicateDetectionAPIIntegration
) -> DuplicateDetectionResponse:
    """
    FastAPI endpoint for duplicate detection.
    
    Args:
        request: Duplicate detection request
        file_content: Raw file content
        duplicate_api: Duplicate detection API integration instance
        
    Returns:
        DuplicateDetectionResponse
    """
    return await duplicate_api.detect_duplicates_with_websocket(request, file_content)

async def handle_duplicate_decision_endpoint(
    job_id: str,
    user_id: str,
    file_hash: str,
    decision: str,
    duplicate_api: DuplicateDetectionAPIIntegration
) -> Dict[str, Any]:
    """
    FastAPI endpoint for handling duplicate decisions.
    
    Args:
        job_id: Job identifier
        user_id: User identifier
        file_hash: File hash
        decision: User's decision
        duplicate_api: Duplicate detection API integration instance
        
    Returns:
        Dictionary with decision result
    """
    return await duplicate_api.handle_duplicate_decision(job_id, user_id, file_hash, decision)

async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
        job_id: Job identifier
    """
    await websocket_manager.connect(websocket, job_id)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(job_id)
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        websocket_manager.disconnect(job_id)
