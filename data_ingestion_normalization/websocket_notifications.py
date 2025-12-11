"""Centralized WebSocket notification system for ingestion pipeline.

This module extracts WebSocket notification logic from ExcelProcessor and other
components to provide a clean, testable interface for real-time progress updates.
"""

import asyncio
from typing import Dict, Any, Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class NotificationType(Enum):
    """Types of WebSocket notifications in the ingestion pipeline."""
    FILE_UPLOAD = "file_upload"
    SHEET_PROCESSING = "sheet_processing"
    ROW_PROCESSING = "row_processing"
    ANOMALY_DETECTED = "anomaly_detected"
    ENTITY_RESOLUTION = "entity_resolution"
    PLATFORM_DETECTION = "platform_detection"
    DOCUMENT_CLASSIFICATION = "document_classification"
    FIELD_MAPPING = "field_mapping"
    DUPLICATE_CHECK = "duplicate_check"
    ENRICHMENT = "enrichment"
    COMPLETION = "completion"
    ERROR = "error"


class IngestionNotificationService:
    """
    Centralized service for sending ingestion pipeline notifications via WebSocket.
    
    This service handles all real-time notifications during file processing, providing
    a unified interface for progress updates, errors, and completion status.
    """
    
    def __init__(self):
        """Initialize notification service with lazy-loaded WebSocket manager."""
        self.manager = self._get_websocket_manager()
    
    def _get_websocket_manager(self):
        """Lazy load WebSocket manager to avoid circular imports."""
        try:
            from core_infrastructure.fastapi_backend_v2 import manager
            logger.info("WebSocket manager loaded successfully")
            return manager
        except ImportError as e:
            logger.warning(f"WebSocket manager not available: {e}")
            return None
    
    async def send_notification(
        self,
        job_id: str,
        notification_type: NotificationType,
        message: str,
        progress: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a notification via WebSocket.
        
        Args:
            job_id: Unique job identifier
            notification_type: Type of notification
            message: Human-readable message
            progress: Optional progress percentage (0-100)
            data: Optional additional data
        """
        if not self.manager:
            logger.debug(f"WebSocket not available, skipping notification: {message}")
            return
        
        notification = {
            "type": notification_type.value,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if progress is not None:
            notification["progress"] = progress
        
        if data:
            notification.update(data)
        
        try:
            await self.manager.send_update(job_id, notification)
            logger.debug(f"Sent WebSocket notification: {notification_type.value}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")
    
    # Convenience methods for common notifications
    
    async def send_file_upload(self, job_id: str, filename: str, file_size: int):
        """Notify file upload started."""
        await self.send_notification(
            job_id,
            NotificationType.FILE_UPLOAD,
            f"Uploading file: {filename}",
            progress=0.0 ,
            data={"filename": filename, "file_size": file_size}
        )
    
    async def send_sheet_processing(self, job_id: str, sheet_name: str, progress: float):
        """Notify sheet processing progress."""
        await self.send_notification(
            job_id,
            NotificationType.SHEET_PROCESSING,
            f"Processing sheet: {sheet_name}",
            progress=progress,
            data={"sheet_name": sheet_name}
        )
    
    async def send_row_processing(self, job_id: str, rows_processed: int, total_rows: int):
        """Notify row processing progress."""
        progress = (rows_processed / total_rows * 100) if total_rows > 0 else 0
        await self.send_notification(
            job_id,
            NotificationType.ROW_PROCESSING,
            f"Processed {rows_processed}/{total_rows} rows",
            progress=progress,
            data={"rows_processed": rows_processed, "total_rows": total_rows}
        )
    
    async def send_duplicate_check(self, job_id: str, is_duplicate: bool, duplicate_type: str = None):
        """Notify duplicate check result."""
        message = f"Duplicate detected ({duplicate_type})" if is_duplicate else "No duplicates found"
        await self.send_notification(
            job_id,
            NotificationType.DUPLICATE_CHECK,
            message,
            data={"is_duplicate": is_duplicate, "duplicate_type": duplicate_type}
        )
    
    async def send_completion(self, job_id: str, total_rows: int, processing_time: float):
        """Notify processing completion."""
        await self.send_notification(
            job_id,
            NotificationType.COMPLETION,
            "Processing completed successfully",
            progress=100.0,
            data={"total_rows": total_rows, "processing_time_ms": int(processing_time * 1000)}
        )
    
    async def send_error(self, job_id: str, error_message: str):
        """Notify error occurred."""
        await self.send_notification(
            job_id,
            NotificationType.ERROR,
            f"Error: {error_message}",
            data={"error": error_message}
        )
    
    async def send_platform_detection(self, job_id: str, platform: str, confidence: float):
        """Notify platform detection result."""
        await self.send_notification(
            job_id,
            NotificationType.PLATFORM_DETECTION,
            f"Platform detected: {platform}",
            data={"platform": platform, "confidence": confidence}
        )
    
    async def send_document_classification(self, job_id: str, document_type: str, confidence: float):
        """Notify document classification result."""
        await self.send_notification(
            job_id,
            NotificationType.DOCUMENT_CLASSIFICATION,
            f"Document type: {document_type}",
            data={"document_type": document_type, "confidence": confidence}
        )
    
    async def send_anomaly_detected(self, job_id: str, anomaly_type: str, details: Dict[str, Any]):
        """Notify anomaly detected."""
        await self.send_notification(
            job_id,
            NotificationType.ANOMALY_DETECTED,
            f"Anomaly detected: {anomaly_type}",
            data={"anomaly_type": anomaly_type, **details}
        )


# Singleton instance
_notification_service = None


def get_notification_service() -> IngestionNotificationService:
    """
    Get singleton notification service instance.
    
    Returns:
        Singleton IngestionNotificationService instance
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = IngestionNotificationService()
    return _notification_service
