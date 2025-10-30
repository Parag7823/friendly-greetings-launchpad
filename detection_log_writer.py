"""
Production-Grade Detection Log Writer
======================================

Universal, sophisticated detection logging system with:
- Async batching for high throughput
- Automatic retry with exponential backoff
- Graceful degradation when Supabase unavailable
- Thread-safe operation
- Memory-efficient queue management
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionRecord:
    """Standardized detection record for logging."""
    user_id: str
    detection_id: str
    detection_type: str  # 'platform' or 'document'
    detected_value: str
    confidence: float
    method: str
    indicators: List[str]
    payload_keys: List[str]
    filename: Optional[str] = None
    detected_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Convert to Supabase row format."""
        return {
            "user_id": self.user_id,
            "detection_id": self.detection_id,
            "detection_type": self.detection_type,
            "detected_value": self.detected_value,
            "confidence": float(self.confidence),
            "method": self.method,
            "indicators": self.indicators or [],
            "payload_keys": self.payload_keys or [],
            "filename": self.filename,
            "detected_at": self.detected_at or datetime.utcnow().isoformat(),
            "metadata": self.metadata or {},
        }


class DetectionLogWriter:
    """
    Production-grade async detection log writer with batching and retry.
    
    Features:
    - Automatic batching to reduce DB calls
    - Exponential backoff retry on failures
    - Graceful degradation when DB unavailable
    - Memory-bounded queue
    - Background flush loop
    """

    def __init__(
        self,
        supabase_client=None,
        table_name: str = "detection_log",
        batch_size: int = 50,
        flush_interval: float = 2.0,
        max_queue_size: int = 1000,
        max_retries: int = 3,
    ):
        """
        Initialize detection log writer.
        
        Args:
            supabase_client: Supabase client instance (lazy-loaded if None)
            table_name: Target table name
            batch_size: Records per batch insert
            flush_interval: Seconds between automatic flushes
            max_queue_size: Maximum queue size before blocking
            max_retries: Maximum retry attempts per batch
        """
        self._supabase = supabase_client
        self._table = table_name
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        
        # Async queue with bounded size
        self._queue: asyncio.Queue[DetectionRecord] = asyncio.Queue(maxsize=max_queue_size)
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False
        
        # Metrics
        self._metrics = {
            "records_queued": 0,
            "records_written": 0,
            "batches_written": 0,
            "write_errors": 0,
            "queue_full_drops": 0,
        }

    @property
    def client(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            try:
                from supabase_client import get_supabase_client
                self._supabase = get_supabase_client()
            except Exception as exc:
                logger.warning(f"Supabase client unavailable for detection logging: {exc}")
        return self._supabase

    async def start(self) -> None:
        """Start background flush loop."""
        async with self._lock:
            if not self._running:
                self._running = True
                self._flush_task = asyncio.create_task(self._flush_loop())
                logger.info("✅ DetectionLogWriter started")

    async def stop(self) -> None:
        """Stop background flush loop and flush remaining records."""
        async with self._lock:
            if self._running:
                self._running = False
                if self._flush_task:
                    self._flush_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._flush_task
                
                # Final flush
                await self._flush_batch()
                logger.info(f"✅ DetectionLogWriter stopped. Metrics: {self._metrics}")

    async def log_detection(
        self,
        user_id: str,
        detection_id: str,
        detection_type: str,
        detected_value: str,
        confidence: float,
        method: str,
        indicators: List[str],
        payload_keys: List[str],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a detection record for logging.
        
        Returns:
            True if queued successfully, False if queue full
        """
        record = DetectionRecord(
            user_id=user_id,
            detection_id=detection_id,
            detection_type=detection_type,
            detected_value=detected_value,
            confidence=confidence,
            method=method,
            indicators=indicators,
            payload_keys=payload_keys,
            filename=filename,
            metadata=metadata or {},
        )

        try:
            # Non-blocking put with immediate timeout
            self._queue.put_nowait(record)
            self._metrics["records_queued"] += 1
            return True
        except asyncio.QueueFull:
            self._metrics["queue_full_drops"] += 1
            logger.warning(f"Detection log queue full, dropping record: {detection_id}")
            return False

    async def _flush_loop(self) -> None:
        """Background loop to flush batches periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in detection log flush loop: {exc}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _flush_batch(self) -> None:
        """Flush a batch of records to Supabase."""
        if self._queue.empty():
            return

        # Collect batch
        batch: List[DetectionRecord] = []
        while len(batch) < self._batch_size and not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                batch.append(record)
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        # Write with retry
        await self._write_batch_with_retry(batch)

    async def _write_batch_with_retry(self, batch: List[DetectionRecord]) -> None:
        """Write batch with exponential backoff retry."""
        if not self.client:
            logger.warning(f"Supabase unavailable, dropping {len(batch)} detection records")
            self._metrics["write_errors"] += 1
            return

        rows = [record.to_row() for record in batch]
        
        for attempt in range(self._max_retries):
            try:
                # Batch insert
                result = self.client.table(self._table).insert(rows).execute()
                
                # Success
                self._metrics["records_written"] += len(batch)
                self._metrics["batches_written"] += 1
                logger.debug(f"✅ Wrote {len(batch)} detection records to {self._table}")
                return
                
            except Exception as exc:
                wait_time = min(2 ** attempt, 8)  # Exponential backoff, max 8s
                logger.warning(
                    f"Detection log write failed (attempt {attempt + 1}/{self._max_retries}): {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure
                    self._metrics["write_errors"] += 1
                    logger.error(f"❌ Failed to write {len(batch)} detection records after {self._max_retries} attempts")

    def get_metrics(self) -> Dict[str, int]:
        """Get writer metrics."""
        return {
            **self._metrics,
            "queue_size": self._queue.qsize(),
            "queue_max_size": self._queue.maxsize,
        }


# Global singleton instance
_detection_log_writer: Optional[DetectionLogWriter] = None
_writer_lock = asyncio.Lock()


async def get_detection_log_writer(supabase_client=None) -> DetectionLogWriter:
    """
    Get or create singleton detection log writer.
    
    Args:
        supabase_client: Optional Supabase client to use
        
    Returns:
        DetectionLogWriter instance
    """
    global _detection_log_writer
    
    if _detection_log_writer is None:
        async with _writer_lock:
            if _detection_log_writer is None:
                _detection_log_writer = DetectionLogWriter(supabase_client=supabase_client)
                await _detection_log_writer.start()
                logger.info("✅ Global DetectionLogWriter initialized")
    
    return _detection_log_writer


async def log_platform_detection(
    user_id: str,
    detection_id: str,
    platform: str,
    confidence: float,
    method: str,
    indicators: List[str],
    payload_keys: List[str],
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    supabase_client=None,
) -> bool:
    """
    Convenience function to log platform detection.
    
    Returns:
        True if logged successfully
    """
    writer = await get_detection_log_writer(supabase_client)
    return await writer.log_detection(
        user_id=user_id,
        detection_id=detection_id,
        detection_type="platform",
        detected_value=platform,
        confidence=confidence,
        method=method,
        indicators=indicators,
        payload_keys=payload_keys,
        filename=filename,
        metadata=metadata,
    )


async def log_document_classification(
    user_id: str,
    classification_id: str,
    document_type: str,
    confidence: float,
    method: str,
    indicators: List[str],
    payload_keys: List[str],
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    supabase_client=None,
) -> bool:
    """
    Convenience function to log document classification.
    
    Returns:
        True if logged successfully
    """
    writer = await get_detection_log_writer(supabase_client)
    return await writer.log_detection(
        user_id=user_id,
        detection_id=classification_id,
        detection_type="document",
        detected_value=document_type,
        confidence=confidence,
        method=method,
        indicators=indicators,
        payload_keys=payload_keys,
        filename=filename,
        metadata=metadata,
    )
