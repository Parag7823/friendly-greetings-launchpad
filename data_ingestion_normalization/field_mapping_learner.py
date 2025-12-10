"""
UNIVERSAL FIELD MAPPING LEARNER
================================
Production-grade field mapping learning system with:
- Async batch learning from successful extractions
- Confidence scoring based on extraction success
- Platform and document type awareness
- Graceful degradation on DB errors
- Exponential backoff retry strategy

Author: AI Assistant
Date: 2025
"""

import asyncio
import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from supabase import Client
from collections import defaultdict

# LIBRARY FIX: Use tenacity for exponential backoff retry (replaces custom asyncio.sleep loop)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = structlog.get_logger(__name__)


@dataclass
class FieldMappingRecord:
    """Structured record for a field mapping observation"""
    user_id: str
    source_column: str
    target_field: str
    platform: Optional[str]
    document_type: Optional[str]
    filename_pattern: Optional[str]  # FIX #10: Add filename pattern support
    confidence: float
    extraction_success: bool
    metadata: Dict[str, Any]


class FieldMappingLearner:
    """
    UNIVERSAL FIELD MAPPING LEARNER
    
    Learns field mappings from successful extractions and stores them
    in the database for future use. Uses async batching and retry logic.
    """
    
    def __init__(
        self,
        supabase: Optional[Client] = None,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_retries: int = 3
    ):
        self.supabase = supabase
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self._total_learned = 0
        self._total_failed = 0
        
    async def learn_mapping(
        self,
        user_id: str,
        source_column: str,
        target_field: str,
        platform: Optional[str] = None,
        document_type: Optional[str] = None,
        filename_pattern: Optional[str] = None,  # FIX #10: Add filename_pattern parameter
        confidence: float = 0.8,
        extraction_success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Learn a field mapping from a successful extraction.
        
        Args:
            user_id: User ID
            source_column: Source column name from the file
            target_field: Target field name (e.g., 'amount', 'vendor', 'date')
            platform: Optional platform name
            document_type: Optional document type
            filename_pattern: Optional filename pattern (e.g., 'invoice_*.csv')
            confidence: Confidence score (0-1)
            extraction_success: Whether the extraction was successful
            metadata: Additional metadata
        """
        if not user_id or not source_column or not target_field:
            return
        
        try:
            from fastapi_backend_v2 import get_arq_pool
            pool = await get_arq_pool()
            
            mapping_data = {
                'user_id': user_id,
                'source_column': source_column,
                'target_field': target_field,
                'platform': platform,
                'document_type': document_type,
                'filename_pattern': filename_pattern,
                'confidence': confidence,
                'extraction_success': extraction_success,
                'metadata': metadata or {},
                'observed_at': datetime.utcnow().isoformat()
            }
            
            await pool.enqueue_job('learn_field_mapping_batch', mappings=[mapping_data])
            self._total_learned += 1
            logger.debug(f"✅ Enqueued field mapping to ARQ: {source_column} → {target_field}")
            
        except Exception as e:
            logger.error(f"Failed to enqueue field mapping to ARQ: {e}")
            self._total_failed += 1
            try:
                if self.supabase:
                    await self._write_mapping_with_retry(mapping_data)
            except Exception as fallback_error:
                logger.error(f"Fallback DB write also failed: {fallback_error}")
    
    def _aggregate_mappings_UNUSED(self, batch: List[FieldMappingRecord]) -> Dict[tuple, Dict]:
        """
        Aggregate multiple observations of the same mapping.
        
        Returns a dict keyed by (user_id, source_column, target_field, platform, document_type)
        with aggregated confidence and metadata.
        """
        aggregated = defaultdict(lambda: {
            'observations': [],
            'total_confidence': 0.0,
            'success_count': 0,
            'total_count': 0
        })
        
        for record in batch:
            key = (
                record.user_id,
                record.source_column.lower(),
                record.target_field.lower(),
                record.platform,
                record.document_type,
                record.filename_pattern
            )
            
            agg = aggregated[key]
            agg['observations'].append(record)
            agg['total_confidence'] += record.confidence
            agg['total_count'] += 1
            if record.extraction_success:
                agg['success_count'] += 1
                
        result = {}
        for key, agg in aggregated.items():
            user_id, source_column, target_field, platform, document_type, filename_pattern = key
            avg_confidence = agg['total_confidence'] / agg['total_count']
            success_rate = agg['success_count'] / agg['total_count']
            final_confidence = avg_confidence * success_rate
            merged_metadata = {}
            for record in agg['observations']:
                merged_metadata.update(record.metadata)
            merged_metadata['observation_count'] = agg['total_count']
            merged_metadata['success_rate'] = success_rate
            
            result[key] = {
                'user_id': user_id,
                'source_column': source_column,
                'target_field': target_field,
                'platform': platform,
                'document_type': document_type,
                'filename_pattern': filename_pattern,  # FIX #10: Include filename_pattern
                'confidence': final_confidence,
                'metadata': merged_metadata
            }
            
        return result
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _write_mapping_with_retry(self, mapping_data: Dict) -> bool:
        """Write mapping to database with exponential backoff retry."""
        if not self.supabase:
            logger.warning("No Supabase client available for field mapping learning")
            return False
        
        try:
            result = self.supabase.rpc(
                'upsert_field_mapping',
                {
                    'p_user_id': mapping_data['user_id'],
                    'p_source_column': mapping_data['source_column'],
                    'p_target_field': mapping_data['target_field'],
                    'p_platform': mapping_data['platform'],
                    'p_document_type': mapping_data['document_type'],
                    'p_confidence': mapping_data['confidence'],
                    'p_mapping_source': 'ai_learned',
                    'p_metadata': mapping_data['metadata']
                }
            ).execute()
            
            if result.data:
                logger.debug(f"Learned field mapping: {mapping_data['source_column']} -> {mapping_data['target_field']}")
                return True
            else:
                logger.warning("Failed to upsert field mapping")
                raise Exception("Upsert returned no data")
                
        except Exception as e:
            logger.warning(f"Error writing field mapping: {e}")
            raise
        
    async def get_mappings(
        self,
        user_id: str,
        platform: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, str]:
        """Retrieve learned field mappings for a user."""
        if not self.supabase or not user_id:
            return {}
        
        try:
            result = self.supabase.rpc(
                'get_user_field_mappings',
                {
                    'p_user_id': user_id,
                    'p_platform': platform
                }
            ).execute()
            
            if not result.data:
                return {}
            
            mappings = {}
            for row in result.data:
                target_field = row.get('target_field')
                source_column = row.get('source_column')
                confidence = row.get('confidence', 0.0)
                
                if confidence < min_confidence:
                    continue
                
                if target_field not in mappings or confidence > mappings[target_field]['confidence']:
                    mappings[target_field] = {
                        'source_column': source_column,
                        'confidence': confidence
                    }
            
            return {
                target: data['source_column']
                for target, data in mappings.items()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving field mappings: {e}")
            return {}


# Global singleton instance
_global_learner: Optional[FieldMappingLearner] = None


def get_field_mapping_learner(supabase: Optional[Client] = None) -> FieldMappingLearner:
    """Get or create the global field mapping learner instance."""
    global _global_learner
    
    if _global_learner is None:
        _global_learner = FieldMappingLearner(supabase=supabase)
    
    if supabase is not None and _global_learner.supabase is None:
        _global_learner.supabase = supabase
        
    return _global_learner


async def learn_field_mapping(
    user_id: str,
    source_column: str,
    target_field: str,
    platform: Optional[str] = None,
    document_type: Optional[str] = None,
    confidence: float = 0.8,
    extraction_success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    supabase: Optional[Client] = None
):
    """Convenience function to learn a field mapping."""
    learner = get_field_mapping_learner(supabase)
    
    # Start learner if not running
    if not learner._running:
        await learner.start()
        
    await learner.learn_mapping(
        user_id=user_id,
        source_column=source_column,
        target_field=target_field,
        platform=platform,
        document_type=document_type,
        confidence=confidence,
        extraction_success=extraction_success,
        metadata=metadata
    )


async def get_learned_mappings(
    user_id: str,
    platform: Optional[str] = None,
    min_confidence: float = 0.5,
    supabase: Optional[Client] = None
) -> Dict[str, str]:
    """Convenience function to retrieve learned field mappings."""
    learner = get_field_mapping_learner(supabase)
    return await learner.get_mappings(user_id, platform, min_confidence)
