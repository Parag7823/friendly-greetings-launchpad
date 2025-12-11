"""Multi-faceted field type detection with format validation, PII detection, and AI fallback."""

import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from functools import lru_cache

import yaml
import polars as pl
import structlog
import validators
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from jinja2 import Template
import instructor
from groq import AsyncGroq
import os

from core_infrastructure.centralized_cache import safe_get_cache
from core_infrastructure.rate_limiter import GlobalRateLimiter

logger = structlog.get_logger(__name__)


# Pydantic models for type-safe configuration

class FieldPattern(BaseModel):
    """Field pattern with validation"""
    patterns: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    
    @validator('patterns')
    def patterns_not_empty(cls, v):
        if not v:
            raise ValueError("patterns cannot be empty")
        return v


class FormatPattern(BaseModel):
    """Format pattern with validation"""
    regex: str
    confidence: float = Field(ge=0.0, le=1.0)
    format: str
    description: str = ""


class FieldDetectorConfig(BaseSettings):
    """Configuration settings"""
    field_patterns_yaml: str = "config/field_patterns.yaml"
    format_patterns_yaml: str = "config/format_patterns.yaml"
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_pii_detection: bool = True
    enable_ai_fallback: bool = True
    confidence_threshold: float = 0.5
    
    class Config:
        env_prefix = "FIELD_DETECTOR_"


class FieldType(BaseModel):
    """Structured AI output from instructor"""
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: str
    reasoning: str = ""


class UniversalFieldDetector:
    """Universal Field Detector with YAML config, validators, presidio, and AI fallback."""
    
    _class_analyzer = None
    _analyzer_preloaded = False
    
    @classmethod
    def _preload_analyzer_sync(cls):
        """Initialize presidio analyzer at module-load time"""
        if cls._analyzer_preloaded:
            return cls._class_analyzer
        
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            analyzer = AnalyzerEngine()
            invoice_pattern = Pattern(name="invoice_pattern",
                                      regex=r"\b(INV|INVOICE)[-\s]?\d{4,10}\b",
                                      score=0.85)
            invoice_recognizer = PatternRecognizer(supported_entity="INVOICE_NUMBER", patterns=[invoice_pattern])
            analyzer.registry.add_recognizer(invoice_recognizer)
            po_pattern = Pattern(name="po_pattern",
                                regex=r"\b(PO|P\.O\.)[-\s]?\d{4,10}\b",
                                score=0.85)
            po_recognizer = PatternRecognizer(supported_entity="PO_NUMBER", patterns=[po_pattern])
            analyzer.registry.add_recognizer(po_recognizer)
            
            cls._class_analyzer = analyzer
            cls._analyzer_preloaded = True
            logger.info("Presidio analyzer built at module-load time")
            return analyzer
        except Exception as e:
            logger.warning(f"Presidio analyzer build failed, will use fallback: {e}")
            cls._analyzer_preloaded = True
            return None
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.config = FieldDetectorConfig()
        from core_infrastructure.utils.helpers import initialize_centralized_cache
        self.cache = initialize_centralized_cache(None)
        self.field_patterns = self._load_field_patterns()
        self.format_patterns = self._load_format_patterns()
        self.analyzer = UniversalFieldDetector._class_analyzer
        if os.getenv('GROQ_API_KEY'):
            self.groq_client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
            self.groq_client = instructor.patch(self.groq_client)
        else:
            self.groq_client = None
        
        self.rate_limiter = GlobalRateLimiter()
        
        logger.info("UniversalFieldDetector initialized",
                   field_patterns=sum(len(v) for v in self.field_patterns.values()),
                   format_patterns=len(self.format_patterns),
                   analyzer_ready=self.analyzer is not None)
    
    def _load_field_patterns(self) -> Dict[str, Dict[str, FieldPattern]]:
        """Load field patterns from YAML"""
        try:
            yaml_path = Path(self.config.field_patterns_yaml)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                return {
                    category: {k: FieldPattern(**v) for k, v in types.items()}
                    for category, types in data.items()
                }
            logger.warning("config/field_patterns.yaml not found, using empty patterns")
        except Exception as e:
            logger.error("Failed to load field patterns", error=str(e))
        return {}
    
    def _load_format_patterns(self) -> Dict[str, FormatPattern]:
        """Load format patterns from YAML"""
        try:
            yaml_path = Path(self.config.format_patterns_yaml)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                return {k: FormatPattern(**v) for k, v in data.items()}
            logger.warning("config/format_patterns.yaml not found, using empty patterns")
        except Exception as e:
            logger.error("Failed to load format patterns", error=str(e))
        return {}
    
    def _add_custom_recognizers(self):
        """Add custom financial field recognizers to presidio."""
        invoice_pattern = Pattern(name="invoice_pattern",
                                  regex=r"\b(INV|INVOICE)[-\s]?\d{4,10}\b",
                                  score=0.85)
        invoice_recognizer = PatternRecognizer(supported_entity="INVOICE_NUMBER",
                                              patterns=[invoice_pattern])
        self.analyzer.registry.add_recognizer(invoice_recognizer)
        po_pattern = Pattern(name="po_pattern",
                            regex=r"\b(PO|P\.O\.)[-\s]?\d{4,10}\b",
                            score=0.85)
        po_recognizer = PatternRecognizer(supported_entity="PO_NUMBER",
                                         patterns=[po_pattern])
        self.analyzer.registry.add_recognizer(po_recognizer)
    
    async def detect_field_types_universal(
        self, 
        data: Dict[str, Any], 
        filename: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Parallel + cached field detection."""
        try:
            if not data:
                return {
                    'field_types': {},
                    'confidence': 0.0,
                    'method': 'no_data',
                    'detected_fields': []
                }
            
            tasks = [
                self._cached_analyze_field(name, value)
                for name, value in data.items() if value is not None
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Build field_types and detected_fields
            field_types = {}
            detected_fields = []
            total_confidence = 0.0
            
            for (field_name, field_value), result in zip(
                [(n, v) for n, v in data.items() if v is not None], results
            ):
                if isinstance(result, Exception):
                    logger.error("Field analysis failed", field=field_name, error=str(result))
                    continue
                
                field_types[field_name] = result
                detected_fields.append({
                    'name': field_name,
                    'type': result['type'],
                    'confidence': result['confidence'],
                    'format': result.get('format'),
                    'category': result.get('category')
                })
                total_confidence += result['confidence']
            
            avg_confidence = total_confidence / len(field_types) if field_types else 0.0
            
            return {
                'field_types': field_types,
                'confidence': avg_confidence,
                'method': 'parallel_cached_analysis',
                'detected_fields': detected_fields,
                'filename': filename,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error("Field detection failed", error=str(e))
            return {
                'field_types': {},
                'confidence': 0.0,
                'method': 'error',
                'error': str(e),
                'detected_fields': []
            }
    
    @cached(ttl=3600, serializer=JsonSerializer())
    async def _cached_analyze_field(self, field_name: str, field_value: Any) -> Dict[str, Any]:
        """Cached field analysis with aiocache."""
        field_name_lower = field_name.lower()
        field_value_str = str(field_value).strip()
        
        analysis = {
            'type': 'unknown',
            'confidence': 0.0,
            'format': None,
            'category': None,
            'patterns_matched': []
        }
        
        format_match = self._detect_format_with_validators(field_value_str)
        if format_match:
            analysis['format'] = format_match['format']
            analysis['confidence'] += format_match['confidence']
        
        semantic_match = self._check_semantic_patterns(field_name_lower)
        if semantic_match:
            analysis['type'] = semantic_match['type']
            analysis['category'] = semantic_match['category']
            analysis['confidence'] += semantic_match['confidence']
            analysis['patterns_matched'] = semantic_match['patterns']
        
        if analysis['confidence'] < self.config.confidence_threshold and self.analyzer:
            pii_match = await self._detect_pii_with_presidio(field_value_str)
            if pii_match:
                analysis.update(pii_match)
        
        if analysis['confidence'] < self.config.confidence_threshold and self.config.enable_ai_fallback and self.groq_client:
            ai_match = await self._ai_fallback_with_instructor(field_name, field_value_str)
            if ai_match:
                analysis.update(ai_match)
        
        # Normalize confidence
        analysis['confidence'] = min(1.0, analysis['confidence'])
        
        return analysis
    
    @lru_cache(maxsize=1000)
    def _detect_format_with_validators(self, value: str) -> Optional[Dict[str, Any]]:
        """Detect format using validators library."""
        if validators.email(value):
            return {'format': 'email', 'confidence': 0.95}
        if validators.url(value):
            return {'format': 'url', 'confidence': 0.92}
        if validators.uuid(value):
            return {'format': 'uuid', 'confidence': 0.98}
        for format_name, pattern in self.format_patterns.items():
            import re
            if re.match(pattern.regex, value, re.IGNORECASE):
                return {'format': format_name, 'confidence': pattern.confidence}
        
        return None
    
    def _check_semantic_patterns(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Check semantic patterns from YAML"""
        import re
        for category, type_patterns in self.field_patterns.items():
            for field_type, pattern_obj in type_patterns.items():
                for pattern in pattern_obj.patterns:
                    if re.search(pattern, field_name, re.IGNORECASE):
                        return {
                            'type': field_type,
                            'category': category,
                            'confidence': pattern_obj.confidence,
                            'patterns': [pattern]
                        }
        return None
    
    async def _detect_pii_with_presidio(self, value: str) -> Optional[Dict[str, Any]]:
        """Detect PII using presidio-analyzer."""
        try:
            results = self.analyzer.analyze(text=value, language='en')
            if results:
                entity = results[0]
                return {
                    'type': entity.entity_type.lower(),
                    'confidence': entity.score,
                    'category': 'pii'
                }
        except Exception as e:
            logger.warning("presidio detection failed", error=str(e))
        return None
    
    async def _ai_fallback_with_instructor(self, field_name: str, sample_value: str) -> Optional[Dict[str, Any]]:
        """AI fallback using instructor + Jinja2."""
        try:
            # Apply rate limiting before AI call
            can_sync, msg = await self.rate_limiter.check_global_rate_limit("groq_field_detection", "system")
            if not can_sync:
                logger.warning(f"Rate limit exceeded for AI field detection: {msg}")
                return None
            
            prompt_template = Template("""
You are a financial data expert. Analyze this field:
Field Name: {{ field_name }}
Sample Value: {{ sample_value }}

Determine the field type, category, and confidence.
""")
            
            prompt = prompt_template.render(field_name=field_name, sample_value=sample_value[:100])
            result: FieldType = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_model=FieldType,
                max_tokens=200,
                temperature=0.1
            )
            
            return {
                'type': result.type,
                'confidence': result.confidence,
                'category': result.category,
                'method': 'ai_instructor'
            }
        except Exception as e:
            logger.warning("AI fallback failed", error=str(e))
        return None
    
    async def get_field_suggestions(self, detected_fields: List[Dict]) -> Dict[str, Any]:
        """Get field suggestions using polars for fast filtering."""
        if not detected_fields:
            return {
                'suggestions': [],
                'total_fields': 0,
                'recognized_fields': 0,
                'confidence_score': 0.0
            }
        
        df = pl.DataFrame(detected_fields)
        
        low_conf = df.filter(pl.col('confidence') < 0.5)
        unknown = df.filter(pl.col('type') == 'unknown')
        
        suggestions = []
        if len(low_conf) > 0:
            suggestions.append({
                'type': 'low_confidence',
                'message': f"Review {len(low_conf)} low-confidence fields",
                'fields': low_conf['name'].to_list()
            })
        
        if len(unknown) > 0:
            suggestions.append({
                'type': 'unknown_fields',
                'message': f"Define patterns for {len(unknown)} unknown types",
                'fields': unknown['name'].to_list()
            })
        
        return {
            'suggestions': suggestions,
            'total_fields': len(df),
            'recognized_fields': len(df.filter(pl.col('type') != 'unknown')),
            'confidence_score': float(df['confidence'].mean()),
            'low_confidence_count': len(low_conf),
            'unknown_count': len(unknown)
        }


# Initialize presidio analyzer at module-load time
try:
    UniversalFieldDetector._preload_analyzer_sync()
except Exception as e:
    logger.warning(f"Module-level analyzer preload failed: {e}")


# ============================================================================
# FIELD MAPPING LEARNER (Merged from field_mapping_learner.py)
# ============================================================================
# Production-grade field mapping learning system with:
# - Async batch learning from successful extractions
# - Confidence scoring based on extraction success
# - Platform and document type awareness
# - Graceful degradation on DB errors
# - Exponential backoff retry strategy

from dataclasses import dataclass
from supabase import Client
from collections import defaultdict

# LIBRARY FIX: Use tenacity for exponential backoff retry (replaces custom asyncio.sleep loop)
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity not installed, field mapping retries will be disabled")


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
        self._running = False  # Track if learner is running
        
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
            from core_infrastructure.fastapi_backend_v2 import get_arq_pool
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
    
    def _write_mapping_with_retry_wrapper(self, mapping_data: Dict) -> bool:
        """Wrapper for retry decorator when tenacity is available."""
        if not TENACITY_AVAILABLE:
            return asyncio.run(self._write_mapping_direct(mapping_data))
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        async def _retry_logic():
            return await self._write_mapping_direct(mapping_data)
        
        return asyncio.run(_retry_logic())
    
    async def _write_mapping_with_retry(self, mapping_data: Dict) -> bool:
        """Write mapping to database with exponential backoff retry."""
        return await self._write_mapping_direct(mapping_data)
    
    async def _write_mapping_direct(self, mapping_data: Dict) -> bool:
        """Direct database write without retry wrapper."""
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
    
    async def start(self):
        """Start the learner."""
        self._running = True


# Global singleton instance
_global_field_mapping_learner: Optional[FieldMappingLearner] = None


def get_field_mapping_learner(supabase: Optional[Client] = None) -> FieldMappingLearner:
    """Get or create the global field mapping learner instance."""
    global _global_field_mapping_learner
    
    if _global_field_mapping_learner is None:
        _global_field_mapping_learner = FieldMappingLearner(supabase=supabase)
    
    if supabase is not None and _global_field_mapping_learner.supabase is None:
        _global_field_mapping_learner.supabase = supabase
        
    return _global_field_mapping_learner


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

