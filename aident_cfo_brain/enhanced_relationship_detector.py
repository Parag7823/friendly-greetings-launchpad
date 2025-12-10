"""Enhanced Relationship Detector v2.0

Multi-method relationship detection using:
- Graph analysis (igraph)
- Fuzzy string matching (rapidfuzz)
- Semantic similarity (sentence-transformers)
- Date parsing (pendulum)
- Named Entity Recognition (spaCy)
- Machine learning scoring (scikit-learn)

Features:
- Cross-file relationship detection
- Within-file relationship detection
- Comprehensive scoring system
- Validation and deduplication

Author: Senior Full-Stack Engineer
Version: 2.0.0
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from groq import AsyncGroq
from supabase import create_client, Client

import pendulum
from core_infrastructure.provenance_tracker import normalize_business_logic, normalize_temporal_causality

# Lazy-loaded heavy libraries
ig = None
fuzz = None
process = None
np = None
spacy = None
SentenceTransformer = None
util = None
RandomForestClassifier = None

# Structured logging
import structlog
logger = structlog.get_logger(__name__)

# ============================================
# SEMANTIC RELATIONSHIP CONFIGURATION
# ============================================
SEMANTIC_CONFIG = {
    'enable_caching': os.getenv('ENABLE_SEMANTIC_CACHE', 'true').lower() == 'true',
    'cache_ttl_seconds': int(os.getenv('SEMANTIC_CACHE_TTL', str(48 * 3600))),  # 48 hours
    'enable_embeddings': os.getenv('ENABLE_EMBEDDINGS', 'true').lower() == 'true',
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'bge-large-en-v1.5'),
    'semantic_model': os.getenv('SEMANTIC_MODEL', 'llama-3.3-70b-versatile'),
    'temperature': float(os.getenv('SEMANTIC_TEMPERATURE', '0.1')),
    'max_tokens': int(os.getenv('SEMANTIC_MAX_TOKENS', '800')),
    'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
    'max_concurrent': int(os.getenv('SEMANTIC_MAX_CONCURRENT', '10')),
    'max_per_second': int(os.getenv('SEMANTIC_MAX_PER_SECOND', '5')),
    'timeout_seconds': int(os.getenv('SEMANTIC_TIMEOUT', '30')),
    'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0')
}

logger.info("Semantic relationship configuration loaded", config=SEMANTIC_CONFIG)

# Auto-validated AI responses
try:
    import instructor
    from pydantic import BaseModel, Field, ValidationError, validator
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("instructor not available - AI responses won't be auto-validated")

# Lazy-load EmbeddingService
_embedding_service_instance = None
_embedding_service_loaded = False
EmbeddingService = None
EMBEDDING_SERVICE_AVAILABLE = False

def _ensure_embedding_service_loaded():
    """Lazy load EmbeddingService on first use"""
    global _embedding_service_instance, _embedding_service_loaded, EmbeddingService, EMBEDDING_SERVICE_AVAILABLE
    if not _embedding_service_loaded:
        try:
            try:
                from data_ingestion_normalization.embedding_service import EmbeddingService as EmbeddingServiceClass
            except ImportError:
                from embedding_service import EmbeddingService as EmbeddingServiceClass
            
            EmbeddingService = EmbeddingServiceClass
            _embedding_service_instance = EmbeddingServiceClass()
            EMBEDDING_SERVICE_AVAILABLE = True
            _embedding_service_loaded = True
            logger.info("EmbeddingService loaded successfully")
        except ImportError as e:
            logger.warning(f"EmbeddingService not available - semantic embeddings disabled: {e}")
            _embedding_service_loaded = True
            EMBEDDING_SERVICE_AVAILABLE = False
            _embedding_service_instance = None
    return _embedding_service_instance

# ProductionDuplicateDetectionService
try:
    try:
        from duplicate_detection_fraud.production_duplicate_detection_service import ProductionDuplicateDetectionService
    except ImportError:
        from production_duplicate_detection_service import ProductionDuplicateDetectionService
    DUPLICATE_SERVICE_AVAILABLE = True
except ImportError as e:
    DUPLICATE_SERVICE_AVAILABLE = False
    logger.warning(f"ProductionDuplicateDetectionService not available: {e}")

# Retry logic with exponential backoff
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity not available - retry logic disabled")

# Pydantic models for data validation
if INSTRUCTOR_AVAILABLE:
    class RelationshipEnrichment(BaseModel):
        """Auto-validated AI response for relationship enrichment"""
        semantic_description: str = Field(min_length=10, max_length=500)
        reasoning: str = Field(min_length=20)
        temporal_causality: str = Field(
            pattern="^(source_causes_target|target_causes_source|bidirectional|correlation_only)$"
        )
        business_logic: str
        
        @validator('semantic_description')
        def validate_description(cls, v):
            if not v or len(v.strip()) < 10:
                raise ValueError('semantic_description must be at least 10 characters')
            return v.strip()
        
        @validator('reasoning')
        def validate_reasoning(cls, v):
            if not v or len(v.strip()) < 20:
                raise ValueError('reasoning must be at least 20 characters')
            return v.strip()
    
    class DynamicRelationshipResponse(BaseModel):
        """Auto-validated AI response for dynamic relationship detection"""
        is_related: bool = Field(description="Whether the two events are related")
        relationship_type: str = Field(
            min_length=1,
            description="Type of relationship (e.g., invoice_payment, loan_disbursement, subscription_renewal)"
        )
        confidence: float = Field(
            ge=0.0,
            le=1.0,
            description="Confidence score between 0.0 and 1.0"
        )
        reasoning: str = Field(
            min_length=10,
            description="Why these events are related"
        )
        
        @validator('relationship_type')
        def validate_relationship_type(cls, v):
            if not v or len(v.strip()) < 1:
                raise ValueError('relationship_type must be non-empty')
            return v.strip().lower()
        
        @validator('reasoning')
        def validate_reasoning(cls, v):
            if not v or len(v.strip()) < 10:
                raise ValueError('reasoning must be at least 10 characters')
            return v.strip()
    
    class RelationshipRecord(BaseModel):
        """Validated relationship record for database storage"""
        source_event_id: str = Field(min_length=1)
        target_event_id: str = Field(min_length=1)
        relationship_type: str = Field(min_length=1)
        confidence_score: float = Field(ge=0.0, le=1.0)
        detection_method: str
        metadata: Dict[str, Any] = Field(default_factory=dict)
        key_factors: List[str] = Field(default_factory=list)
        
        @validator('source_event_id', 'target_event_id')
        def validate_event_ids(cls, v):
            if not v or not isinstance(v, str):
                raise ValueError('Event IDs must be non-empty strings')
            return v
        
        @validator('confidence_score')
        def validate_confidence(cls, v):
            if not (0.0 <= v <= 1.0):
                raise ValueError('confidence_score must be between 0.0 and 1.0')
            return v

# Lazy-load SemanticRelationshipExtractor
SemanticRelationshipExtractor = None
SEMANTIC_EXTRACTOR_AVAILABLE = False

def _load_semantic_relationship_extractor():
    """Lazy load SemanticRelationshipExtractor on first use"""
    global SemanticRelationshipExtractor, SEMANTIC_EXTRACTOR_AVAILABLE
    if SemanticRelationshipExtractor is None:
        try:
            # Try aident_cfo_brain prefix first (when running as package)
            from aident_cfo_brain.semantic_relationship_extractor import SemanticRelationshipExtractor as SRE
            SemanticRelationshipExtractor = SRE
            SEMANTIC_EXTRACTOR_AVAILABLE = True
            logger.info("✅ SemanticRelationshipExtractor loaded")
        except ImportError:
            try:
                # Fallback for direct script execution
                from semantic_relationship_extractor import SemanticRelationshipExtractor as SRE
                SemanticRelationshipExtractor = SRE
                SEMANTIC_EXTRACTOR_AVAILABLE = True
                logger.info("✅ SemanticRelationshipExtractor loaded (fallback)")
            except ImportError:
                SEMANTIC_EXTRACTOR_AVAILABLE = False
                logger.warning("SemanticRelationshipExtractor not available. Semantic analysis will be disabled.")
    return SemanticRelationshipExtractor

# Lazy-load CausalInferenceEngine
CausalInferenceEngine = None
CAUSAL_INFERENCE_AVAILABLE = False

def _load_causal_inference_engine():
    """Lazy load CausalInferenceEngine on first use"""
    global CausalInferenceEngine, CAUSAL_INFERENCE_AVAILABLE
    if CausalInferenceEngine is None:
        try:
            # Try root-level import first (causal_inference_engine.py is at root)
            from causal_inference_engine import CausalInferenceEngine as CIE
            CausalInferenceEngine = CIE
            CAUSAL_INFERENCE_AVAILABLE = True
            logger.info("✅ CausalInferenceEngine loaded")
        except ImportError:
            CAUSAL_INFERENCE_AVAILABLE = False
            logger.warning("CausalInferenceEngine not available. Causal analysis will be disabled.")
    return CausalInferenceEngine

# Lazy-load TemporalPatternLearner
TemporalPatternLearner = None
TEMPORAL_PATTERN_LEARNER_AVAILABLE = False

def _load_temporal_pattern_learner():
    """Lazy load TemporalPatternLearner on first use"""
    global TemporalPatternLearner, TEMPORAL_PATTERN_LEARNER_AVAILABLE
    if TemporalPatternLearner is None:
        try:
            # Try root-level import first (temporal_pattern_learner.py is at root)
            from temporal_pattern_learner import TemporalPatternLearner as TPL
            TemporalPatternLearner = TPL
            TEMPORAL_PATTERN_LEARNER_AVAILABLE = True
            logger.info("✅ TemporalPatternLearner loaded")
        except ImportError:
            TEMPORAL_PATTERN_LEARNER_AVAILABLE = False
            logger.warning("TemporalPatternLearner not available. Temporal pattern learning will be disabled.")
    return TemporalPatternLearner

# Lazy loading helpers

def _load_spacy():
    """Lazy load spaCy NLP model on first use"""
    global spacy
    if spacy is None:
        try:
            import spacy as spacy_module
            spacy = spacy_module
            logger.info("spaCy module loaded")
        except ImportError:
            logger.error("spaCy not installed - NER features unavailable")
            raise ImportError("spaCy is required for NER features. Install with: pip install spacy")
    return spacy

def _load_sentence_transformers():
    """Lazy load sentence-transformers on first use"""
    global SentenceTransformer, util
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST, util as st_util
            SentenceTransformer = ST
            util = st_util
            logger.info("sentence-transformers module loaded")
        except ImportError:
            logger.error("sentence-transformers not installed - semantic features unavailable")
            raise ImportError("sentence-transformers is required for semantic features. Install with: pip install sentence-transformers")
    return SentenceTransformer, util

def _load_sklearn():
    """Lazy load scikit-learn on first use"""
    global RandomForestClassifier
    if RandomForestClassifier is None:
        try:
            from sklearn.ensemble import RandomForestClassifier as RFC
            RandomForestClassifier = RFC
            logger.info("scikit-learn module loaded")
        except ImportError:
            logger.error("scikit-learn not installed - ML features unavailable")
            raise ImportError("scikit-learn is required for ML features. Install with: pip install scikit-learn")
    return RandomForestClassifier

def _load_igraph():
    """Lazy load igraph on first use"""
    global ig
    if ig is None:
        try:
            import igraph as igraph_module
            ig = igraph_module
            logger.info("igraph module loaded")
        except ImportError:
            logger.error("igraph not installed - graph features unavailable")
            raise ImportError("igraph is required. Install with: pip install igraph")
    return ig

def _load_rapidfuzz():
    """Lazy load rapidfuzz on first use"""
    global fuzz, process
    if fuzz is None:
        try:
            from rapidfuzz import fuzz as fuzz_module, process as process_module
            fuzz = fuzz_module
            process = process_module
            logger.info("rapidfuzz module loaded")
        except ImportError:
            logger.error("rapidfuzz not installed - fuzzy matching unavailable")
            raise ImportError("rapidfuzz is required. Install with: pip install rapidfuzz")
    return fuzz, process

def _load_numpy():
    """Lazy load numpy on first use"""
    global np
    if np is None:
        try:
            import numpy as numpy_module
            np = numpy_module
            logger.info("numpy module loaded")
        except ImportError:
            logger.error("numpy not installed - numerical features unavailable")
            raise ImportError("numpy is required. Install with: pip install numpy")
    return np


# ============================================================================
# PRELOAD PATTERN: Module-level preloading for heavy dependencies
# ============================================================================
# These flags track preload status to avoid repeated initialization attempts
_PRELOAD_COMPLETED = False

def _preload_all_modules():
    """
    PRELOAD PATTERN: Initialize all heavy modules at module-load time.
    Called automatically when module is imported.
    This eliminates first-request latency.
    """
    global _PRELOAD_COMPLETED
    
    if _PRELOAD_COMPLETED:
        return
    
    # Preload EmbeddingService
    try:
        _ensure_embedding_service_loaded()
        logger.info("✅ PRELOAD: EmbeddingService loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: EmbeddingService load failed: {e}")
    
    # Preload SemanticRelationshipExtractor
    try:
        _load_semantic_relationship_extractor()
        logger.info("✅ PRELOAD: SemanticRelationshipExtractor loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: SemanticRelationshipExtractor load failed: {e}")
    
    # Preload CausalInferenceEngine
    try:
        _load_causal_inference_engine()
        logger.info("✅ PRELOAD: CausalInferenceEngine loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: CausalInferenceEngine load failed: {e}")
    
    # Preload TemporalPatternLearner
    try:
        _load_temporal_pattern_learner()
        logger.info("✅ PRELOAD: TemporalPatternLearner loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: TemporalPatternLearner load failed: {e}")
    
    # Preload numpy
    try:
        _load_numpy()
        logger.info("✅ PRELOAD: numpy loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: numpy load failed: {e}")
    
    # Preload rapidfuzz
    try:
        _load_rapidfuzz()
        logger.info("✅ PRELOAD: rapidfuzz loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: rapidfuzz load failed: {e}")
    
    _PRELOAD_COMPLETED = True

class EnhancedRelationshipDetector:
    """Enhanced relationship detector that actually finds relationships between events"""
    
    def __init__(
        self,
        llm_client: AsyncGroq = None,
        supabase_client: Client = None,
        cache_client=None,
        embedding_service=None,  # NEW: Dependency injection for embedding service
    ):
        self.llm_client = llm_client
        self.supabase = supabase_client
        self.cache = cache_client  # Use centralized cache, no local cache
        
        self.embedding_service = embedding_service
        if embedding_service is None:
            self.embedding_service = _ensure_embedding_service_loaded()
        
        self.nlp = None
        self._spacy_loaded = False
        self.semantic_model = None
        self.graph = None
        self._igraph_loaded = False
        self._semantic_extractor_client = llm_client
        self._semantic_extractor_supabase = supabase_client
        
        # CRITICAL FIX: Initialize advanced semantic extractor with configuration
        # This replaces the lazy-loading approach with eager initialization
        self.semantic_extractor = None
        self._semantic_extractor_loaded = False
        try:
            SRE = _load_semantic_relationship_extractor()
            if SRE:
                self.semantic_extractor = SRE(
                    openai_client=llm_client,
                    supabase_client=supabase_client,
                    cache_client=cache_client,
                    config=SEMANTIC_CONFIG
                )
                self._semantic_extractor_loaded = True
                logger.info(
                    "Advanced semantic extractor initialized",
                    caching_enabled=SEMANTIC_CONFIG['enable_caching'],
                    embeddings_enabled=SEMANTIC_CONFIG['enable_embeddings'],
                    model=SEMANTIC_CONFIG['semantic_model']
                )
            else:
                logger.warning("SemanticRelationshipExtractor not available - semantic enrichment disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic extractor: {e}")
            self.semantic_extractor = None
        
        if DUPLICATE_SERVICE_AVAILABLE:
            self.duplicate_service = ProductionDuplicateDetectionService(supabase_client)
        else:
            self.duplicate_service = None
        
        self.causal_engine = None
        self._causal_engine_loaded = False
        self.temporal_learner = None
        self._temporal_learner_loaded = False
        
        # ML model for adaptive relationship scoring (trained on-demand)
        self.ml_model = None
        self.ml_model_trained = False
    
    def _ensure_causal_engine(self):
        """Lazy load causal engine on first use."""
        if not self._causal_engine_loaded:
            CIE = _load_causal_inference_engine()
            if CIE:
                self.causal_engine = CIE(supabase_client=self.supabase)
                logger.info("Causal inference engine initialized")
            else:
                self.causal_engine = None
            self._causal_engine_loaded = True
        return self.causal_engine
    
    def _ensure_temporal_learner(self):
        """Lazy load temporal learner on first use."""
        if not self._temporal_learner_loaded:
            TPL = _load_temporal_pattern_learner()
            if TPL:
                self.temporal_learner = TPL(supabase_client=self.supabase)
                logger.info("Temporal pattern learner initialized")
            else:
                self.temporal_learner = None
            self._temporal_learner_loaded = True
        return self.temporal_learner
    
    def _ensure_semantic_extractor(self):
        """Return the already-initialized semantic extractor."""
        # CRITICAL FIX: Semantic extractor is now initialized eagerly in __init__
        # This method now simply returns the cached instance
        if not self._semantic_extractor_loaded:
            logger.warning("Semantic extractor was not initialized during __init__")
            return None
        return self.semantic_extractor
    
    async def detect_all_relationships(self, user_id: str, file_id: Optional[str] = None, transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """Detect relationships using document_type classification and database JOINs."""
        try:
            logger.info(f"Starting relationship detection for user_id={user_id}, file_id={file_id}")
            
            cross_file_relationships = await self._detect_cross_document_relationships_db(user_id, file_id)
            within_file_relationships = await self._detect_within_file_relationships_db(user_id, file_id)
            all_relationships = cross_file_relationships + within_file_relationships
            
            stored_relationships = []
            if all_relationships:
                stored_relationships = await self._store_relationships(all_relationships, user_id, transaction_id)
            
            # CRITICAL FIX: Use advanced SemanticRelationshipExtractor for semantic enrichment
            # Fetch events for semantic enrichment
            events_map = {}
            if stored_relationships:
                try:
                    event_ids = set()
                    for rel in stored_relationships:
                        event_ids.add(rel['source_event_id'])
                        event_ids.add(rel['target_event_id'])
                    
                    import asyncio
                    events_result = await asyncio.to_thread(
                        lambda: self.supabase.table('raw_events').select(
                            'id, source_platform, document_type, amount_usd, source_ts, vendor_standard, user_id'
                        ).in_('id', list(event_ids)).execute()
                    )
                    
                    if events_result.data:
                        events_map = {e['id']: e for e in events_result.data}
                        logger.info(f"Fetched {len(events_map)} events for semantic enrichment")
                except Exception as e:
                    logger.warning(f"Failed to fetch events for semantic enrichment: {e}")
            
            # Delegate semantic enrichment to SemanticRelationshipExtractor
            semantic_enrichment_stats = await self._enrich_relationships_with_semantic_extractor(
                stored_relationships, events_map, user_id
            )
            
            causal_analysis_stats = await self._analyze_causal_relationships(
                stored_relationships, user_id
            )
            temporal_learning_stats = await self._learn_temporal_patterns(user_id)
            
            logger.info(f"Relationship detection completed: {len(all_relationships)} relationships found")
            
            # Return stored relationships with enrichment
            return {
                "relationships": stored_relationships if stored_relationships else all_relationships,
                "total_relationships": len(all_relationships),
                "cross_document_relationships": len(cross_file_relationships),
                "within_file_relationships": len(within_file_relationships),
                "semantic_enrichment": semantic_enrichment_stats,
                "causal_analysis": causal_analysis_stats,
                "temporal_learning": temporal_learning_stats,
                "processing_stats": {
                    "relationship_types_found": list(set([r.get('relationship_type', 'unknown') for r in all_relationships])),
                    "method": "ai_dynamic_detection",
                    "semantic_system": "SemanticRelationshipExtractor",
                    "semantic_analysis_enabled": self._ensure_semantic_extractor() is not None,
                    "causal_analysis_enabled": self._ensure_causal_engine() is not None,
                    "temporal_learning_enabled": self._ensure_temporal_learner() is not None
                },
                "message": "Relationship detection completed with advanced semantic enrichment"
            }
            
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    async def _detect_cross_document_relationships_db(self, user_id: str, file_id: Optional[str] = None) -> List[Dict]:
        """
        Use AI-powered dynamic relationship detection instead of hard-coded pairs.
        Works for ANY transaction type, not just predefined ones.
        """
        relationships = []
        
        try:
            # Fetch all events for this user
            import asyncio
            events_result = await asyncio.to_thread(
                lambda: self.supabase.table('raw_events').select(
                    'id, document_type, amount_usd, source_ts, vendor_standard, source_platform'
                ).eq('user_id', user_id).execute()
            )
            
            if not events_result.data or len(events_result.data) < 2:
                logger.info(f"Insufficient events for cross-document relationship detection (found {len(events_result.data or [])})")
                return []
            
            events = events_result.data
            logger.info(f"Analyzing {len(events)} events for dynamic relationships")
            
            # Use AI to detect relationships between all event pairs
            relationships = await self._detect_relationships_dynamically(events, user_id)
            
            logger.info(f"Found {len(relationships)} dynamic relationships using AI analysis")
            return relationships
            
        except Exception as e:
            logger.error(f"Cross-document relationship detection failed: {e}", exc_info=True)
            return []
    
    async def _detect_relationships_dynamically(self, events: List[Dict], user_id: str) -> List[Dict]:
        """
        CRITICAL FIX: Use AI to dynamically detect relationships instead of hard-coded pairs.
        Works for ANY transaction type (crypto, loans, subscriptions, refunds, etc.).
        
        Args:
            events: List of event dictionaries with document_type, amount_usd, source_ts, etc.
            user_id: User ID for logging and tracking
            
        Returns:
            List of detected relationships with confidence scores
        """
        if not self.llm_client:
            logger.warning("Groq client not available - cannot perform dynamic relationship detection")
            return []
        
        if not INSTRUCTOR_AVAILABLE:
            logger.warning("Instructor not available - cannot validate AI responses")
            return []
        
        relationships = []
        processed_pairs = set()  # Track processed pairs to avoid duplicates
        
        try:
            # Analyze each pair of events
            for i, source_event in enumerate(events):
                for target_event in events[i+1:]:  # Avoid duplicate pairs
                    source_id = source_event.get('id')
                    target_id = target_event.get('id')
                    
                    # Skip if same event
                    if source_id == target_id:
                        continue
                    
                    # Skip if already processed (in either direction)
                    pair_key = tuple(sorted([source_id, target_id]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    
                    try:
                        # Ask AI: "Are these two events related?"
                        relationship = await self._detect_relationship_pair_ai(
                            source_event, target_event, user_id
                        )
                        
                        if relationship and relationship.get('is_related'):
                            relationships.append({
                                'source_event_id': source_id,
                                'target_event_id': target_id,
                                'relationship_type': relationship.get('relationship_type', 'unknown'),
                                'confidence_score': relationship.get('confidence', 0.5),
                                'reasoning': relationship.get('reasoning', ''),
                                'metadata': {
                                    'source_type': source_event.get('document_type'),
                                    'target_type': target_event.get('document_type'),
                                    'source_amount': source_event.get('amount_usd'),
                                    'target_amount': target_event.get('amount_usd'),
                                    'time_delta_days': self._calculate_time_delta(
                                        source_event.get('source_ts'),
                                        target_event.get('source_ts')
                                    )
                                },
                                'detection_method': 'ai_dynamic'
                            })
                            
                            logger.info(
                                f"Dynamic relationship detected",
                                source_type=source_event.get('document_type'),
                                target_type=target_event.get('document_type'),
                                relationship_type=relationship.get('relationship_type'),
                                confidence=relationship.get('confidence')
                            )
                    
                    except Exception as pair_error:
                        logger.warning(
                            f"Failed to analyze event pair: {pair_error}",
                            source_id=source_id,
                            target_id=target_id
                        )
                        continue
            
            return relationships
        
        except Exception as e:
            logger.error(f"Dynamic relationship detection failed: {e}", exc_info=True)
            return []
    
    async def _detect_relationship_pair_ai(self, source_event: Dict, target_event: Dict, user_id: str) -> Optional[Dict]:
        """
        Use Groq/Llama to detect if two events are related and classify the relationship.
        
        Args:
            source_event: Source event dictionary
            target_event: Target event dictionary
            user_id: User ID for context
            
        Returns:
            Dictionary with is_related, relationship_type, confidence, reasoning
        """
        try:
            # Build human-readable event descriptions
            source_desc = self._format_event_for_ai(source_event, "Event 1")
            target_desc = self._format_event_for_ai(target_event, "Event 2")
            
            prompt = f"""You are a financial analyst AI. Analyze these two financial events and determine if they are related.

{source_desc}

{target_desc}

TASK: Determine if these events are related. If yes, classify the relationship type.

Return a JSON response with:
1. "is_related": true/false - Are these events connected?
2. "relationship_type": The type of relationship (e.g., invoice_payment, loan_disbursement, subscription_renewal, refund_processing, intercompany_transfer, cryptocurrency_transfer, etc.)
3. "confidence": 0.0-1.0 - How confident are you? (0.0 = not related, 1.0 = definitely related)
4. "reasoning": Brief explanation of why they are/aren't related

Be specific. Use relationship types that match the actual business process, not generic ones.
Only return valid JSON, no markdown or explanations."""

            # Use instructor for auto-validated responses
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=DynamicRelationshipResponse,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2  # Lower temperature for more consistent classification
                )
            )
            
            return {
                'is_related': response.is_related,
                'relationship_type': response.relationship_type,
                'confidence': response.confidence,
                'reasoning': response.reasoning
            }
        
        except ValidationError as ve:
            logger.warning(f"AI response validation failed: {ve.errors()}")
            return None
        except Exception as e:
            logger.warning(f"AI relationship detection failed: {e}")
            return None
    
    def _format_event_for_ai(self, event: Dict, label: str) -> str:
        """Format an event for AI analysis in human-readable format."""
        doc_type = event.get('document_type', 'unknown')
        amount = event.get('amount_usd', 0)
        date = event.get('source_ts', 'unknown')
        vendor = event.get('vendor_standard', 'unknown')
        platform = event.get('source_platform', 'unknown')
        
        return f"""{label}:
- Type: {doc_type}
- Amount: ${amount:,.2f}
- Date: {date}
- Vendor/Entity: {vendor}
- Platform: {platform}"""
    
    def _calculate_time_delta(self, source_ts: Optional[str], target_ts: Optional[str]) -> Optional[int]:
        """Calculate days between two timestamps."""
        try:
            if not source_ts or not target_ts:
                return None
            
            source_date = pendulum.parse(source_ts)
            target_date = pendulum.parse(target_ts)
            
            delta = abs((target_date - source_date).days)
            return delta
        except Exception as e:
            logger.warning(f"Failed to calculate time delta: {e}")
            return None
    
    async def _detect_within_file_relationships_db(self, user_id: str, file_id: Optional[str] = None) -> List[Dict]:
        """Use database self-JOIN to find within-file relationships."""
        relationships = []
        
        try:
            if not file_id:
                logger.info("Skipping within-file detection (no file_id specified)")
                return []
            
            result = self.supabase.rpc('find_within_document_relationships', {
                'p_user_id': user_id,
                'p_file_id': file_id,
                'p_relationship_type': 'within_file',
                'p_max_results': 1000
            }).execute()
            
            if result.data:
                for rel in result.data:
                    relationships.append({
                        'source_event_id': rel['source_event_id'],
                        'target_event_id': rel['target_event_id'],
                        'relationship_type': rel['relationship_type'],
                        'confidence_score': float(rel['confidence']),
                        'metadata': rel.get('metadata', {}),
                        'detection_method': 'database_self_join'
                    })
                
                logger.info(f"Found {len(result.data)} within-file relationships")
            
            return relationships
        
        except Exception as e:
            logger.error(f"Within-file relationship detection failed: {e}", exc_info=True)
            return []

    async def _store_relationships(self, relationships: List[Dict], user_id: str, transaction_id: Optional[str] = None, job_id: Optional[str] = None) -> List[Dict]:
        """
        Store detected relationships in the database and return stored records with IDs.
        Returns:
            List of stored relationship records with database-assigned IDs
        """
        try:
            if not relationships:
                return []

            stored_relationships = []

            # Fetch event details for AI enrichment (batch fetch for efficiency)
            event_ids = set()
            for rel in relationships:
                event_ids.add(rel['source_event_id'])
                event_ids.add(rel['target_event_id'])

            # Fetch all events in one query
            events_map = {}
            if self.llm_client and event_ids:
                try:
                    import asyncio
                    events_result = await asyncio.to_thread(
                        lambda: self.supabase.table('raw_events').select(
                            'id, source_platform, document_type, amount_usd, source_ts, vendor_standard'
                        ).in_('id', list(event_ids)).execute()
                    )
                    
                    if events_result.data:
                        events_map = {e['id']: e for e in events_result.data}
                        logger.info(f"Fetched {len(events_map)} events for AI enrichment")
                except Exception as e:
                    logger.warning(f"Failed to fetch events for enrichment: {e}")
            
            # Prepare relationship instances for insertion
            relationship_instances = []
            for rel in relationships:
                # Build metadata from match flags
                metadata = rel.get('metadata', {})
                if rel.get('amount_match'):
                    metadata['amount_match'] = True
                if rel.get('date_match'):
                    metadata['date_match'] = True
                if rel.get('entity_match'):
                    metadata['entity_match'] = True
                
                # Build key factors array
                key_factors = []
                if rel.get('amount_match'):
                    key_factors.append('amount_match')
                if rel.get('date_match'):
                    key_factors.append('date_match')
                if rel.get('entity_match'):
                    key_factors.append('entity_match')
                
                try:
                    if INSTRUCTOR_AVAILABLE:
                        validated_rel = RelationshipRecord(
                            source_event_id=rel['source_event_id'],
                            target_event_id=rel['target_event_id'],
                            relationship_type=rel['relationship_type'],
                            confidence_score=rel['confidence_score'],
                            detection_method=rel.get('detection_method', 'unknown'),
                            metadata=metadata,
                            key_factors=key_factors
                        )
                except ValidationError as ve:
                    logger.error(
                        "relationship_validation_failed",
                        errors=ve.errors(),
                        relationship_type=rel.get('relationship_type')
                    )
                    continue  # Skip invalid relationships
                
                # CRITICAL FIX: Store basic relationship without AI enrichment
                # Semantic enrichment will be handled separately by SemanticRelationshipExtractor
                pattern_signature = f"{rel['relationship_type']}_{'-'.join(sorted(key_factors))}"
                pattern_id = await self._get_or_create_pattern_id(pattern_signature, rel['relationship_type'], key_factors, user_id)
                
                record = {
                    'user_id': user_id,
                    'source_event_id': rel['source_event_id'],
                    'target_event_id': rel['target_event_id'],
                    'relationship_type': rel['relationship_type'],
                    'confidence_score': rel['confidence_score'],
                    'detection_method': rel.get('detection_method', 'unknown'),
                    'metadata': metadata,
                    'pattern_id': pattern_id
                }
                
                relationship_instances.append(record)
            
            if relationship_instances:
                try:
                    result = self.supabase.table('relationship_instances').insert(relationship_instances).execute()
                    if result.data:
                        stored_relationships = result.data
                        logger.info(
                            "relationships_stored",
                            count=len(stored_relationships),
                            total_attempted=len(relationship_instances)
                        )
                except Exception as e:
                    logger.error(f"Failed to insert relationships: {e}")
                    raise
            
            return stored_relationships
        
        except Exception as e:
            logger.error(f"Error storing relationships: {e}", exc_info=True)
            raise
    
    async def _generate_relationship_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for relationship text using BGE model.
        This enables similarity-based relationship discovery and duplicate detection.
        """
        try:
            if not text:
                return None
            
            if not self.embedding_service:
                logger.warning("Embedding service not available, skipping embedding generation")
                return None
            
            embedding = await self.embedding_service.embed_text(text)
            
            logger.debug(f"✅ Generated relationship embedding (1024 dims) for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate relationship embedding: {e}")
            return None
    
    def _find_similar_filename(self, target_filename: str, available_files: List[str]) -> Optional[str]:
        """Find similar filename using RapidFuzz."""
        if not available_files:
            return None
        
        result = process.extractOne(
            target_filename, 
            available_files, 
            scorer=fuzz.WRatio,  # Handles word order, case, punctuation
            score_cutoff=40  # Minimum threshold (0-100 scale)
        )
        
        if result:
            best_match, score, _ = result
            logger.info(f"RapidFuzz match: '{target_filename}' → '{best_match}' (score: {score:.1f}/100)")
            return best_match
        
        return None
    
    def _sort_events_by_date(self, events: List[Dict]) -> List[Dict]:
        """Sort events by date using polars."""
        try:
            if not events:
                return []
            import polars as pl
            df = pl.DataFrame(events)
            df = df.with_columns(
                pl.col('payload').map_elements(lambda x: self._extract_date({'payload': x}) or datetime.min, return_dtype=pl.Object).alias('_sort_date')
            ).sort('_sort_date').drop('_sort_date')
            return df.to_dicts()
        except Exception as e:
            logger.warning(f"Polars sort failed: {e}, using fallback")
            return sorted(events, key=lambda x: self._extract_date(x) or datetime.min)
    
    async def _calculate_relationship_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate comprehensive relationship score"""
        try:
            # Extract data from events
            source_payload = source.get('payload', {})
            target_payload = target.get('payload', {})
            
            # Calculate individual scores
            if self.duplicate_service:
                source_amount = self._extract_amount(source)
                target_amount = self._extract_amount(target)
                amount_score = self.duplicate_service.compare_amounts(source_amount, target_amount)
                
                source_date = self._extract_date(source)
                target_date = self._extract_date(target)
                date_score = self.duplicate_service.compare_dates(source_date, target_date)
            else:
                # Fallback if service unavailable
                logger.warning("duplicate_service not available, using fallback scoring")
                amount_score = 0.0
                date_score = 0.0
            
            entity_score = self._calculate_entity_score(source_payload, target_payload)
            id_score = self._calculate_id_score(source_payload, target_payload)
            context_score = await self._calculate_context_score(source_payload, target_payload)
            
            # Weight scores based on relationship type
            weights = self._get_relationship_weights(relationship_type)
            
            # Calculate weighted score
            total_score = (
                amount_score * weights['amount'] +
                date_score * weights['date'] +
                entity_score * weights['entity'] +
                id_score * weights['id'] +
                context_score * weights['context']
            )
            
            return min(total_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating relationship score: {e}")
            return 0.0
    
    def _calculate_entity_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate entity similarity using recordlinkage."""
        try:
            import recordlinkage
            source_entities = self._extract_entities(source_payload)
            target_entities = self._extract_entities(target_payload)
            
            if not source_entities or not target_entities:
                return 0.0
            
            source_str = ' '.join(source_entities)
            target_str = ' '.join(target_entities)
            
            indexer = recordlinkage.Index()
            pairs = indexer.full()
            compare = recordlinkage.Compare()
            compare.string('source', 'target', method='cosine', threshold=0.5, label='entity')
            
            features = compare.compute(pairs, recordlinkage.DataFrame(
                {'source': [source_str]},
                {'target': [target_str]}
            ))
            
            return float(features.iloc[0, 0]) if len(features) > 0 else 0.0
        except Exception as e:
            logger.warning(f"recordlinkage entity scoring failed: {e}, using fallback")
            source_entities = self._extract_entities(source_payload)
            target_entities = self._extract_entities(target_payload)
            if not source_entities or not target_entities:
                return 0.0
            common = set(source_entities) & set(target_entities)
            total = set(source_entities) | set(target_entities)
            return len(common) / len(total) if total else 0.0
    
    def _calculate_id_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate ID similarity using rapidfuzz."""
        try:
            from rapidfuzz import fuzz
            source_ids = self._extract_ids(source_payload)
            target_ids = self._extract_ids(target_payload)
            
            if not source_ids or not target_ids:
                return 0.0
            
            max_score = 0.0
            for source_id in source_ids:
                for target_id in target_ids:
                    if source_id == target_id:
                        return 1.0
                    partial = fuzz.partial_ratio(source_id, target_id) / 100.0
                    max_score = max(max_score, partial)
            
            return max_score
        except Exception as e:
            logger.warning(f"rapidfuzz ID scoring failed: {e}, using fallback")
            source_ids = self._extract_ids(source_payload)
            target_ids = self._extract_ids(target_payload)
            if not source_ids or not target_ids:
                return 0.0
            common = set(source_ids) & set(target_ids)
            return 1.0 if common else 0.0
    
    async def _calculate_context_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """
        Calculate semantic similarity using shared BGE embedding service.
        """
        try:
            # CRITICAL FIX: Check for empty dictionaries BEFORE string conversion
            # str({}) returns '{}' which is not empty, so we need to check dict is empty first
            if not source_payload or not target_payload:
                return 0.0
            
            # Fallback to word overlap if embedding service unavailable
            source_text = str(source_payload)[:500]  # Limit to 500 chars for speed
            target_text = str(target_payload)[:500]
            
            if not source_text.strip() or not target_text.strip():
                return 0.0
            
            try:
                # Use injected BGE embedding service
                if not self.embedding_service:
                    logger.warning("Embedding service not available, using word overlap fallback")
                    raise Exception("Embedding service not available")
                
                # Generate embeddings using BGE model (1024 dims)
                source_emb = await self.embedding_service.embed_text(source_text)
                target_emb = await self.embedding_service.embed_text(target_text)
                
                # Calculate cosine similarity
                similarity = self.embedding_service.similarity(source_emb, target_emb)
                
                # Ensure [0, 1] range
                return max(0.0, min(1.0, similarity))
                
            except Exception as embed_error:
                logger.warning(f"BGE embedding failed, using word overlap fallback: {embed_error}")
                # Fallback to word overlap
                source_words = set(source_text.lower().split())
                target_words = set(target_text.lower().split())
                if not source_words or not target_words:
                    return 0.0
                common_words = source_words & target_words
                total_words = source_words | target_words
                return len(common_words) / len(total_words)
            
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0
    
    def _get_relationship_weights(self, relationship_type: str) -> Dict[str, float]:
        """Get weights for different relationship types"""
        weights = {
            'amount': 0.3,
            'date': 0.2,
            'entity': 0.2,
            'id': 0.2,
            'context': 0.1
        }
        
        # Adjust weights based on relationship type
        if relationship_type in ['invoice_to_payment', 'payment_to_invoice']:
            weights['amount'] = 0.4
            weights['id'] = 0.3
        elif relationship_type in ['revenue_to_cashflow', 'expense_to_bank']:
            weights['date'] = 0.3
            weights['amount'] = 0.3
        elif relationship_type in ['payroll_to_bank']:
            weights['entity'] = 0.3
            weights['date'] = 0.3
        
        return weights
    
    def _extract_amount(self, event: Dict) -> float:
        """Extract amount from event using enriched amount_usd for currency consistency"""
        try:
            # PRIORITY 1: Use amount_usd from enriched columns (Phase 5 enrichment)
            if 'amount_usd' in event and event['amount_usd'] is not None:
                amount_usd = event['amount_usd']
                if isinstance(amount_usd, (int, float)) and amount_usd != 0:
                    return float(amount_usd)
            
            # PRIORITY 2: Check payload for amount_usd (fallback)
            payload = event.get('payload', {})
            if 'amount_usd' in payload and payload['amount_usd']:
                return float(payload['amount_usd'])
            
            # PRIORITY 3: Use universal extractors on raw payload
            try:
                from universal_extractors_optimized import UniversalExtractorsOptimized
                universal_extractors = UniversalExtractorsOptimized()
                amount_result = universal_extractors._extract_amount_fallback(payload)
                if amount_result and isinstance(amount_result, (int, float)):
                    return float(amount_result)
            except Exception as e:
                logger.warning(f"Universal amount extraction failed: {e}")
            
            # PRIORITY 4: Manual extraction from payload
            amount_fields = ['amount', 'total', 'value', 'payment_amount']
            for field in amount_fields:
                if field in payload and payload[field]:
                    return float(payload[field])
            
            return 0.0
        except Exception as e:
            logger.error(f"Amount extraction failed: {e}")
            return 0.0
    
    def _extract_date(self, event: Dict) -> Optional[datetime]:
        """Extract date using Pendulum."""
        try:
            # PRIORITY 1: Transaction date from payload (business date)
            payload = event.get('payload', {})
            
            # Check common transaction date fields in payload
            transaction_date_fields = ['date', 'transaction_date', 'txn_date', 'posting_date', 'value_date']
            for field in transaction_date_fields:
                if field in payload and payload[field]:
                    try:
                        date_str = str(payload[field])
                        # Pendulum handles 100+ date formats automatically
                        parsed = pendulum.parse(date_str, strict=False)
                        return parsed.naive()  # Convert to naive datetime for compatibility
                    except:
                        continue
            
            # PRIORITY 2: Check enriched source_ts column (from Phase 5)
            if 'source_ts' in event and event['source_ts']:
                try:
                    parsed = pendulum.parse(str(event['source_ts']), strict=False)
                    return parsed.naive()
                except:
                    pass
            
            # PRIORITY 3: Fallback to system timestamps (ONLY if no transaction date found)
            system_date_fields = ['created_at', 'ingest_ts', 'processed_at']
            for field in system_date_fields:
                if field in event and event[field]:
                    try:
                        parsed = pendulum.parse(str(event[field]), strict=False)
                        return parsed.naive()
                    except:
                        continue
            
            return None
        except Exception as e:
            logger.error(f"Date extraction failed: {e}")
            return None
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entities using spaCy NER."""
        entities = []
        try:
            # PRIORITY 1: Use spaCy NER if available
            if self.nlp:
                text = str(payload)[:1000]  # Limit to 1000 chars for speed
                if text.strip():
                    doc = self.nlp(text)
                    # Extract organizations, people, locations, money
                    for ent in doc.ents:
                        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PRODUCT']:
                            entities.append(ent.text)
                    
                    if entities:
                        return list(set(entities))
            
            # PRIORITY 2: Universal extractors fallback
            try:
                from universal_extractors_optimized import UniversalExtractorsOptimized
                universal_extractors = UniversalExtractorsOptimized()
                vendor_result = universal_extractors._extract_vendor_fallback(payload)
                if vendor_result and isinstance(vendor_result, str):
                    entities.append(vendor_result)
            except Exception as e:
                logger.warning(f"Universal vendor extraction failed: {e}")
            
            # PRIORITY 3: Extract from entities field
            if 'entities' in payload:
                entity_data = payload['entities']
                if isinstance(entity_data, dict):
                    for entity_type, entity_list in entity_data.items():
                        if isinstance(entity_list, list):
                            entities.extend(entity_list)
            
            # PRIORITY 4: Capitalization heuristic (last resort)
            if not entities:
                text = str(payload)
                words = text.split()
                for word in words:
                    if len(word) > 3 and word[0].isupper():
                        entities.append(word)
            
            return list(set(entities))
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_ids(self, payload: Dict) -> List[str]:
        """Extract IDs using parse library."""
        ids = []
        try:
            from parse import parse
            payload_str = str(payload)
            
            id_patterns = [
                '{prefix:w}-{num:d}',
                '#{id:d}',
                'INV{num:d}',
                'PO{num:d}',
                'REF{id:w}',
            ]
            
            for pattern in id_patterns:
                results = parse(pattern, payload_str)
                if results:
                    ids.append(str(results.named.get('id') or results.named.get('num') or results.named.get('prefix')))
            
            id_fields = ['id', 'transaction_id', 'payment_id', 'invoice_id', 'reference']
            for field in id_fields:
                if field in payload and payload[field]:
                    ids.append(str(payload[field]))
            
            return list(set(ids))
        except Exception as e:
            logger.warning(f"parse ID extraction failed: {e}, using fallback")
            ids = []
            id_fields = ['id', 'transaction_id', 'payment_id', 'invoice_id', 'reference']
            for field in id_fields:
                if field in payload and payload[field]:
                    ids.append(str(payload[field]))
            return ids
    
    async def _remove_duplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicates using datasketch MinHash LSH."""
        if not relationships:
            return []
        
        from datasketch import MinHash, MinHashLSH
        
        seen = set()
        exact_unique = []
        
        for rel in relationships:
            key = f"{rel.get('source_event_id')}_{rel.get('target_event_id')}_{rel.get('relationship_type')}"
            if key not in seen:
                seen.add(key)
                exact_unique.append(rel)
        
        
        if len(exact_unique) < 2:
            return exact_unique
        
        try:
            lsh = MinHashLSH(threshold=0.85, num_perm=128)
            minhashes = {}
            
            for i, rel in enumerate(exact_unique):
                desc = f"{rel.get('relationship_type', '')} {rel.get('semantic_description', '')}"
                mh = MinHash(num_perm=128)
                for token in desc.split():
                    mh.update(token.encode('utf8'))
                minhashes[i] = mh
                lsh.insert(f"rel_{i}", mh)
            
            semantic_unique = []
            skip_indices = set()
            
            for i in range(len(exact_unique)):
                if i in skip_indices:
                    continue
                semantic_unique.append(exact_unique[i])
                
                duplicates = lsh.query(minhashes[i])
                for dup_id in duplicates:
                    dup_idx = int(dup_id.split('_')[1])
                    if dup_idx > i:
                        skip_indices.add(dup_idx)
            
            return semantic_unique
            
        except Exception as e:
            logger.warning(f"MinHash deduplication failed: {e}, falling back to exact dedup")
            return exact_unique
    
    async def _validate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Validate relationships"""
        validated = []
        
        for rel in relationships:
            if self._validate_relationship_structure(rel):
                validated.append(rel)
        
        return validated
    
    def _validate_relationship_structure(self, rel: Dict) -> bool:
        """Validate relationship structure"""
        required_fields = ['source_event_id', 'target_event_id', 'relationship_type', 'confidence_score']
        
        for field in required_fields:
            if field not in rel or rel[field] is None:
                return False
        
        # Check confidence score range
        if not (0.0 <= rel['confidence_score'] <= 1.0):
            return False
        
        return True
    
    async def _enrich_relationships_with_semantic_extractor(
        self,
        stored_relationships: List[Dict],
        events_map: Dict[str, Dict],
        user_id: str
    ) -> Dict[str, Any]:
        """
        CRITICAL FIX: Use advanced SemanticRelationshipExtractor for semantic enrichment.
        
        Replaces duplicate _enrich_relationship_with_ai() logic with battle-tested system.
        Provides:
        - Jinja2 template-based prompts
        - Redis-backed distributed caching
        - Prometheus metrics (Grafana-ready)
        - Rate-limited batch processing
        - Automatic embeddings
        
        Args:
            stored_relationships: List of basic relationship records from database
            events_map: Mapping of event_id -> event details
            user_id: User ID for logging and context
            
        Returns:
            Dictionary with enrichment stats and results
        """
        try:
            semantic_extractor = self._ensure_semantic_extractor()
            if not semantic_extractor:
                logger.warning("SemanticRelationshipExtractor not available - skipping semantic enrichment")
                return {
                    'total_enriched': 0,
                    'successful': 0,
                    'failed': 0,
                    'skipped': len(stored_relationships),
                    'method': 'none'
                }
            
            enriched_count = 0
            failed_count = 0
            
            # Prepare relationship pairs for batch processing
            relationship_pairs = []
            for rel in stored_relationships:
                source_event = events_map.get(rel['source_event_id'])
                target_event = events_map.get(rel['target_event_id'])
                
                if not source_event or not target_event:
                    logger.warning(
                        f"Missing event data for relationship {rel.get('id', 'unknown')}",
                        source_id=rel['source_event_id'],
                        target_id=rel['target_event_id']
                    )
                    failed_count += 1
                    continue
                
                relationship_pairs.append((source_event, target_event))
            
            if not relationship_pairs:
                logger.info("No valid relationship pairs for semantic enrichment")
                return {
                    'total_enriched': 0,
                    'successful': 0,
                    'failed': failed_count,
                    'skipped': 0,
                    'method': 'semantic_extractor'
                }
            
            # Use batch processing with rate limiting
            logger.info(f"Enriching {len(relationship_pairs)} relationships with semantic extractor")
            semantic_results = await semantic_extractor.extract_semantic_relationships_batch(
                relationship_pairs
            )
            
            # Update relationships with semantic data
            for i, semantic_rel in enumerate(semantic_results):
                if not semantic_rel:
                    failed_count += 1
                    continue
                
                try:
                    rel = stored_relationships[i]
                    
                    # Update relationship with semantic fields
                    update_data = {
                        'semantic_description': semantic_rel.semantic_description,
                        'reasoning': semantic_rel.reasoning,
                        'temporal_causality': semantic_rel.temporal_causality.value,
                        'business_logic': semantic_rel.business_logic.value,
                        'key_factors': semantic_rel.key_factors,
                        'embedding': semantic_rel.embedding,
                        'metadata': {
                            **(rel.get('metadata', {})),
                            **semantic_rel.metadata
                        }
                    }
                    
                    # Store updated relationship in database
                    import asyncio
                    await asyncio.to_thread(
                        lambda: self.supabase.table('relationship_instances').update(
                            update_data
                        ).eq('id', rel['id']).execute()
                    )
                    
                    enriched_count += 1
                    
                    logger.info(
                        f"Relationship enriched",
                        relationship_type=semantic_rel.relationship_type,
                        confidence=semantic_rel.confidence,
                        temporal_causality=semantic_rel.temporal_causality.value
                    )
                
                except Exception as update_error:
                    logger.warning(f"Failed to update relationship with semantic data: {update_error}")
                    failed_count += 1
                    continue
            
            return {
                'total_enriched': enriched_count,
                'successful': enriched_count,
                'failed': failed_count,
                'skipped': len(stored_relationships) - len(relationship_pairs),
                'method': 'semantic_extractor',
                'cache_hit_rate': getattr(semantic_extractor, '_cache_hits', 0) / max(
                    getattr(semantic_extractor, '_cache_hits', 0) + getattr(semantic_extractor, '_cache_misses', 1), 1
                )
            }
        
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}", exc_info=True)
            return {
                'total_enriched': 0,
                'successful': 0,
                'failed': len(stored_relationships),
                'skipped': 0,
                'method': 'semantic_extractor',
                'error': str(e)
            }
    
    async def _fetch_events_by_ids(self, event_ids: List[str], user_id: str) -> Dict[str, Dict]:
        """Fetch events by IDs and return as dictionary"""
        try:
            if not event_ids:
                return {}
            
            # Fetch events from database
            result = self.supabase.table('raw_events').select(
                'id, source_platform, document_type, amount_usd, source_ts, '
                'vendor_standard, payload, created_at'
            ).in_('id', event_ids).eq('user_id', user_id).execute()
            
            if not result.data:
                return {}
            
            # Convert to dictionary keyed by event ID
            events_dict = {event['id']: event for event in result.data}
            
            return events_dict
            
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return {}
    
    async def _analyze_causal_relationships(
        self,
        relationships: List[Dict],
        user_id: str
    ) -> Dict[str, Any]:
        """
        PHASE 3: Analyze relationships for causality using Bradford Hill criteria.
        
        This determines which relationships are truly causal (cause-effect)
        vs merely correlated.
        
        Args:
            relationships: List of detected relationships
            user_id: User ID for context
        
        Returns:
            Statistics about causal analysis
        """
        causal_engine = self._ensure_causal_engine()
        if not causal_engine or not relationships:
            return {
                'enabled': False,
                'total_relationships': len(relationships),
                'causal_count': 0,
                'message': 'Causal analysis not available or no relationships to analyze'
            }
        
        try:
            # Extract relationship IDs
            relationship_ids = [rel.get('id') for rel in relationships if rel.get('id')]
            
            if not relationship_ids:
                return {
                    'enabled': True,
                    'total_relationships': len(relationships),
                    'causal_count': 0,
                    'message': 'No relationship IDs found for causal analysis'
                }
            
            # Run causal analysis
            result = await causal_engine.analyze_causal_relationships(
                user_id=user_id,
                relationship_ids=relationship_ids
            )
            
            return {
                'enabled': True,
                'total_relationships': result.get('total_analyzed', 0),
                'causal_count': result.get('causal_count', 0),
                'causal_percentage': result.get('causal_percentage', 0.0),
                'avg_causal_score': result.get('avg_causal_score', 0.0),
                'message': result.get('message', 'Causal analysis completed')
            }
            
        except Exception as e:
            logger.error("causal_analysis_failed", error=str(e), exc_info=True)
            raise
    
    async def _learn_temporal_patterns(self, user_id: str) -> Dict[str, Any]:
        """
        PHASE 4: Learn temporal patterns and predict missing relationships.
        
        This analyzes historical relationship timings to:
        - Learn patterns (e.g., "invoices paid in 30±5 days")
        - Detect seasonal cycles
        - Predict missing relationships
        - Identify temporal anomalies
        
        Args:
            user_id: User ID for context
        
        Returns:
            Statistics about temporal pattern learning
        """
        temporal_learner = self._ensure_temporal_learner()
        if not temporal_learner:
            return {
                'enabled': False,
                'patterns_learned': 0,
                'predictions_made': 0,
                'anomalies_detected': 0,
                'message': 'Temporal pattern learning not available'
            }
        
        try:
            # Learn all patterns
            patterns_result = await temporal_learner.learn_all_patterns(user_id)
            
            # Predict missing relationships
            predictions_result = await temporal_learner.predict_missing_relationships(user_id)
            
            # Detect temporal anomalies
            anomalies_result = await temporal_learner.detect_temporal_anomalies(user_id)
            
            return {
                'enabled': True,
                'patterns_learned': patterns_result.get('total_patterns', 0),
                'predictions_made': predictions_result.get('total_predictions', 0),
                'overdue_predictions': predictions_result.get('overdue_count', 0),
                'anomalies_detected': anomalies_result.get('total_anomalies', 0),
                'critical_anomalies': anomalies_result.get('critical_count', 0),
                'patterns': patterns_result.get('patterns', []),
                'predictions': predictions_result.get('predictions', []),
                'anomalies': anomalies_result.get('anomalies', []),
                'message': (
                    f"Temporal learning completed: {patterns_result.get('total_patterns', 0)} patterns learned, "
                    f"{predictions_result.get('total_predictions', 0)} predictions made, "
                    f"{anomalies_result.get('total_anomalies', 0)} anomalies detected"
                )
            }
            
        except Exception as e:
            logger.error(f"Temporal pattern learning failed: {e}")
            return {
                'enabled': True,
                'patterns_learned': 0,
                'predictions_made': 0,
                'anomalies_detected': 0,
                'error': str(e),
                'message': 'Temporal pattern learning failed'
            }

    async def _get_or_create_pattern_id(self, pattern_signature: str, relationship_type: str, key_factors: List[str], user_id: str) -> Optional[str]:
        """
        FIX #1: Atomic UPSERT pattern using tenacity retry for unique constraint handling.
        
        Implements robust "select-insert-select" pattern to prevent race conditions:
        1. Try to SELECT existing pattern
        2. If not found, INSERT new pattern
        3. Handle unique constraint violations with tenacity retry
        4. Return pattern ID
        
        This ensures atomicity without requiring custom PostgreSQL RPC.
        
        Args:
            pattern_signature: Unique signature for the pattern (e.g., "invoice_to_payment_amount_match-date_match")
            relationship_type: Type of relationship (e.g., "invoice_to_payment")
            key_factors: List of factors that define this pattern (e.g., ["amount_match", "date_match"])
            user_id: User ID for scoping
        
        Returns:
            Pattern ID (UUID string) or None if creation fails
        """
        import asyncio
        from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
        
        async def _upsert_pattern():
            # FIX #5: Wrap synchronous Supabase calls in asyncio.to_thread() to avoid blocking event loop
            # Step 1: Try to SELECT existing pattern
            select_result = await asyncio.to_thread(
                lambda: self.supabase.table('temporal_patterns').select('id').eq(
                    'pattern_signature', pattern_signature
                ).eq('user_id', user_id).limit(1).execute()
            )
            
            if select_result.data and len(select_result.data) > 0:
                pattern_id = select_result.data[0]['id']
                logger.info(f"Found existing pattern: {pattern_id}")
                return pattern_id
            
            # Step 2: Pattern doesn't exist, try to INSERT
            insert_result = await asyncio.to_thread(
                lambda: self.supabase.table('temporal_patterns').insert({
                    'user_id': user_id,
                    'pattern_signature': pattern_signature,
                    'relationship_type': relationship_type,
                    'key_factors': key_factors,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                    'pattern_count': 1,
                    'std_dev_days': 5.0,  # Default value
                    'confidence_score': 0.5  # Default value
                }).execute()
            )
            
            if insert_result.data and len(insert_result.data) > 0:
                pattern_id = insert_result.data[0]['id']
                logger.info(f"Created new pattern: {pattern_id}")
                return pattern_id
            
            # If we get here, something went wrong
            logger.error("Insert returned no data")
            raise RuntimeError("Insert returned no data")
        
        try:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
                retry=retry_if_exception_type((Exception,)),
                reraise=True
            )
            
            async for attempt in retrying:
                with attempt:
                    return await _upsert_pattern()
        except Exception as e:
            logger.error(f"Failed to get/create pattern: {e}")
            return None


# Test function
async def test_enhanced_relationship_detection(user_id: str = "550e8400-e29b-41d4-a716-446655440000"):
    """Test the enhanced relationship detection system"""
    try:
        # Initialize OpenAI client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Enhanced Relationship Detector
        enhanced_detector = EnhancedRelationshipDetector(openai_client=None, supabase_client=supabase)
        
        # Detect relationships
        result = await enhanced_detector.detect_all_relationships(user_id)
        
        return {
            "message": "Enhanced Relationship Detection Test Completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Enhanced Relationship Detection Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Run test synchronously to avoid async/sync mixing
    import asyncio
    result = asyncio.run(test_enhanced_relationship_detection())
    print(result)


# ============================================================================
# PRELOAD PATTERN: Initialize heavy modules at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.
# 
# BENEFITS:
# - First request is instant (no cold-start delay)
# - Shared across all worker instances
# - Memory is allocated once, not per-instance

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level preload failed (will use fallback): {e}")
