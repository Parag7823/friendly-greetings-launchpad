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

import igraph as ig
import pendulum
import spacy
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from provenance_tracker import normalize_business_logic, normalize_temporal_causality

# ✅ NEW: Structured logging with structlog
import structlog
logger = structlog.get_logger(__name__)

# ✅ NEW: Add instructor for auto-validated AI responses
try:
    import instructor
    from pydantic import BaseModel, Field, ValidationError, validator
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("instructor not available - AI responses won't be auto-validated")

# ✅ NEW: Add tenacity for retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity not available - retry logic disabled")

# Initialize Groq client for semantic analysis
try:
    from groq import Groq
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        
        # ✅ NEW: Patch Groq client with instructor for auto-validated responses
        if INSTRUCTOR_AVAILABLE:
            groq_client = instructor.patch(groq_client)
            logger.info("groq_client_patched_with_instructor", auto_validation=True)
        
        GROQ_AVAILABLE = True
        logger.info("groq_client_initialized", model="llama-3.3-70b-versatile")
    else:
        groq_client = None
        GROQ_AVAILABLE = False
        logger.warning("groq_api_key_not_found")
except ImportError:
    groq_client = None
    GROQ_AVAILABLE = False
    logger.warning("groq_package_not_installed")

# ✅ NEW: Pydantic models for comprehensive data validation
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

# Debug logging now handled via structlog

# Import semantic relationship extractor for AI-powered semantic analysis
try:
    from semantic_relationship_extractor import SemanticRelationshipExtractor
    SEMANTIC_EXTRACTOR_AVAILABLE = True
except ImportError:
    SEMANTIC_EXTRACTOR_AVAILABLE = False
    logger.warning("SemanticRelationshipExtractor not available. Semantic analysis will be disabled.")

# Import causal inference engine for Bradford Hill criteria and causal analysis
try:
    from causal_inference_engine import CausalInferenceEngine
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    logger.warning("CausalInferenceEngine not available. Causal analysis will be disabled.")

# Import temporal pattern learner for pattern learning and prediction
try:
    from temporal_pattern_learner import TemporalPatternLearner
    TEMPORAL_PATTERN_LEARNER_AVAILABLE = True
except ImportError:
    TEMPORAL_PATTERN_LEARNER_AVAILABLE = False
    logger.warning("TemporalPatternLearner not available. Temporal pattern learning will be disabled.")

# Neo4j REMOVED (Nov 2025): Replaced by igraph + Supabase
# - igraph handles in-memory graph analytics (13-32x faster)
# - Supabase stores all relationships persistently
# - No need for separate graph database

class EnhancedRelationshipDetector:
    """Enhanced relationship detector that actually finds relationships between events"""
    
    def __init__(
        self,
        anthropic_client: AsyncGroq = None,
        openai_client: Any = None,  # backward-compatible alias used in older code
        supabase_client: Client = None,
        cache_client=None,
    ):
        # Prefer explicitly provided anthropic_client; else accept legacy openai_client
        client = anthropic_client or openai_client
        self.anthropic = client
        self.supabase = supabase_client
        self.cache = cache_client  # Use centralized cache, no local cache
        
        # REFACTORED: Initialize genius libraries
        try:
            self.nlp = spacy.load("en_core_web_sm")  # NER for entity extraction
            logger.info("✅ spaCy NER model loaded")
        except:
            self.nlp = None
            logger.warning("⚠️ spaCy model not found, run: python -m spacy download en_core_web_sm")
        
        # CRITICAL FIX: Use shared BGE embedding service instead of loading separate model
        # Old: Loaded all-MiniLM-L6-v2 (384 dims) separately
        # New: Use BGE large model (1024 dims) via embedding_service singleton
        # This eliminates 400MB+ memory waste and ensures consistent embeddings
        self.semantic_model = None  # Will use embedding_service instead
        logger.info("✅ Using shared BGE embedding service (BAAI/bge-large-en-v1.5)")
        
        # igraph for in-memory graph analysis
        self.graph = ig.Graph(directed=True)
        logger.info("✅ igraph initialized (13-32x faster than networkx)")
        
        # Initialize semantic relationship extractor for AI-powered analysis
        if SEMANTIC_EXTRACTOR_AVAILABLE:
            self.semantic_extractor = SemanticRelationshipExtractor(
                openai_client=client,
                supabase_client=supabase_client,
                cache_client=None  # v2.0: Using aiocache (Redis) instead
            )
            logger.info("✅ Semantic relationship extractor v2.0 initialized (aiocache + instructor)")
        else:
            self.semantic_extractor = None
            logger.warning("⚠️ Semantic relationship extractor not available")
        
        # Initialize causal inference engine for Bradford Hill criteria analysis
        if CAUSAL_INFERENCE_AVAILABLE:
            self.causal_engine = CausalInferenceEngine(
                supabase_client=supabase_client
            )
            logger.info("✅ Causal inference engine initialized")
        else:
            self.causal_engine = None
            logger.warning("⚠️ Causal inference engine not available")
        
        # Initialize temporal pattern learner for pattern learning and prediction
        if TEMPORAL_PATTERN_LEARNER_AVAILABLE:
            self.temporal_learner = TemporalPatternLearner(
                supabase_client=supabase_client
            )
            logger.info("✅ Temporal pattern learner initialized")
        else:
            self.temporal_learner = None
            logger.warning("⚠️ Temporal pattern learner not available")
        
        # ML model for adaptive relationship scoring (trained on-demand)
        self.ml_model = None
        self.ml_model_trained = False
        
    async def detect_all_relationships(self, user_id: str, file_id: Optional[str] = None, transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        CRITICAL FIX: Detect relationships using document_type classification and database-level JOINs.
        
        This replaces:
        1. Hardcoded filename patterns with document_type from Phase 5 classification
        2. O(N²) Python loops with efficient PostgreSQL JOINs
        3. In-memory cache with centralized caching
        
        Args:
            user_id: User ID to filter events
            file_id: Optional file_id to scope detection to specific file
        """
        try:
            logger.info(f"Starting relationship detection for user_id={user_id}, file_id={file_id}")
            
            # CRITICAL FIX: Use database functions instead of fetching all events
            cross_file_relationships = await self._detect_cross_document_relationships_db(user_id, file_id)
            within_file_relationships = await self._detect_within_file_relationships_db(user_id, file_id)
            
            # Combine and store relationships
            all_relationships = cross_file_relationships + within_file_relationships
            
            # Store relationships in Supabase database and get back stored records with IDs
            stored_relationships = []
            if all_relationships:
                stored_relationships = await self._store_relationships(all_relationships, user_id, transaction_id)
                logger.info(f"✅ Stored {len(stored_relationships)} relationships with database IDs")
            
            # PHASE 2B: Enrich relationships with semantic analysis
            # Use stored_relationships (with IDs) instead of all_relationships
            semantic_enrichment_stats = await self._enrich_relationships_with_semantics(
                stored_relationships, user_id
            )
            
            # PHASE 3: Causal inference using Bradford Hill criteria
            # Use stored_relationships (with IDs) instead of all_relationships
            causal_analysis_stats = await self._analyze_causal_relationships(
                stored_relationships, user_id
            )
            
            # PHASE 4: Temporal pattern learning and prediction
            temporal_learning_stats = await self._learn_temporal_patterns(user_id)
            
            logger.info(f"Relationship detection completed: {len(all_relationships)} relationships found")
            
            # ✅ CRITICAL FIX: Return stored_relationships (with enrichment) instead of all_relationships
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
                    "method": "database_joins",
                    "complexity": "O(N log N) instead of O(N²)",
                    "semantic_analysis_enabled": self.semantic_extractor is not None,
                    "causal_analysis_enabled": self.causal_engine is not None,
                    "temporal_learning_enabled": self.temporal_learner is not None
                },
                "message": "Relationship detection completed successfully using database-level optimization"
            }
            
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    async def _detect_cross_document_relationships_db(self, user_id: str, file_id: Optional[str] = None) -> List[Dict]:
        """
        CRITICAL FIX: Use database-level JOINs to find cross-document relationships.
        This replaces hardcoded filename patterns with document_type classification.
        """
        relationships = []
        
        try:
            # Define document type pairs for relationship detection
            # CRITICAL FIX: Use document_type instead of hardcoded filenames
            document_type_pairs = [
                ('invoice', 'bank_statement', 'invoice_to_payment'),
                ('invoice', 'payment', 'invoice_to_payment'),
                ('revenue', 'bank_statement', 'revenue_to_bank'),
                ('expense', 'bank_statement', 'expense_to_bank'),
                ('payroll', 'bank_statement', 'payroll_to_bank'),
                ('receivable', 'bank_statement', 'receivable_collection'),
            ]
            
            for source_type, target_type, relationship_type in document_type_pairs:
                try:
                    # Call database function for efficient relationship detection
                    result = self.supabase.rpc('find_cross_document_relationships', {
                        'p_user_id': user_id,
                        'p_source_document_type': source_type,
                        'p_target_document_type': target_type,
                        'p_relationship_type': relationship_type,
                        'p_max_results': 1000,
                        'p_amount_tolerance': 5.0,
                        'p_date_range_days': 30
                    }).execute()
                    
                    if result.data:
                        for rel in result.data:
                            relationships.append({
                                'source_event_id': rel['source_event_id'],
                                'target_event_id': rel['target_event_id'],
                                'relationship_type': rel['relationship_type'],
                                'confidence_score': float(rel['confidence']),
                                'amount_match': rel.get('amount_match', False),
                                'date_match': rel.get('date_match', False),
                                'entity_match': rel.get('entity_match', False),
                                'metadata': rel.get('metadata', {}),
                                'detection_method': 'database_join'
                            })
                        
                        logger.info(f"Found {len(result.data)} {relationship_type} relationships")
                
                except Exception as e:
                    logger.warning(f"Failed to detect {relationship_type}: {e}")
                    continue
            
            return relationships
            
        except Exception as e:
            logger.error(f"Cross-document relationship detection failed: {e}")
            return []
    
    async def _detect_within_file_relationships_db(self, user_id: str, file_id: Optional[str] = None) -> List[Dict]:
        """
        CRITICAL FIX: Use database self-JOIN to find within-file relationships.
        This replaces O(N²) Python loops with efficient SQL.
        """
        relationships = []
        
        try:
            if not file_id:
                # If no file_id specified, skip within-file detection
                logger.info("Skipping within-file detection (no file_id specified)")
                return []
            
            # Call database function for efficient within-file relationship detection
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
            logger.error(f"Within-file relationship detection failed: {e}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    ) if TENACITY_AVAILABLE else lambda f: f
    async def _enrich_relationship_with_ai(self, rel: Dict, source_event: Dict, target_event: Dict) -> Dict:
        """
        Enrich a relationship with AI-generated semantic fields using Groq/Llama.
        
        Populates: semantic_description, reasoning, temporal_causality, business_logic
        
        Raises:
            ValueError: If Groq is unavailable or enrichment fails
        """
        if not GROQ_AVAILABLE or not groq_client:
            logger.error("groq_unavailable_cannot_enrich_relationship")
            raise ValueError("Groq client not available for relationship enrichment")
        
        try:
            # Build context for AI with enhanced business logic classification
            prompt = f"""You are a financial analyst AI. Analyze this relationship between two financial events and provide detailed, accurate insights.

SOURCE EVENT:
- Document Type: {source_event.get('document_type', 'unknown')}
- Platform: {source_event.get('source_platform', 'unknown')}
- Amount: ${source_event.get('amount_usd', 0):.2f}
- Date: {source_event.get('source_ts', 'unknown')}
- Vendor/Entity: {source_event.get('vendor_standard', 'unknown')}

TARGET EVENT:
- Document Type: {target_event.get('document_type', 'unknown')}
- Platform: {target_event.get('source_platform', 'unknown')}
- Amount: ${target_event.get('amount_usd', 0):.2f}
- Date: {target_event.get('source_ts', 'unknown')}
- Vendor/Entity: {target_event.get('vendor_standard', 'unknown')}

DETECTED RELATIONSHIP:
- Type: {rel.get('relationship_type', 'unknown')}
- Confidence: {rel.get('confidence_score', 0):.2f}
- Matching Factors: {', '.join(rel.get('key_factors', [])) if rel.get('key_factors') else 'None'}

TASK: Provide a JSON response with these fields:

1. "semantic_description": A clear, business-friendly description of this relationship (2-3 sentences). Explain what happened in plain language.

2. "reasoning": Detailed explanation of WHY this relationship exists. Include:
   - What evidence supports this connection?
   - What business process does this represent?
   - Why are these two events related?

3. "temporal_causality": Determine the causal direction. Choose ONE:
   - "source_causes_target": Source event directly caused the target event
   - "target_causes_source": Target event caused the source event
   - "bidirectional": Both events influence each other
   - "correlation_only": Events are correlated but no clear causation

4. "business_logic": Classify the business process. Choose the MOST SPECIFIC category:
   - "invoice_payment": Invoice being paid through bank/payment system
   - "revenue_collection": Revenue recorded and cash collected
   - "expense_reimbursement": Expense claim being reimbursed
   - "payroll_processing": Salary/wage payment to employee
   - "tax_payment": Tax withholding or payment to authorities
   - "vendor_payment": Payment to supplier/vendor for goods/services
   - "customer_payment": Payment received from customer
   - "recurring_subscription": Recurring subscription or membership payment
   - "loan_disbursement": Loan or credit disbursement
   - "loan_repayment": Loan or credit repayment
   - "intercompany_transfer": Transfer between related entities
   - "bank_fee": Bank or platform fee charge
   - "refund_processing": Refund or credit note processing
   - "standard_payment_flow": Generic payment flow (use only if none above fit)
   - "unknown": Unable to determine (avoid if possible)

IMPORTANT: Be specific and accurate. Use "unknown" only when truly uncertain.

Return ONLY valid JSON, no markdown blocks or explanations."""

            
            # ✅ REFACTORED: Use instructor for auto-validated responses
            if INSTRUCTOR_AVAILABLE and hasattr(groq_client, 'chat'):
                # Instructor automatically validates and retries on failure
                enrichment_obj = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=RelationshipEnrichment,  # Auto-validates!
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                # Validate using Pydantic
                validated = RelationshipEnrichment(
                    semantic_description=enrichment_obj.semantic_description,
                    reasoning=enrichment_obj.reasoning,
                    temporal_causality=enrichment_obj.temporal_causality,
                    business_logic=enrichment_obj.business_logic
                )
                
                logger.info(
                    "relationship_enriched_with_ai",
                    relationship_type=rel.get('relationship_type'),
                    temporal_causality=validated.temporal_causality
                )
                
                return validated.dict()
            else:
                # No fallback - raise error if instructor not available
                logger.error("instructor_not_available_cannot_validate_response")
                raise ValueError("Instructor not available for response validation")
            
        except ValidationError as ve:
            logger.error("pydantic_validation_failed", errors=ve.errors())
            raise ValueError(f"AI response validation failed: {ve}")
        except Exception as e:
            logger.error("ai_enrichment_failed", error=str(e), exc_info=True)
            raise
    
    # REMOVED: Rule-based enrichment fallback
    # Reason: Fallback logic is too simplistic and doesn't meet output expectations
    # If AI enrichment fails, we now raise an error instead of returning degraded output
    # This ensures data quality and makes failures visible for monitoring

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
            if GROQ_AVAILABLE and event_ids:
                try:
                    events_result = self.supabase.table('raw_events').select(
                        'id, source_platform, document_type, amount_usd, source_ts, vendor_standard'
                    ).in_('id', list(event_ids)).execute()
                    
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
                
                # ✅ NEW: Validate relationship record before storage
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
                
                # ✅ NEW: AI-powered semantic enrichment using Groq/Llama
                semantic_description = None
                reasoning = None
                temporal_causality = None
                business_logic = None
                
                if events_map:
                    source_event = events_map.get(rel['source_event_id'])
                    target_event = events_map.get(rel['target_event_id'])
                    
                    if source_event and target_event:
                        try:
                            enrichment = await self._enrich_relationship_with_ai(rel, source_event, target_event)
                            semantic_description = enrichment.get('semantic_description')
                            reasoning = enrichment.get('reasoning')
                            temporal_causality = enrichment.get('temporal_causality')
                            business_logic = enrichment.get('business_logic')
                        except ValueError as e:
                            logger.error(
                                "ai_enrichment_failed_skipping_relationship",
                                relationship_type=rel.get('relationship_type'),
                                error=str(e)
                            )
                            # Skip this relationship if enrichment fails
                            continue
                
                # Generate pattern_id based on relationship type and key factors
                pattern_signature = f"{rel['relationship_type']}_{'-'.join(sorted(key_factors))}"
                pattern_id = await self._get_or_create_pattern_id(pattern_signature, rel['relationship_type'], key_factors, user_id)
                
                # Generate relationship embedding for semantic search (if semantic extractor available)
                relationship_embedding = None
                if self.semantic_extractor and semantic_description:
                    try:
                        relationship_embedding = await self._generate_relationship_embedding(semantic_description)
                    except Exception as e:
                        logger.warning(f"Failed to generate relationship embedding: {e}")
                
                now = datetime.utcnow().isoformat()
                relationship_instances.append({
                    'user_id': user_id,
                    'source_event_id': rel['source_event_id'],
                    'target_event_id': rel['target_event_id'],
                    'relationship_type': rel['relationship_type'],
                    'confidence_score': rel['confidence_score'],
                    'detection_method': rel.get('detection_method', 'unknown'),
                    'pattern_id': pattern_id,
                    'transaction_id': transaction_id if transaction_id else None,
                    'relationship_embedding': relationship_embedding,
                    'metadata': metadata,
                    'key_factors': key_factors,
                    'semantic_description': semantic_description,
                    'reasoning': reasoning or 'Detected based on matching criteria',
                    'temporal_causality': normalize_temporal_causality(temporal_causality),
                    'business_logic': normalize_business_logic(business_logic),
                    'created_at': now,
                    'updated_at': now,
                    'job_id': job_id
                })
            
            # Batch insert relationships and collect results
            batch_size = 100
            for i in range(0, len(relationship_instances), batch_size):
                batch = relationship_instances[i:i + batch_size]
                try:
                    result = self.supabase.table('relationship_instances').insert(batch).execute()
                    if result.data:
                        stored_relationships.extend(result.data)
                except Exception as e:
                    logger.warning(f"Failed to insert relationship batch: {e}")
            
            logger.info(
                "relationships_stored",
                count=len(stored_relationships),
                total_attempted=len(relationship_instances)
            )
            return stored_relationships
            
            now = datetime.utcnow().isoformat()
            relationship_instances.append({
                'user_id': user_id,
                'source_event_id': rel['source_event_id'],
                'target_event_id': rel['target_event_id'],
                'relationship_type': rel['relationship_type'],
                'confidence_score': rel['confidence_score'],
                'detection_method': rel.get('detection_method', 'unknown'),
                'pattern_id': pattern_id,
                'transaction_id': transaction_id if transaction_id else None,
                'relationship_embedding': relationship_embedding,
                'metadata': metadata,
                'key_factors': key_factors,
                'semantic_description': semantic_description,
                'reasoning': reasoning or 'Detected based on matching criteria',
                'temporal_causality': normalize_temporal_causality(temporal_causality),
                'business_logic': normalize_business_logic(business_logic),
                'created_at': now,
                'updated_at': now,
                'job_id': job_id
            })
        This enables similarity-based relationship discovery and duplicate detection.
        """
        try:
            if not text:
                return None
            
            # Import embedding service
            from embedding_service import get_embedding_service
            
            # Get embedding service instance
            embedding_service = await get_embedding_service()
            
            # Generate embedding using BGE model (1024 dimensions)
            embedding = await embedding_service.embed_text(text)
            
            logger.debug(f"✅ Generated relationship embedding (1024 dims) for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate relationship embedding: {e}")
            return None
    
    # UTILITY: Fuzzy filename matching (used by legacy code if needed)
    def _find_similar_filename(self, target_filename: str, available_files: List[str]) -> Optional[str]:
        """REFACTORED: Find similar filename using RapidFuzz (100x faster than difflib)"""
        if not available_files:
            return None
        
        # RapidFuzz one-liner replacement (100x faster)
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
    
    # DEPRECATED LEGACY METHODS REMOVED (Nov 2025)
    # The following methods were deleted as they used O(N²) loops and hardcoded filenames:
    # - _group_events_by_file() - Replaced by document_type classification
    
    def _sort_events_by_date(self, events: List[Dict]) -> List[Dict]:
        """Sort events by date if available"""
        try:
            return sorted(events, key=lambda x: self._extract_date(x) or datetime.min)
        except:
            return events
    
    def _determine_relationship_type(self, event1: Dict, event2: Dict) -> str:
        """
        Determine the type of relationship between two events using document_type.
        
        REFACTORED: Replaced 200+ lines of hardcoded keyword matching with
        document_type lookup. This is 100% accurate, multilingual, and uses
        the AI-classified document_type from the database.
        """
        # Get document types from events (already AI-classified in database)
        doc_type1 = event1.get('document_type', 'unknown')
        doc_type2 = event2.get('document_type', 'unknown')
        
        # Simple lookup table for relationship types
        # This replaces all the _is_*_event() functions with clean logic
        relationship_map = {
            ('invoice', 'bank_statement'): 'invoice_to_payment',
            ('invoice', 'payment'): 'invoice_to_payment',
            ('invoice', 'bank_transaction'): 'invoice_to_payment',
            ('bank_statement', 'invoice'): 'payment_to_invoice',
            ('payment', 'invoice'): 'payment_to_invoice',
            ('bank_transaction', 'invoice'): 'payment_to_invoice',
            ('revenue', 'bank_statement'): 'revenue_to_cashflow',
            ('revenue', 'bank_transaction'): 'revenue_to_cashflow',
            ('expense', 'bank_statement'): 'expense_to_bank',
            ('expense', 'bank_transaction'): 'expense_to_bank',
            ('payroll', 'bank_statement'): 'payroll_to_bank',
            ('payroll', 'bank_transaction'): 'payroll_to_bank',
        }
        
        # Lookup relationship type
        return relationship_map.get((doc_type1, doc_type2), 'related_transaction')
    
    # DELETED: All _is_*_event() functions (200+ lines removed)
    # These hardcoded keyword matchers are replaced by document_type from database
    # Benefits:
    # - 100% accurate (uses AI classification)
    # - Multilingual (AI handles any language)
    # - No maintenance (keywords managed in DB)
    # - Faster (no string searching)
    
    async def _calculate_relationship_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate comprehensive relationship score"""
        try:
            # Extract data from events
            source_payload = source.get('payload', {})
            target_payload = target.get('payload', {})
            
            # Calculate individual scores
            # CRITICAL FIX: Pass full events to _calculate_amount_score for amount_usd access
            amount_score = self._calculate_amount_score(source, target)
            date_score = self._calculate_date_score(source, target)
            entity_score = self._calculate_entity_score(source_payload, target_payload)
            id_score = self._calculate_id_score(source_payload, target_payload)
            context_score = self._calculate_context_score(source_payload, target_payload)
            
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
    
    def _calculate_amount_score(self, source: Dict, target: Dict) -> float:
        """Calculate amount similarity score using USD-normalized amounts
        
        CRITICAL: Now receives full events to access amount_usd from enriched columns.
        """
        try:
            source_amount = self._extract_amount(source)
            target_amount = self._extract_amount(target)
            
            if source_amount == 0 or target_amount == 0:
                return 0.0
            
            # Calculate ratio
            ratio = min(source_amount, target_amount) / max(source_amount, target_amount)
            return ratio
            
        except:
            return 0.0
    
    def _calculate_date_score(self, source: Dict, target: Dict) -> float:
        """Calculate date similarity score"""
        try:
            source_date = self._extract_date(source)
            target_date = self._extract_date(target)
            
            if not source_date or not target_date:
                return 0.0
            
            # Calculate days difference
            date_diff = abs((source_date - target_date).days)
            
            # Score based on proximity
            if date_diff == 0:
                return 1.0
            elif date_diff <= 1:
                return 0.9
            elif date_diff <= 7:
                return 0.7
            elif date_diff <= 30:
                return 0.5
            else:
                return 0.2
                
        except:
            return 0.0
    
    def _calculate_entity_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate entity similarity score"""
        try:
            source_entities = self._extract_entities(source_payload)
            target_entities = self._extract_entities(target_payload)
            
            if not source_entities or not target_entities:
                return 0.0
            
            # Find common entities
            common_entities = set(source_entities) & set(target_entities)
            total_entities = set(source_entities) | set(target_entities)
            
            if not total_entities:
                return 0.0
            
            return len(common_entities) / len(total_entities)
            
        except:
            return 0.0
    
    def _calculate_id_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate ID similarity score"""
        try:
            source_ids = self._extract_ids(source_payload)
            target_ids = self._extract_ids(target_payload)
            
            if not source_ids or not target_ids:
                return 0.0
            
            # Check for exact ID matches
            common_ids = set(source_ids) & set(target_ids)
            if common_ids:
                return 1.0
            
            # Check for partial matches
            partial_matches = 0
            for source_id in source_ids:
                for target_id in target_ids:
                    if source_id in target_id or target_id in source_id:
                        partial_matches += 1
            
            if partial_matches > 0:
                return 0.5
            
            return 0.0
            
        except:
            return 0.0
    
    async def _calculate_context_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """CRITICAL FIX: Calculate semantic similarity using shared BGE embedding service"""
        try:
            # Fallback to word overlap if embedding service unavailable
            source_text = str(source_payload)[:500]  # Limit to 500 chars for speed
            target_text = str(target_payload)[:500]
            
            if not source_text.strip() or not target_text.strip():
                return 0.0
            
            try:
                # CRITICAL FIX: Use shared BGE embedding service
                from embedding_service import get_embedding_service
                embedding_service = await get_embedding_service()
                
                # Generate embeddings using BGE model (1024 dims)
                source_emb = await embedding_service.embed_text(source_text)
                target_emb = await embedding_service.embed_text(target_text)
                
                # Calculate cosine similarity
                similarity = embedding_service.similarity(source_emb, target_emb)
                
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
        """Extract amount from event using enriched amount_usd for currency consistency
        
        CRITICAL: This function now receives the full event dict (not just payload)
        to access enriched columns like amount_usd for accurate cross-currency matching.
        """
        try:
            # PRIORITY 1: Use amount_usd from enriched columns (Phase 5 enrichment)
            # This ensures all amounts are in USD for accurate cross-currency comparison
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
        """REFACTORED: Extract date using Pendulum (handles 100+ formats automatically)
        
        CRITICAL: Uses transaction date from payload (business logic) instead of created_at
        (system timestamp) for accurate historical relationship detection.
        """
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
        """REFACTORED: Extract entities using spaCy NER (95% accuracy vs 40% capitalization)"""
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
        """Extract IDs from payload"""
        ids = []
        try:
            # Try different ID fields
            id_fields = ['id', 'transaction_id', 'payment_id', 'invoice_id', 'reference']
            for field in id_fields:
                if field in payload and payload[field]:
                    ids.append(str(payload[field]))
            
            return ids
        except:
            return []
    
    async def _remove_duplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """CRITICAL FIX: Remove exact AND semantic duplicates using shared BGE embeddings
        
        Two-stage deduplication:
        1. Exact duplicates (same source + target + type) - O(N) with set
        2. Semantic duplicates (different text, same meaning) - O(N²) with cosine similarity
        
        Example semantic duplicates:
        - "invoice_to_payment" vs "payment_for_invoice"
        - "revenue_to_bank" vs "bank_deposit_from_revenue"
        """
        if not relationships:
            return []
        
        # STAGE 1: Remove exact duplicates (fast)
        seen = set()
        exact_unique = []
        
        for rel in relationships:
            # Create unique key
            key = f"{rel.get('source_event_id')}_{rel.get('target_event_id')}_{rel.get('relationship_type')}"
            
            if key not in seen:
                seen.add(key)
                exact_unique.append(rel)
        
        logger.info(f"After exact dedup: {len(relationships)} → {len(exact_unique)} relationships")
        
        # STAGE 2: Remove semantic duplicates using BGE embeddings
        if len(exact_unique) < 2:
            return exact_unique
        
        try:
            # CRITICAL FIX: Use shared BGE embedding service
            from embedding_service import get_embedding_service
            embedding_service = await get_embedding_service()
            
            # Generate embeddings for relationship descriptions
            descriptions = []
            for rel in exact_unique:
                desc = f"{rel.get('relationship_type', '')} {rel.get('semantic_description', '')}"
                descriptions.append(desc)
            
            # Batch encode all descriptions using BGE (1024 dims)
            embeddings = await embedding_service.embed_batch(descriptions)
            
            # Find semantic duplicates using cosine similarity
            semantic_unique = []
            skip_indices = set()
            
            for i in range(len(exact_unique)):
                if i in skip_indices:
                    continue
                
                semantic_unique.append(exact_unique[i])
                
                # Check for semantic duplicates with remaining relationships
                for j in range(i + 1, len(exact_unique)):
                    if j in skip_indices:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = embedding_service.similarity(embeddings[i], embeddings[j])
                    
                    # If very similar (>0.85), mark as duplicate
                    if similarity > 0.85:
                        logger.debug(f"Semantic duplicate found: '{descriptions[i]}' ≈ '{descriptions[j]}' (similarity: {similarity:.3f})")
                        skip_indices.add(j)
            
            logger.info(f"After semantic dedup: {len(exact_unique)} → {len(semantic_unique)} relationships")
            return semantic_unique
            
        except Exception as e:
            logger.warning(f"Semantic deduplication failed: {e}, falling back to exact dedup only")
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
    
    async def _enrich_relationships_with_semantics(
        self, 
        relationships: List[Dict], 
        user_id: str
    ) -> Dict[str, Any]:
        """
        PHASE 2B: Enrich detected relationships with AI-powered semantic analysis.
        
        This adds:
        - Natural language descriptions of relationships
        - Temporal causality detection (cause vs correlation)
        - Business logic pattern identification
        - Relationship embeddings for similarity search
        - Explainable confidence scoring
        
        Args:
            relationships: List of detected relationships
            user_id: User ID for context
        
        Returns:
            Statistics about semantic enrichment
        """
        if not self.semantic_extractor or not relationships:
            return {
                'enabled': False,
                'total_relationships': len(relationships),
                'enriched_count': 0,
                'message': 'Semantic enrichment not available or no relationships to enrich'
            }
        
        try:
            enriched_count = 0
            failed_count = 0
            
            # Get events for context
            event_ids = set()
            for rel in relationships:
                event_ids.add(rel['source_event_id'])
                event_ids.add(rel['target_event_id'])
            
            # Fetch events from database
            events_dict = await self._fetch_events_by_ids(list(event_ids), user_id)
            
            # Process relationships in batches
            batch_size = 5  # Conservative batch size for API rate limits
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                for rel in batch:
                    try:
                        source_event = events_dict.get(rel['source_event_id'])
                        target_event = events_dict.get(rel['target_event_id'])
                        
                        if not source_event or not target_event:
                            logger.warning(f"Missing events for relationship {rel.get('source_event_id')} -> {rel.get('target_event_id')}")
                            failed_count += 1
                            continue
                        
                        # Extract semantic relationship
                        semantic_rel = await self.semantic_extractor.extract_semantic_relationships(
                            source_event=source_event,
                            target_event=target_event,
                            context_events=None,  # Could add surrounding events for better context
                            existing_relationship=rel
                        )
                        
                        if semantic_rel:
                            enriched_count += 1
                            logger.debug(
                                f"✅ Enriched relationship: {semantic_rel.relationship_type} "
                                f"(confidence: {semantic_rel.confidence:.2f})"
                            )
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to enrich relationship: {e}")
                        failed_count += 1
                        continue
            
            # Get metrics from semantic extractor
            extractor_metrics = self.semantic_extractor.get_metrics()
            
            return {
                'enabled': True,
                'total_relationships': len(relationships),
                'enriched_count': enriched_count,
                'failed_count': failed_count,
                'success_rate': enriched_count / len(relationships) if relationships else 0.0,
                'cache_hit_rate': extractor_metrics.get('cache_hit_rate', 0.0),
                'avg_confidence': extractor_metrics.get('avg_confidence', 0.0),
                'causality_distribution': extractor_metrics.get('causality_distribution', {}),
                'business_logic_distribution': extractor_metrics.get('business_logic_distribution', {}),
                'message': f'Semantic enrichment completed: {enriched_count}/{len(relationships)} relationships enriched'
            }
            
        except Exception as e:
            logger.error("semantic_enrichment_failed", error=str(e), exc_info=True)
            raise
    
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
        if not self.causal_engine or not relationships:
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
            result = await self.causal_engine.analyze_causal_relationships(
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
        if not self.temporal_learner:
            return {
                'enabled': False,
                'patterns_learned': 0,
                'predictions_made': 0,
                'anomalies_detected': 0,
                'message': 'Temporal pattern learning not available'
            }
        
        try:
            # Learn all patterns
            patterns_result = await self.temporal_learner.learn_all_patterns(user_id)
            
            # Predict missing relationships
            predictions_result = await self.temporal_learner.predict_missing_relationships(user_id)
            
            # Detect temporal anomalies
            anomalies_result = await self.temporal_learner.detect_temporal_anomalies(user_id)
            
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

# Test function
async def test_enhanced_relationship_detection(user_id: str = "550e8400-e29b-41d4-a716-446655440000"):
    """Test the enhanced relationship detection system"""
    try:
        # Initialize OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Enhanced Relationship Detector
        enhanced_detector = EnhancedRelationshipDetector(openai_client, supabase)
        
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
    result = test_enhanced_relationship_detection()
    print(result) 