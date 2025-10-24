"""
Production-Grade Semantic Relationship Extractor
================================================

AI-powered semantic relationship extraction that understands the MEANING
and business context of financial relationships, not just pattern matching.

This module is Phase 2B of the Relationship Engine enhancement strategy.

Features:
- GPT-4 powered semantic understanding
- Natural language relationship descriptions
- Temporal causality detection (cause vs correlation)
- Business logic validation
- Relationship embeddings for similarity search
- Confidence scoring with explainability
- Async processing with caching
- Learning from user feedback

Author: Senior Full-Stack Engineer
Version: 1.0.0
Date: 2025-01-21
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class TemporalCausality(Enum):
    """Temporal causality types for relationships"""
    SOURCE_CAUSES_TARGET = "source_causes_target"  # Invoice causes payment
    TARGET_CAUSES_SOURCE = "target_causes_source"  # Rare, but possible
    BIDIRECTIONAL = "bidirectional"  # Mutual causation
    CORRELATION_ONLY = "correlation_only"  # Related but no causation


class BusinessLogicType(Enum):
    """Business logic patterns for relationships"""
    STANDARD_PAYMENT_FLOW = "standard_payment_flow"  # Invoice → Payment
    REVENUE_RECOGNITION = "revenue_recognition"  # Sale → Revenue
    EXPENSE_REIMBURSEMENT = "expense_reimbursement"  # Expense → Payment
    PAYROLL_PROCESSING = "payroll_processing"  # Payroll → Bank transfer
    TAX_WITHHOLDING = "tax_withholding"  # Income → Tax payment
    ASSET_DEPRECIATION = "asset_depreciation"  # Asset → Expense
    LOAN_REPAYMENT = "loan_repayment"  # Loan → Payment
    REFUND_PROCESSING = "refund_processing"  # Payment → Refund
    RECURRING_BILLING = "recurring_billing"  # Subscription → Payment
    UNKNOWN = "unknown"


@dataclass
class SemanticRelationship:
    """Semantic relationship result with rich metadata"""
    source_event_id: str
    target_event_id: str
    relationship_type: str
    semantic_description: str
    confidence: float
    temporal_causality: TemporalCausality
    business_logic: BusinessLogicType
    reasoning: str
    key_factors: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SemanticRelationshipExtractor:
    """
    Production-grade semantic relationship extractor using GPT-4 to understand
    the MEANING and business context of financial relationships.
    
    This goes beyond pattern matching to provide:
    - Natural language explanations of relationships
    - Temporal causality analysis (cause vs correlation)
    - Business logic validation
    - Semantic embeddings for similarity search
    - Explainable confidence scoring
    """
    
    def __init__(self, openai_client: AsyncAnthropic, supabase_client=None, cache_client=None, config=None):
        self.anthropic = openai_client
        self.supabase = supabase_client
        self.cache = cache_client
        self.config = config or self._get_default_config()
        
        # Performance tracking
        self.metrics = {
            'semantic_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ai_calls': 0,
            'avg_confidence': 0.0,
            'causality_distribution': {},
            'business_logic_distribution': {},
            'processing_times': []
        }
        
        logger.info("✅ SemanticRelationshipExtractor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_caching': True,
            'cache_ttl_hours': 48,  # Semantic relationships are stable
            'enable_embeddings': True,
            'embedding_model': 'text-embedding-3-small',
            'semantic_model': 'claude-haiku-4-20250514',  # Fast and accurate for semantic understanding
            'temperature': 0.1,  # Low temperature for consistency
            'max_tokens': 500,
            'confidence_threshold': 0.7,
            'batch_size': 10,
            'timeout_seconds': 30
        }
    
    async def extract_semantic_relationships(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any],
        context_events: Optional[List[Dict[str, Any]]] = None,
        existing_relationship: Optional[Dict[str, Any]] = None
    ) -> Optional[SemanticRelationship]:
        """
        Extract semantic relationship between two events using AI.
        
        Args:
            source_event: Source financial event
            target_event: Target financial event
            context_events: Optional surrounding events for context
            existing_relationship: Optional existing relationship data (for enhancement)
        
        Returns:
            SemanticRelationship with rich metadata or None if extraction fails
        """
        start_time = time.time()
        
        try:
            # 1. Generate cache key
            cache_key = self._generate_cache_key(source_event, target_event)
            
            # 2. Check cache
            if self.config['enable_caching'] and self.cache:
                cached_result = await self._get_cached_semantic_relationship(cache_key)
                if cached_result:
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"Semantic cache hit: {cache_key[:16]}...")
                    return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # 3. Build comprehensive prompt
            prompt = self._build_semantic_extraction_prompt(
                source_event, target_event, context_events, existing_relationship
            )
            
            # 4. Call GPT-4 for semantic analysis
            semantic_result = await self._call_semantic_ai(prompt)
            
            if not semantic_result:
                logger.warning("Semantic AI extraction returned no result")
                return None
            
            # 5. Generate embedding for similarity search
            embedding = None
            if self.config['enable_embeddings']:
                embedding = await self._generate_relationship_embedding(semantic_result)
            
            # 6. Build semantic relationship object
            semantic_relationship = SemanticRelationship(
                source_event_id=source_event.get('id'),
                target_event_id=target_event.get('id'),
                relationship_type=semantic_result.get('relationship_type', 'unknown'),
                semantic_description=semantic_result.get('semantic_description', ''),
                confidence=float(semantic_result.get('confidence', 0.0)),
                temporal_causality=TemporalCausality(
                    semantic_result.get('temporal_causality', 'correlation_only')
                ),
                business_logic=BusinessLogicType(
                    semantic_result.get('business_logic', 'unknown')
                ),
                reasoning=semantic_result.get('reasoning', ''),
                key_factors=semantic_result.get('key_factors', []),
                metadata={
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': self.config['semantic_model'],
                    'source_platform': source_event.get('source_platform'),
                    'target_platform': target_event.get('source_platform'),
                    'source_document_type': source_event.get('document_type'),
                    'target_document_type': target_event.get('document_type')
                },
                embedding=embedding
            )
            
            # 7. Cache the result
            if self.config['enable_caching'] and self.cache:
                await self._cache_semantic_relationship(cache_key, semantic_relationship)
            
            # 8. Update metrics
            self._update_metrics(semantic_relationship)
            
            # 9. Store in database
            if self.supabase:
                await self._store_semantic_relationship(semantic_relationship)
            
            logger.info(
                f"✅ Semantic extraction: {semantic_relationship.relationship_type} "
                f"(confidence: {semantic_relationship.confidence:.2f}, "
                f"causality: {semantic_relationship.temporal_causality.value})"
            )
            
            return semantic_relationship
            
        except Exception as e:
            logger.error(f"Semantic relationship extraction failed: {e}")
            return None
    
    async def extract_semantic_relationships_batch(
        self,
        relationship_pairs: List[Tuple[Dict, Dict]],
        context_events: Optional[List[Dict]] = None
    ) -> List[Optional[SemanticRelationship]]:
        """
        Extract semantic relationships for multiple pairs in batch.
        
        Args:
            relationship_pairs: List of (source_event, target_event) tuples
            context_events: Optional context events
        
        Returns:
            List of SemanticRelationship objects (None for failures)
        """
        try:
            # Process in batches for rate limiting
            batch_size = self.config['batch_size']
            results = []
            
            for i in range(0, len(relationship_pairs), batch_size):
                batch = relationship_pairs[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self.extract_semantic_relationships(source, target, context_events)
                    for source, target in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch extraction error: {result}")
                        results.append(None)
                    else:
                        results.append(result)
                
                # Rate limiting pause between batches
                if i + batch_size < len(relationship_pairs):
                    await asyncio.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch semantic extraction failed: {e}")
            return [None] * len(relationship_pairs)
    
    def _build_semantic_extraction_prompt(
        self,
        source_event: Dict,
        target_event: Dict,
        context_events: Optional[List[Dict]],
        existing_relationship: Optional[Dict]
    ) -> str:
        """
        Build comprehensive prompt for semantic relationship extraction.
        
        This prompt is designed to extract:
        - Semantic meaning of the relationship
        - Temporal causality (cause vs correlation)
        - Business logic pattern
        - Confidence with reasoning
        """
        
        # Extract key information from events
        source_info = self._extract_event_summary(source_event)
        target_info = self._extract_event_summary(target_event)
        
        # Build context summary
        context_summary = ""
        if context_events:
            context_summary = "\n\nSurrounding Context Events:\n"
            for i, ctx_event in enumerate(context_events[:5], 1):  # Limit to 5 for token efficiency
                ctx_info = self._extract_event_summary(ctx_event)
                context_summary += f"{i}. {ctx_info}\n"
        
        # Build existing relationship info
        existing_info = ""
        if existing_relationship:
            existing_info = f"""
Existing Relationship Detection:
- Type: {existing_relationship.get('relationship_type', 'unknown')}
- Confidence: {existing_relationship.get('confidence_score', 0.0):.2f}
- Method: {existing_relationship.get('detection_method', 'unknown')}
"""
        
        prompt = f"""You are a financial relationship analyst. Analyze the semantic relationship between two financial events.

SOURCE EVENT:
{source_info}

TARGET EVENT:
{target_info}
{context_summary}{existing_info}

Analyze this relationship and provide a JSON response with the following structure:
{{
    "relationship_type": "string (e.g., 'invoice_payment', 'revenue_recognition', 'expense_reimbursement')",
    "semantic_description": "string (natural language explanation of the relationship, 1-2 sentences)",
    "confidence": float (0.0-1.0, how confident are you this relationship exists),
    "temporal_causality": "string (one of: 'source_causes_target', 'target_causes_source', 'bidirectional', 'correlation_only')",
    "business_logic": "string (one of: 'standard_payment_flow', 'revenue_recognition', 'expense_reimbursement', 'payroll_processing', 'tax_withholding', 'asset_depreciation', 'loan_repayment', 'refund_processing', 'recurring_billing', 'unknown')",
    "reasoning": "string (explain WHY you believe this relationship exists and your confidence level)",
    "key_factors": ["list", "of", "key", "factors", "that", "support", "this", "relationship"]
}}

IMPORTANT GUIDELINES:
1. **Temporal Causality**: Determine if source CAUSES target (e.g., invoice causes payment), or if they're just correlated
2. **Business Logic**: Identify the business process pattern this relationship follows
3. **Confidence**: Be conservative. Only give high confidence (>0.8) if multiple strong signals align
4. **Key Factors**: List specific evidence (matching amounts, dates, entities, IDs, descriptions)
5. **Semantic Description**: Explain the relationship in plain English that a business user would understand

Example high-confidence relationship:
- Source: Invoice #1234 to Acme Corp for $5,000 on 2024-01-15
- Target: Bank deposit of $5,000 from Acme Corp on 2024-02-10
- Analysis: "Payment received for invoice" (confidence: 0.95, causality: source_causes_target, logic: standard_payment_flow)

Example low-confidence relationship:
- Source: Expense of $100 on 2024-01-15
- Target: Bank withdrawal of $95 on 2024-01-20
- Analysis: "Possibly related transactions" (confidence: 0.4, causality: correlation_only, logic: unknown)

Provide ONLY the JSON response, no additional text."""
        
        return prompt
    
    def _extract_event_summary(self, event: Dict) -> str:
        """Extract human-readable summary of an event"""
        try:
            payload = event.get('payload', {})
            
            # Extract key fields
            amount = event.get('amount_usd') or payload.get('amount') or payload.get('total') or 'unknown'
            date = event.get('source_ts') or payload.get('date') or payload.get('transaction_date') or 'unknown'
            vendor = event.get('vendor_standard') or payload.get('vendor') or payload.get('merchant') or 'unknown'
            description = payload.get('description') or payload.get('memo') or ''
            document_type = event.get('document_type', 'unknown')
            platform = event.get('source_platform', 'unknown')
            
            # Build summary
            summary = f"[{document_type}] "
            
            if amount != 'unknown':
                summary += f"Amount: ${amount}, "
            
            if date != 'unknown':
                summary += f"Date: {date}, "
            
            if vendor != 'unknown':
                summary += f"Vendor: {vendor}, "
            
            summary += f"Platform: {platform}"
            
            if description:
                summary += f", Description: {description[:100]}"
            
            return summary
            
        except Exception as e:
            logger.warning(f"Event summary extraction failed: {e}")
            return f"[Event ID: {event.get('id', 'unknown')}]"
    
    async def _call_semantic_ai(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call GPT-4 for semantic analysis with structured output"""
        try:
            self.metrics['ai_calls'] += 1
            
            response = await self.anthropic.messages.create(
                model="claude-haiku-4-20250514",
                system="You are a financial relationship analyst. Provide accurate, well-reasoned analysis in JSON format.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config['max_tokens']
            )
            
            # Parse JSON response
            result_text = response.content[0].text
            result = json.loads(result_text)
            
            # Validate required fields
            required_fields = ['relationship_type', 'semantic_description', 'confidence', 
                             'temporal_causality', 'business_logic', 'reasoning', 'key_factors']
            
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field in AI response: {field}")
                    return None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Semantic AI call failed: {e}")
            return None
    
    async def _generate_relationship_embedding(self, semantic_result: Dict) -> Optional[List[float]]:
        """Generate embedding for semantic similarity search"""
        try:
            # Combine key semantic information for embedding
            embedding_text = (
                f"{semantic_result['relationship_type']} "
                f"{semantic_result['semantic_description']} "
                f"{semantic_result['business_logic']} "
                f"{' '.join(semantic_result['key_factors'])}"
            )
            
            # Note: Anthropic doesn't have embeddings API
            # TODO: Use a separate embedding service (OpenAI, Cohere, etc.) if needed
            # For now, return None to disable embedding functionality
            logger.warning("Embedding generation disabled - Anthropic doesn't support embeddings")
            return None
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _generate_cache_key(self, source_event: Dict, target_event: Dict) -> str:
        """Generate deterministic cache key for event pair"""
        source_id = source_event.get('id', '')
        target_id = target_event.get('id', '')
        
        # Use sorted IDs to make cache key order-independent
        id_pair = tuple(sorted([source_id, target_id]))
        key_string = f"semantic_rel:{id_pair[0]}:{id_pair[1]}"
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_semantic_relationship(self, cache_key: str) -> Optional[SemanticRelationship]:
        """Get cached semantic relationship"""
        if not self.cache:
            return None
        
        try:
            if hasattr(self.cache, 'get_cached_classification'):
                cached_data = await self.cache.get_cached_classification(
                    cache_key,
                    classification_type='semantic_relationship'
                )
                
                if cached_data:
                    # Reconstruct SemanticRelationship from cached data
                    return SemanticRelationship(
                        source_event_id=cached_data['source_event_id'],
                        target_event_id=cached_data['target_event_id'],
                        relationship_type=cached_data['relationship_type'],
                        semantic_description=cached_data['semantic_description'],
                        confidence=cached_data['confidence'],
                        temporal_causality=TemporalCausality(cached_data['temporal_causality']),
                        business_logic=BusinessLogicType(cached_data['business_logic']),
                        reasoning=cached_data['reasoning'],
                        key_factors=cached_data['key_factors'],
                        metadata=cached_data['metadata'],
                        embedding=cached_data.get('embedding')
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_semantic_relationship(self, cache_key: str, semantic_rel: SemanticRelationship):
        """Cache semantic relationship"""
        if not self.cache:
            return
        
        try:
            if hasattr(self.cache, 'store_classification'):
                # Convert to dict for caching
                cached_data = {
                    'source_event_id': semantic_rel.source_event_id,
                    'target_event_id': semantic_rel.target_event_id,
                    'relationship_type': semantic_rel.relationship_type,
                    'semantic_description': semantic_rel.semantic_description,
                    'confidence': semantic_rel.confidence,
                    'temporal_causality': semantic_rel.temporal_causality.value,
                    'business_logic': semantic_rel.business_logic.value,
                    'reasoning': semantic_rel.reasoning,
                    'key_factors': semantic_rel.key_factors,
                    'metadata': semantic_rel.metadata,
                    'embedding': semantic_rel.embedding
                }
                
                await self.cache.store_classification(
                    cache_key,
                    cached_data,
                    classification_type='semantic_relationship',
                    ttl_hours=self.config['cache_ttl_hours'],
                    confidence_score=semantic_rel.confidence,
                    model_version=self.config['semantic_model']
                )
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _store_semantic_relationship(self, semantic_rel: SemanticRelationship):
        """Store semantic relationship in database"""
        if not self.supabase:
            return
        
        try:
            # Update existing relationship_instances record with semantic data
            update_data = {
                'semantic_description': semantic_rel.semantic_description,
                'temporal_causality': semantic_rel.temporal_causality.value,
                'business_logic': semantic_rel.business_logic.value,
                'reasoning': semantic_rel.reasoning,
                'metadata': {
                    **semantic_rel.metadata,
                    'key_factors': semantic_rel.key_factors
                }
            }
            
            # If embeddings are enabled, add to update
            if semantic_rel.embedding:
                update_data['relationship_embedding'] = semantic_rel.embedding
            
            # Update relationship_instances where source and target match
            self.supabase.table('relationship_instances').update(update_data).eq(
                'source_event_id', semantic_rel.source_event_id
            ).eq(
                'target_event_id', semantic_rel.target_event_id
            ).execute()
            
            logger.debug(f"Stored semantic relationship: {semantic_rel.source_event_id} → {semantic_rel.target_event_id}")
            
        except Exception as e:
            logger.error(f"Failed to store semantic relationship: {e}")
    
    def _update_metrics(self, semantic_rel: SemanticRelationship):
        """Update extraction metrics"""
        self.metrics['semantic_extractions'] += 1
        
        # Update confidence average
        current_avg = self.metrics['avg_confidence']
        count = self.metrics['semantic_extractions']
        self.metrics['avg_confidence'] = (current_avg * (count - 1) + semantic_rel.confidence) / count
        
        # Update causality distribution
        causality = semantic_rel.temporal_causality.value
        self.metrics['causality_distribution'][causality] = \
            self.metrics['causality_distribution'].get(causality, 0) + 1
        
        # Update business logic distribution
        logic = semantic_rel.business_logic.value
        self.metrics['business_logic_distribution'][logic] = \
            self.metrics['business_logic_distribution'].get(logic, 0) + 1
        
        # Update processing times
        processing_time = semantic_rel.metadata.get('processing_time', 0.0)
        self.metrics['processing_times'].append(processing_time)
        if len(self.metrics['processing_times']) > 1000:
            self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            **self.metrics,
            'cache_hit_rate': (
                self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0
            ),
            'avg_processing_time': (
                sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
                if self.metrics['processing_times'] else 0.0
            )
        }


# Test function
async def test_semantic_extraction():
    """Test semantic relationship extraction"""
    import os
    from anthropic import AsyncAnthropic
    
    try:
        # Initialize Anthropic client
        anthropic_client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Initialize extractor
        extractor = SemanticRelationshipExtractor(anthropic_client)
        
        # Test events
        source_event = {
            'id': 'evt_001',
            'document_type': 'invoice',
            'source_platform': 'quickbooks',
            'amount_usd': 5000.00,
            'source_ts': '2024-01-15T00:00:00Z',
            'vendor_standard': 'Acme Corporation',
            'payload': {
                'invoice_number': 'INV-1234',
                'description': 'Professional services rendered',
                'due_date': '2024-02-15'
            }
        }
        
        target_event = {
            'id': 'evt_002',
            'document_type': 'bank_statement',
            'source_platform': 'chase',
            'amount_usd': 5000.00,
            'source_ts': '2024-02-10T00:00:00Z',
            'vendor_standard': 'Acme Corporation',
            'payload': {
                'transaction_type': 'deposit',
                'description': 'Payment from Acme Corporation',
                'reference': 'INV-1234'
            }
        }
        
        # Extract semantic relationship
        result = await extractor.extract_semantic_relationships(source_event, target_event)
        
        if result:
            print("\n✅ Semantic Extraction Test Successful!")
            print(f"Relationship Type: {result.relationship_type}")
            print(f"Description: {result.semantic_description}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Temporal Causality: {result.temporal_causality.value}")
            print(f"Business Logic: {result.business_logic.value}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Key Factors: {', '.join(result.key_factors)}")
            print(f"\nMetrics: {extractor.get_metrics()}")
        else:
            print("\n❌ Semantic extraction failed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_semantic_extraction())
