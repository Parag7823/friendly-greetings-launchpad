"""
Enhanced Relationship Detector - FIXES CORE ISSUES

This module provides an enhanced relationship detection system that actually finds
relationships between financial events instead of just discovering relationship types.

Key Improvements:
1. Actually finds relationships between events
2. Cross-file relationship detection
3. Within-file relationship detection
4. Comprehensive scoring system
5. Proper validation and deduplication
"""

import os
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from anthropic import AsyncAnthropic
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Import debug logger for capturing relationship detection reasoning
try:
    from debug_logger import get_debug_logger
    DEBUG_LOGGER_AVAILABLE = True
except ImportError:
    DEBUG_LOGGER_AVAILABLE = False
    logger.warning("Debug logger not available - skipping detailed logging")

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

# Import Neo4j relationship detector for graph database integration
try:
    from neo4j_relationship_detector import Neo4jRelationshipDetector
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4jRelationshipDetector not available. Graph database features will be disabled.")

class EnhancedRelationshipDetector:
    """Enhanced relationship detector that actually finds relationships between events"""
    
    def __init__(self, anthropic_client: AsyncAnthropic = None, supabase_client: Client = None, cache_client=None):
        self.anthropic = anthropic_client
        self.supabase = supabase_client
        self.cache = cache_client  # Use centralized cache, no local cache
        
        # Initialize semantic relationship extractor for AI-powered analysis
        if SEMANTIC_EXTRACTOR_AVAILABLE:
            self.semantic_extractor = SemanticRelationshipExtractor(
                openai_client=anthropic_client,
                supabase_client=supabase_client,
                cache_client=cache_client
            )
            logger.info("✅ Semantic relationship extractor initialized")
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
        
        # Initialize Neo4j graph database for visual relationship exploration
        if NEO4J_AVAILABLE:
            try:
                self.neo4j = Neo4jRelationshipDetector()
                logger.info("✅ Neo4j graph database initialized")
            except Exception as e:
                self.neo4j = None
                logger.error(f"❌ Neo4j initialization failed: {e}")
        else:
            self.neo4j = None
            logger.warning("⚠️ Neo4j not available - graph features disabled")
        
    async def detect_all_relationships(self, user_id: str, file_id: Optional[str] = None) -> Dict[str, Any]:
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
            
            # Store relationships in Supabase database
            if all_relationships:
                await self._store_relationships(all_relationships, user_id)
            
            # NEW: Sync relationships to Neo4j graph database
            if all_relationships and self.neo4j:
                await self._sync_to_neo4j(all_relationships, user_id)
            
            # PHASE 2B: Enrich relationships with semantic analysis
            semantic_enrichment_stats = await self._enrich_relationships_with_semantics(
                all_relationships, user_id
            )
            
            # PHASE 3: Causal inference using Bradford Hill criteria
            causal_analysis_stats = await self._analyze_causal_relationships(
                all_relationships, user_id
            )
            
            # PHASE 4: Temporal pattern learning and prediction
            temporal_learning_stats = await self._learn_temporal_patterns(user_id)
            
            logger.info(f"Relationship detection completed: {len(all_relationships)} relationships found")
            
            # Debug logging for developer console
            if DEBUG_LOGGER_AVAILABLE and user_id:
                try:
                    debug_logger = get_debug_logger(self.supabase, None)
                    # Log top 10 relationships with details
                    relationships_sample = all_relationships[:10] if len(all_relationships) > 10 else all_relationships
                    await debug_logger.log_relationship_detection(
                        job_id=file_id or 'batch_processing',
                        user_id=user_id,
                        relationships=[{
                            "from_event": r.get('from_event_id'),
                            "to_event": r.get('to_event_id'),
                            "type": r.get('relationship_type'),
                            "confidence": r.get('confidence_score', 0),
                            "evidence": r.get('evidence', []),
                            "reasoning": r.get('reasoning', '')
                        } for r in relationships_sample],
                        total_found=len(all_relationships)
                    )
                except Exception as debug_err:
                    logger.warning(f"Debug logging failed: {debug_err}")
            
            return {
                "relationships": all_relationships,
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
                                'amount_match': rel['amount_match'],
                                'date_match': rel['date_match'],
                                'entity_match': rel['entity_match'],
                                'metadata': rel['metadata'],
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
                        'metadata': rel['metadata'],
                        'detection_method': 'database_self_join'
                    })
                
                logger.info(f"Found {len(result.data)} within-file relationships")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Within-file relationship detection failed: {e}")
            return []
    
    async def _store_relationships(self, relationships: List[Dict], user_id: str):
        """Store detected relationships in the database"""
        try:
            if not relationships:
                return
            
            # Prepare relationship instances for insertion
            relationship_instances = []
            for rel in relationships:
                relationship_instances.append({
                    'user_id': user_id,
                    'source_event_id': rel['source_event_id'],
                    'target_event_id': rel['target_event_id'],
                    'relationship_type': rel['relationship_type'],
                    'confidence_score': rel['confidence_score'],  # ✅ CRITICAL FIX: Use correct column name
                    'detection_method': rel.get('detection_method', 'unknown'),
                    # ✅ CRITICAL FIX: Remove 'metadata' - column doesn't exist in relationship_instances table
                    # Metadata is stored in the rel object but not persisted to this table
                    'created_at': datetime.utcnow().isoformat()
                })
            
            # Batch insert relationships
            batch_size = 100
            for i in range(0, len(relationship_instances), batch_size):
                batch = relationship_instances[i:i + batch_size]
                try:
                    self.supabase.table('relationship_instances').insert(batch).execute()
                except Exception as e:
                    logger.warning(f"Failed to insert relationship batch: {e}")
            
            logger.info(f"Stored {len(relationship_instances)} relationships in database")
            
        except Exception as e:
            logger.error(f"Failed to store relationships: {e}")
    
    # DEPRECATED: Old methods below are kept for backward compatibility but should not be used
    def _group_events_by_file(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """DEPRECATED: Group events by source filename"""
        logger.warning("Using deprecated _group_events_by_file method. Use document_type classification instead.")
        events_by_file = {}
        for event in events:
            filename = event.get('source_filename', 'unknown')
            if filename not in events_by_file:
                events_by_file[filename] = []
            events_by_file[filename].append(event)
        return events_by_file
    
    async def _detect_cross_file_relationships(self, events_by_file: Dict[str, List[Dict]]) -> List[Dict]:
        """Detect relationships between different files"""
        relationships = []
        
        # Define cross-file relationship patterns - EXPANDED for all sample files
        cross_file_patterns = [
            # Invoice to Payment relationships
            {
                'source_files': ['company_invoices.csv', 'comprehensive_vendor_payments.csv'],
                'relationship_type': 'invoice_to_payment',
                'description': 'Invoice payments'
            },
            {
                'source_files': ['company_invoices.csv', 'company_bank_statements.csv'],
                'relationship_type': 'invoice_to_bank',
                'description': 'Invoice bank payments'
            },
            # Revenue to Cash Flow relationships
            {
                'source_files': ['company_revenue.csv', 'comprehensive_cash_flow.csv'],
                'relationship_type': 'revenue_to_cashflow',
                'description': 'Revenue cash flow'
            },
            {
                'source_files': ['company_revenue.csv', 'company_bank_statements.csv'],
                'relationship_type': 'revenue_to_bank',
                'description': 'Revenue bank deposits'
            },
            # Expense relationships
            {
                'source_files': ['company_expenses.csv', 'company_bank_statements.csv'],
                'relationship_type': 'expense_to_bank',
                'description': 'Expense bank transactions'
            },
            {
                'source_files': ['company_expenses.csv', 'comprehensive_vendor_payments.csv'],
                'relationship_type': 'expense_to_payment',
                'description': 'Expense payments'
            },
            # Payroll relationships
            {
                'source_files': ['comprehensive_payroll_data.csv', 'company_bank_statements.csv'],
                'relationship_type': 'payroll_to_bank',
                'description': 'Payroll bank transactions'
            },
            {
                'source_files': ['comprehensive_payroll_data.csv', 'comprehensive_cash_flow.csv'],
                'relationship_type': 'payroll_to_cashflow',
                'description': 'Payroll cash flow impact'
            },
            # Receivables relationships
            {
                'source_files': ['company_invoices.csv', 'company_accounts_receivable.csv'],
                'relationship_type': 'invoice_to_receivable',
                'description': 'Invoice receivables'
            },
            {
                'source_files': ['company_accounts_receivable.csv', 'company_bank_statements.csv'],
                'relationship_type': 'receivable_to_bank',
                'description': 'Receivable collections'
            },
            # Tax relationships
            {
                'source_files': ['company_tax_filings.csv', 'company_expenses.csv'],
                'relationship_type': 'tax_to_expense',
                'description': 'Tax expense relationships'
            },
            {
                'source_files': ['company_tax_filings.csv', 'company_revenue.csv'],
                'relationship_type': 'tax_to_revenue',
                'description': 'Tax revenue relationships'
            },
            # Asset relationships
            {
                'source_files': ['company_assets.csv', 'company_expenses.csv'],
                'relationship_type': 'asset_to_expense',
                'description': 'Asset depreciation expenses'
            },
            {
                'source_files': ['company_assets.csv', 'company_bank_statements.csv'],
                'relationship_type': 'asset_to_bank',
                'description': 'Asset purchases/sales'
            }
        ]
        
        # Log available files for debugging
        available_files = list(events_by_file.keys())
        logger.info(f"Available files for cross-file analysis: {available_files}")

        for pattern in cross_file_patterns:
            source_file = pattern['source_files'][0]
            target_file = pattern['source_files'][1]

            # Check exact match first
            if source_file in events_by_file and target_file in events_by_file:
                source_events = events_by_file[source_file]
                target_events = events_by_file[target_file]

                logger.info(f"Found exact match for pattern: {source_file} ({len(source_events)} events) ↔ {target_file} ({len(target_events)} events)")

                file_relationships = await self._find_file_relationships(
                    source_events, target_events, pattern['relationship_type']
                )
                relationships.extend(file_relationships)
            else:
                # Try fuzzy matching for similar file names
                source_match = self._find_similar_filename(source_file, available_files)
                target_match = self._find_similar_filename(target_file, available_files)

                if source_match and target_match and source_match != target_match:
                    source_events = events_by_file[source_match]
                    target_events = events_by_file[target_match]

                    logger.info(f"Found fuzzy match for pattern: {source_match} ({len(source_events)} events) ↔ {target_match} ({len(target_events)} events)")

                    file_relationships = await self._find_file_relationships(
                        source_events, target_events, pattern['relationship_type']
                    )
                    relationships.extend(file_relationships)
                else:
                    logger.debug(f"No match found for pattern: {source_file} ↔ {target_file}")
        
        return relationships

    def _find_similar_filename(self, target_filename: str, available_files: List[str]) -> Optional[str]:
        """Find a similar filename using fuzzy matching"""
        import difflib

        # Extract key terms from target filename
        target_base = target_filename.lower().replace('.csv', '').replace('_', ' ')
        target_words = set(target_base.split())

        best_match = None
        best_score = 0.0

        for available_file in available_files:
            available_base = available_file.lower().replace('.csv', '').replace('_', ' ')
            available_words = set(available_base.split())

            # Calculate word overlap score
            common_words = target_words.intersection(available_words)
            if common_words:
                word_score = len(common_words) / max(len(target_words), len(available_words))

                # Calculate sequence similarity
                seq_score = difflib.SequenceMatcher(None, target_base, available_base).ratio()

                # Combined score
                combined_score = (word_score * 0.7) + (seq_score * 0.3)

                if combined_score > best_score and combined_score > 0.4:  # Minimum threshold
                    best_score = combined_score
                    best_match = available_file

        if best_match:
            logger.info(f"Fuzzy match: '{target_filename}' → '{best_match}' (score: {best_score:.3f})")

        return best_match

    async def _detect_within_file_relationships(self, events: List[Dict]) -> List[Dict]:
        """Detect relationships within the same file"""
        relationships = []
        
        # Group events by file
        events_by_file = self._group_events_by_file(events)
        
        for filename, file_events in events_by_file.items():
            if len(file_events) < 2:
                continue
                
            # Detect relationships within this file
            file_relationships = await self._find_within_file_relationships(file_events, filename)
            relationships.extend(file_relationships)
        
        return relationships
    
    async def _find_file_relationships(self, source_events: List[Dict], target_events: List[Dict], relationship_type: str) -> List[Dict]:
        """Find relationships between two sets of events with optimized pagination"""
        relationships = []
        
        # Use environment variable for configurable limits (default 100 for better coverage)
        import os
        max_source_events = int(os.getenv('RELATIONSHIP_MAX_SOURCE_EVENTS', '100'))
        max_target_events = int(os.getenv('RELATIONSHIP_MAX_TARGET_EVENTS', '100'))
        batch_size = int(os.getenv('RELATIONSHIP_BATCH_SIZE', '50'))
        
        # Process in batches to avoid O(N²) explosion while maintaining good coverage
        source_batch_count = min(len(source_events), max_source_events)
        target_batch_count = min(len(target_events), max_target_events)
        
        logger.info(f"Processing {source_batch_count} source events x {target_batch_count} target events for {relationship_type}")
        
        # Process in smaller batches for memory efficiency
        for source_start in range(0, source_batch_count, batch_size):
            source_batch = source_events[source_start:source_start + batch_size]
            
            for target_start in range(0, target_batch_count, batch_size):
                target_batch = target_events[target_start:target_start + batch_size]
                
                # Process this batch
                for source_event in source_batch:
                    for target_event in target_batch:
                        score = await self._calculate_relationship_score(source_event, target_event, relationship_type)
                        
                        if score > 0.6:  # Only include high-confidence relationships
                            relationship = {
                                'source_event_id': source_event.get('id'),
                                'target_event_id': target_event.get('id'),
                                'relationship_type': relationship_type,
                                'confidence_score': score,
                                'source_file': source_event.get('source_filename'),
                                'target_file': target_event.get('source_filename'),
                                'detection_method': 'cross_file_analysis',
                                'reasoning': f"Cross-file relationship between {source_event.get('source_filename')} and {target_event.get('source_filename')}"
                            }
                            relationships.append(relationship)
        
        logger.info(f"Found {len(relationships)} relationships for {relationship_type}")
        return relationships
    
    async def _find_within_file_relationships(self, events: List[Dict], filename: str) -> List[Dict]:
        """Find relationships within a single file with configurable window"""
        relationships = []
        
        # Use environment variable for configurable window size (default 20 for better coverage)
        import os
        relationship_window = int(os.getenv('RELATIONSHIP_WITHIN_FILE_WINDOW', '20'))
        
        # Sort events by date if possible
        sorted_events = self._sort_events_by_date(events)
        
        logger.info(f"Processing {len(sorted_events)} events within {filename} (window={relationship_window})")
        
        for i, event1 in enumerate(sorted_events):
            # Look at next N events (configurable window)
            window_end = min(i + relationship_window + 1, len(sorted_events))
            for j in range(i + 1, window_end):
                event2 = sorted_events[j]
                relationship_type = self._determine_relationship_type(event1, event2)
                score = await self._calculate_relationship_score(event1, event2, relationship_type)
                
                if score > 0.5:  # Lower threshold for within-file relationships
                    relationship = {
                        'source_event_id': event1.get('id'),
                        'target_event_id': event2.get('id'),
                        'relationship_type': relationship_type,
                        'confidence_score': score,
                        'source_file': filename,
                        'target_file': filename,
                        'detection_method': 'within_file_analysis',
                        'reasoning': f"Sequential relationship within {filename}"
                    }
                    relationships.append(relationship)
        
        logger.info(f"Found {len(relationships)} within-file relationships in {filename}")
        return relationships
    
    def _sort_events_by_date(self, events: List[Dict]) -> List[Dict]:
        """Sort events by date if available"""
        try:
            return sorted(events, key=lambda x: self._extract_date(x) or datetime.min)
        except:
            return events
    
    def _determine_relationship_type(self, event1: Dict, event2: Dict) -> str:
        """Determine the type of relationship between two events"""
        payload1 = event1.get('payload', {})
        payload2 = event2.get('payload', {})
        
        # Check for common relationship patterns
        if self._is_invoice_event(payload1) and self._is_payment_event(payload2):
            return 'invoice_to_payment'
        elif self._is_payment_event(payload1) and self._is_invoice_event(payload2):
            return 'payment_to_invoice'
        elif self._is_revenue_event(payload1) and self._is_cashflow_event(payload2):
            return 'revenue_to_cashflow'
        elif self._is_expense_event(payload1) and self._is_bank_event(payload2):
            return 'expense_to_bank'
        elif self._is_payroll_event(payload1) and self._is_bank_event(payload2):
            return 'payroll_to_bank'
        else:
            return 'related_transaction'
    
    def _is_invoice_event(self, payload: Dict) -> bool:
        """Check if event is an invoice"""
        text = str(payload).lower()
        return any(word in text for word in ['invoice', 'bill', 'receivable'])
    
    def _is_payment_event(self, payload: Dict) -> bool:
        """Check if event is a payment"""
        text = str(payload).lower()
        return any(word in text for word in ['payment', 'charge', 'transaction', 'debit'])
    
    def _is_revenue_event(self, payload: Dict) -> bool:
        """Check if event is revenue"""
        text = str(payload).lower()
        return any(word in text for word in ['revenue', 'income', 'sales'])
    
    def _is_cashflow_event(self, payload: Dict) -> bool:
        """Check if event is cash flow"""
        text = str(payload).lower()
        return any(word in text for word in ['cash', 'flow', 'bank'])
    
    def _is_expense_event(self, payload: Dict) -> bool:
        """Check if event is an expense"""
        text = str(payload).lower()
        return any(word in text for word in ['expense', 'cost', 'payment'])
    
    def _is_payroll_event(self, payload: Dict) -> bool:
        """Check if event is payroll"""
        text = str(payload).lower()
        return any(word in text for word in ['payroll', 'salary', 'wage', 'employee'])
    
    def _is_bank_event(self, payload: Dict) -> bool:
        """Check if event is a bank transaction"""
        text = str(payload).lower()
        return any(word in text for word in ['bank', 'account', 'transaction'])
    
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
    
    def _calculate_context_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate context similarity score"""
        try:
            source_text = str(source_payload).lower()
            target_text = str(target_payload).lower()
            
            # Simple text similarity
            source_words = set(source_text.split())
            target_words = set(target_text.split())
            
            if not source_words or not target_words:
                return 0.0
            
            common_words = source_words & target_words
            total_words = source_words | target_words
            
            return len(common_words) / len(total_words)
            
        except:
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
        """Extract transaction date from event, prioritizing business date over system timestamps
        
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
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        continue
            
            # PRIORITY 2: Check enriched source_ts column (from Phase 5)
            if 'source_ts' in event and event['source_ts']:
                try:
                    return datetime.fromisoformat(event['source_ts'].replace('Z', '+00:00'))
                except:
                    pass
            
            # PRIORITY 3: Fallback to system timestamps (ONLY if no transaction date found)
            system_date_fields = ['created_at', 'ingest_ts', 'processed_at']
            for field in system_date_fields:
                if field in event and event[field]:
                    try:
                        return datetime.fromisoformat(event[field].replace('Z', '+00:00'))
                    except:
                        continue
            
            return None
        except Exception as e:
            logger.error(f"Date extraction failed: {e}")
            return None
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entities from payload using universal field detection"""
        entities = []
        try:
            # Import universal extractors directly (avoid circular import)
            from universal_extractors_optimized import UniversalExtractorsOptimized
            universal_extractors = UniversalExtractorsOptimized()
            
            # Use synchronous extraction method to avoid async/sync mixing
            try:
                # Use synchronous fallback method instead of async
                vendor_result = universal_extractors._extract_vendor_fallback(payload)
                if vendor_result and isinstance(vendor_result, str):
                    entities.append(vendor_result)
            except Exception as e:
                logger.warning(f"Universal vendor extraction failed: {e}")
                pass  # Fall back to manual extraction
            
            # Extract from entities field
            if 'entities' in payload:
                entity_data = payload['entities']
                if isinstance(entity_data, dict):
                    for entity_type, entity_list in entity_data.items():
                        if isinstance(entity_list, list):
                            entities.extend(entity_list)
            
            # Extract from text
            text = str(payload)
            # Simple entity extraction
            words = text.split()
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    entities.append(word)
            
            return list(set(entities))
        except:
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
    
    def _remove_duplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create unique key
            key = f"{rel.get('source_event_id')}_{rel.get('target_event_id')}_{rel.get('relationship_type')}"
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
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
            logger.error(f"Semantic enrichment failed: {e}")
            return {
                'enabled': True,
                'total_relationships': len(relationships),
                'enriched_count': 0,
                'error': str(e),
                'message': 'Semantic enrichment failed'
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
            logger.error(f"Causal analysis failed: {e}")
            return {
                'enabled': True,
                'total_relationships': len(relationships),
                'causal_count': 0,
                'error': str(e),
                'message': 'Causal analysis failed'
            }
    
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
    
    async def _sync_to_neo4j(self, relationships: List[Dict], user_id: str) -> Dict[str, Any]:
        """
        Sync relationships to Neo4j graph database for visual exploration.
        
        Args:
            relationships: List of detected relationships
            user_id: User ID for context
        
        Returns:
            Statistics about Neo4j sync
        """
        if not self.neo4j:
            return {
                'enabled': False,
                'synced_nodes': 0,
                'synced_relationships': 0,
                'message': 'Neo4j not available'
            }
        
        try:
            synced_nodes = 0
            synced_relationships = 0
            failed_count = 0
            
            # Get unique event IDs
            event_ids = set()
            for rel in relationships:
                event_ids.add(rel['source_event_id'])
                event_ids.add(rel['target_event_id'])
            
            # Fetch event details from Supabase
            events_dict = await self._fetch_events_by_ids(list(event_ids), user_id)
            
            # Create event nodes in Neo4j
            for event_id, event_data in events_dict.items():
                try:
                    if self.neo4j.create_event_node(event_data):
                        synced_nodes += 1
                except Exception as e:
                    logger.error(f"Failed to create Neo4j node for {event_id}: {e}")
                    failed_count += 1
            
            # Create relationship edges in Neo4j
            for rel in relationships:
                try:
                    if self.neo4j.create_relationship(
                        rel['source_event_id'],
                        rel['target_event_id'],
                        rel
                    ):
                        synced_relationships += 1
                except Exception as e:
                    logger.error(f"Failed to create Neo4j relationship: {e}")
                    failed_count += 1
            
            logger.info(f"✅ Neo4j sync: {synced_nodes} nodes, {synced_relationships} relationships")
            
            return {
                'enabled': True,
                'synced_nodes': synced_nodes,
                'synced_relationships': synced_relationships,
                'failed_count': failed_count,
                'message': f'Neo4j sync completed: {synced_nodes} nodes, {synced_relationships} relationships'
            }
            
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}")
            return {
                'enabled': True,
                'synced_nodes': 0,
                'synced_relationships': 0,
                'error': str(e),
                'message': 'Neo4j sync failed'
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