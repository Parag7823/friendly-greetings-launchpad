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
from openai import AsyncOpenAI
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class EnhancedRelationshipDetector:
    """Enhanced relationship detector that actually finds relationships between events"""
    
    def __init__(self, openai_client: AsyncOpenAI, supabase_client: Client):
        self.openai = openai_client
        self.supabase = supabase_client
        self.relationship_cache = {}
        
    async def detect_all_relationships(self, user_id: str) -> Dict[str, Any]:
        """Detect actual relationships between financial events"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"relationships": [], "message": "No data found for relationship analysis"}
            
            logger.info(f"Processing {len(events.data)} events for enhanced relationship detection")
            
            # Group events by file type for cross-file analysis
            events_by_file = self._group_events_by_file(events.data)
            
            # Detect cross-file relationships
            cross_file_relationships = await self._detect_cross_file_relationships(events_by_file)
            
            # Detect within-file relationships
            within_file_relationships = await self._detect_within_file_relationships(events.data)
            
            # Combine all relationships
            all_relationships = cross_file_relationships + within_file_relationships
            
            # Remove duplicates and validate
            unique_relationships = self._remove_duplicate_relationships(all_relationships)
            validated_relationships = await self._validate_relationships(unique_relationships)
            
            logger.info(f"Enhanced relationship detection completed: {len(validated_relationships)} relationships found")
            
            return {
                "relationships": validated_relationships,
                "total_relationships": len(validated_relationships),
                "cross_file_relationships": len(cross_file_relationships),
                "within_file_relationships": len(within_file_relationships),
                "processing_stats": {
                    "total_events": len(events.data),
                    "files_analyzed": len(events_by_file),
                    "relationship_types_found": list(set([r.get('relationship_type', 'unknown') for r in validated_relationships]))
                },
                "message": "Enhanced relationship detection completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Enhanced relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    def _group_events_by_file(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Group events by source filename"""
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
        """Find relationships between two sets of events"""
        relationships = []
        
        for source_event in source_events[:10]:  # Limit for performance
            for target_event in target_events[:10]:  # Limit for performance
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
        
        return relationships
    
    async def _find_within_file_relationships(self, events: List[Dict], filename: str) -> List[Dict]:
        """Find relationships within a single file"""
        relationships = []
        
        # Sort events by date if possible
        sorted_events = self._sort_events_by_date(events)
        
        for i, event1 in enumerate(sorted_events):
            for j, event2 in enumerate(sorted_events[i+1:i+6]):  # Look at next 5 events
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
            amount_score = self._calculate_amount_score(source_payload, target_payload)
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
    
    def _calculate_amount_score(self, source_payload: Dict, target_payload: Dict) -> float:
        """Calculate amount similarity score"""
        try:
            source_amount = self._extract_amount(source_payload)
            target_amount = self._extract_amount(target_payload)
            
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
    
    def _extract_amount(self, payload: Dict) -> float:
        """Extract amount from payload"""
        try:
            # Try different amount fields
            amount_fields = ['amount', 'amount_usd', 'total', 'value', 'payment_amount']
            for field in amount_fields:
                if field in payload and payload[field]:
                    return float(payload[field])
            
            # Try to extract from text
            text = str(payload)
            matches = re.findall(r'[\d,]+\.?\d*', text)
            if matches:
                return float(matches[0].replace(',', ''))
            
            return 0.0
        except:
            return 0.0
    
    def _extract_date(self, event: Dict) -> Optional[datetime]:
        """Extract date from event"""
        try:
            # Try different date fields
            date_fields = ['created_at', 'date', 'timestamp', 'processed_at']
            for field in date_fields:
                if field in event and event[field]:
                    return datetime.fromisoformat(event[field].replace('Z', '+00:00'))
            
            return None
        except:
            return None
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entities from payload"""
        entities = []
        try:
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
    import asyncio
    result = asyncio.run(test_enhanced_relationship_detection())
    print(result) 