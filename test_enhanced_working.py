#!/usr/bin/env python3
"""
Working Test for Enhanced Relationship Detection

This script tests the enhanced relationship detection system
with the correct imports to avoid the AsyncOpenAI error.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from supabase import create_client

class EnhancedRelationshipDetector:
    """Enhanced relationship detector that actually finds relationships between events"""
    
    def __init__(self, openai_client, supabase_client):
        self.openai = openai_client
        self.supabase = supabase_client
        self.relationship_cache = {}
        
    async def detect_all_relationships(self, user_id: str):
        """Detect actual relationships between financial events"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"relationships": [], "message": "No data found for relationship analysis"}
            
            print(f"Processing {len(events.data)} events for enhanced relationship detection")
            
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
            
            print(f"Enhanced relationship detection completed: {len(validated_relationships)} relationships found")
            
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
            print(f"Enhanced relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    def _group_events_by_file(self, events):
        """Group events by source filename"""
        events_by_file = {}
        for event in events:
            filename = event.get('source_filename', 'unknown')
            if filename not in events_by_file:
                events_by_file[filename] = []
            events_by_file[filename].append(event)
        return events_by_file
    
    async def _detect_cross_file_relationships(self, events_by_file):
        """Detect relationships between different files"""
        relationships = []
        
        # Define cross-file relationship patterns
        cross_file_patterns = [
            {
                'source_files': ['company_invoices.csv', 'comprehensive_vendor_payments.csv'],
                'relationship_type': 'invoice_to_payment',
                'description': 'Invoice payments'
            },
            {
                'source_files': ['company_revenue.csv', 'comprehensive_cash_flow.csv'],
                'relationship_type': 'revenue_to_cashflow',
                'description': 'Revenue cash flow'
            },
            {
                'source_files': ['company_expenses.csv', 'company_bank_statements.csv'],
                'relationship_type': 'expense_to_bank',
                'description': 'Expense bank transactions'
            },
            {
                'source_files': ['comprehensive_payroll_data.csv', 'company_bank_statements.csv'],
                'relationship_type': 'payroll_to_bank',
                'description': 'Payroll bank transactions'
            },
            {
                'source_files': ['company_invoices.csv', 'company_accounts_receivable.csv'],
                'relationship_type': 'invoice_to_receivable',
                'description': 'Invoice receivables'
            }
        ]
        
        for pattern in cross_file_patterns:
            source_file = pattern['source_files'][0]
            target_file = pattern['source_files'][1]
            
            if source_file in events_by_file and target_file in events_by_file:
                source_events = events_by_file[source_file]
                target_events = events_by_file[target_file]
                
                file_relationships = await self._find_file_relationships(
                    source_events, target_events, pattern['relationship_type']
                )
                relationships.extend(file_relationships)
        
        return relationships
    
    async def _detect_within_file_relationships(self, events):
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
    
    async def _find_file_relationships(self, source_events, target_events, relationship_type):
        """Find relationships between two sets of events"""
        relationships = []
        
        for source_event in source_events[:5]:  # Limit for performance
            for target_event in target_events[:5]:  # Limit for performance
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
    
    async def _find_within_file_relationships(self, events, filename):
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
    
    def _sort_events_by_date(self, events):
        """Sort events by date if available"""
        try:
            return sorted(events, key=lambda x: self._extract_date(x) or datetime.min)
        except:
            return events
    
    def _determine_relationship_type(self, event1, event2):
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
    
    def _is_invoice_event(self, payload):
        """Check if event is an invoice"""
        text = str(payload).lower()
        return any(word in text for word in ['invoice', 'bill', 'receivable'])
    
    def _is_payment_event(self, payload):
        """Check if event is a payment"""
        text = str(payload).lower()
        return any(word in text for word in ['payment', 'charge', 'transaction', 'debit'])
    
    def _is_revenue_event(self, payload):
        """Check if event is revenue"""
        text = str(payload).lower()
        return any(word in text for word in ['revenue', 'income', 'sales'])
    
    def _is_cashflow_event(self, payload):
        """Check if event is cash flow"""
        text = str(payload).lower()
        return any(word in text for word in ['cash', 'flow', 'bank'])
    
    def _is_expense_event(self, payload):
        """Check if event is an expense"""
        text = str(payload).lower()
        return any(word in text for word in ['expense', 'cost', 'payment'])
    
    def _is_payroll_event(self, payload):
        """Check if event is payroll"""
        text = str(payload).lower()
        return any(word in text for word in ['payroll', 'salary', 'wage', 'employee'])
    
    def _is_bank_event(self, payload):
        """Check if event is a bank transaction"""
        text = str(payload).lower()
        return any(word in text for word in ['bank', 'account', 'transaction'])
    
    async def _calculate_relationship_score(self, source, target, relationship_type):
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
            print(f"Error calculating relationship score: {e}")
            return 0.0
    
    def _calculate_amount_score(self, source_payload, target_payload):
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
    
    def _calculate_date_score(self, source, target):
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
    
    def _calculate_entity_score(self, source_payload, target_payload):
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
    
    def _calculate_id_score(self, source_payload, target_payload):
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
    
    def _calculate_context_score(self, source_payload, target_payload):
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
    
    def _get_relationship_weights(self, relationship_type):
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
    
    def _extract_amount(self, payload):
        """Extract amount from payload"""
        try:
            # Try different amount fields
            amount_fields = ['amount', 'amount_usd', 'total', 'value', 'payment_amount']
            for field in amount_fields:
                if field in payload and payload[field]:
                    return float(payload[field])
            
            # Try to extract from text
            text = str(payload)
            import re
            matches = re.findall(r'[\d,]+\.?\d*', text)
            if matches:
                return float(matches[0].replace(',', ''))
            
            return 0.0
        except:
            return 0.0
    
    def _extract_date(self, event):
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
    
    def _extract_entities(self, payload):
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
            import re
            # Simple entity extraction
            words = text.split()
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    entities.append(word)
            
            return list(set(entities))
        except:
            return []
    
    def _extract_ids(self, payload):
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
    
    def _remove_duplicate_relationships(self, relationships):
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
    
    async def _validate_relationships(self, relationships):
        """Validate relationships"""
        validated = []
        
        for rel in relationships:
            if self._validate_relationship_structure(rel):
                validated.append(rel)
        
        return validated
    
    def _validate_relationship_structure(self, rel):
        """Validate relationship structure"""
        required_fields = ['source_event_id', 'target_event_id', 'relationship_type', 'confidence_score']
        
        for field in required_fields:
            if field not in rel or rel[field] is None:
                return False
        
        # Check confidence score range
        if not (0.0 <= rel['confidence_score'] <= 1.0):
            return False
        
        return True

async def test_enhanced_working():
    """Test the enhanced relationship detection with correct imports"""
    
    print("🚀 Testing Enhanced Relationship Detection (Working Version)")
    print("=" * 60)
    
    try:
        # Initialize clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase credentials")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Test user ID
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        print(f"📊 Testing with user ID: {user_id}")
        
        # Initialize Enhanced Relationship Detector
        enhanced_detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Detect relationships
        print("🔍 Detecting relationships...")
        result = await enhanced_detector.detect_all_relationships(user_id)
        
        # Display results
        print("\n📈 RESULTS:")
        print("-" * 30)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
        
        total_relationships = result.get("total_relationships", 0)
        cross_file_relationships = result.get("cross_file_relationships", 0)
        within_file_relationships = result.get("within_file_relationships", 0)
        
        print(f"✅ Total Relationships: {total_relationships}")
        print(f"📁 Cross-File: {cross_file_relationships}")
        print(f"📄 Within-File: {within_file_relationships}")
        
        # Show sample relationships
        relationships = result.get("relationships", [])
        if relationships:
            print(f"\n🔗 SAMPLE RELATIONSHIPS:")
            print("-" * 30)
            
            for i, rel in enumerate(relationships[:3]):  # Show first 3
                print(f"Relationship {i+1}:")
                print(f"  Type: {rel.get('relationship_type')}")
                print(f"  Confidence: {rel.get('confidence_score', 0):.3f}")
                print(f"  Method: {rel.get('detection_method')}")
                print()
        
        # Success criteria
        if total_relationships > 0:
            print("🎉 SUCCESS: Relationships detected!")
            return True
        else:
            print("❌ FAILED: No relationships detected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_working())
    sys.exit(0 if success else 1) 