#!/usr/bin/env python3
"""
Fix entity resolution issues and prevent over-merging problems
"""

import re

def fix_entity_resolution():
    """Fix entity resolution issues and prevent over-merging"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Enhance EntityResolver class with better over-merging prevention
    entity_resolver_pattern = r'class EntityResolver:\s+"""Resolves and normalizes entities across different platforms"""\s+\s+def __init__\(self, supabase: Client\):\s+self\.supabase = supabase'
    
    entity_resolver_replacement = '''class EntityResolver:
    """Resolves and normalizes entities across different platforms with over-merging prevention"""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.similarity_threshold = 0.85  # Higher threshold to prevent over-merging
        self.name_variations_cache = {}
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names with enhanced logic"""
        if not name1 or not name2:
            return 0.0
        
        # Normalize names
        name1_norm = self._normalize_entity_name(name1)
        name2_norm = self._normalize_entity_name(name2)
        
        # Exact match after normalization
        if name1_norm == name2_norm:
            return 1.0
        
        # Check for exact substring match (but not too short)
        if len(name1_norm) > 5 and len(name2_norm) > 5:
            if name1_norm in name2_norm or name2_norm in name1_norm:
                return 0.9
        
        # Use sequence matcher for fuzzy matching
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, name1_norm, name2_norm).ratio()
        
        # Apply business logic penalties
        if self._are_different_entities(name1_norm, name2_norm):
            similarity *= 0.5  # Reduce similarity for likely different entities
        
        return similarity
    
    def _normalize_entity_name(self, name: str) -> str:
        """Enhanced entity name normalization"""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = name.lower().strip()
        
        # Remove common business suffixes
        suffixes_to_remove = [
            ' inc', ' corp', ' llc', ' ltd', ' co', ' company', ' pvt', ' private',
            ' limited', ' corporation', ' incorporated', ' enterprises', ' solutions',
            ' services', ' systems', ' technologies', ' tech', ' group', ' holdings'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove punctuation and extra spaces
        import re
        normalized = re.sub(r'[^a-zA-Z0-9\\s]', '', normalized)
        normalized = re.sub(r'\\s+', ' ', normalized).strip()
        
        return normalized
    
    def _are_different_entities(self, name1: str, name2: str) -> bool:
        """Check if two names likely represent different entities"""
        # Check for different business types
        business_indicators1 = self._extract_business_indicators(name1)
        business_indicators2 = self._extract_business_indicators(name2)
        
        if business_indicators1 and business_indicators2 and business_indicators1 != business_indicators2:
            return True
        
        # Check for different locations
        location1 = self._extract_location(name1)
        location2 = self._extract_location(name2)
        
        if location1 and location2 and location1 != location2:
            return True
        
        return False
    
    def _extract_business_indicators(self, name: str) -> str:
        """Extract business type indicators from name"""
        indicators = []
        if 'bank' in name or 'credit' in name or 'financial' in name:
            indicators.append('financial')
        if 'tech' in name or 'software' in name or 'digital' in name:
            indicators.append('technology')
        if 'food' in name or 'restaurant' in name or 'cafe' in name:
            indicators.append('food')
        if 'health' in name or 'medical' in name or 'pharmacy' in name:
            indicators.append('healthcare')
        
        return '_'.join(indicators) if indicators else ""
    
    def _extract_location(self, name: str) -> str:
        """Extract location indicators from name"""
        # Simple location extraction - can be enhanced
        location_keywords = ['new york', 'ny', 'california', 'ca', 'texas', 'tx', 'florida', 'fl']
        for location in location_keywords:
            if location in name.lower():
                return location
        return ""'''
    
    content = re.sub(entity_resolver_pattern, entity_resolver_replacement, content, flags=re.DOTALL)
    
    # Enhance the resolve_entity method
    resolve_entity_pattern = r'async def resolve_entity\(self, entity_name: str, entity_type: str, platform: str, \s+user_id: str, row_data: Dict, column_names: List\[str\], \s+source_file: str, row_id: str\) -> Dict\[str, Any\]:'
    
    resolve_entity_replacement = '''async def resolve_entity(self, entity_name: str, entity_type: str, platform: str, 
                           user_id: str, row_data: Dict, column_names: List[str], 
                           source_file: str, row_id: str) -> Dict[str, Any]:
        """Resolve entity using enhanced logic with over-merging prevention"""
        
        # Extract strong identifiers
        identifiers = self.extract_strong_identifiers(row_data, column_names)
        
        try:
            # First, try to find exact matches
            exact_match = await self._find_exact_match(entity_name, user_id)
            if exact_match:
                return {
                    'entity_id': exact_match['id'],
                    'normalized_name': exact_match['normalized_name'],
                    'confidence_score': 1.0,
                    'match_type': 'exact',
                    'reasoning': 'Exact name match found'
                }
            
            # Then, try to find similar matches with higher threshold
            similar_matches = await self._find_similar_matches(entity_name, user_id)
            
            if similar_matches:
                best_match = similar_matches[0]
                if best_match['similarity'] >= self.similarity_threshold:
                    return {
                        'entity_id': best_match['id'],
                        'normalized_name': best_match['normalized_name'],
                        'confidence_score': best_match['similarity'],
                        'match_type': 'similar',
                        'reasoning': f"Similar match found with {best_match['similarity']:.2f} similarity"
                    }
            
            # If no good match found, create new entity
            return await self._create_new_entity(entity_name, entity_type, platform, user_id, identifiers, source_file)
            
        except Exception as e:
            logger.error(f"Entity resolution failed for {entity_name}: {e}")
            return {
                'entity_id': None,
                'normalized_name': entity_name,
                'confidence_score': 0.0,
                'match_type': 'failed',
                'reasoning': f"Resolution failed: {str(e)}"
            }'''
    
    content = re.sub(resolve_entity_pattern, resolve_entity_replacement, content, flags=re.DOTALL)
    
    # Add helper methods for entity resolution
    helper_methods = '''
    
    async def _find_exact_match(self, entity_name: str, user_id: str) -> Optional[Dict]:
        """Find exact match for entity name"""
        normalized_name = self._normalize_entity_name(entity_name)
        
        result = self.supabase.table('normalized_entities').select('*').eq('user_id', user_id).eq('normalized_name', normalized_name).execute()
        
        if result.data:
            return result.data[0]
        return None
    
    async def _find_similar_matches(self, entity_name: str, user_id: str) -> List[Dict]:
        """Find similar matches with enhanced similarity calculation"""
        normalized_name = self._normalize_entity_name(entity_name)
        
        # Get all entities for the user
        result = self.supabase.table('normalized_entities').select('*').eq('user_id', user_id).execute()
        
        if not result.data:
            return []
        
        # Calculate similarity for each entity
        matches = []
        for entity in result.data:
            similarity = self._calculate_name_similarity(normalized_name, entity['normalized_name'])
            if similarity > 0.7:  # Lower threshold for candidate selection
                matches.append({
                    'id': entity['id'],
                    'normalized_name': entity['normalized_name'],
                    'similarity': similarity
                })
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:5]  # Return top 5 matches
    
    async def _create_new_entity(self, entity_name: str, entity_type: str, platform: str, 
                                user_id: str, identifiers: Dict, source_file: str) -> Dict[str, Any]:
        """Create new entity with enhanced validation"""
        normalized_name = self._normalize_entity_name(entity_name)
        
        # Validate entity name
        if len(normalized_name) < 2:
            return {
                'entity_id': None,
                'normalized_name': entity_name,
                'confidence_score': 0.0,
                'match_type': 'invalid',
                'reasoning': 'Entity name too short'
            }
        
        # Create new entity
        new_entity = {
            'user_id': user_id,
            'entity_name': entity_name,
            'entity_type': entity_type,
            'normalized_name': normalized_name,
            'confidence_score': 0.8,
            'source_file': source_file,
            'platform': platform,
            'identifiers': identifiers,
            'detected_at': datetime.utcnow().isoformat()
        }
        
        result = self.supabase.table('normalized_entities').insert(new_entity).execute()
        
        if result.data:
            return {
                'entity_id': result.data[0]['id'],
                'normalized_name': normalized_name,
                'confidence_score': 0.8,
                'match_type': 'new',
                'reasoning': 'New entity created'
            }
        else:
            return {
                'entity_id': None,
                'normalized_name': normalized_name,
                'confidence_score': 0.0,
                'match_type': 'creation_failed',
                'reasoning': 'Failed to create new entity'
            }'''
    
    # Find the end of the EntityResolver class and add helper methods
    class_end_pattern = r'return \{\s+\'entity_id\': None,\s+\'normalized_name\': entity_name,\s+\'confidence_score\': 0\.0,\s+\'match_type\': \'failed\',\s+\'reasoning\': f"Resolution failed: \{str\(e\)\}"\s+\}'
    
    class_end_replacement = '''return {
                'entity_id': None,
                'normalized_name': entity_name,
                'confidence_score': 0.0,
                'match_type': 'failed',
                'reasoning': f"Resolution failed: {str(e)}"
            }''' + helper_methods
    
    content = re.sub(class_end_pattern, class_end_replacement, content, flags=re.DOTALL)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed entity resolution issues and over-merging prevention")

if __name__ == "__main__":
    fix_entity_resolution()
