#!/usr/bin/env python3
"""
Fix entity resolution issues with a simpler approach
"""

def fix_entity_resolution_simple():
    """Fix entity resolution issues with a simpler approach"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the EntityResolver class initialization
    old_init = '''    def __init__(self, supabase: Client):
        self.supabase = supabase'''
    
    new_init = '''    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.similarity_threshold = 0.85  # Higher threshold to prevent over-merging
        self.name_variations_cache = {}'''
    
    content = content.replace(old_init, new_init)
    
    # Add enhanced similarity calculation method
    similarity_method = '''
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
        normalized = re.sub(r'[^a-zA-Z0-9 ]', '', normalized)
        normalized = re.sub(r' +', ' ', normalized).strip()
        
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
    
    # Find the end of the EntityResolver class and add the new methods
    # Look for the end of the class (before the next class definition)
    class_end_marker = '''    def extract_strong_identifiers(self, row_data: Dict, column_names: List[str]) -> Dict[str, Any]:'''
    
    if class_end_marker in content:
        content = content.replace(class_end_marker, similarity_method + '\n\n    ' + class_end_marker)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed entity resolution issues with simplified approach")

if __name__ == "__main__":
    fix_entity_resolution_simple()
