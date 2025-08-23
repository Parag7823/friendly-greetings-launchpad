#!/usr/bin/env python3
"""
Script to remove duplicate class definitions and methods from fastapi_backend.py
This addresses the systematic duplication issue causing deployment failures
"""

import re

def remove_duplicates():
    """Remove duplicate class definitions and methods from fastapi_backend.py"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the classes we want to keep (first occurrence)
    classes_to_keep = [
        'Config',
        'CurrencyNormalizer', 
        'VendorStandardizer',
        'PlatformIDExtractor',
        'DataEnrichmentProcessor',
        'ConnectionInfo',
        'ConnectionManager',
        'ProcessRequest',
        'StreamingFileProcessor',
        'DocumentAnalyzer',
        'PlatformDetector',
        'AIRowClassifier',
        'BatchAIRowClassifier',
        'RowProcessor',
        'ExcelProcessor',
        'EntityResolver',
        'EnhancedRelationshipDetector'
    ]
    
    # Find all class definitions
    class_pattern = r'^class\s+(\w+)\s*[:\(]'
    class_matches = list(re.finditer(class_pattern, content, re.MULTILINE))
    
    # Track which classes we've seen
    seen_classes = set()
    classes_to_remove = []
    
    for match in class_matches:
        class_name = match.group(1)
        if class_name in classes_to_keep:
            if class_name in seen_classes:
                # This is a duplicate of a class we want to keep
                classes_to_remove.append(match.start())
            else:
                seen_classes.add(class_name)
        else:
            # Unknown class, keep it for now
            pass
    
    # Sort removal positions in reverse order to avoid index shifting
    classes_to_remove.sort(reverse=True)
    
    # Find the end of each duplicate class (next class definition or end of file)
    for start_pos in classes_to_remove:
        # Find the next class definition after this one
        next_class_match = re.search(r'^class\s+\w+\s*[:\(]', content[start_pos+1:], re.MULTILINE)
        if next_class_match:
            end_pos = start_pos + 1 + next_class_match.start()
        else:
            # If no next class, remove to end of file
            end_pos = len(content)
        
        # Remove the duplicate class
        content = content[:start_pos] + content[end_pos:]
        print(f"Removed duplicate class starting at position {start_pos}")
    
    # Write back to file
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully removed duplicate classes")
    print(f"Removed {len(classes_to_remove)} duplicate class definitions")

if __name__ == "__main__":
    remove_duplicates()
