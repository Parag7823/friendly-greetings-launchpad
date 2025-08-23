#!/usr/bin/env python3
"""
Script to remove orphaned code block from fastapi_backend.py
This removes the large block of orphaned methods that are causing IndentationError
"""

def remove_orphaned_code():
    """Remove the orphaned code block from fastapi_backend.py"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the start and end of the orphaned code block
    # The orphaned code starts around line 1953 and ends before the RowProcessor class at line 2559
    
    # Look for the start marker (the orphaned detect_platform method)
    start_marker = "def detect_platform(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:"
    
    # Look for the end marker (the start of RowProcessor class)
    end_marker = "class RowProcessor:"
    
    # Find positions
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos == -1:
        print("Start marker not found")
        return
    
    if end_pos == -1:
        print("End marker not found")
        return
    
    if start_pos >= end_pos:
        print("Invalid positions: start >= end")
        return
    
    print(f"Found orphaned code block from position {start_pos} to {end_pos}")
    
    # Remove the orphaned code block
    before_orphaned = content[:start_pos]
    after_orphaned = content[end_pos:]
    
    # Combine the parts
    new_content = before_orphaned + after_orphaned
    
    # Write back to file
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully removed orphaned code block")
    print(f"Removed approximately {end_pos - start_pos} characters")

if __name__ == "__main__":
    remove_orphaned_code()
