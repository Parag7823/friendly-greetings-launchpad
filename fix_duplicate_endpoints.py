#!/usr/bin/env python3
"""
Fix Duplicate Endpoints

This script removes duplicate endpoints from fastapi_backend.py
"""

def fix_duplicate_endpoints():
    """Remove duplicate endpoints from fastapi_backend.py"""
    
    print("🔧 Fixing duplicate endpoints...")
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and remove duplicate endpoints
    endpoints_to_remove = []
    seen_endpoints = set()
    
    for i, line in enumerate(lines):
        if line.strip().startswith('@app.get("/test-cross-file-relationships/{user_id}")'):
            if '/test-cross-file-relationships/{user_id}' in seen_endpoints:
                # This is a duplicate, mark for removal
                endpoints_to_remove.append(i)
                print(f"📝 Found duplicate endpoint at line {i+1}")
            else:
                seen_endpoints.add('/test-cross-file-relationships/{user_id}')
    
    # Remove duplicate endpoints (in reverse order to maintain line numbers)
    for line_num in reversed(endpoints_to_remove):
        # Find the end of the function (look for the next @app.get or class definition)
        start_line = line_num
        end_line = start_line
        
        for j in range(start_line + 1, len(lines)):
            if (lines[j].strip().startswith('@app.get') or 
                lines[j].strip().startswith('class ') or
                lines[j].strip().startswith('def ') and j > start_line + 5):
                end_line = j - 1
                break
        
        # Remove the duplicate function
        del lines[start_line:end_line + 1]
        print(f"🗑️ Removed duplicate endpoint from lines {start_line+1}-{end_line+1}")
    
    # Write back to file
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Successfully fixed duplicate endpoints!")

if __name__ == "__main__":
    fix_duplicate_endpoints() 