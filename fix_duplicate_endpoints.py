#!/usr/bin/env python3
"""
Fix duplicate endpoints and hardcoded user ID issues in fastapi_backend.py
"""

import re

def fix_fastapi_backend():
    """Fix the duplicate endpoints and hardcoded user ID issues"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the first occurrence of hardcoded user ID
    content = re.sub(
        r'user_id: str = Form\("550e8400-e29b-41d4-a716-446655440000"\),  # Default test user ID',
        'user_id: str = Form(...),  # Required user ID - no default',
        content,
        count=1  # Only fix the first occurrence
    )
    
    # Remove the duplicate upload-and-process endpoint
    # Find the second occurrence and remove it
    pattern = r'@app\.post\("/upload-and-process"\)\s+async def upload_and_process\([^)]*\):\s+"""[^"]*"""\s+try:[^}]*except Exception as e:[^}]*raise HTTPException\([^)]*\)'
    
    # More specific pattern to match the duplicate
    duplicate_pattern = r'@app\.post\("/upload-and-process"\)\s+async def upload_and_process\([^)]*\):\s+"""Direct file upload and processing endpoint for testing"""\s+try:[^}]*except Exception as e:\s+logger\.error\(f"Upload and process error: \{e\}"\)\s+raise HTTPException\(status_code=500, detail=str\(e\)\)'
    
    # Remove the duplicate
    content = re.sub(duplicate_pattern, '', content, flags=re.DOTALL)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed duplicate endpoints and hardcoded user ID issues")

if __name__ == "__main__":
    fix_fastapi_backend() 