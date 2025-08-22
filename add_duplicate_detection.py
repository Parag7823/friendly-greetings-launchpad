#!/usr/bin/env python3
"""
Add duplicate file detection to the upload process
"""

import re

def add_duplicate_detection():
    """Add duplicate file detection to the upload process"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the pattern for file hash calculation and raw_records storage
    pattern = r'# Calculate file hash for duplicate detection\s+file_hash = hashlib\.sha256\(file_content\)\.hexdigest\(\)\s+\s+# Store in raw_records'
    
    replacement = '''# Calculate file hash for duplicate detection
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check for duplicate files
        duplicate_check = supabase.table('raw_records').select('id, file_name, created_at').eq('user_id', user_id).eq('content->file_hash', file_hash).execute()
        
        if duplicate_check.data:
            duplicate_file = duplicate_check.data[0]
            await manager.send_update(job_id, {
                "step": "error",
                "message": f"❌ Duplicate file detected! This file was already uploaded on {duplicate_file['created_at'][:10]}",
                "progress": 0
            })
            raise HTTPException(
                status_code=400, 
                detail=f"Duplicate file detected. File '{duplicate_file['file_name']}' was already uploaded on {duplicate_file['created_at'][:10]}"
            )
        
        # Store in raw_records'''
    
    # Replace the first occurrence only (in the main process_file method)
    content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added duplicate file detection to upload process")

if __name__ == "__main__":
    add_duplicate_detection()
