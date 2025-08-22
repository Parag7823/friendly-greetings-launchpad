#!/usr/bin/env python3
"""
Fix missing await manager.send_update calls
"""

# Read the file
with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix missing await manager.send_update calls
content = content.replace(
    '        # Step 1: Read the file\n            "step": "reading",',
    '        # Step 1: Read the file\n        await manager.send_update(job_id, {\n            "step": "reading",'
)

content = content.replace(
    '        # Step 2: Detect platform and document type\n            "step": "analyzing",',
    '        # Step 2: Detect platform and document type\n        await manager.send_update(job_id, {\n            "step": "analyzing",'
)

content = content.replace(
    '        # Step 3: Create raw_records entry\n            "step": "storing",',
    '        # Step 3: Create raw_records entry\n        await manager.send_update(job_id, {\n            "step": "storing",'
)

# Write back
with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed missing await manager.send_update calls")
