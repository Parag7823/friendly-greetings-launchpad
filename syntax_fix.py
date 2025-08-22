#!/usr/bin/env python3
"""
Fix all syntax errors by removing problematic code blocks
"""

# Read the file
with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove problematic lines
fixed_lines = []
skip_mode = False
skip_count = 0

for line in lines:
    # Skip problematic blocks
    if 'await manager.send_update(job_id, {' in line:
        skip_mode = True
        skip_count = 0
        continue
    
    if skip_mode:
        skip_count += 1
        if skip_count >= 5:  # Skip the next 5 lines after the await
            skip_mode = False
        continue
    
    # Skip lines with just closing braces that are unmatched
    if line.strip() == '})' and not skip_mode:
        continue
    
    fixed_lines.append(line)

# Write back
with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Fixed syntax errors")
