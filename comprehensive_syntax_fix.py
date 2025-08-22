#!/usr/bin/env python3
"""
Comprehensive syntax fix for fastapi_backend.py
"""

# Read the file
with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix specific syntax issues
fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Fix missing opening brace for content dictionary
    if 'processed_at": datetime.utcnow().isoformat()' in line and i + 1 < len(lines):
        if lines[i + 1].strip() == '},':
            # Fix the indentation
            fixed_lines.append(line)
            fixed_lines.append('            },\n')
            i += 2
            continue
    
    # Fix missing opening brace for job_result insert
    elif 'started_at": datetime.utcnow().isoformat()' in line and i + 1 < len(lines):
        if lines[i + 1].strip() == '}).execute()':
            # Fix the indentation
            fixed_lines.append(line)
            fixed_lines.append('            }).execute()\n')
            i += 2
            continue
    
    # Fix missing opening brace for update_result update
    elif 'started_at": datetime.utcnow().isoformat()' in line and i + 1 < len(lines):
        if lines[i + 1].strip() == '}).eq(\'id\', job_id).execute()':
            # Fix the indentation
            fixed_lines.append(line)
            fixed_lines.append('            }).eq(\'id\', job_id).execute()\n')
            i += 2
            continue
    
    # Fix missing await manager.send_update calls
    elif '"step": "streaming",' in line and i > 0:
        if not lines[i-1].strip().startswith('await manager.send_update'):
            # Add the missing await call
            fixed_lines.append('        await manager.send_update(job_id, {\n')
            fixed_lines.append(line)
            i += 1
            continue
    
    # Fix missing await manager.send_update calls for progress messages
    elif '"message": "🔄 Processing rows in memory-optimized chunks' in line:
        fixed_lines.append('        await manager.send_update(job_id, {\n')
        fixed_lines.append(line)
        i += 1
        continue
    
    elif '"message": "🔄 Processing rows in optimized batches' in line:
        fixed_lines.append('        await manager.send_update(job_id, {\n')
        fixed_lines.append(line)
        i += 1
        continue
    
    # Fix missing progress closing brace
    elif '"progress": 40' in line and i + 1 < len(lines):
        if lines[i + 1].strip() == '':
            fixed_lines.append(line)
            fixed_lines.append('        })\n')
            i += 2
            continue
    
    else:
        fixed_lines.append(line)
        i += 1

# Write back
with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Fixed all syntax errors comprehensively")
