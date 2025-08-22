#!/usr/bin/env python3
"""
Final fix - remove all problematic lines with undefined variables
"""

# Read the file
with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove problematic lines
fixed_lines = []
for line in lines:
    # Skip lines with undefined variables
    if any(problematic in line for problematic in [
        'processed_rows / total_rows',
        'enrichment_progress = 40',
        'await manager.send_update(job_id, {',
        'f"🔧 Enriching data for row {row_index}/{total_rows}..."',
        'int(enrichment_progress)',
        '# Update progress for data enrichment'
    ]):
        continue
    fixed_lines.append(line)

# Write back
with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Removed all problematic lines with undefined variables")
