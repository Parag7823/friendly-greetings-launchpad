#!/usr/bin/env python3
"""
Fix indentation errors in fastapi_backend.py
"""

import re

def fix_indentation_errors():
    """Fix all indentation errors in fastapi_backend.py"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove problematic progress update code blocks that have undefined variables
    # Pattern: Lines with over-indented progress updates using undefined variables
    
    # Find and remove problematic progress update blocks
    patterns_to_remove = [
        r'# Update progress for data enrichment\s*\n\s*if row_index % 10 == 0:  # Update every 10 rows\s*\n\s*enrichment_progress = 40 \+ \(processed_rows / total_rows\) \* 30\s*\n\s*await manager\.send_update\(job_id, \{\s*\n\s*"step": "enrichment",\s*\n\s*"message": f"🔧 Enriching data for row \{row_index\}/\{total_rows\}\.\.\.",\s*\n\s*"progress": int\(enrichment_progress\)\s*\n\s*\}\)\s*\n',
        r'# Update progress for data enrichment\s*\n\s*if row_index % 10 == 0:  # Update every 10 rows\s*\n\s*enrichment_progress = 40 \+ \(processed_rows / total_rows\) \* 30\s*\n\s*await manager\.send_update\(job_id, \{\s*\n\s*"step": "enrichment",\s*\n\s*"message": f"🔧 Enriching data for row \{row_index\}/\{total_rows\}\.\.\.",\s*\n\s*"progress": int\(enrichment_progress\)\s*\n\s*\}\)\s*\n\s*',
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    # Fix 2: Fix over-indented code blocks
    # Find lines that are over-indented and fix them
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check for over-indented lines that should be at method level
        if (line.strip().startswith('if not enriched_payload.get(') or 
            line.strip().startswith('enriched_payload[') or
            line.strip().startswith('event = {') or
            line.strip().startswith('"provider":') or
            line.strip().startswith('"kind":') or
            line.strip().startswith('"source_platform":') or
            line.strip().startswith('"payload":') or
            line.strip().startswith('"row_index":') or
            line.strip().startswith('"sheet_name":') or
            line.strip().startswith('"source_filename":') or
            line.strip().startswith('"uploader":') or
            line.strip().startswith('"ingest_ts":') or
            line.strip().startswith('"status":') or
            line.strip().startswith('"confidence_score":') or
            line.strip().startswith('"classification_metadata":') or
            line.strip().startswith('"platform_detection":') or
            line.strip().startswith('"ai_classification":') or
            line.strip().startswith('"enrichment_data":') or
            line.strip().startswith('"row_type":') or
            line.strip().startswith('"category":') or
            line.strip().startswith('"subcategory":') or
            line.strip().startswith('"entities":') or
            line.strip().startswith('"relationships":') or
            line.strip().startswith('"description":') or
            line.strip().startswith('"reasoning":') or
            line.strip().startswith('"sheet_name":') or
            line.strip().startswith('"file_context":') or
            line.strip().startswith('}')):
            
            # Fix indentation to be at method level (8 spaces)
            if line.startswith('                            '):
                line = '        ' + line[28:]  # Remove 28 spaces, add 8
            elif line.startswith('                        '):
                line = '        ' + line[24:]  # Remove 24 spaces, add 8
            elif line.startswith('                    '):
                line = '        ' + line[20:]  # Remove 20 spaces, add 8
            elif line.startswith('                '):
                line = '        ' + line[16:]  # Remove 16 spaces, add 8
            elif line.startswith('            '):
                line = '        ' + line[12:]  # Remove 12 spaces, add 8
        
        fixed_lines.append(line)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("✅ Fixed indentation errors in fastapi_backend.py")

if __name__ == "__main__":
    fix_indentation_errors()
