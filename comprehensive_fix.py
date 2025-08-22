#!/usr/bin/env python3
"""
Comprehensive fix for all indentation and syntax errors in fastapi_backend.py
"""

import re

def fix_all_errors():
    """Fix all indentation and syntax errors"""
    
    print("🔧 Starting comprehensive fix...")
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove all problematic progress update blocks with undefined variables
    print("📝 Fixing progress update blocks...")
    
    # Pattern to match the problematic progress update blocks
    problematic_pattern = r'(\s*)# Update progress for data enrichment\s*\n\s*if row_index % 10 == 0:  # Update every 10 rows\s*\n\s*enrichment_progress = 40 \+ \(processed_rows / total_rows\) \* 30\s*\n\s*await manager\.send_update\(job_id, \{\s*\n\s*"step": "enrichment",\s*\n\s*"message": f"🔧 Enriching data for row \{row_index\}/\{total_rows\}\.\.\.",\s*\n\s*"progress": int\(enrichment_progress\)\s*\n\s*\}\)\s*\n'
    
    # Remove all instances
    content = re.sub(problematic_pattern, '', content, flags=re.MULTILINE)
    
    # Fix 2: Fix indentation issues in the event creation block
    print("📝 Fixing event creation indentation...")
    
    # Find and fix the event creation block
    event_pattern = r'(\s*)# Create the event payload with enhanced metadata\s*\n\s*event = \{\s*\n(\s*)"provider": "excel-upload",\s*\n(\s*)"kind": enriched_payload\.get\('kind', 'transaction'\),\s*\n(\s*)"source_platform": platform_info\.get\('platform', 'unknown'\),\s*\n(\s*)"payload": enriched_payload,  # Use enriched payload instead of raw\s*\n(\s*)"row_index": row_index,\s*\n(\s*)"sheet_name": sheet_name,\s*\n(\s*)"source_filename": file_context\['filename'\],\s*\n(\s*)"uploader": file_context\['user_id'\],\s*\n(\s*)"ingest_ts": datetime\.utcnow\(\)\.isoformat\(\),\s*\n(\s*)"status": "pending",\s*\n(\s*)"confidence_score": enriched_payload\.get\('ai_confidence', 0\.5\),\s*\n(\s*)"classification_metadata": \{\s*\n(\s*)"platform_detection": platform_info,\s*\n(\s*)"ai_classification": ai_classification,\s*\n(\s*)"enrichment_data": enriched_payload,\s*\n(\s*)"row_type": enriched_payload\.get\('kind', 'transaction'\),\s*\n(\s*)"category": enriched_payload\.get\('category', 'other'\),\s*\n(\s*)"subcategory": enriched_payload\.get\('subcategory', 'general'\),\s*\n(\s*)"entities": enriched_payload\.get\('entities', \{\}\),\s*\n(\s*)"relationships": enriched_payload\.get\('relationships', \{\}\),\s*\n(\s*)"description": enriched_payload\.get\('standard_description', ''\),\s*\n(\s*)"reasoning": enriched_payload\.get\('ai_reasoning', ''\),\s*\n(\s*)"sheet_name": sheet_name,\s*\n(\s*)"file_context": file_context\s*\n(\s*)\}\s*\n(\s*)\}\s*\n'
    
    def fix_event_indentation(match):
        indent = match.group(1)
        return f'{indent}# Create the event payload with enhanced metadata\n{indent}event = {{\n{indent}    "provider": "excel-upload",\n{indent}    "kind": enriched_payload.get("kind", "transaction"),\n{indent}    "source_platform": platform_info.get("platform", "unknown"),\n{indent}    "payload": enriched_payload,  # Use enriched payload instead of raw\n{indent}    "row_index": row_index,\n{indent}    "sheet_name": sheet_name,\n{indent}    "source_filename": file_context["filename"],\n{indent}    "uploader": file_context["user_id"],\n{indent}    "ingest_ts": datetime.utcnow().isoformat(),\n{indent}    "status": "pending",\n{indent}    "confidence_score": enriched_payload.get("ai_confidence", 0.5),\n{indent}    "classification_metadata": {{\n{indent}        "platform_detection": platform_info,\n{indent}        "ai_classification": ai_classification,\n{indent}        "enrichment_data": enriched_payload,\n{indent}        "row_type": enriched_payload.get("kind", "transaction"),\n{indent}        "category": enriched_payload.get("category", "other"),\n{indent}        "subcategory": enriched_payload.get("subcategory", "general"),\n{indent}        "entities": enriched_payload.get("entities", {{}}),\n{indent}        "relationships": enriched_payload.get("relationships", {{}}),\n{indent}        "description": enriched_payload.get("standard_description", ""),\n{indent}        "reasoning": enriched_payload.get("ai_reasoning", ""),\n{indent}        "sheet_name": sheet_name,\n{indent}        "file_context": file_context\n{indent}    }}\n{indent}}}\n'
    
    content = re.sub(event_pattern, fix_event_indentation, content, flags=re.MULTILINE)
    
    # Fix 3: Fix any remaining over-indented lines
    print("📝 Fixing remaining indentation issues...")
    
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix over-indented lines that should be at method level
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
    
    print("✅ Comprehensive fix completed!")

if __name__ == "__main__":
    fix_all_errors()
