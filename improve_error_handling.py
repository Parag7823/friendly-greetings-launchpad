#!/usr/bin/env python3
"""
Improve error handling and progress feedback throughout the system
"""

import re

def improve_error_handling():
    """Improve error handling and progress feedback"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add comprehensive error handling to the main process_file method
    # Find the pattern where we process rows and add better error handling
    
    # Add better progress feedback for data enrichment
    enrichment_pattern = r'# Data enrichment - create enhanced payload\s+enriched_payload = await self\.enrichment_processor\.enrich_row_data\('
    
    enrichment_replacement = '''# Data enrichment - create enhanced payload
                            # Update progress for data enrichment
                            if row_index % 10 == 0:  # Update every 10 rows
                                enrichment_progress = 40 + (processed_rows / total_rows) * 30
                                await manager.send_update(job_id, {
                                    "step": "enrichment",
                                    "message": f"🔧 Enriching data for row {row_index}/{total_rows}...",
                                    "progress": int(enrichment_progress)
                                })
                            
                            enriched_payload = await self.enrichment_processor.enrich_row_data('''
    
    content = re.sub(enrichment_pattern, enrichment_replacement, content, flags=re.DOTALL)
    
    # Add better error handling for AI classification failures
    ai_error_pattern = r'except Exception as e:\s+logger\.error\(f"Error processing row \{row_index\} in sheet \{sheet_name\}: \{str\(e\)\}"\)\s+errors\.append\(error_msg\)'
    
    ai_error_replacement = '''except Exception as e:
                            error_msg = f"Error processing row {row_index} in sheet {sheet_name}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            
                            # Continue processing other rows instead of failing completely
                            continue'''
    
    content = re.sub(ai_error_pattern, ai_error_replacement, content, flags=re.DOTALL)
    
    # Add validation for required fields before storing events
    validation_pattern = r'# Create the event payload with enhanced metadata\s+event = \{'
    
    validation_replacement = '''# Validate required fields before creating event
                            if not enriched_payload.get('kind'):
                                enriched_payload['kind'] = 'transaction'
                            if not enriched_payload.get('category'):
                                enriched_payload['category'] = 'other'
                            
                            # Create the event payload with enhanced metadata
                            event = {'''
    
    content = re.sub(validation_pattern, validation_replacement, content, flags=re.DOTALL)
    
    # Add better success indicators
    success_pattern = r'"message": f"✅ Processing completed! \{events_created\} events created, \{insights\.get\(\'automatic_relationships\', \{\}\)\.get\(\'total_relationships\', 0\)\} relationships detected\."'
    
    success_replacement = '''"message": f"✅ Processing completed! {events_created} events created, {insights.get('automatic_relationships', {}).get('total_relationships', 0)} relationships detected. Data enrichment: {sum(1 for e in insights.get('processing_stats', {}).get('enrichment_stats', {}).values() if e > 0)} fields enriched."'''
    
    content = re.sub(success_pattern, success_replacement, content)
    
    # Write the improved content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Improved error handling and progress feedback")

if __name__ == "__main__":
    improve_error_handling()
