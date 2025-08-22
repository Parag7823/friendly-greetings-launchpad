#!/usr/bin/env python3
"""
Add automatic relationship detection to the upload flow
"""

import re

def add_automatic_relationships():
    """Add automatic relationship detection to the upload flow"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the first occurrence of Step 8 (in the main process_file method)
    # Look for the pattern that includes the insights return
    pattern = r'# Step 8: Update ingestion_jobs with completion\s+supabase\.table\(\'ingestion_jobs\'\)\.update\(\{\s+\'status\': \'completed\',\s+\'updated_at\': datetime\.utcnow\(\)\.isoformat\(\)\s+\}\)\.eq\(\'id\', job_id\)\.execute\(\)\s+\s+await manager\.send_update\(job_id, \{\s+"step": "completed",\s+"message": f"✅ Processing completed! \{events_created\} events created from \{processed_rows\} rows\.",\s+"progress": 100\s+\}\)\s+\s+return insights'
    
    replacement = '''# Step 8: Automatic Relationship Detection
        await manager.send_update(job_id, {
            "step": "relationships",
            "message": "🔗 Detecting financial relationships automatically...",
            "progress": 98
        })
        
        try:
            # Initialize Enhanced Relationship Detector
            enhanced_detector = EnhancedRelationshipDetector(openai, supabase)
            
            # Detect relationships automatically
            relationship_results = await enhanced_detector.detect_all_relationships(user_id)
            
            # Store relationship results in insights
            insights['automatic_relationships'] = {
                'total_relationships': relationship_results.get('total_relationships', 0),
                'relationship_types': relationship_results.get('relationship_types', []),
                'detection_method': 'automatic_upload_processing',
                'detected_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Automatic relationship detection completed: {relationship_results.get('total_relationships', 0)} relationships found")
            
        except Exception as e:
            logger.error(f"Automatic relationship detection failed: {e}")
            insights['automatic_relationships'] = {
                'error': str(e),
                'detection_method': 'automatic_upload_processing_failed',
                'detected_at': datetime.utcnow().isoformat()
            }
        
        # Step 9: Update ingestion_jobs with completion
        supabase.table('ingestion_jobs').update({
            'status': 'completed',
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', job_id).execute()
        
        await manager.send_update(job_id, {
            "step": "completed",
            "message": f"✅ Processing completed! {events_created} events created, {insights.get('automatic_relationships', {}).get('total_relationships', 0)} relationships detected.",
            "progress": 100
        })
        
        return insights'''
    
    # Replace the first occurrence only
    content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added automatic relationship detection to upload flow")

if __name__ == "__main__":
    add_automatic_relationships()
