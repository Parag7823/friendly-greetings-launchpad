#!/usr/bin/env python3
"""
Fix database table population to ensure all tables receive data during upload
"""

import re

def fix_database_population():
    """Ensure all database tables are properly populated"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add database population after relationship detection
    db_population_pattern = r'# Step 9: Update ingestion_jobs with completion'
    
    db_population_replacement = '''# Step 9: Complete Database Population
        await manager.send_update(job_id, {
            "step": "database_population",
            "message": "💾 Populating all database tables with processed data...",
            "progress": 99
        })
        
        try:
            # Store normalized entities
            if insights.get('automatic_relationships', {}).get('total_relationships', 0) > 0:
                # Extract entities from relationships
                entities_to_store = []
                for rel in insights.get('automatic_relationships', {}).get('relationships', []):
                    if rel.get('source_entity'):
                        entities_to_store.append({
                            'user_id': user_id,
                            'entity_name': rel['source_entity'],
                            'entity_type': 'vendor',
                            'normalized_name': rel['source_entity'],
                            'confidence_score': rel.get('confidence_score', 0.8),
                            'source_file': filename,
                            'detected_at': datetime.utcnow().isoformat()
                        })
                    if rel.get('target_entity'):
                        entities_to_store.append({
                            'user_id': user_id,
                            'entity_name': rel['target_entity'],
                            'entity_type': 'vendor',
                            'normalized_name': rel['target_entity'],
                            'confidence_score': rel.get('confidence_score', 0.8),
                            'source_file': filename,
                            'detected_at': datetime.utcnow().isoformat()
                        })
                
                if entities_to_store:
                    # Store normalized entities
                    supabase.table('normalized_entities').insert(entities_to_store).execute()
                    logger.info(f"Stored {len(entities_to_store)} normalized entities")
                
                # Store entity matches
                entity_matches = []
                for i, entity1 in enumerate(entities_to_store):
                    for j, entity2 in enumerate(entities_to_store[i+1:], i+1):
                        if entity1['normalized_name'] == entity2['normalized_name']:
                            entity_matches.append({
                                'user_id': user_id,
                                'entity1_id': entity1.get('id'),
                                'entity2_id': entity2.get('id'),
                                'match_confidence': 0.9,
                                'match_type': 'exact_name',
                                'matched_at': datetime.utcnow().isoformat()
                            })
                
                if entity_matches:
                    supabase.table('entity_matches').insert(entity_matches).execute()
                    logger.info(f"Stored {len(entity_matches)} entity matches")
            
            # Store relationship patterns
            relationship_types = insights.get('automatic_relationships', {}).get('relationship_types', [])
            for rel_type in relationship_types:
                pattern_data = {
                    'id_pattern': None,
                    'date_window': 1,
                    'description': f"Auto-detected pattern for {rel_type}",
                    'amount_match': True,
                    'entity_match': False,
                    'relationship_type': rel_type,
                    'confidence_threshold': 0.7
                }
                
                supabase.table('relationship_patterns').upsert({
                    'user_id': user_id,
                    'relationship_type': rel_type,
                    'pattern_data': pattern_data,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }).execute()
            
            # Store platform patterns
            platform_info = insights.get('processing_stats', {}).get('platform_detected', 'unknown')
            if platform_info != 'unknown':
                platform_pattern = {
                    'user_id': user_id,
                    'platform_name': platform_info,
                    'pattern_data': {
                        'columns': insights.get('processing_stats', {}).get('matched_columns', []),
                        'patterns': insights.get('processing_stats', {}).get('matched_patterns', []),
                        'confidence': insights.get('processing_stats', {}).get('platform_confidence', 0.0)
                    },
                    'detected_at': datetime.utcnow().isoformat()
                }
                
                supabase.table('platform_patterns').upsert(platform_pattern).execute()
                
                # Store discovered platform
                supabase.table('discovered_platforms').upsert({
                    'user_id': user_id,
                    'platform_name': platform_info,
                    'detection_count': 1,
                    'first_detected': datetime.utcnow().isoformat(),
                    'last_detected': datetime.utcnow().isoformat()
                }).execute()
            
            # Store metrics
            metrics_data = {
                'user_id': user_id,
                'metric_type': 'file_processing',
                'metric_value': events_created,
                'metric_details': {
                    'file_name': filename,
                    'total_rows': total_rows,
                    'events_created': events_created,
                    'relationships_detected': insights.get('automatic_relationships', {}).get('total_relationships', 0),
                    'platform_detected': platform_info,
                    'processing_time': datetime.utcnow().isoformat()
                },
                'recorded_at': datetime.utcnow().isoformat()
            }
            
            supabase.table('metrics').insert(metrics_data).execute()
            
            logger.info("Database population completed successfully")
            
        except Exception as e:
            logger.error(f"Database population failed: {e}")
            # Continue processing even if database population fails
        
        # Step 10: Update ingestion_jobs with completion'''
    
    content = re.sub(db_population_pattern, db_population_replacement, content)
    
    # Update the final step number
    final_step_pattern = r'# Step 9: Update ingestion_jobs with completion'
    
    final_step_replacement = '''# Step 10: Update ingestion_jobs with completion'''
    
    content = re.sub(final_step_pattern, final_step_replacement, content)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed database table population")

if __name__ == "__main__":
    fix_database_population()
