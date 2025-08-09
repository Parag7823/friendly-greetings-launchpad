-- Fix over-merged entities by separating them based on name similarity
-- This script should be run after applying the migration to fix the function

-- First, let's see what we have
SELECT 
    entity_type,
    canonical_name,
    array_length(aliases, 1) as alias_count,
    email,
    bank_account,
    platform_sources
FROM normalized_entities 
WHERE user_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY alias_count DESC;

-- Create a function to separate over-merged entities
CREATE OR REPLACE FUNCTION separate_overmerged_entities(p_user_id UUID) 
RETURNS TABLE(entity_id UUID, action TEXT, details TEXT) AS $$
DECLARE
    v_entity RECORD;
    v_alias TEXT;
    v_new_entity_id UUID;
    v_similarity DECIMAL(3,2);
    v_should_merge BOOLEAN;
BEGIN
    -- Process each entity with many aliases
    FOR v_entity IN 
        SELECT id, entity_type, canonical_name, aliases, email, bank_account, platform_sources, source_files
        FROM normalized_entities 
        WHERE user_id = p_user_id 
        AND array_length(aliases, 1) > 10  -- Only process entities with many aliases
    LOOP
        -- For each alias, check if it should be a separate entity
        FOREACH v_alias IN ARRAY v_entity.aliases
        LOOP
            -- Skip if it's the canonical name
            IF v_alias = v_entity.canonical_name THEN
                CONTINUE;
            END IF;
            
            -- Calculate similarity with canonical name
            v_similarity := GREATEST(
                -- Exact match
                CASE WHEN v_alias = v_entity.canonical_name THEN 1.0 ELSE 0.0 END,
                -- Contains match
                CASE WHEN v_alias ILIKE '%' || v_entity.canonical_name || '%' THEN 0.8 ELSE 0.0 END,
                CASE WHEN v_entity.canonical_name ILIKE '%' || v_alias || '%' THEN 0.8 ELSE 0.0 END
            );
            
            -- If similarity is low, create new entity
            IF v_similarity < 0.8 THEN
                -- Create new entity for this alias
                INSERT INTO normalized_entities (
                    user_id, entity_type, canonical_name, aliases, email, bank_account,
                    platform_sources, source_files, confidence_score
                ) VALUES (
                    p_user_id, v_entity.entity_type, v_alias, ARRAY[v_alias], 
                    v_entity.email, v_entity.bank_account, v_entity.platform_sources, 
                    v_entity.source_files, 0.5
                ) RETURNING id INTO v_new_entity_id;
                
                -- Return the action
                entity_id := v_new_entity_id;
                action := 'created_new_entity';
                details := format('Separated "%s" from "%s" (similarity: %s)', v_alias, v_entity.canonical_name, v_similarity);
                RETURN NEXT;
            END IF;
        END LOOP;
        
        -- Clean up the original entity by removing separated aliases
        -- This is a simplified approach - in production you'd want to be more careful
        UPDATE normalized_entities 
        SET aliases = ARRAY[canonical_name]  -- Keep only the canonical name
        WHERE id = v_entity.id;
        
        entity_id := v_entity.id;
        action := 'cleaned_original_entity';
        details := format('Cleaned aliases for "%s"', v_entity.canonical_name);
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Run the separation
SELECT * FROM separate_overmerged_entities('550e8400-e29b-41d4-a716-446655440000');

-- Show the results
SELECT 
    entity_type,
    canonical_name,
    array_length(aliases, 1) as alias_count,
    email,
    bank_account,
    platform_sources
FROM normalized_entities 
WHERE user_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY alias_count DESC, canonical_name;

-- Clean up the function
DROP FUNCTION separate_overmerged_entities(UUID); 