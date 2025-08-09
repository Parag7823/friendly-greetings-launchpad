-- Fix entity resolution over-merging by adding name similarity checks
-- This prevents entities with different names from being merged just because they share identifiers

-- Drop the old function
DROP FUNCTION IF EXISTS find_or_create_entity(UUID, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT);

-- Create the improved function with name similarity checks
CREATE OR REPLACE FUNCTION find_or_create_entity(
    p_user_id UUID,
    p_entity_name TEXT,
    p_entity_type TEXT,
    p_platform TEXT,
    p_email TEXT DEFAULT NULL,
    p_bank_account TEXT DEFAULT NULL,
    p_phone TEXT DEFAULT NULL,
    p_tax_id TEXT DEFAULT NULL,
    p_source_file TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_entity_id UUID;
    v_similarity_score DECIMAL(3,2);
    v_best_match_id UUID;
    v_best_score DECIMAL(3,2) := 0;
    v_match_reason TEXT;
    v_matched_fields TEXT[];
    v_name_similarity DECIMAL(3,2);
    v_should_merge BOOLEAN := FALSE;
BEGIN
    -- First, try exact match by canonical name
    SELECT id INTO v_entity_id
    FROM public.normalized_entities
    WHERE user_id = p_user_id 
    AND entity_type = p_entity_type 
    AND canonical_name = p_entity_name;
    
    IF v_entity_id IS NOT NULL THEN
        v_match_reason := 'exact_match';
        v_matched_fields := ARRAY['name'];
        v_similarity_score := 1.0;
        v_should_merge := TRUE;
    ELSE
        -- CRITICAL FIX: Check for entities with matching identifiers but require name similarity
        IF p_email IS NOT NULL THEN
            -- Find entities with matching email
            FOR v_entity_id, v_name_similarity IN
                SELECT id, 
                       GREATEST(
                           -- Exact name match
                           CASE WHEN p_entity_name = canonical_name THEN 1.0 ELSE 0.0 END,
                           -- Contains match
                           CASE WHEN p_entity_name ILIKE '%' || canonical_name || '%' THEN 0.8 ELSE 0.0 END,
                           CASE WHEN canonical_name ILIKE '%' || p_entity_name || '%' THEN 0.8 ELSE 0.0 END,
                           -- Alias match
                           (SELECT COALESCE(MAX(
                               CASE 
                                   WHEN p_entity_name = alias THEN 1.0
                                   WHEN p_entity_name ILIKE '%' || alias || '%' THEN 0.8
                                   WHEN alias ILIKE '%' || p_entity_name || '%' THEN 0.8
                                   ELSE 0.0
                               END
                           ), 0.0)
                           FROM unnest(aliases) AS alias)
                       ) as name_similarity
                FROM public.normalized_entities
                WHERE user_id = p_user_id 
                AND entity_type = p_entity_type 
                AND email = p_email
            LOOP
                IF v_name_similarity > v_best_score THEN
                    v_best_match_id := v_entity_id;
                    v_best_score := v_name_similarity;
                END IF;
            END LOOP;
            
            -- Only merge if name similarity is high enough (≥0.8)
            IF v_best_match_id IS NOT NULL AND v_best_score >= 0.8 THEN
                v_entity_id := v_best_match_id;
                v_match_reason := 'email_match_with_name_similarity';
                v_matched_fields := ARRAY['email', 'name'];
                v_similarity_score := v_best_score;
                v_should_merge := TRUE;
            END IF;
        END IF;
        
        -- Check bank account with name similarity
        IF v_entity_id IS NULL AND p_bank_account IS NOT NULL THEN
            v_best_match_id := NULL;
            v_best_score := 0;
            
            FOR v_entity_id, v_name_similarity IN
                SELECT id, 
                       GREATEST(
                           -- Exact name match
                           CASE WHEN p_entity_name = canonical_name THEN 1.0 ELSE 0.0 END,
                           -- Contains match
                           CASE WHEN p_entity_name ILIKE '%' || canonical_name || '%' THEN 0.8 ELSE 0.0 END,
                           CASE WHEN canonical_name ILIKE '%' || p_entity_name || '%' THEN 0.8 ELSE 0.0 END,
                           -- Alias match
                           (SELECT COALESCE(MAX(
                               CASE 
                                   WHEN p_entity_name = alias THEN 1.0
                                   WHEN p_entity_name ILIKE '%' || alias || '%' THEN 0.8
                                   WHEN alias ILIKE '%' || p_entity_name || '%' THEN 0.8
                                   ELSE 0.0
                               END
                           ), 0.0)
                           FROM unnest(aliases) AS alias)
                       ) as name_similarity
                FROM public.normalized_entities
                WHERE user_id = p_user_id 
                AND entity_type = p_entity_type 
                AND bank_account = p_bank_account
            LOOP
                IF v_name_similarity > v_best_score THEN
                    v_best_match_id := v_entity_id;
                    v_best_score := v_name_similarity;
                END IF;
            END LOOP;
            
            -- Only merge if name similarity is high enough (≥0.8)
            IF v_best_match_id IS NOT NULL AND v_best_score >= 0.8 THEN
                v_entity_id := v_best_match_id;
                v_match_reason := 'bank_match_with_name_similarity';
                v_matched_fields := ARRAY['bank_account', 'name'];
                v_similarity_score := v_best_score;
                v_should_merge := TRUE;
            END IF;
        END IF;
        
        -- Try fuzzy matching by name similarity (no identifier requirement)
        IF v_entity_id IS NULL THEN
            v_best_match_id := NULL;
            v_best_score := 0;
            
            FOR v_entity_id, v_name_similarity IN
                SELECT id, 
                       GREATEST(
                           -- Exact name match
                           CASE WHEN p_entity_name = canonical_name THEN 1.0 ELSE 0.0 END,
                           -- Contains match
                           CASE WHEN p_entity_name ILIKE '%' || canonical_name || '%' THEN 0.8 ELSE 0.0 END,
                           CASE WHEN canonical_name ILIKE '%' || p_entity_name || '%' THEN 0.8 ELSE 0.0 END,
                           -- Alias match
                           (SELECT COALESCE(MAX(
                               CASE 
                                   WHEN p_entity_name = alias THEN 1.0
                                   WHEN p_entity_name ILIKE '%' || alias || '%' THEN 0.8
                                   WHEN alias ILIKE '%' || p_entity_name || '%' THEN 0.8
                                   ELSE 0.0
                               END
                           ), 0.0)
                           FROM unnest(aliases) AS alias)
                       ) as name_similarity
                FROM public.normalized_entities
                WHERE user_id = p_user_id 
                AND entity_type = p_entity_type
            LOOP
                IF v_name_similarity > v_best_score AND v_name_similarity >= 0.8 THEN
                    v_best_match_id := v_entity_id;
                    v_best_score := v_name_similarity;
                END IF;
            END LOOP;
            
            IF v_best_match_id IS NOT NULL THEN
                v_entity_id := v_best_match_id;
                v_match_reason := 'fuzzy_name_match';
                v_matched_fields := ARRAY['name'];
                v_similarity_score := v_best_score;
                v_should_merge := TRUE;
            END IF;
        END IF;
    END IF;
    
    -- If no match found or similarity too low, create new entity
    IF v_entity_id IS NULL OR NOT v_should_merge THEN
        INSERT INTO public.normalized_entities (
            user_id, entity_type, canonical_name, aliases, email, phone, bank_account, tax_id,
            platform_sources, source_files, confidence_score
        ) VALUES (
            p_user_id, p_entity_type, p_entity_name, ARRAY[p_entity_name], p_email, p_phone, p_bank_account, p_tax_id,
            ARRAY[p_platform], ARRAY[p_source_file], 0.5
        ) RETURNING id INTO v_entity_id;
        
        v_match_reason := 'new_entity';
        v_matched_fields := ARRAY['name'];
        v_similarity_score := 1.0;
    ELSE
        -- Update existing entity with new information
        UPDATE public.normalized_entities
        SET 
            aliases = CASE 
                WHEN NOT (p_entity_name = ANY(aliases)) THEN aliases || p_entity_name
                ELSE aliases
            END,
            platform_sources = CASE 
                WHEN NOT (p_platform = ANY(platform_sources)) THEN platform_sources || p_platform
                ELSE platform_sources
            END,
            source_files = CASE 
                WHEN p_source_file IS NOT NULL AND NOT (p_source_file = ANY(source_files)) 
                THEN source_files || p_source_file
                ELSE source_files
            END,
            email = COALESCE(p_email, email),
            phone = COALESCE(p_phone, phone),
            bank_account = COALESCE(p_bank_account, bank_account),
            tax_id = COALESCE(p_tax_id, tax_id),
            last_seen_at = now(),
            updated_at = now()
        WHERE id = v_entity_id;
    END IF;
    
    -- Log the match for debugging
    INSERT INTO public.entity_matches (
        user_id, normalized_entity_id, source_entity_name, source_entity_type, 
        source_platform, source_file, match_confidence, match_reason, 
        similarity_score, matched_fields
    ) VALUES (
        p_user_id, v_entity_id, p_entity_name, p_entity_type, 
        p_platform, p_source_file, v_similarity_score, v_match_reason, 
        v_similarity_score, v_matched_fields
    );
    
    RETURN v_entity_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add comment explaining the fix
COMMENT ON FUNCTION find_or_create_entity IS 'Fixed entity resolution to prevent over-merging by requiring name similarity (≥0.8) when matching by email/bank_account'; 