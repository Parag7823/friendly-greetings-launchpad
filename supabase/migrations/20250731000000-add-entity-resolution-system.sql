-- Entity Resolution System Migration
-- This adds comprehensive entity resolution capabilities for cross-platform entity matching

-- Create normalized_entities table for unified entity management
CREATE TABLE public.normalized_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Entity identification
    entity_type TEXT NOT NULL CHECK (entity_type IN ('employee', 'vendor', 'customer', 'project')),
    canonical_name TEXT NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    
    -- Strong identifiers for cross-platform matching
    email TEXT,
    phone TEXT,
    bank_account TEXT,
    tax_id TEXT,
    
    -- Platform sources where this entity was found
    platform_sources TEXT[] DEFAULT '{}',
    source_files TEXT[] DEFAULT '{}',
    
    -- Entity metadata
    confidence_score DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    UNIQUE(user_id, entity_type, canonical_name)
);

-- Create entity_matches table for tracking entity resolution decisions
CREATE TABLE public.entity_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Source entity information
    source_entity_name TEXT NOT NULL,
    source_entity_type TEXT NOT NULL,
    source_platform TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_row_id UUID REFERENCES public.raw_events(id) ON DELETE CASCADE,
    
    -- Resolved entity
    normalized_entity_id UUID REFERENCES public.normalized_entities(id) ON DELETE CASCADE,
    match_confidence DECIMAL(3,2) NOT NULL CHECK (match_confidence >= 0 AND match_confidence <= 1),
    match_reason TEXT NOT NULL, -- 'exact_match', 'fuzzy_match', 'email_match', 'bank_match', 'manual'
    
    -- Match metadata
    similarity_score DECIMAL(3,2),
    matched_fields TEXT[], -- ['name', 'email', 'bank_account']
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_normalized_entities_user_id ON public.normalized_entities(user_id);
CREATE INDEX idx_normalized_entities_type ON public.normalized_entities(entity_type);
CREATE INDEX idx_normalized_entities_canonical_name ON public.normalized_entities(canonical_name);
CREATE INDEX idx_normalized_entities_email ON public.normalized_entities(email) WHERE email IS NOT NULL;
CREATE INDEX idx_normalized_entities_bank_account ON public.normalized_entities(bank_account) WHERE bank_account IS NOT NULL;

CREATE INDEX idx_entity_matches_user_id ON public.entity_matches(user_id);
CREATE INDEX idx_entity_matches_source_entity ON public.entity_matches(source_entity_name, source_entity_type);
CREATE INDEX idx_entity_matches_normalized_entity ON public.entity_matches(normalized_entity_id);

-- RLS Policies for normalized_entities
CREATE POLICY "_service_access_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "_user_access_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for entity_matches
CREATE POLICY "_service_access_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "_user_access_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.uid() = user_id);

-- Function to find or create normalized entity
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
    ELSE
        -- Try exact match by email (strong identifier)
        IF p_email IS NOT NULL THEN
            SELECT id INTO v_entity_id
            FROM public.normalized_entities
            WHERE user_id = p_user_id 
            AND entity_type = p_entity_type 
            AND email = p_email;
            
            IF v_entity_id IS NOT NULL THEN
                v_match_reason := 'email_match';
                v_matched_fields := ARRAY['email'];
                v_similarity_score := 0.95;
            END IF;
        END IF;
        
        -- Try exact match by bank account (strong identifier)
        IF v_entity_id IS NULL AND p_bank_account IS NOT NULL THEN
            SELECT id INTO v_entity_id
            FROM public.normalized_entities
            WHERE user_id = p_user_id 
            AND entity_type = p_entity_type 
            AND bank_account = p_bank_account;
            
            IF v_entity_id IS NOT NULL THEN
                v_match_reason := 'bank_match';
                v_matched_fields := ARRAY['bank_account'];
                v_similarity_score := 0.95;
            END IF;
        END IF;
        
        -- Try fuzzy matching by name similarity
        IF v_entity_id IS NULL THEN
            -- Find best fuzzy match
            FOR v_entity_id, v_similarity_score IN
                SELECT id, 
                       GREATEST(
                           -- Jaro-Winkler similarity
                           CASE 
                               WHEN p_entity_name = canonical_name THEN 1.0
                               WHEN p_entity_name ILIKE '%' || canonical_name || '%' THEN 0.9
                               WHEN canonical_name ILIKE '%' || p_entity_name || '%' THEN 0.9
                               ELSE 0.0
                           END,
                           -- Check aliases
                           (SELECT COALESCE(MAX(
                               CASE 
                                   WHEN p_entity_name = alias THEN 1.0
                                   WHEN p_entity_name ILIKE '%' || alias || '%' THEN 0.9
                                   WHEN alias ILIKE '%' || p_entity_name || '%' THEN 0.9
                                   ELSE 0.0
                               END
                           ), 0.0)
                           FROM unnest(aliases) AS alias)
                       ) as similarity
                FROM public.normalized_entities
                WHERE user_id = p_user_id 
                AND entity_type = p_entity_type
            LOOP
                IF v_similarity_score > v_best_score AND v_similarity_score >= 0.8 THEN
                    v_best_match_id := v_entity_id;
                    v_best_score := v_similarity_score;
                END IF;
            END LOOP;
            
            IF v_best_match_id IS NOT NULL THEN
                v_entity_id := v_best_match_id;
                v_match_reason := 'fuzzy_match';
                v_matched_fields := ARRAY['name'];
                v_similarity_score := v_best_score;
            END IF;
        END IF;
    END IF;
    
    -- If no match found, create new entity
    IF v_entity_id IS NULL THEN
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
    
    RETURN v_entity_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get entity resolution statistics
CREATE OR REPLACE FUNCTION get_entity_resolution_stats(user_uuid UUID)
RETURNS TABLE(
    total_entities INTEGER,
    employees_count INTEGER,
    vendors_count INTEGER,
    customers_count INTEGER,
    projects_count INTEGER,
    total_matches INTEGER,
    exact_matches INTEGER,
    fuzzy_matches INTEGER,
    email_matches INTEGER,
    bank_matches INTEGER,
    new_entities INTEGER,
    avg_confidence DECIMAL(3,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT ne.id)::INTEGER as total_entities,
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'employee' THEN ne.id END)::INTEGER as employees_count,
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'vendor' THEN ne.id END)::INTEGER as vendors_count,
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'customer' THEN ne.id END)::INTEGER as customers_count,
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'project' THEN ne.id END)::INTEGER as projects_count,
        COUNT(em.id)::INTEGER as total_matches,
        COUNT(CASE WHEN em.match_reason = 'exact_match' THEN 1 END)::INTEGER as exact_matches,
        COUNT(CASE WHEN em.match_reason = 'fuzzy_match' THEN 1 END)::INTEGER as fuzzy_matches,
        COUNT(CASE WHEN em.match_reason = 'email_match' THEN 1 END)::INTEGER as email_matches,
        COUNT(CASE WHEN em.match_reason = 'bank_match' THEN 1 END)::INTEGER as bank_matches,
        COUNT(CASE WHEN em.match_reason = 'new_entity' THEN 1 END)::INTEGER as new_entities,
        AVG(em.match_confidence)::DECIMAL(3,2) as avg_confidence
    FROM public.normalized_entities ne
    LEFT JOIN public.entity_matches em ON ne.id = em.normalized_entity_id
    WHERE ne.user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to search entities by name similarity
CREATE OR REPLACE FUNCTION search_entities_by_name(user_uuid UUID, search_term TEXT, p_entity_type TEXT DEFAULT NULL)
RETURNS TABLE(
    id UUID,
    entity_type TEXT,
    canonical_name TEXT,
    aliases TEXT[],
    email TEXT,
    platform_sources TEXT[],
    confidence_score DECIMAL(3,2),
    similarity_score DECIMAL(3,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ne.id,
        ne.entity_type,
        ne.canonical_name,
        ne.aliases,
        ne.email,
        ne.platform_sources,
        ne.confidence_score,
        GREATEST(
            -- Exact match
            CASE WHEN ne.canonical_name ILIKE search_term THEN 1.0 ELSE 0.0 END,
            -- Contains match
            CASE WHEN ne.canonical_name ILIKE '%' || search_term || '%' THEN 0.8 ELSE 0.0 END,
            -- Alias match
            (SELECT COALESCE(MAX(
                CASE 
                    WHEN alias ILIKE search_term THEN 1.0
                    WHEN alias ILIKE '%' || search_term || '%' THEN 0.8
                    ELSE 0.0
                END
            ), 0.0)
            FROM unnest(ne.aliases) AS alias)
        ) as similarity_score
    FROM public.normalized_entities ne
    WHERE ne.user_id = user_uuid
    AND (p_entity_type IS NULL OR ne.entity_type = p_entity_type)
    AND (
        ne.canonical_name ILIKE '%' || search_term || '%'
        OR ne.aliases && ARRAY[search_term]
        OR EXISTS (
            SELECT 1 FROM unnest(ne.aliases) AS alias 
            WHERE alias ILIKE '%' || search_term || '%'
        )
    )
    ORDER BY similarity_score DESC, ne.canonical_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get entity details with all related events
CREATE OR REPLACE FUNCTION get_entity_details(user_uuid UUID, entity_id UUID)
RETURNS TABLE(
    entity_info JSONB,
    related_events JSONB,
    match_history JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        -- Entity information
        jsonb_build_object(
            'id', ne.id,
            'entity_type', ne.entity_type,
            'canonical_name', ne.canonical_name,
            'aliases', ne.aliases,
            'email', ne.email,
            'phone', ne.phone,
            'bank_account', ne.bank_account,
            'tax_id', ne.tax_id,
            'platform_sources', ne.platform_sources,
            'source_files', ne.source_files,
            'confidence_score', ne.confidence_score,
            'first_seen_at', ne.first_seen_at,
            'last_seen_at', ne.last_seen_at
        ) as entity_info,
        
        -- Related events
        COALESCE(
            (SELECT jsonb_agg(
                jsonb_build_object(
                    'id', re.id,
                    'kind', re.kind,
                    'category', re.category,
                    'subcategory', re.subcategory,
                    'source_platform', re.source_platform,
                    'payload', re.payload,
                    'source_filename', re.source_filename,
                    'created_at', re.created_at
                )
            )
            FROM public.raw_events re
            WHERE re.user_id = user_uuid
            AND re.entities ? ne.entity_type
            AND re.entities->ne.entity_type ? ne.canonical_name
            ), '[]'::jsonb
        ) as related_events,
        
        -- Match history
        COALESCE(
            (SELECT jsonb_agg(
                jsonb_build_object(
                    'source_entity_name', em.source_entity_name,
                    'source_platform', em.source_platform,
                    'source_file', em.source_file,
                    'match_confidence', em.match_confidence,
                    'match_reason', em.match_reason,
                    'similarity_score', em.similarity_score,
                    'matched_fields', em.matched_fields,
                    'created_at', em.created_at
                )
            )
            FROM public.entity_matches em
            WHERE em.user_id = user_uuid
            AND em.normalized_entity_id = entity_id
            ), '[]'::jsonb
        ) as match_history
    FROM public.normalized_entities ne
    WHERE ne.user_id = user_uuid
    AND ne.id = entity_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 