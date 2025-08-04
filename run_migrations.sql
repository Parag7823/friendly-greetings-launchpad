-- Run this in your Supabase SQL Editor to fix the issues

-- 1. Create the test user
INSERT INTO auth.users (
    id,
    email,
    encrypted_password,
    email_confirmed_at,
    created_at,
    updated_at,
    raw_app_meta_data,
    raw_user_meta_data,
    is_super_admin,
    confirmation_token,
    email_change,
    email_change_token_new,
    recovery_token
) VALUES (
    '550e8400-e29b-41d4-a716-446655440000',
    'test@finley.ai',
    crypt('testpassword', gen_salt('bf')),
    now(),
    now(),
    now(),
    '{"provider": "email", "providers": ["email"]}',
    '{"name": "Test User"}',
    false,
    '',
    '',
    '',
    ''
) ON CONFLICT (id) DO NOTHING;

-- 2. Fix the aggregate function issue
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
DECLARE
    v_total_entities INTEGER;
    v_employees_count INTEGER;
    v_vendors_count INTEGER;
    v_customers_count INTEGER;
    v_projects_count INTEGER;
    v_total_matches INTEGER;
    v_exact_matches INTEGER;
    v_fuzzy_matches INTEGER;
    v_email_matches INTEGER;
    v_bank_matches INTEGER;
    v_new_entities INTEGER;
    v_avg_confidence DECIMAL(3,2);
BEGIN
    -- Get entity counts
    SELECT 
        COUNT(DISTINCT ne.id),
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'employee' THEN ne.id END),
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'vendor' THEN ne.id END),
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'customer' THEN ne.id END),
        COUNT(DISTINCT CASE WHEN ne.entity_type = 'project' THEN ne.id END)
    INTO v_total_entities, v_employees_count, v_vendors_count, v_customers_count, v_projects_count
    FROM public.normalized_entities ne
    WHERE ne.user_id = user_uuid;
    
    -- Get match counts and average confidence
    SELECT 
        COUNT(em.id),
        COUNT(CASE WHEN em.match_reason = 'exact_match' THEN 1 END),
        COUNT(CASE WHEN em.match_reason = 'fuzzy_match' THEN 1 END),
        COUNT(CASE WHEN em.match_reason = 'email_match' THEN 1 END),
        COUNT(CASE WHEN em.match_reason = 'bank_match' THEN 1 END),
        COUNT(CASE WHEN em.match_reason = 'new_entity' THEN 1 END),
        AVG(em.match_confidence)
    INTO v_total_matches, v_exact_matches, v_fuzzy_matches, v_email_matches, v_bank_matches, v_new_entities, v_avg_confidence
    FROM public.normalized_entities ne
    LEFT JOIN public.entity_matches em ON ne.id = em.normalized_entity_id
    WHERE ne.user_id = user_uuid;
    
    RETURN QUERY SELECT 
        v_total_entities,
        v_employees_count,
        v_vendors_count,
        v_customers_count,
        v_projects_count,
        v_total_matches,
        v_exact_matches,
        v_fuzzy_matches,
        v_email_matches,
        v_bank_matches,
        v_new_entities,
        v_avg_confidence;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 3. Also insert into public.users table if it exists
INSERT INTO public.users (
    id,
    email,
    created_at,
    updated_at
) VALUES (
    '550e8400-e29b-41d4-a716-446655440000',
    'test@finley.ai',
    now(),
    now()
) ON CONFLICT (id) DO NOTHING; 