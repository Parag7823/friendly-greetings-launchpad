-- Update the get_entity_details function to use proper table references
DROP FUNCTION IF EXISTS public.get_entity_details(uuid, uuid);

CREATE OR REPLACE FUNCTION public.get_entity_details(user_uuid uuid, entity_id uuid)
RETURNS TABLE(
  entity_info jsonb,
  related_events jsonb,
  match_history jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
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
        
        -- Related events as jsonb array
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
        
        -- Match history as jsonb array
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
$$;