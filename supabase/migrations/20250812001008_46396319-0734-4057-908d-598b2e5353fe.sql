-- Enable RLS on tables that already have policies defined
ALTER TABLE public.entity_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.normalized_entities ENABLE ROW LEVEL SECURITY;