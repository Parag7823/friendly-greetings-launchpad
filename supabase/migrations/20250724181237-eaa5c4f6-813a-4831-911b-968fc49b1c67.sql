-- Create a special policy for service-level test inserts
-- This will allow inserts when the source is 'integration_test'
CREATE POLICY "Allow integration test inserts" 
ON public.raw_records 
FOR INSERT 
WITH CHECK (source = 'integration_test');

-- Also create a test table specifically for integration tests that doesn't need user auth
CREATE TABLE public.integration_test_logs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  message TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  source TEXT NOT NULL DEFAULT 'fastapi_test',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS but make it publicly accessible for testing
ALTER TABLE public.integration_test_logs ENABLE ROW LEVEL SECURITY;

-- Allow anyone to insert test logs (this is for testing only)
CREATE POLICY "Anyone can insert test logs" 
ON public.integration_test_logs 
FOR INSERT 
WITH CHECK (true);

-- Allow anyone to read test logs (this is for testing only)
CREATE POLICY "Anyone can read test logs" 
ON public.integration_test_logs 
FOR SELECT 
USING (true);