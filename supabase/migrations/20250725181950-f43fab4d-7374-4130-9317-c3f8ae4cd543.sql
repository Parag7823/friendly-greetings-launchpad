-- Drop all existing storage policies
DROP POLICY IF EXISTS "Allow authenticated uploads to finley-uploads" ON storage.objects;
DROP POLICY IF EXISTS "Allow authenticated access to finley-uploads" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to view their own files in finley-uploads" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to update their own files in finley-uploads" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to delete their own files in finley-uploads" ON storage.objects;

-- Create completely permissive policies for the finley-uploads bucket
CREATE POLICY "Allow all operations on finley-uploads"
ON storage.objects
FOR ALL
USING (bucket_id = 'finley-uploads')
WITH CHECK (bucket_id = 'finley-uploads');