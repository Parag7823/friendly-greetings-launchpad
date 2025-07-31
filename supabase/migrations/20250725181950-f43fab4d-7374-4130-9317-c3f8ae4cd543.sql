-- Drop all existing storage policies
DROP POLICY IF EXISTS "Allow authenticated uploads to finely-upload" ON storage.objects;
DROP POLICY IF EXISTS "Allow authenticated access to finely-upload" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to view their own files in finely-upload" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to update their own files in finely-upload" ON storage.objects;
DROP POLICY IF EXISTS "Allow users to delete their own files in finely-upload" ON storage.objects;

-- Create completely permissive policies for the finely-upload bucket
CREATE POLICY "Allow all operations on finely-upload"
ON storage.objects
FOR ALL
USING (bucket_id = 'finely-upload')
WITH CHECK (bucket_id = 'finely-upload');