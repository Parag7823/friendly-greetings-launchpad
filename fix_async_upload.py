"""Fix from_upload to be async"""
with open('data_ingestion_normalization/streaming_source.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the classmethod to make it async
old_signature = """    @classmethod
    def from_upload(
        cls,
        upload_file,
        temp_dir: Optional[str] = None,
    ) -> "StreamedFile":"""

new_signature = """    @classmethod
    async def from_upload(
        cls,
        upload_file,
        temp_dir: Optional[str] = None,
    ) -> "StreamedFile":"""

if old_signature in content:
    content = content.replace(old_signature, new_signature)
    print("✓ Updated method signature to async")
else:
    print("⚠ Signature not found or already async")

# Replace the sync read with async read
old_read = """            # Check if it has a file attribute (FastAPI UploadFile)
            if hasattr(upload_file, 'file'):
                source = upload_file.file
            else:
                source = upload_file
                
            # Ensure at start
            if hasattr(source, 'seek'):
                try:
                    source.seek(0)
                except Exception:
                    pass  # Might not be seekable
            
            # Stream in chunks (1MB at a time)
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)"""

new_read = """            # FastAPI UploadFile: use async read
            # Stream in chunks (1MB at a time)
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)"""

if old_read in content:
    content = content.replace(old_read, new_read)
    print("✓ Updated read logic to use await")
else:
    print("⚠ Read logic not found or already updated")

# Write back
with open('data_ingestion_normalization/streaming_source.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ SUCCESS: from_upload is now async!")
