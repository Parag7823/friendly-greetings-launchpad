"""Add await to from_upload call"""
with open('core_infrastructure/fastapi_backend_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = "        streamed_file = StreamedFile.from_upload(file)"
new = "        streamed_file = await StreamedFile.from_upload(file)"

if old in content:
    content = content.replace(old, new)
    with open('core_infrastructure/fastapi_backend_v2.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ SUCCESS: Added await to from_upload call")
else:
    if new in content:
        print("✓ Already has await")
    else:
        print("⚠ Pattern not found")
