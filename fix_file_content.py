"""Fix undefined file_content variable in process-with-websocket endpoint"""
with open('core_infrastructure/fastapi_backend_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# Fix 1: Change "file_content": file_content to use first_sheet_data or streamed_file.read_bytes()
# Since file_content is undefined, we need to replace it with actual data

# In platform_payload (line 5469), replace file_content with None or sample data
old1 = '"file_content": file_content,'
new1 = '"file_content": first_sheet_data.to_dicts() if first_sheet_data is not None else [],'
if old1 in content:
    content = content.replace(old1, new1)
    fixes += 1
    print("Fixed 1: platform_payload file_content")

# In document classifier payload (line 5501), same fix
old2 = 'payload={"file_content": file_content, "filename": filename}'
new2 = 'payload={"file_content": first_sheet_data.to_dicts() if first_sheet_data is not None else [], "filename": filename}'
if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1
    print("Fixed 2: document_classifier payload file_content")

if fixes > 0:
    with open('core_infrastructure/fastapi_backend_v2.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"SUCCESS: Fixed {fixes} undefined file_content references!")
else:
    print("No patterns found to fix")
