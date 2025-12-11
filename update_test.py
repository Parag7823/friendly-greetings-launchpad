"""Update load test to use simple endpoint"""
with open('tests/test_load_stages1to3.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change endpoint from /api/process-with-websocket to /api/upload-simple
old = '"/api/process-with-websocket"'
new = '"/api/upload-simple"'

if old in content:
    content = content.replace(old, new)
    with open('tests/test_load_stages1to3.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Updated load test to use /api/upload-simple")
else:
    print("Pattern not found")
