"""Remove temporary simple endpoint code"""
with open('core_infrastructure/fastapi_backend_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Marker used to insert
marker = '# Simple upload endpoint for testing (no complex dependencies)'

if marker in content:
    # Remove everything from marker to @app.get("/health")
    parts = content.split(marker)
    if len(parts) > 1:
        prefix = parts[0]
        # Find where it ends (at next endpoint or health)
        rest = parts[1]
        if '@app.get("/health")' in rest:
            suffix = rest.split('@app.get("/health")', 1)[1]
            new_content = prefix + '\n\n@app.get("/health")' + suffix
            
            with open('core_infrastructure/fastapi_backend_v2.py', 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("SUCCESS: Removed simple endpoint")
        else:
            print("Could not find end of simple endpoint")
    else:
        print("Marker not found correctly")
else:
    print("Simple endpoint not found (already clean)")
