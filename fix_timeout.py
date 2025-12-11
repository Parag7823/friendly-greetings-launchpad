"""Fix timeout in fastapi_backend_v2.py"""
with open('core_infrastructure/fastapi_backend_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix timeout from 30 to 120 seconds
old = "timeout = app_config.request_timeout if 'app_config' in globals() else 30"
new = "timeout = app_config.request_timeout if 'app_config' in globals() else 120"

if old in content:
    content = content.replace(old, new)
    with open('core_infrastructure/fastapi_backend_v2.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Timeout increased from 30s to 120s")
else:
    print("Pattern not found - checking current value...")
    if "else 120" in content:
        print("Already fixed to 120s!")
    else:
        print("Need manual fix")
