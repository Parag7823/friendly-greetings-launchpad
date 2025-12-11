"""Update rate limit default from 5 to 10"""
import re

with open('core_infrastructure/rate_limiter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change default from 5 to 10
old = "def __init__(self, max_uploads_per_minute: int = 5):"
new = "def __init__(self, max_uploads_per_minute: int = 10):"

if old in content:
    content = content.replace(old, new)
    with open('core_infrastructure/rate_limiter.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Updated rate limit to 10")
else:
    print("Pattern not found, checking if already updated...")
    if "max_uploads_per_minute: int = 10" in content:
        print("Already set to 10")
    else:
        print("Could not find pattern to update")
