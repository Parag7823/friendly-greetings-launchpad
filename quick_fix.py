#!/usr/bin/env python3
"""
Quick fix for the indentation error
"""

# Read the file
with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove line 6336 (index 6335)
if len(lines) > 6335:
    lines.pop(6335)

# Write back
with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Removed problematic line 6336")
