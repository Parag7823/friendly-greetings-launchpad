#!/usr/bin/env python3
"""
Quick Fix Script for AsyncOpenAI Import Error

This script replaces all instances of AsyncOpenAI with OpenAI in fastapi_backend.py
"""

import re

def fix_async_openai():
    """Fix AsyncOpenAI import errors in fastapi_backend.py"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace AsyncOpenAI with OpenAI
    fixed_content = content.replace('AsyncOpenAI', 'OpenAI')
    
    # Write back to file
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Fixed AsyncOpenAI import errors in fastapi_backend.py")
    
    # Count replacements
    original_count = content.count('AsyncOpenAI')
    fixed_count = fixed_content.count('OpenAI')
    
    print(f"📊 Replaced {original_count} instances of AsyncOpenAI with OpenAI")
    print(f"📊 Total OpenAI instances now: {fixed_count}")

if __name__ == "__main__":
    fix_async_openai() 