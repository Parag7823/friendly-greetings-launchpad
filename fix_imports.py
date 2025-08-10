#!/usr/bin/env python3
"""
Fix AsyncOpenAI Import Errors

This script reads fastapi_backend.py and replaces all AsyncOpenAI with OpenAI
"""

def fix_async_openai_imports():
    """Fix AsyncOpenAI import errors in fastapi_backend.py"""
    
    print("🔧 Fixing AsyncOpenAI import errors...")
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original instances
    original_count = content.count('AsyncOpenAI')
    print(f"📊 Found {original_count} instances of AsyncOpenAI")
    
    # Replace AsyncOpenAI with OpenAI
    fixed_content = content.replace('AsyncOpenAI', 'OpenAI')
    
    # Count fixed instances
    fixed_count = fixed_content.count('OpenAI')
    print(f"📊 Total OpenAI instances after fix: {fixed_count}")
    
    # Write back to file
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Successfully fixed AsyncOpenAI import errors!")
    print(f"🔄 Replaced {original_count} instances of AsyncOpenAI with OpenAI")

if __name__ == "__main__":
    fix_async_openai_imports() 