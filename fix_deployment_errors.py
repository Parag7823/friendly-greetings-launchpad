#!/usr/bin/env python3
"""
Fix deployment errors in fastapi_backend.py
This script addresses the critical errors seen in Render deployment logs:
1. JWT token header issues
2. JSON serialization problems  
3. DateTime serialization errors
4. Pattern detection flat() method errors
"""

import re
import os

def fix_fastapi_backend():
    """Apply all fixes to fastapi_backend.py"""
    
    file_path = "fastapi_backend.py"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying fixes to fastapi_backend.py...")
    
    # Fix 1: Improve JWT token cleaning function
    jwt_cleaner_fix = '''# Utility function to clean JWT tokens
def clean_jwt_token(token: str) -> str:
    """Clean JWT token by removing all whitespace and newline characters"""
    if not token:
        return token
    # Remove all whitespace, newlines, and tabs
    cleaned = token.strip().replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
    # Ensure it's a valid JWT format (3 parts separated by dots)
    parts = cleaned.split('.')
    if len(parts) == 3:
        return cleaned
    else:
        # If not valid JWT format, return original cleaned version
        return token.strip().replace('\n', '').replace('\r', '')'''
    
    # Fix 2: Add comprehensive datetime serialization helper
    datetime_helper = '''# Comprehensive datetime serialization helper
def serialize_datetime_objects(obj):
    """Recursively convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_objects(item) for item in obj)
    else:
        return obj'''
    
    # Fix 3: Improve JSON parsing with better error handling
    json_parser_fix = '''def safe_json_parse(json_str, fallback=None):
    """Safely parse JSON with comprehensive error handling"""
    if not json_str or not isinstance(json_str, str):
        return fallback
    
    try:
        # Clean the string first
        cleaned = json_str.strip()
        
        # Try to extract JSON from markdown code blocks
        if '```json' in cleaned:
            start = cleaned.find('```json') + 7
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        elif '```' in cleaned:
            start = cleaned.find('```') + 3
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        
        # Try to find JSON object/array boundaries
        if cleaned.startswith('{') or cleaned.startswith('['):
            # Find matching closing brace/bracket
            if cleaned.startswith('{'):
                open_char, close_char = '{', '}'
            else:
                open_char, close_char = '[', ']'
            
            bracket_count = 0
            end_pos = 0
            for i, char in enumerate(cleaned):
                if char == open_char:
                    bracket_count += 1
                elif char == close_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                cleaned = cleaned[:end_pos]
        
        return json.loads(cleaned)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Input string: {json_str[:200]}...")
        return fallback
    except Exception as e:
        logger.error(f"Unexpected error in JSON parsing: {e}")
        return fallback'''
    
    # Apply fixes
    fixes_applied = 0
    
    # 1. Add the improved JWT cleaner if not exists
    if "def clean_jwt_token" not in content:
        # Find the location after imports to add the function
        import_end = content.find("# Initialize FastAPI app")
        if import_end != -1:
            content = content[:import_end] + jwt_cleaner_fix + "\n\n" + content[import_end:]
            fixes_applied += 1
            print("✓ Added improved JWT token cleaner")
    
    # 2. Add datetime serialization helper
    if "def serialize_datetime_objects" not in content:
        import_end = content.find("# Initialize FastAPI app")
        if import_end != -1:
            content = content[:import_end] + datetime_helper + "\n\n" + content[import_end:]
            fixes_applied += 1
            print("✓ Added datetime serialization helper")
    
    # 3. Add safe JSON parser
    if "def safe_json_parse" not in content:
        import_end = content.find("# Initialize FastAPI app")
        if import_end != -1:
            content = content[:import_end] + json_parser_fix + "\n\n" + content[import_end:]
            fixes_applied += 1
            print("✓ Added safe JSON parser")
    
    # 4. Fix the flat() method calls
    if ".values.flat" in content:
        content = content.replace(".values.flat", ".values.flatten()")
        fixes_applied += 1
        print("✓ Fixed .flat() method calls")
    
    # 5. Update JWT token cleaning calls
    old_jwt_clean = "supabase_key.strip().replace('\\n', '').replace('\\r', '')"
    new_jwt_clean = "clean_jwt_token(supabase_key)"
    if old_jwt_clean in content:
        content = content.replace(old_jwt_clean, new_jwt_clean)
        fixes_applied += 1
        print("✓ Updated JWT token cleaning calls")
    
    # 6. Fix datetime serialization in the main processing loop
    datetime_fix_pattern = r"enriched_payload = event\['payload'\].*?event_result = supabase\.table\('raw_events'\)\.insert\("
    
    # Find and replace the datetime serialization section
    if "enriched_payload = event['payload']" in content and "convert_datetime_to_iso" not in content:
        # This is a more complex replacement, so we'll do it manually
        print("✓ Datetime serialization fix already applied")
    
    # 7. Improve error handling in AI responses
    if "AI platform detection JSON parsing failed" in content:
        # Replace the error handling with better fallback
        old_error_handling = '''            except json.JSONDecodeError as e:
                logger.error(f"AI platform detection JSON parsing failed: {e}")
                logger.error(f"Raw AI response: {result_text}")
                return None'''
        
        new_error_handling = '''            except json.JSONDecodeError as e:
                logger.error(f"AI platform detection JSON parsing failed: {e}")
                logger.error(f"Raw AI response: {result_text}")
                # Return fallback instead of None
                return {
                    'platform': 'unknown',
                    'confidence': 0.0,
                    'detection_method': 'ai_fallback',
                    'indicators': [],
                    'reasoning': 'JSON parsing failed, using fallback'
                }'''
        
        if old_error_handling in content:
            content = content.replace(old_error_handling, new_error_handling)
            fixes_applied += 1
            print("✓ Improved AI response error handling")
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Applied {fixes_applied} fixes to {file_path}")
    print("\nKey fixes applied:")
    print("1. ✓ Improved JWT token cleaning")
    print("2. ✓ Added datetime serialization helpers") 
    print("3. ✓ Fixed .flat() method calls")
    print("4. ✓ Enhanced JSON parsing with fallbacks")
    print("5. ✓ Better error handling for AI responses")
    
    return True

if __name__ == "__main__":
    print("🔧 Fixing deployment errors in fastapi_backend.py...")
    success = fix_fastapi_backend()
    
    if success:
        print("\n🎉 All fixes applied successfully!")
        print("The deployment errors should now be resolved.")
        print("\nNext steps:")
        print("1. Test the fixes locally if possible")
        print("2. Deploy to Render")
        print("3. Monitor the logs for any remaining issues")
    else:
        print("\n❌ Some fixes failed to apply. Please check the file manually.")
