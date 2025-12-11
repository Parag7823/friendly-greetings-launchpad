"""Fix fastapi_backend_v2.py by adding missing _get_excel_processor_instance"""
import re

# Read the file
with open('core_infrastructure/fastapi_backend_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if '_get_excel_processor_instance' in content and 'def _get_excel_processor_instance' in content:
    print("Already fixed!")
else:
    # Find the location after ExcelProcessor import
    pattern = r'(except Exception as e:\s*\n\s*print\(f"\[ERROR\] Failed to import ExcelProcessor: \{e\}", flush=True\)\s*\n\s*ExcelProcessor = None\s*\n)'
    
    replacement = r'''\1
# Singleton pattern for ExcelProcessor instance
_excel_processor_instance = None
_excel_processor_lock = threading.Lock()

def _get_excel_processor_instance():
    """Get or create singleton ExcelProcessor instance."""
    global _excel_processor_instance
    if _excel_processor_instance is None:
        with _excel_processor_lock:
            if _excel_processor_instance is None:
                if ExcelProcessor is not None:
                    _excel_processor_instance = ExcelProcessor()
                    print("[OK] ExcelProcessor singleton created", flush=True)
                else:
                    raise RuntimeError("ExcelProcessor not available")
    return _excel_processor_instance

'''
    
    new_content = re.sub(pattern, replacement, content, count=1)
    
    if new_content == content:
        print("Pattern not found, trying simpler approach...")
        # Simpler: just find "ExcelProcessor = None" and add after it
        old_text = '    ExcelProcessor = None\n\nprint("[DEBUG] Importing EnhancedRelationshipDetector..."'
        new_text = '''    ExcelProcessor = None

# Singleton pattern for ExcelProcessor instance
_excel_processor_instance = None
_excel_processor_lock = threading.Lock()

def _get_excel_processor_instance():
    """Get or create singleton ExcelProcessor instance."""
    global _excel_processor_instance
    if _excel_processor_instance is None:
        with _excel_processor_lock:
            if _excel_processor_instance is None:
                if ExcelProcessor is not None:
                    _excel_processor_instance = ExcelProcessor()
                    print("[OK] ExcelProcessor singleton created", flush=True)
                else:
                    raise RuntimeError("ExcelProcessor not available")
    return _excel_processor_instance

print("[DEBUG] Importing EnhancedRelationshipDetector..."'''
        new_content = content.replace(old_text, new_text)
    
    with open('core_infrastructure/fastapi_backend_v2.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Fixed! Added _get_excel_processor_instance function")
