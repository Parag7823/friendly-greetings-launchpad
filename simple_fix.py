#!/usr/bin/env python3
"""
Simple fix for indentation errors in fastapi_backend.py
"""

def fix_indentation():
    """Fix indentation errors by removing problematic code blocks"""
    
    print("🔧 Starting simple fix...")
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove problematic lines
    fixed_lines = []
    skip_next_n_lines = 0
    
    for i, line in enumerate(lines):
        # Skip lines if we're in a problematic block
        if skip_next_n_lines > 0:
            skip_next_n_lines -= 1
            continue
        
        # Check for problematic patterns
        if '# Update progress for data enrichment' in line:
            # Skip this line and the next 6 lines (the problematic block)
            skip_next_n_lines = 6
            continue
        
        # Fix specific indentation issues
        if line.strip() == 'enriched_payload[\'kind\'] = \'transaction\'':
            # Fix indentation
            line = '            enriched_payload[\'kind\'] = \'transaction\'\n'
        elif line.strip() == 'enriched_payload[\'category\'] = \'other\'':
            # Fix indentation
            line = '            enriched_payload[\'category\'] = \'other\'\n'
        
        fixed_lines.append(line)
    
    # Write the fixed content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Simple fix completed!")

if __name__ == "__main__":
    fix_indentation()
