#!/usr/bin/env python3
"""
Script to fix ExcelProcessor method binding issue.
Moves nested helper methods into the class and removes monkey-patching.
"""

def fix_file():
    filepath = "fastapi_backend_v2.py"
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the insertion point (after _build_platform_id_map method, before the comment)
    insertion_line = None
    for i, line in enumerate(lines):
        if line.strip() == "return name_to_id" and i < 10000:
            # Found the end of _build_platform_id_map
            insertion_line = i + 1
            break
    
    if not insertion_line:
        print("ERROR: Could not find insertion point")
        return False
    
    print(f"Found insertion point at line {insertion_line + 1}")
    
    # Find the nested methods (inside get_performance_optimization_status)
    nested_start = None
    nested_end = None
    for i, line in enumerate(lines):
        if i > 10180 and line.strip().startswith("def _normalize_entity_type(self"):
            nested_start = i
            print(f"Found nested methods start at line {i + 1}")
            break
    
    if not nested_start:
        print("ERROR: Could not find nested methods")
        return False
    
    # Find where nested methods end (before "# ExcelProcessor class ends here")
    for i in range(nested_start, len(lines)):
        if "# ExcelProcessor class ends here" in lines[i]:
            nested_end = i
            print(f"Found nested methods end at line {i + 1}")
            break
    
    if not nested_end:
        print("ERROR: Could not find end of nested methods")
        return False
    
    # Extract the nested methods
    nested_methods = lines[nested_start:nested_end]
    print(f"Extracted {len(nested_methods)} lines of nested methods")
    
    # Find and remove monkey-patching lines
    monkey_patch_start = None
    monkey_patch_end = None
    for i in range(nested_end, min(nested_end + 50, len(lines))):
        if "ExcelProcessor._normalize_entity_type" in lines[i]:
            monkey_patch_start = i
            print(f"Found monkey-patching start at line {i + 1}")
            break
    
    if monkey_patch_start:
        # Find end of monkey-patching (empty line after last assignment)
        for i in range(monkey_patch_start, min(monkey_patch_start + 20, len(lines))):
            if lines[i].strip() == "" and i > monkey_patch_start:
                monkey_patch_end = i + 1
                print(f"Found monkey-patching end at line {i + 1}")
                break
    
    # Build the new file
    new_lines = []
    
    # Part 1: Everything before insertion point
    new_lines.extend(lines[:insertion_line])
    
    # Part 2: Add the methods to the class (with proper indentation)
    new_lines.append("\n")
    for line in nested_methods:
        # These methods already have correct indentation (4 spaces for class methods)
        new_lines.append(line)
    
    # Part 3: Skip to after the nested methods section
    # Continue from the line that says "# ExcelProcessor class ends here"
    new_lines.extend(lines[nested_end:nested_start])
    
    # Part 4: Skip monkey-patching if found
    if monkey_patch_start and monkey_patch_end:
        # Skip the monkey-patching lines
        new_lines.extend(lines[monkey_patch_end:])
    else:
        # No monkey-patching found, continue normally
        new_lines.extend(lines[nested_end:])
    
    # Write the fixed file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ File fixed successfully!")
    print(f"   - Inserted {len(nested_methods)} lines of methods into ExcelProcessor class")
    print(f"   - Removed nested duplicate definitions")
    if monkey_patch_start:
        print(f"   - Removed monkey-patching lines")
    
    return True

if __name__ == "__main__":
    success = fix_file()
    exit(0 if success else 1)
