import ast
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'fastapi_backend_v2.py'
src = p.read_text(encoding='utf-8')
mod = ast.parse(src)

cls_methods = []
global_funcs = []
nested_found = []

target_names = {
    '_normalize_entity_type',
    '_store_entity_matches',
    '_store_platform_patterns',
    '_store_discovered_platforms',
}

for node in mod.body:
    if isinstance(node, ast.ClassDef) and node.name == 'ExcelProcessor':
        for b in node.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls_methods.append(b.name)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        global_funcs.append(node.name)
        # search nested
        for b in ast.walk(node):
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b is not node:
                if b.name in target_names:
                    nested_found.append((node.name, b.name))

print('ExcelProcessor method count:', len(cls_methods))
for t in sorted(target_names):
    print(f'Class has {t}:', t in cls_methods)
print('Global function names (first 30):', global_funcs[:30])
print('Nested target functions found in:', nested_found)
