#!/usr/bin/env python3
"""Index helper functions from 3b1b videos."""
import ast
from pathlib import Path
import chromadb

VIDEOS_PATH = Path('/app/data/external/3b1b-videos')

client = chromadb.HttpClient(host='chromadb', port=8000)
coll = client.get_or_create_collection(name='manim_helpers')

prefixes = ('get_', 'create_', 'make_', 'build_')
helpers = []

for py_file in VIDEOS_PATH.rglob('*.py'):
    try:
        content = py_file.read_text(errors='ignore')
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(prefixes):
                lines = content.split('\n')
                end = min(node.end_lineno or node.lineno+30, node.lineno+40)
                code = '\n'.join(lines[node.lineno-1:end])
                if len(code) > 50:
                    helpers.append({
                        'name': node.name,
                        'code': code[:1500],
                        'file': py_file.name,
                    })
    except:
        pass

print(f'Found {len(helpers)} helpers')

# Index
for i, h in enumerate(helpers):
    doc_id = f"helper_{h['name']}_{i}"
    doc = f"# Function: {h['name']}\nFile: {h['file']}\n\n```python\n{h['code']}\n```"
    try:
        coll.upsert(ids=[doc_id], documents=[doc], metadatas={'name': h['name'], 'file': h['file']})
    except Exception as e:
        print(f"Error {h['name']}: {e}")

print(f'Indexed: {coll.count()}')
