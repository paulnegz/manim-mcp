#!/usr/bin/env python3
"""Index video series from 3b1b."""
from pathlib import Path
import chromadb

VIDEOS_PATH = Path('/app/data/external/3b1b-videos')

client = chromadb.HttpClient(host='chromadb', port=8000)
coll = client.get_or_create_collection(name='video_series')
print(f"Collection: {coll.name}")

series = []

for year_dir in VIDEOS_PATH.iterdir():
    if not year_dir.is_dir() or not year_dir.name.startswith('_20'):
        continue

    year = year_dir.name[1:]

    for topic_dir in year_dir.iterdir():
        if not topic_dir.is_dir():
            continue

        topic = topic_dir.name
        parts = [d.name for d in topic_dir.iterdir() if d.is_dir() and d.name.startswith('part')]
        shared = [f.name for f in topic_dir.glob('*shared*.py')]
        py_files = list(topic_dir.glob('*.py'))

        series.append({
            'id': f"{year}_{topic}",
            'year': year,
            'topic': topic,
            'parts': sorted(parts),
            'shared': shared,
            'num_files': len(py_files),
        })

print(f'Found {len(series)} series')

for s in series:
    doc = f"Series: {s['topic']} ({s['year']})\nParts: {len(s['parts'])}\nShared files: {len(s['shared'])}"
    meta = {'year': s['year'], 'topic': s['topic'], 'num_parts': len(s['parts'])}
    coll.upsert(ids=[s['id']], documents=[doc], metadatas=[meta])

print(f'Indexed: {coll.count()}')
