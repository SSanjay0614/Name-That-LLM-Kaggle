import json
from pathlib import Path

for name in ['judith007_0664', 'omer_0663', 'synthreaper_0670']:
    p = Path(f'name_that_llm/public_notebooks/{name}')
    meta_path = p / 'kernel-metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        print(f"=== {name} ===")
        print(f"Title: {meta.get('title')}")
        print(f"Code file: {meta.get('code_file')}")
        print("Dataset sources:")
        for d in meta.get('dataset_sources', []):
            print(f"  {d}")
        print("Kernel sources:")
        for k in meta.get('kernel_sources', []):
            print(f"  {k}")
        print("Model sources:")
        for m in meta.get('model_sources', []):
            print(f"  {m}")
        print()
