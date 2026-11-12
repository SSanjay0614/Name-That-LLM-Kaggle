import json
from pathlib import Path

nb = json.loads(Path('name_that_llm/public_notebooks/omer_0663/0-663-0-660-v31-atlas-20.ipynb').read_text(encoding='utf-8'))
for c in [2, 10, 12]:
    cell = nb['cells'][c]
    print(f"=== Cell {c} ===")
    print("".join(cell.get('source', [])))
