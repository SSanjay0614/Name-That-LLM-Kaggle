import json
from pathlib import Path

for name in ['omer_0663', 'judith007_0664', 'synthreaper_0670']:
    p = Path(f'name_that_llm/public_notebooks/{name}')
    nb_files = list(p.glob('*.ipynb'))
    if not nb_files:
        continue
    nb = json.loads(nb_files[0].read_text(encoding='utf-8'))
    print(f"=== {name}: {nb_files[0].name} (cells: {len(nb['cells'])}) ===")
    for i, cell in enumerate(nb['cells']):
        ctype = cell.get('cell_type')
        src = "".join(cell.get('source', []))
        lines = src.splitlines()
        first = lines[0] if lines else ""
        print(f"  Cell {i} ({ctype}, {len(lines)} lines): {first[:100]}")
