import json
from pathlib import Path

for name in ['judith007_0664', 'synthreaper_0670', 'omer_0663']:
    p = Path(f'name_that_llm/public_notebooks/{name}')
    nb_files = list(p.glob('*.ipynb'))
    if not nb_files:
        continue
    nb = json.loads(nb_files[0].read_text(encoding='utf-8'))
    print(f"=== {name}: {nb_files[0].name} (cells: {len(nb['cells'])}) ===")
    
    # check first few cells and preview/submission logic
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get("source", []))
        if any(w in src.lower() for w in ['emit_known_preview', 'preview =', 'submission.parquet', 'variant =', 'shares =', '0.66', '0.67']):
            first = src.splitlines()[0] if src.splitlines() else ""
            print(f"  Cell {i}: {first[:80]}")
