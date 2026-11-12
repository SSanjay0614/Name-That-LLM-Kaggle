import json
from pathlib import Path

nb_path = Path("name_that_llm/public_notebooks/areebbb_0643/0-643-same-stack-sharper-blend-enjoy.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell_id in [5, 7, 75, 76, 77, 78, 79, 80]:
    if cell_id < len(nb["cells"]):
        cell = nb["cells"][cell_id]
        src = "".join(cell.get("source", []))
        print(f"=== Cell {cell_id} ({cell.get('cell_type')}) ===")
        print(src[:1000])
        print("...\n" if len(src) > 1000 else "\n")
