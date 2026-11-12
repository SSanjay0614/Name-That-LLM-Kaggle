import json
from pathlib import Path

nb_path = Path("name_that_llm/public_notebooks/areebbb_0643/0-643-same-stack-sharper-blend-enjoy.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    first_line = lines[0] if lines else ""
    if any(k in src.lower() for k in ["%%writefile", "test.csv", "submission", "infer.py", "comp_stack", "subprocess", "parquet"]):
        print(f"Cell {i} ({cell.get('cell_type')}): {first_line[:80]} (lines: {len(lines)})")
