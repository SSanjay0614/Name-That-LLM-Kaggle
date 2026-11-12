"""Dump full code cells of a notebook, with optional per-cell truncation."""
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
nb = json.loads(path.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] != "code":
        continue
    print(f"===== CELL {i} code ({len(src)} chars) =====")
    print(src[:limit])
    print()
