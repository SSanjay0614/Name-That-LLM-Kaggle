"""Inspect the commit/preview path of the canonical 0.684 mirror."""

import json
from pathlib import Path

p = Path("public_new3/ntl-684-ozertuu-mirror.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i in [0, 1, len(code) - 2, len(code) - 1]:
    s = "".join(code[i]["source"])
    print("=" * 30, "code cell", i, len(s), "chars")
    print(s[:4000])
