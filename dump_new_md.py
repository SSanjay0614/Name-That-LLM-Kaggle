import json
from pathlib import Path

for p in sorted(Path("public_new3").glob("*.ipynb")):
    nb = json.loads(p.read_text(encoding="utf-8"))
    md = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
    print("#" * 40, p.name, "| markdown cells", len(md))
    for i, s in md[:3]:
        print(f"--- md {i} ---")
        print(s[:1800])
