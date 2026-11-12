"""Summarise a public notebook: markdown notes and the head of each code cell."""
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"{path.name}: {len(cells)} cells "
      f"({sum(c['cell_type']=='code' for c in cells)} code)\n")
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        keep = [l for l in src.splitlines() if l.strip()]
        if keep:
            print(f"--- md[{i}] ---")
            print("\n".join(keep[:14]))
            print()
    else:
        first = [l for l in src.splitlines() if l.strip()]
        if first:
            print(f"--- code[{i}] ({len(src)} chars) ---")
            print("\n".join(first[:6]))
            print()
