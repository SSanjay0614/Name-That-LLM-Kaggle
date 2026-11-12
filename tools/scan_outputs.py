"""What artifact does each public notebook write, and does it have a cheap preview path?"""

import json
import re
from pathlib import Path

for p in sorted(Path("public_new3").glob("*.ipynb")):
    nb = json.loads(p.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    text = "".join("".join(c["source"]) for c in code)
    print("=" * 30, p.name, "| code cells", len(code))
    for pat in [
        r"submission\.(parquet|csv)",
        r"to_parquet\(",
        r"to_csv\(",
        r"mode['\"]?\s*[:=]\s*['\"][^'\"]+",
        r"FAST[_A-Z]*\s*=\s*[^\n]{0,60}",
        r"sample_submission[^\n]{0,60}",
    ]:
        hits = sorted(set(re.findall(pat, text)))[:8]
        print(f"  {pat}: {hits}")
