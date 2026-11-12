"""Dump the config / run-command surface of the 0.684-lineage notebooks."""

import json
import re
from pathlib import Path

FILES = [
    "ntl-684-ozertuu-mirror.ipynb",
    "name-that-llm-lv683-a15.ipynb",
    "the-prompt-is-not-the-author.ipynb",
    "lora-trained27-evidence-for-close-calls.ipynb",
    "ntl-fuse33-0-668-fork-response-corpus.ipynb",
]

PATTERNS = [
    r"--sinkhorn[^\n]{0,40}",
    r"--comp_w[^\n]{0,40}",
    r"--native_share[^\n]{0,40}",
    r"--gsq[^\n]{0,40}",
    r"--share[^\n]{0,40}",
    r"--target_mix[^\n]{0,40}",
    r"'--[a-z_]+'[^\n]{0,60}",
    r"target_mix\s*=\s*[^\n]{0,40}",
    r"strength\s*=\s*[^\n]{0,40}",
]

for name in FILES:
    p = Path("public_new3") / name
    nb = json.loads(p.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print("=" * 30, name, "| code cells", len(code))
    for i, c in enumerate(code):
        s = "".join(c["source"])
        lines = [ln for ln in s.splitlines() if any(re.search(pt, ln) for pt in PATTERNS)]
        if lines:
            print(f"  --- cell {i} ({len(s)} chars)")
            for ln in lines[:12]:
                print("   ", ln.strip()[:150])
