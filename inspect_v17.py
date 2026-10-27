"""What does the 0.640 kernel (V17) actually run at inference time?"""

import json
import re
import sys
from pathlib import Path

NB = Path(r"C:\Users\ssanj\OneDrive\Desktop\Sanjay\Antigravity\name_that_llm\kernel_v17\ntl_v17_glm_svc_pool.ipynb")

KEYS = [
    "from_pretrained", "device_map", "dtype", "cuda", "8b", "14b", "4b", "qwen", "llama",
    "SVC", "LinearSVC", "LogisticRegression", "TfidfVectorizer", "LGBM", "sentence",
    "submission", "test.csv", "test.shape", "smoke", "SMOKE",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    print(f"cells {len(cells)}")
    for i, c in enumerate(cells):
        t = "".join(c["source"])
        head = t.splitlines()[0][:100] if t else ""
        print(f"{i:3d} {c['cell_type'][:4]} {len(t):6d} | {head}")
    print("\n== mentions ==")
    hits = {}
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        for ln, line in enumerate("".join(c["source"]).splitlines(), 1):
            low = line.lower()
            if any(k.lower() in low for k in KEYS):
                print(f"c{i}:{ln}: {line.strip()[:140]}")


if __name__ == "__main__":
    main()

