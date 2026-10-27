"""Inspect the NTL V19 (V27 reproduction) notebook: cell map and heavy-compute markers."""

import json
import re
import sys
from pathlib import Path

NB = Path(r"C:\Users\ssanj\OneDrive\Desktop\Sanjay\Antigravity\name_that_llm\kernel_v19\ntl_v19_nlm_v27_reproduction.ipynb")

PATTERNS = [
    r"from_pretrained",
    r"\.cuda\(|device\s*=\s*['\"]cuda|to\(['\"]cuda",
    r"AutoModel|AutoTokenizer|SentenceTransformer",
    r"batch_size|bs\s*=\s*\d+",
    r"torch\.(no_grad|inference_mode)",
    r"TEST|csv|parquet",
    r"max_length|truncation",
    r"SVC|LogisticRegression|TfidfVectorizer",
    r"tqdm|print\(",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    print(f"cells: {len(cells)}")
    for i, cell in enumerate(cells):
        text = "".join(cell["source"])
        if cell["cell_type"] != "code":
            print(f"{i:3d} mark {len(text):6d} | {text.splitlines()[0][:90] if text else ''}")
            continue
        head = text.splitlines()[0][:80] if text else ""
        print(f"{i:3d} code {len(text):6d} | {head}")
    print("\n== markers ==")
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        for lineno, line in enumerate("".join(cell["source"]).splitlines(), 1):
            s = line.strip()
            if any(re.search(p, s) for p in PATTERNS):
                print(f"c{i}:{lineno}: {s[:150]}")


if __name__ == "__main__":
    main()
