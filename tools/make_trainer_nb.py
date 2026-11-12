"""Turn trainer/train_q8_lora.py into a Kaggle notebook plus metadata.

The `.py` file is the source of truth so it can be syntax-checked and diffed; cells are split on
`# %%` markers. Markdown cells keep their `# ` comment prefixes stripped.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "trainer" / "train_q8_lora.py"
OUT_DIR = HERE / "kernel_t1_q8_lora"
CODE_FILE = "ntl_q8_lora_pool.ipynb"

META = {
    "id": "sanjay0614/ntl-v43-q8-lora-pool",
    "title": "NTL v43 q8 lora pool",
    "code_file": CODE_FILE,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": ["sanjay0614/ntl-pool-v4"],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"],
    "machine_shape": "NvidiaTeslaT4",
}


def split_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None or not any(line.strip() for line in buf):
            return
        body = [l for l in buf]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            return
        if kind == "markdown":
            body = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in body]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": body})
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped == "# %% [markdown]":
            flush()
            kind, buf = "markdown", []
            continue
        if stripped == "# %%":
            flush()
            kind, buf = "code", []
            continue
        if kind is None:
            continue
        buf.append(line)
    flush()
    return cells


def main():
    text = SRC.read_text(encoding="utf-8")
    cells = split_cells(text)
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("\n".join(cell["source"]))  # fails loudly on a syntax error
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / CODE_FILE).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(META, indent=2) + "\n", encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT_DIR / CODE_FILE} | cells {len(cells)} ({code_cells} code)")
    for i, c in enumerate(cells):
        head = "".join(c["source"]).splitlines()
        print(f"  {i:2d} {c['cell_type'][:4]} {len(head):4d} lines | {head[0][:80]}")


if __name__ == "__main__":
    main()
