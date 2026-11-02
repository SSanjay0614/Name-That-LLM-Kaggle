import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "public_notebooks" / "avazaku_0651"
SRC_NB = SRC_DIR / "name-that-llm-0-651-verified-reproduction.ipynb"
SRC_META = SRC_DIR / "kernel-metadata.json"

DEST_DIR = ROOT / "kernel_v21"
DEST_DIR.mkdir(parents=True, exist_ok=True)
DEST_NB = DEST_DIR / "ntl-v21-sota-0651.ipynb"
SLUG = "ntl-v21-sota-0651"
TITLE = "NTL V21 0.651 SOTA NLM V27"

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# Patch Cell 5 (comp_stack with commit smoke guard)
c5_src = cells[5]["source"] if isinstance(cells[5]["source"], str) else "".join(cells[5]["source"])
new_c5 = """# complementary public stack in its own process
import os, time as _t
_c0 = _t.time()
TRUE_SUBMISSION = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))

if TRUE_SUBMISSION:
    os.environ['NTL_DROP_Q35'] = '1'
    !python comp_stack.py 2>&1 | grep -vE 'Warning|warn'
    COMP_SEC = _t.time() - _c0
    print('comp stack wall time', round(COMP_SEC))
else:
    print('[smoke] commit mode: skipping heavy comp_stack to protect GPU quota')
    import numpy as np, pandas as pd
    test_df = pd.read_csv(DATA + '/test.csv')
    np.save('/kaggle/working/comp_logit.npy', np.zeros((len(test_df), 18)))
    COMP_SEC = 0
"""
cells[5]["source"] = [l + "\n" for l in new_c5.split("\n")]

# Patch Cell 6 (infer with commit smoke guard)
c6_src = cells[6]["source"] if isinstance(cells[6]["source"], str) else "".join(cells[6]["source"])
old_infer_run = "!python infer.py"

# Find where infer.py is called in Cell 6 and guard it
c6_lines = c6_src.split("\n")
infer_idx = next(i for i, l in enumerate(c6_lines) if old_infer_run in l)

head_c6 = c6_lines[:infer_idx]
infer_cmd = c6_lines[infer_idx]
tail_c6 = c6_lines[infer_idx+1:]

new_c6 = "\n".join(head_c6) + f"""
if TRUE_SUBMISSION:
    {infer_cmd}
    print(pd.read_parquet('submission.parquet').head())
else:
    print('[smoke] commit mode: fast submission.parquet generation (~0.01h GPU)')
    test_df = pd.read_csv(DATA + '/test.csv')
    # Default top candidate for schema check
    sub = pd.DataFrame({{'id': test_df['id'], 'model': ['qwen3.8-max'] * len(test_df)}})
    sub.to_parquet('submission.parquet', index=False)
    print('Smoke submission created with shape:', sub.shape)
    print(sub.head())
"""
cells[6]["source"] = [l + "\n" for l in new_c6.split("\n")]

# Clean outputs
for i, cell in enumerate(cells):
    cell.setdefault("id", f"cell-{i}")
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Metadata
nb.setdefault("metadata", {})["kaggle"] = {
    "id": f"sanjay0614/{SLUG}",
    "title": TITLE,
    "code_file": DEST_NB.name,
    "language": "python",
    "sourceType": "notebook",
    "isGpuEnabled": True,
    "isInternetEnabled": False,
    "accelerator": "NvidiaTeslaT4"
}

DEST_NB.write_text(json.dumps(nb, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")

meta = json.loads(SRC_META.read_text(encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = f"sanjay0614/{SLUG}"
meta["title"] = TITLE
meta["code_file"] = DEST_NB.name
meta["is_private"] = True
meta["machine_shape"] = "NvidiaTeslaT4"

(DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"Built {DEST_NB} successfully with fast commit smoke guard!")
