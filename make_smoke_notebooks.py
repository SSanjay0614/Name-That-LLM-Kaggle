import json
from pathlib import Path

ROOT = Path(__file__).parent

def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}

for version, slug, title, filename in [
    (19, "ntl-v19-nlm-v27-smoke", "NTL V19 NLM V27 Smoke", "ntl_v19_nlm_v27_smoke.ipynb"),
    (20, "ntl-v20-v27-glm-kimi-svc-smoke", "NTL V20 V27 GLM Kimi SVC Smoke", "ntl_v20_v27_glm_kimi_svc_smoke.ipynb"),
]:
    original = ROOT / f"kernel_v{version}" / ("ntl_v19_nlm_v27_reproduction.ipynb" if version == 19 else "ntl_v20_v27_glm_kimi_svc_pool.ipynb")
    notebook = json.loads(original.read_text(encoding="utf-8"))
    cells = [source(c) for c in notebook["cells"] if c.get("cell_type") == "code"]
    gauss = next(c for c in cells if c.startswith("%%writefile gauss.py"))
    infer = next(c for c in cells if c.startswith("%%writefile infer.py"))
    comp = next(c for c in cells if c.startswith("%%writefile comp_stack.py"))
    v20_pool = next((c for c in cells if c.startswith("# V20: add only")), None)

    smoke_setup = r'''# Lightweight cloud smoke: validate mounted data and code inputs; no model inference or official submission file.
import glob, os, pathlib, pandas as pd, numpy as np, sys, py_compile
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
TRAIN = pd.read_csv(DATA + '/train.csv')
TEST = pd.read_csv(DATA + '/test.csv')
assert {'id','text'}.issubset(TEST.columns) and TEST.id.is_unique
assert {'id','text','model'}.issubset(TRAIN.columns)
assert TRAIN.model.nunique() == 18
POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))
print('SMOKE_INPUTS_OK', {'train': len(TRAIN), 'test': len(TEST), 'pool': POOL, 'models': TRAIN.model.nunique()}, flush=True)
'''
    smoke_run = r'''# Exercise the complementary stack's deterministic smoke branch, without neural checkpoints.
import os, subprocess, sys
os.environ['NTL_SMOKE'] = '1'
os.environ['NTL_DROP_Q35'] = '1'
os.environ['NTL_TINY'] = '1'
os.environ['NTL_DATA'] = DATA
subprocess.run([sys.executable, 'comp_stack.py'], check=True)
assert os.path.exists('/kaggle/working/comp_logit.npy')
print('COMP_STACK_SMOKE_OK', flush=True)
'''
    final = r'''# Compile the exact inference source and validate only the visible-test output contract.
for name in ('infer.py', 'comp_stack.py', 'gauss.py'):
    py_compile.compile(name, doraise=True)
labels = set(TRAIN.model.astype(str).unique())
preview = pd.DataFrame({'id': TEST.id, 'model': [next(iter(labels))] * len(TEST)})
assert preview.id.is_unique and preview.model.isin(labels).all()
assert not os.path.exists('submission.parquet'), 'Smoke notebooks must not create a submit-ready artifact.'
print('SMOKE_PASS: scripts compile; data, 18 classes and preview schema are valid; no model inference ran.', flush=True)
'''
    selected = [md(f"# {title}\n\nCloud smoke only. This checks mounted competition inputs, the V20 pool augmentation (when applicable), and the complementary stack's synthetic smoke path. It deliberately skips all transformer inference and creates no competition submission artifact."),
                code(smoke_setup), code(gauss), code(infer), code(comp), code(smoke_run)]
    if version == 20 and v20_pool:
        selected.append(code(v20_pool))
    selected.append(code(final))
    outdir = ROOT / f"kernel_v{version}_smoke"
    outdir.mkdir(exist_ok=True)
    (outdir / filename).write_text(json.dumps({"cells": selected, "metadata": notebook.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = json.loads((ROOT / f"kernel_v{version}" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update({"id": f"sanjay0614/{slug}", "title": title, "code_file": filename, "enable_gpu": False, "machine_shape": "CPU"})
    meta.pop("model_sources", None)
    (outdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (outdir / "README.md").write_text(f"# {title}\n\nCloud smoke copy of V{version}. It validates inputs and code shape only; it intentionally skips full inference and must not be submitted to the competition.\n", encoding="utf-8")
    print(outdir)
