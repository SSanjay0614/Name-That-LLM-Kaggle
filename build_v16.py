import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernel_v16"
NB = OUT / "ntl_v16_grouped_svc_source_ablation.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Name That LLM — V16 grouped SVC source ablation (validation only)

This Kaggle notebook compares the V10/V13 four-view SVC component with the same five `GroupKFold` prompt folds: base public pool, base + GLM, base + Kimi, and base + both. It reports balanced accuracy and per-label recall. It does not run or score the full neural/SVC/ATLAS ensemble, and its output is not a competition submission.

The added texts come from Daniel Andres Miranda's [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. The source uses third-party provider routing and the dataset README documents the generation settings. Only `model` and `text` are used here.

Public reference inspected: Areeb Ali's [0.643 Same Stack, Sharper Blend](https://www.kaggle.com/code/areebbb/0-643-same-stack-sharper-blend-enjoy). Its reported 0.643 recipe uses a spectral bank whose dataset metadata lists no license; this experiment does not copy that bank or notebook code.
"""),
    code("""from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.metrics import balanced_accuracy_score, recall_score

LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
])

def find_unique(name):
    hits = list(Path('/kaggle/input').rglob(name))
    assert len(hits) == 1, f'Expected exactly one {name}, found {hits}'
    return hits[0]

train = pd.read_csv(find_unique('train.csv'))
pool = pd.read_parquet(find_unique('pool_v4.parquet'))[['model', 'text']].copy()
extra = pd.read_csv(find_unique('ntl_openrouter_glm_kimi.csv'))[['model', 'text']].copy()
assert train.prompt_id.notna().all() and train.model.isin(LABELS).all()
assert pool.model.isin(LABELS).all() and extra.model.isin(LABELS).all()
assert len(extra) == 1516
print('train', train.shape, 'pool', pool.shape, 'GLM/Kimi', extra.groupby('model').size().to_dict())
"""),
    code("""import re

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here when where why how what whether then now still yet even much many few little".split())

def shape_text(t):
    return re.sub(r"[^\\W\\d_]+", lambda m: "X" if m[0].isupper() else "x", t)

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\\w+|[^\\w\\s]", t))

VIEWS = [
    ('char', 'char', (3, 5), 250000, 1.0),
    ('word', 'word', (1, 3), 180000, 0.7),
    ('shape', 'char', (2, 6), 60000, 0.5),
    ('small', 'word', (1, 3), 50000, 0.5),
]

def make_matrix(fit_texts, val_texts):
    train_blocks, val_blocks = [], []
    for name, analyzer, ngram_range, max_features, weight in VIEWS:
        prep = shape_text if name == 'shape' else small_words_text if name == 'small' else (lambda s: s)
        opts = dict(analyzer=analyzer, ngram_range=ngram_range, lowercase=False,
                    sublinear_tf=True, min_df=2, max_features=max_features, dtype=np.float32)
        if analyzer == 'word':
            opts['token_pattern'] = r'(?u)\\b\\w+\\b|[^\\w\\s]'
        vec = TfidfVectorizer(**opts)
        train_blocks.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        val_blocks.append(vec.transform([prep(t) for t in val_texts]) * weight)
        del vec
    return hstack(train_blocks, format='csr'), hstack(val_blocks, format='csr')

def rows_for(source):
    if source == 'base':
        return pool
    if source == 'glm':
        return pd.concat([pool, extra[extra.model.str.startswith('glm-')]], ignore_index=True)
    if source == 'kimi':
        return pd.concat([pool, extra[extra.model.eq('kimi-k3')]], ignore_index=True)
    if source == 'both':
        return pd.concat([pool, extra], ignore_index=True)
    raise ValueError(source)

"""),
    code("""from sklearn.preprocessing import LabelEncoder

y_all = train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
groups = train.prompt_id.to_numpy()
folds = list(GroupKFold(n_splits=5).split(np.zeros(len(train)), y_all, groups))
assert all(not (set(groups[a]) & set(groups[b])) for a, b in folds)
results = {}
per_class = {}
started = time.time()

for source in ('base', 'glm', 'kimi', 'both'):
    addon = rows_for(source)
    oof = np.full(len(train), -1, dtype=np.int32)
    print('\\nSOURCE', source, 'external rows', len(addon) - len(pool), flush=True)
    for fold, (fit_idx, val_idx) in enumerate(folds):
        fold_train = train.iloc[fit_idx]
        fit_texts = fold_train.text.fillna('').astype(str).tolist() + addon.text.fillna('').astype(str).tolist()
        fit_y = np.concatenate([
            fold_train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy(),
            addon.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
        ])
        val_texts = train.iloc[val_idx].text.fillna('').astype(str).tolist()
        X_fit, X_val = make_matrix(fit_texts, val_texts)
        model = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        model.fit(X_fit, fit_y)
        oof[val_idx] = model.predict(X_val)
        print(f'  fold {fold + 1}/5 rows={len(val_idx)} complete', flush=True)
        del X_fit, X_val, model
    assert (oof >= 0).all()
    results[source] = float(balanced_accuracy_score(y_all, oof))
    per_class[source] = recall_score(y_all, oof, labels=np.arange(len(LABELS)), average=None, zero_division=0).tolist()
    print(source, 'OOF BA', results[source], flush=True)

print('elapsed_seconds', round(time.time() - started, 1))
print('SUMMARY', json.dumps({'balanced_accuracy': results,
                           'delta_vs_base': {k: results[k] - results['base'] for k in results},
                           'labels': LABELS, 'recall_by_label': per_class}, indent=2), flush=True)
"""),
    code("""out = Path('/kaggle/working/v16_svc_source_ablation.json')
out.write_text(json.dumps({'metric_scope': 'SVC component only; not full ensemble OOF',
                           'folds': 5, 'group': 'prompt_id', 'balanced_accuracy': results,
                           'delta_vs_base': {k: results[k] - results['base'] for k in results},
                           'labels': LABELS, 'recall_by_label': per_class,
                           'external_rows': extra.groupby('model').size().to_dict()}, indent=2), encoding='utf-8')
print('WROTE', out)
"""),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for idx, cell in enumerate(cells, start=1):
        cell["id"] = f"v16-cell-{idx}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "kaggle": {"id": "sanjay0614/ntl-v16-grouped-svc-source-ablation",
                       "title": "NTL V16 Grouped SVC Source Ablation",
                       "code_file": NB.name},
            "language_info": {"name": "python", "version": "3.10.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "sanjay0614/ntl-v16-grouped-svc-source-ablation",
        "title": "NTL V16 Grouped SVC Source Ablation",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "nishantdahal/ntl-handwriting-blend",
            "danielandresmiranda/ntl-openrouter-glm-kimi",
        ],
        "competition_sources": ["name-that-llm"],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("""# V16 — Grouped SVC source ablation (validation only)

Runs five prompt-grouped folds on Kaggle for four SVC data recipes: V10/V13 base pool, base plus GLM only, base plus Kimi only, and base plus both. Records pooled balanced accuracy and per-label recall in `/kaggle/working/v16_svc_source_ablation.json`.

This is a component diagnostic, not full-ensemble OOF and not a competition submission. It cannot pass the action plan's submission gate on its own. The actual feature fitting runs in Kaggle; the local builder only writes notebook and metadata files.

Baseline pool: Nishant Dahal, [NTL two-stage LoRA part two](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend), CC BY-SA 4.0 (the pool includes ShareAlike material). Added data: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. Prompt bank attribution remains with Seyit Kaan Güneş, [NTL Extra API Passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages), CC BY 4.0.
""", encoding="utf-8")
    print(NB)


if __name__ == "__main__":
    main()
