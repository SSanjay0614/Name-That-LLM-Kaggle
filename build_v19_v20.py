"""Prepare two Kaggle runs from the verified public NLM V27 recipe.

This builder edits notebook files only. All model work runs on Kaggle.
V19 reproduces the public 0.651 recipe unchanged; V20 adds one SVC-pool source.
"""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "public_notebooks" / "avazaku_0651"
SOURCE_NOTEBOOK = SOURCE_DIR / "name-that-llm-0-651-verified-reproduction.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"

V20_POOL_CELL = r'''# V20: add only the CC BY 4.0 GLM/Kimi rows to the V27 SVC pool.
# The matched-passage source used by TF-IDF and all other stack components is unchanged.
import glob as _v20_glob
import numpy as _v20_np
import pandas as _v20_pd

_v20_base = _v20_pd.read_parquet(POOL)
_v20_external_paths = _v20_glob.glob('/kaggle/input/**/ntl_openrouter_glm_kimi.csv', recursive=True)
assert len(_v20_external_paths) == 1, f'Expected one GLM/Kimi source, found {_v20_external_paths}'
_v20_external = _v20_pd.read_csv(_v20_external_paths[0], usecols=['model', 'text'])
_v20_external = _v20_external[_v20_external.model.isin(['glm-5.3', 'glm-5.3-flash', 'kimi-k3'])].copy()
_v20_external['model'] = _v20_external.model.astype(str)
_v20_external['text'] = _v20_external.text.fillna('').astype(str)
_v20_external = _v20_external.drop_duplicates(['model', 'text'])

_v20_base_keys = set(zip(
    _v20_base.model.fillna('').astype(str),
    _v20_base.text.fillna('').astype(str),
))
_v20_external = _v20_external[
    ~_v20_external.apply(lambda r: (r.model, r.text) in _v20_base_keys, axis=1)
].copy()

for _v20_col in _v20_base.columns:
    if _v20_col not in _v20_external.columns:
        _v20_external[_v20_col] = _v20_np.nan
_v20_combined = _v20_pd.concat(
    [_v20_base, _v20_external[_v20_base.columns]], ignore_index=True
)
_v20_pool_path = '/kaggle/working/ntl_v20_pool_glm_kimi.parquet'
_v20_combined.to_parquet(_v20_pool_path, index=False)
POOL = _v20_pool_path
print('V20_SVC_POOL', {
    'base_rows': len(_v20_base),
    'external_rows_added': len(_v20_external),
    'external_exact_duplicates_removed': 1516 - len(_v20_external),
    'combined_rows': len(_v20_combined),
    'added_by_model': _v20_external.groupby('model').size().to_dict(),
    'pool_path': POOL,
}, flush=True)
'''


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_readme(version, title, change):
    return (
        f"# {title}\n\n"
        "Parent: the public [NLM V27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) recipe, "
        "reproduced with pinned inputs by [Avazaku](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction). "
        "That author's official 0.651 score is a reference, not our result.\n\n"
        f"{change}\n\n"
        "The public reproduction recorded a visible Kaggle run of about 4 h 4 min on T4 x2, "
        "and its hidden-run time was not reported. This project run must remain within the live "
        "9-hour notebook limit. It is a Kaggle cloud run; no local model fitting is performed.\n\n"
        "The upstream notebook code is Apache 2.0 and the input datasets/models keep their own terms. "
        "The reproduced input inventory includes CC BY-SA 4.0 handwriting-blend data and Llama 3.1 "
        "Community License assets; preserve the original notices and source links. V20 adds only "
        "the CC BY 4.0 GLM/Kimi passage dataset.\n"
    )


def build(version, title, slug, experiment_text, *, add_pool=False):
    folder = ROOT / f"kernel_v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    # The upstream executable cells remain unchanged. The new note labels this fork's scope.
    notebook["cells"].insert(0, markdown(
        f"# {title}\n\n"
        "Benchmark reference: the public Avazaku reproduction reports 0.651 for the upstream "
        "NLM V27 notebook. That is not a score for this fork.\n\n"
        f"{experiment_text}"
    ))
    if add_pool:
        # Insert after setup has discovered POOL and before the final inference cell consumes it.
        run_cells = [i for i, c in enumerate(notebook["cells"]) if c.get("cell_type") == "code"
                     and "!python infer.py" in "".join(c.get("source", []))]
        if len(run_cells) != 1:
            raise RuntimeError("Could not uniquely locate the public V27 inference cell")
        notebook["cells"].insert(run_cells[0], code_cell(V20_POOL_CELL))

    # Kaggle's notebook renderer rejects the optional cell IDs in this export,
    # including IDs inherited from the upstream notebook. Remove them uniformly.
    for cell in notebook["cells"]:
        cell.pop("id", None)

    notebook_path = folder / f"{slug.replace('-', '_')}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata["id"] = f"sanjay0614/{slug}"
    metadata["title"] = title
    metadata["code_file"] = notebook_path.name
    metadata["is_private"] = True
    if add_pool:
        metadata["dataset_sources"] = list(dict.fromkeys(
            metadata["dataset_sources"] + ["danielandresmiranda/ntl-openrouter-glm-kimi"]
        ))
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (folder / "README.md").write_text(make_readme(version, title, experiment_text), encoding="utf-8")
    return notebook_path


def main():
    p19 = build(
        19,
        "NTL V19 NLM V27 Reproduction",
        "ntl-v19-nlm-v27-reproduction",
        "Reproduce the verified public 0.651 NLM V27 recipe with the same pinned dataset, kernel, "
        "and model inputs. No inference code, weights, input versions, or blend settings are changed. "
        "This is a high-scoring public baseline reproduction, not a new method.",
    )
    p20 = build(
        20,
        "NTL V20 V27 GLM Kimi SVC Pool",
        "ntl-v20-v27-glm-kimi-svc-pool",
        "Keep the reproduced V27 recipe fixed and append the 1,516 CC BY 4.0 GLM-5.3, GLM-5.3-flash, "
        "and Kimi-K3 passages to its SVC pool only. Remove exact (model, text) duplicates against the "
        "V27 pool in Kaggle before fitting. The change is motivated by our prompt-grouped SVC-only V16 "
        "ablation (both sources +0.00333 BA) and full-pipeline V13 result (0.640 vs V10's 0.639); "
        "neither establishes a gain on the V27 stack.",
        add_pool=True,
    )
    print(p19)
    print(p20)


if __name__ == "__main__":
    main()
