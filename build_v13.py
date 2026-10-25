import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "kernel_v10" / "ntl_v10_prompt_residual_readout.ipynb"
SOURCE_METADATA = ROOT / "kernel_v10" / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kernel_v13"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "ntl_v13_glm_kimi_svc_pool.ipynb"

OLD_POOL_BLOCK = """pool = pd.read_parquet(find("pool_v4.parquet"))
fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]"""

NEW_POOL_BLOCK = """pool = pd.read_parquet(find("pool_v4.parquet"))
_glm_kimi = pd.read_csv(find("ntl_openrouter_glm_kimi.csv"))[["model", "text"]]
pool = pd.concat([pool[["model", "text"]], _glm_kimi], ignore_index=True)
print(f"SVC pool rows including GLM/Kimi: {len(pool)}", flush=True)
fit_texts = train.text.astype(str).tolist() + pool.text.astype(str).tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]"""


def cell_text(cell):
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def set_cell_text(cell, text):
    source = cell.get("source", [])
    cell["source"] = text if isinstance(source, str) else text.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    svc_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and OLD_POOL_BLOCK in cell_text(cell)
    ]
    if len(svc_cells) != 1:
        raise RuntimeError(f"Expected one V10 SVC pool cell, found {len(svc_cells)}")
    svc = cell_text(svc_cells[0]).replace(OLD_POOL_BLOCK, NEW_POOL_BLOCK, 1)
    set_cell_text(svc_cells[0], svc)

    title_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown" and "Name That LLM" in cell_text(cell)
        and "V10" in cell_text(cell)
    ]
    if len(title_cells) != 1:
        raise RuntimeError(f"Expected one V10 title cell, found {len(title_cells)}")
    set_cell_text(
        title_cells[0],
        """# Name That LLM — V13: GLM/Kimi SVC Pool Augmentation

V13 keeps V10's full model ensemble, official-only prompt-residual readout, blend weights, and Sinkhorn balancing unchanged. Its only model change is adding 1,516 CC BY 4.0 passages from GLM-5.3, GLM-5.3-flash, and Kimi-K3 to the existing n-gram SVC fit pool. This directly tests a separately published pool-augmentation experiment that improved a different 0.639 notebook by one displayed leaderboard point; that result is motivation, not a score forecast for this V10-based branch.

Dataset attribution: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. The dataset README documents the source prompt bank and notes third-party provider routing. No passages or model weights are copied into this repository.
""",
    )

    metadata = notebook["metadata"]
    kernel = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    kernel["id"] = "sanjay0614/ntl-v13-glm-kimi-svc-pool"
    kernel["title"] = "NTL V13 GLM Kimi SVC Pool"
    kernel["code_file"] = OUTPUT_NOTEBOOK.name
    sources = kernel.setdefault("dataset_sources", [])
    source = "danielandresmiranda/ntl-openrouter-glm-kimi"
    if source not in sources:
        sources.append(source)
    metadata["kaggle"] = {
        "id": kernel["id"],
        "title": kernel["title"],
        "code_file": kernel["code_file"],
    }

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(kernel, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(
        """# V13 — GLM/Kimi SVC Pool Augmentation

V13 uses V10 as its baseline and leaves all neural members, the official-only prompt-residual readout, blend weights, and Sinkhorn balancing unchanged. It adds the 1,516 CC BY 4.0 GLM-5.3, GLM-5.3-flash, and Kimi-K3 passages to the existing four-view n-gram SVC training pool.

A public GLM/Kimi pool-patch experiment scored 0.640 versus its 0.639 control, but it used a different 0.639 notebook. That finding motivates this controlled transfer test; it does not establish an expected V13 score. Dataset attribution: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0.

The Kaggle notebook's visible three-row fixture verifies execution and formatting only. It is not validation evidence. The full competition score must come from the hidden-test rerun.
""",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
