import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "kernel_v10" / "ntl_v10_prompt_residual_readout.ipynb"
SOURCE_METADATA = ROOT / "kernel_v10" / "kernel-metadata.json"

EXPERIMENTS = (
    {
        "version": 14,
        "slug": "name-that-llm-v14-quality-svc-pool",
        "folder": "kernel_v14",
        "notebook": "ntl_v14_quality_svc_pool.ipynb",
        "title": "Name That LLM V14 Quality SVC Pool",
        "pool_block": """pool = pd.read_parquet(find("quality_pool_public.parquet"))[["model", "text"]]
print(f"Using curated public SVC pool: {len(pool)} rows", flush=True)
fit_texts = train.text.astype(str).tolist() + pool.text.astype(str).tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]""",
        "datasets": ["fleongg/ntl-quality-selected-public-svc-pool"],
        "description": (
            "Replace V10's 75,783-row SVC fit pool with the 40,129-row curated "
            "public-only SVC pool. Its published prompt-grouped, SVC-only five-fold "
            "validation reported balanced accuracy 0.4582 versus 0.4430 for the "
            "original pool. This is component OOF evidence, not full-ensemble or "
            "leaderboard evidence."
        ),
    },
    {
        "version": 15,
        "slug": "name-that-llm-v15-quality-glm-kimi-svc-pool",
        "folder": "kernel_v15",
        "notebook": "ntl_v15_quality_glm_kimi_svc_pool.ipynb",
        "title": "Name That LLM V15 Quality GLM Kimi SVC Pool",
        "pool_block": """pool = pd.read_parquet(find("quality_pool_public.parquet"))[["model", "text"]]
_glm_kimi = pd.read_csv(find("ntl_openrouter_glm_kimi.csv"))[["model", "text"]]
pool = pd.concat([pool, _glm_kimi], ignore_index=True).drop_duplicates(subset=["text"])
print(f"Using curated public SVC pool plus GLM/Kimi: {len(pool)} rows", flush=True)
fit_texts = train.text.astype(str).tolist() + pool.text.astype(str).tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]""",
        "datasets": [
            "fleongg/ntl-quality-selected-public-svc-pool",
            "danielandresmiranda/ntl-openrouter-glm-kimi",
        ],
        "description": (
            "Combine V14's curated public-only SVC pool with 1,516 GLM/Kimi "
            "passages. Exact text overlap between these two corpora was checked "
            "locally and was zero. This tests whether the separately observed "
            "curated-pool OOF gain and GLM/Kimi leaderboard gain are complementary; "
            "the combined effect is unvalidated."
        ),
    },
)

OLD_POOL_BLOCK = """pool = pd.read_parquet(find("pool_v4.parquet"))
fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
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


def build_one(experiment):
    output_dir = ROOT / experiment["folder"]
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    svc_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and OLD_POOL_BLOCK in cell_text(cell)
    ]
    if len(svc_cells) != 1:
        raise RuntimeError(
            f"Expected one V10 SVC pool cell for V{experiment['version']}, found {len(svc_cells)}"
        )
    set_cell_text(
        svc_cells[0],
        cell_text(svc_cells[0]).replace(OLD_POOL_BLOCK, experiment["pool_block"], 1),
    )

    title_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown" and "Name That LLM" in cell_text(cell)
        and "V10" in cell_text(cell)
    ]
    if len(title_cells) != 1:
        raise RuntimeError(f"Expected one V10 title cell, found {len(title_cells)}")

    attribution = (
        "Dataset attribution: fleongg/NTL Quality Selected Public SVC Pool; its per-row "
        "source and license catalog is retained in the attached dataset. "
    )
    if experiment["version"] == 15:
        attribution += (
            "The additional OpenRouter passages are from Daniel Andres Miranda, "
            "[NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/"
            "danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. "
        )
    set_cell_text(
        title_cells[0],
        f"""# Name That LLM — V{experiment['version']}: {experiment['title']}

V{experiment['version']} keeps V10's neural members, official-only prompt-residual readout, blend weights, and Sinkhorn balancing unchanged. It changes only the public text used to fit the four-view n-gram SVC.

{experiment['description']}

{attribution}No corpus is copied into this repository. The Kaggle run uses the attached datasets directly.
""",
    )

    kernel = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    kernel["id"] = f"sanjay0614/{experiment['slug']}"
    kernel["title"] = experiment["title"]
    kernel["code_file"] = experiment["notebook"]
    sources = kernel.setdefault("dataset_sources", [])
    for source in experiment["datasets"]:
        if source not in sources:
            sources.append(source)
    notebook["metadata"]["kaggle"] = {
        "id": kernel["id"],
        "title": kernel["title"],
        "code_file": kernel["code_file"],
    }
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    notebook_path = output_dir / experiment["notebook"]
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        f"""# V{experiment['version']} — {experiment['title']}

{experiment['description']}

This is a controlled branch from V10. All neural members, the official-only prompt-residual readout, blend weights, and Sinkhorn balancing remain unchanged; only the four-view n-gram SVC training pool changes.

{attribution}The attached dataset provides per-row source/license provenance. No corpus is copied into this repository. A visible three-row Kaggle smoke run checks execution/format only; the full hidden-test rerun determines leaderboard performance.
""",
        encoding="utf-8",
    )
    print(f"Wrote {notebook_path}")


def main():
    for experiment in EXPERIMENTS:
        build_one(experiment)


if __name__ == "__main__":
    main()
