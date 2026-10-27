import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "kernel_v10" / "ntl_v10_prompt_residual_readout.ipynb"
SOURCE_METADATA = ROOT / "kernel_v10" / "kernel-metadata.json"

OLD_POOL_BLOCK = """pool = pd.read_parquet(find("pool_v4.parquet"))
fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]"""


def cell_text(cell):
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def set_cell_text(cell, value):
    old = cell.get("source", [])
    cell["source"] = value if isinstance(old, str) else value.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def build(version, source, filter_expr, source_label, expected_rows):
    folder = ROOT / f"kernel_v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replacement = f"""pool = pd.read_parquet(find("pool_v4.parquet"))[["model", "text"]]
_external = pd.read_csv(find("ntl_openrouter_glm_kimi.csv"))[["model", "text"]]
_external = _external.loc[{filter_expr}].copy()
assert len(_external) == {expected_rows}, f"Expected {expected_rows} {source_label} passages, found {{len(_external)}}"
pool = pd.concat([pool, _external], ignore_index=True)
print(f"SVC pool rows including {source_label}: {{len(pool)}}", flush=True)
fit_texts = train.text.astype(str).tolist() + pool.text.astype(str).tolist()
fit_y = np.r_[y, pool.model.map({{l: i for i, l in enumerate(LABELS)}}).values]"""

    matching = [cell for cell in notebook["cells"]
                if cell.get("cell_type") == "code" and OLD_POOL_BLOCK in cell_text(cell)]
    if len(matching) != 1:
        raise RuntimeError(f"V{version}: expected one V10 pool block, found {len(matching)}")
    set_cell_text(matching[0], cell_text(matching[0]).replace(OLD_POOL_BLOCK, replacement, 1))

    title = [cell for cell in notebook["cells"]
             if cell.get("cell_type") == "markdown" and "Name That LLM" in cell_text(cell)
             and "V10" in cell_text(cell)]
    if len(title) != 1:
        raise RuntimeError(f"V{version}: expected one V10 title/attribution cell, found {len(title)}")
    details = (
        f"V{version} keeps V10's complete neural/SVC/ATLAS ensemble, official-only readout, blend weights, "
        f"and Sinkhorn balancing unchanged. The only model change is adding {expected_rows:,} CC BY 4.0 "
        f"{source_label} passages to the original SVC pool. This isolates that source from V13's combined "
        "GLM/Kimi addition; it is an exploratory leaderboard test, not a full-ensemble grouped-OOF result.\n\n"
        "Dataset attribution: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages] "
        "(https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. "
        "The dataset README records third-party provider routing and generation settings.\n"
    )
    set_cell_text(title[0], f"# Name That LLM — V{version}: {source_label} SVC augmentation\n\n{details}")

    source_meta = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    kernel_title = f"NTL V{version} {source_label.title()} SVC Pool"
    slug = ("glm-5-3-and-glm-5-3-flash" if version == 17 else "kimi-k3")
    kernel_id = f"sanjay0614/ntl-v{version}-{slug}-svc-pool"
    code_file = f"ntl_v{version}_{source}_svc_pool.ipynb"
    notebook.setdefault("metadata", {})["kaggle"] = {
        "id": kernel_id, "title": kernel_title, "code_file": code_file,
    }
    source_meta.update({"id": kernel_id, "title": kernel_title, "code_file": code_file})
    datasets = source_meta.setdefault("dataset_sources", [])
    extra_dataset = "danielandresmiranda/ntl-openrouter-glm-kimi"
    if extra_dataset not in datasets:
        datasets.append(extra_dataset)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    notebook_path = folder / code_file
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps(source_meta, indent=2) + "\n", encoding="utf-8")
    (folder / "README.md").write_text(
        f"""# V{version} — {source_label} SVC augmentation

Parent: V10. The complete neural/SVC/ATLAS ensemble, official-only residual readout, calibration, and Sinkhorn settings are unchanged. This version adds only {expected_rows:,} {source_label} passages to the original V10 SVC pool.

This is an exploratory leaderboard candidate. No full-ensemble grouped OOF result is claimed. The public score should be compared with V10 (0.639) and V13 (0.640), with the small public leaderboard sample treated as noisy evidence.

Data attribution: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. The dataset documents third-party provider routing. No hidden labels or leaderboard-derived labels are used.
""", encoding="utf-8")
    print(notebook_path)


def main():
    build(17, "glm", "_external.model.str.startswith('glm-')", "GLM-5.3 and GLM-5.3-flash", 1200)
    build(18, "kimi", "_external.model.eq('kimi-k3')", "Kimi-K3", 316)


if __name__ == "__main__":
    main()
