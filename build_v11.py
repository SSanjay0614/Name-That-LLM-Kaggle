import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "kernel_v10"
OUTPUT_DIR = ROOT / "kernel_v11"
SOURCE_NOTEBOOK = SOURCE_DIR / "ntl_v10_prompt_residual_readout.ipynb"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "ntl_v11_residual_weight_ablation.ipynb"


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

    title_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
        and "Name That LLM" in cell_text(cell)
        and "V10" in cell_text(cell)
    ]
    if len(title_cells) != 1:
        raise RuntimeError(f"Expected one V10 title cell, found {len(title_cells)}")
    title = cell_text(title_cells[0])
    title = title.replace("V10", "V11", 1).replace("5%-weight", "10%-weight", 1)
    title = title.replace(
        "The OOF diagnostic uses GroupKFold on `prompt_id`, so prompt siblings remain together.",
        "The OOF diagnostic uses GroupKFold on `prompt_id`, so prompt siblings remain together. "
        "V11 changes only the readout mixture weight from 5% to 10%; all V6 and V10 "
        "features, model fits, and other ensemble weights remain fixed.",
    )
    set_cell_text(title_cells[0], title)

    blend_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "_readout = np.load('/kaggle/working/official_residual_readout.npz'" in cell_text(cell)
    ]
    if len(blend_cells) != 1:
        raise RuntimeError(f"Expected one readout blend cell, found {len(blend_cells)}")
    blend = cell_text(blend_cells[0])
    old_expression = "P = 0.95 * P + 0.05 * _readout_prob"
    if blend.count(old_expression) != 1 or "weight=0.05" not in blend:
        raise RuntimeError("Could not uniquely locate V10's 5% readout mixture")
    blend = blend.replace(old_expression, "P = 0.90 * P + 0.10 * _readout_prob", 1)
    blend = blend.replace("weight=0.05", "weight=0.10", 1)
    set_cell_text(blend_cells[0], blend)

    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    metadata = json.loads((SOURCE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = "sanjay0614/ntl-v11-residual-weight-ablation"
    metadata["title"] = "NTL V11 Residual Weight Ablation"
    metadata["code_file"] = OUTPUT_NOTEBOOK.name
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
