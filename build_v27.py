import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "kernel_v26"
OUTPUT_DIR = ROOT / "kernel_v27"
SOURCE_NOTEBOOK = SOURCE_DIR / "ntl_v26_apex_trio_blend.ipynb"
NOTEBOOK_NAME = "ntl_v27_qwen20_atlas18.ipynb"
KERNEL_ID = "sanjay0614/ntl-v27-qwen20-atlas18"
TITLE = "NTL V27 Qwen20 Atlas18"


def main():
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    settings = cells[2]
    source = "".join(settings["source"]) if isinstance(settings["source"], list) else settings["source"]
    old = "VARIANT = 7"
    if source.count(old) != 1:
        raise ValueError(f"Expected one variant selector, found {source.count(old)}")
    source = source.replace(old, "VARIANT = 8")
    old_variant = "    7: dict(qwen=0.18, helper=0.12, atlas=0.18),"
    new_variant = (
        "    7: dict(qwen=0.18, helper=0.12, atlas=0.18),\n"
        "    8: dict(qwen=0.20, helper=0.10, atlas=0.18),"
    )
    if source.count(old_variant) != 1:
        raise ValueError("Expected one V26 blend definition")
    settings["source"] = source.replace(old_variant, new_variant).splitlines(keepends=True)

    for cell in cells:
        if cell.get("cell_type") == "markdown":
            content = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
            content = content.replace("NTL V26 Apex Trio Blend", TITLE)
            content = content.replace(
                "qwen=0.18, helper=0.12, atlas=0.18",
                "qwen=0.20, helper=0.10, atlas=0.18",
            )
            cell["source"] = content.splitlines(keepends=True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    notebook["metadata"]["kaggle"].update(
        {"id": KERNEL_ID, "title": TITLE, "code_file": NOTEBOOK_NAME}
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / NOTEBOOK_NAME).write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
    )
    metadata = json.loads((SOURCE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update({"id": KERNEL_ID, "title": TITLE, "code_file": NOTEBOOK_NAME})
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built {NOTEBOOK_NAME}: qwen=0.20, helper=0.10, atlas=0.18")


if __name__ == "__main__":
    main()
