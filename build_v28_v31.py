import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "kernel_v26"
SOURCE_NOTEBOOK = SOURCE_DIR / "ntl_v26_apex_trio_blend.ipynb"
CONFIGS = [
    (28, "qwen16-atlas18", 0.16, 0.12, 0.18),
    (29, "helper10-qwen18-atlas18", 0.18, 0.10, 0.18),
    (30, "atlas20-qwen18", 0.18, 0.12, 0.20),
    (31, "atlas16-qwen18", 0.18, 0.12, 0.16),
]


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def main():
    base = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for version, slug, qwen, helper, atlas in CONFIGS:
        notebook = json.loads(json.dumps(base))
        title = f"NTL V{version} {slug}"
        kernel_id = f"sanjay0614/ntl-v{version}-{slug}"
        settings = notebook["cells"][2]
        text = source_text(settings)
        old_selector = "VARIANT = 7"
        if text.count(old_selector) != 1:
            raise ValueError(f"V{version}: expected one selector")
        text = text.replace(old_selector, f"VARIANT = {version}")
        old_entry = "    7: dict(qwen=0.18, helper=0.12, atlas=0.18),"
        new_entry = (
            f"    7: dict(qwen=0.18, helper=0.12, atlas=0.18),\n"
            f"    {version}: dict(qwen={qwen:.2f}, helper={helper:.2f}, atlas={atlas:.2f}),"
        )
        if text.count(old_entry) != 1:
            raise ValueError(f"V{version}: expected one V26 setting")
        settings["source"] = text.replace(old_entry, new_entry).splitlines(keepends=True)

        for cell in notebook["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            content = source_text(cell)
            content = content.replace("NTL V26 Apex Trio Blend", title)
            content = content.replace(
                "qwen=0.18, helper=0.12, atlas=0.18",
                f"qwen={qwen:.2f}, helper={helper:.2f}, atlas={atlas:.2f}",
            )
            cell["source"] = content.splitlines(keepends=True)
        kaggle = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
        kaggle.update({"id": kernel_id, "title": title, "code_file": f"ntl_v{version}_{slug}.ipynb"})

        output_dir = ROOT / f"kernel_v{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        code_file = f"ntl_v{version}_{slug}.ipynb"
        (output_dir / code_file).write_text(
            json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
        )
        metadata = json.loads((SOURCE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
        metadata.update({"id": kernel_id, "title": title, "code_file": code_file})
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "README.md").write_text(
            f"# NTL V{version}: {slug}\n\n"
            f"- Parent: V26, public score 0.644 (`57831956`).\n"
            f"- Blend: Qwen {qwen:.2f}, Helper {helper:.2f}, ATLAS {atlas:.2f}.\n"
            "- This is a one-weight-at-a-time probe relative to V26; it is not a validated improvement.\n"
            "- Fast commit is a format preview only; leaderboard evaluation uses the full scoring rerun.\n"
            "- Score: pending.\n",
            encoding="utf-8",
        )
        print(f"Built V{version}: {kernel_id}")


if __name__ == "__main__":
    main()
