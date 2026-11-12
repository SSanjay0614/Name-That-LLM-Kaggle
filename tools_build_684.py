"""Build today's NTL slate on the 0.684 public lineage.

Why this lineage: the account's best is 0.678 (v41), and every one of our previous
five slots came from the *0.683* "Apex Sovereign" notebook, which is a different
pipeline from the 0.684 release (lbjyang's mirror of ozertuu's 0.684). The 0.684
pipeline adds the native Qwen3-Embedding expert at share 0.40 and the Qwen3.8-27B
classifier at 3%, fits its complementary stack at run time, and has no WeMM expert
or hard typography rules. We have never run it.

All five candidates attach exactly the author's own inputs (the pull API's
dataset/kernel/model lists), so no attachment has to be guessed.

Candidates
  v43  0.684 mirror verbatim (canonical; the mount shim + exact preview emitter)
  v44  prompt-is-not-the-author v37: Sinkhorn strength 0.30 -> 0.20
  v45  lv683-a15 v14: Sinkhorn target mix 0.35 -> 0.50
  v46  LoRA + Trained27 v19: punctuation prior over the 0.684 ensemble
  v47  native-share 0.40 -> 0.45 (v33's one-argument test, Sinkhorn left at 0.30)

Every edit is an asserted exact match, and the notebook must be byte-identical
everywhere else, so a score delta is attributable to the named setting.

Run from the name_that_llm folder:  python tools_build_684.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE / "public_new3"
META = json.loads((PUB / "metadata.json").read_text(encoding="utf-8"))

PROVENANCE = """# {title}

Provenance: this kernel is a copy of the public Kaggle notebook `{source}`
(version {version}, Apache-2.0), re-published for this account's daily submission
slate. {edit_note}

The author's own markdown and credits follow unchanged, including the lineage of
every input asset. No code was modified except the setting(s) named above.

---
"""


def source_lines(text: str) -> list[str]:
    lines = text.splitlines()
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def patch(text: str, old: str, new: str, tag: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{tag}: expected exactly one match for {old!r}, got {count}")
    return text.replace(old, new)


CANDIDATES = [
    {
        "folder": "kernel_v43_684_verbatim",
        "slug": "ntl-v43-684-verbatim",
        "title": "NTL v43 684 verbatim",
        "source": "lbjyang/ntl-684-ozertuu-mirror",
        "edits": [],
        "note": "No code edits at all, so this is a clean reproduction draw of the public 0.684.",
    },
    {
        "folder": "kernel_v44_684_sinkhorn20",
        "slug": "ntl-v44-684-sinkhorn20",
        "title": "NTL v44 684 sinkhorn20",
        "source": "prvsiyan/the-prompt-is-not-the-author",
        "edits": [],
        "note": (
            "The author's current version is itself the one-argument test: Sinkhorn balancing "
            "strength 0.30 -> 0.20, everything else the 0.684 code. Forum reports put the gain at "
            "about +0.003 because the hidden class mix is uneven and strong balancing over-corrects."
        ),
    },
    {
        "folder": "kernel_v45_684_targetmix50",
        "slug": "ntl-v45-684-targetmix50",
        "title": "NTL v45 684 targetmix50",
        "source": "heliosli/name-that-llm-lv683-a15",
        "edits": [],
        "note": (
            "The author's version 14 moves the Sinkhorn target mix 0.35 -> 0.50 (further toward the "
            "uniform training prior) on top of the 0.684 copy, and version 13 of the same notebook "
            "reproduced 0.684 exactly, so this is a calibrated base plus one documented knob."
        ),
    },
    {
        "folder": "kernel_v46_684_punct",
        "slug": "ntl-v46-684-punct",
        "title": "NTL v46 684 punct",
        "source": "kirilltym/lora-trained27-evidence-for-close-calls",
        "edits": [],
        "note": (
            "Author's version 19: the 0.684 ensemble plus a punctuation prior fitted on the official "
            "training answers (spaced em dash, em dash, curly quotes, curly apostrophe; a model with "
            ">=80 answers and zero occurrences of a pattern present in the query loses 0.70 log "
            "score). Prompt-held-out checks added 4-10 correct answers out of 2,159."
        ),
    },
    {
        "folder": "kernel_v47_684_native45",
        "slug": "ntl-v47-684-native45",
        "title": "NTL v47 684 native45",
        "source": "prvsiyan/the-prompt-is-not-the-author",
        "edits": [
            ("'--sinkhorn', '0.20'", "'--sinkhorn', '0.30'",
             "restore the 0.684 Sinkhorn strength so only one setting differs from 0.684"),
            ("'--native_share', '0.40'", "'--native_share', '0.45'",
             "the v33 one-argument test: native Qwen3-Embedding expert share 0.40 -> 0.45"),
        ],
        "note": (
            "One setting away from the 0.684 code: the native Qwen3-Embedding expert share moves "
            "0.40 -> 0.45 (the author's v33 test), with the Sinkhorn strength restored to the 0.684 "
            "value so the two readings stay separable."
        ),
    },
]


def build(cand: dict) -> None:
    meta = META[cand["source"]]
    notebook = json.loads((PUB / f"{cand['source'].split('/')[1]}.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

    for old, new, why in cand["edits"]:
        hit = False
        for cell in code_cells:
            text = "".join(cell["source"])
            if old in text:
                cell["source"] = source_lines(patch(text, old, new, cand["slug"]))
                hit = True
                break
        if not hit:
            raise ValueError(f"{cand['slug']}: {old!r} not found in any code cell")

    for cell in code_cells:
        text = "".join(cell["source"])
        if text.lstrip().startswith("%%"):
            # cell magic (%%writefile ...): the payload after the first line is plain Python
            body = text.split("\n", 1)[1] if "\n" in text else ""
            if body.strip():
                ast.parse(body)
        elif not any(line.lstrip().startswith(("%", "!")) for line in text.splitlines()):
            ast.parse(text)
        cell["execution_count"] = None
        cell["outputs"] = []

    header = PROVENANCE.format(
        title=cand["title"],
        source=cand["source"],
        version=meta["currentVersionNumber"],
        edit_note=cand["note"],
    )
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(header),
    })

    out_dir = HERE / cand["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{cand['slug'].replace('-', '_')}.ipynb"
    (out_dir / code_file).write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "id": f"sanjay0614/{cand['slug']}",
        "title": cand["title"],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": list(meta["datasetDataSources"]),
        "kernel_sources": list(meta["kernelDataSources"]),
        "competition_sources": ["name-that-llm"],
        "model_sources": list(meta["modelDataSources"]),
        "docker_image": meta["dockerImage"],
        "machine_shape": meta["machineShape"],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    edits_md = (
        "\n".join(f"- `{old}` -> `{new}`  ({why})" for old, new, why in cand["edits"])
        or "- none (verbatim copy)"
    )
    (out_dir / "README.md").write_text(
        f"# {cand['title']}\n\n"
        f"- Kaggle kernel: `sanjay0614/{cand['slug']}` (private, GPU, no internet)\n"
        f"- Source: `{cand['source']}` v{meta['currentVersionNumber']} (public, Apache-2.0)\n"
        f"- Built by: `tools_build_684.py` (2026-11-12)\n\n"
        f"{cand['note']}\n\n"
        "## Edits\n\n"
        f"{edits_md}\n\n"
        "## Attachments\n\n"
        f"- {len(metadata['dataset_sources'])} datasets, "
        f"{len(metadata['kernel_sources'])} kernel outputs, "
        f"{len(metadata['model_sources'])} models - exactly the author's own lists, read from "
        "the pull API rather than assumed.\n"
        f"- Docker image: `{metadata['docker_image']}`\n"
        f"- Competition: `name-that-llm`\n\n"
        "## Contract\n\n"
        "- Output: `submission.parquet` with `id, model`; the hidden scoring rerun executes the "
        "full two-GPU pipeline.\n"
        "- The commit run must hit the notebook's own public-fixture preview path before the slot "
        "is spent.\n",
        encoding="utf-8",
    )
    print(f"built {cand['folder']:26s} <- {cand['source']} v{meta['currentVersionNumber']} "
          f"({len(notebook['cells'])} cells, {len(cand['edits'])} edit(s))")


if __name__ == "__main__":
    for candidate in CANDIDATES:
        build(candidate)
