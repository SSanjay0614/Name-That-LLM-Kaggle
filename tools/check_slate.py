import json
from pathlib import Path

FOLDERS = [
    "kernel_v43_684_verbatim",
    "kernel_v44_684_sinkhorn20",
    "kernel_v45_684_targetmix50",
    "kernel_v46_684_punct",
    "kernel_v47_684_native45",
]

KEYS = [
    "emit_known_preview",
    "exact_public_format_preview",
    "--sinkhorn', '0.20'",
    "--sinkhorn', '0.30'",
    "--sinkhorn_target_mix', '0.50'",
    "--sinkhorn_target_mix', '0.35'",
    "--native_share', '0.45'",
    "--native_share', '0.40'",
    "--comp_w', '0.20'",
]

for d in FOLDERS:
    nb = json.loads(next(Path(d).glob("*.ipynb")).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    text = "".join("".join(c["source"]) for c in code)
    counts = {k: text.count(k) for k in KEYS}
    print(d, "| cells", len(nb["cells"]), "|", counts)
