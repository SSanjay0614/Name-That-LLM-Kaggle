"""Build NTL probe kernels from the verbatim 0.683 notebook.

One change per kernel, each an axis this notebook leaves untested but that a neighbouring
public notebook documents:

  v39  gsq_share      0.06 -> 0.12   (omerfarukmerey's 0.682 recipe used GSQ 12%)
  v40  wemm share     0.07 -> 0.14   (CHEN Han's WeMM expert was added at 0.07 unprobed)
  v41  sinkhorn       .30  -> .45    (judith007's 0.664 stack ran .45; .60 cost -0.003)
  v42  gsq 0.12 + wemm 0.14          (both expert axes at once)

Every patch asserts its exact match count, and the notebook must stay byte-identical
everywhere else, so a probe result is attributable to the named change.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "kernel_v38" / "ntl_v38_0683_verbatim.ipynb"
META = HERE / "kernel_v38" / "kernel-metadata.json"

GSQ_OLD = "CONFIG['gsq_share'] = 0.06"
GSQ_ASSERT_OLD = "assert config['gsq_share'] == 0.06, 'Unexpected GSQ share'"
WEMM_OLD = "'share': 0.07, 'deadline_seconds': 30240"
SINK_OLD = "def balance_logits(P, train_labels, strength=.30, target_mix=.35):"


def patch(text: str, old: str, new: str, want: int = 1) -> str:
    got = text.count(old)
    assert got == want, f"expected {want} match(es) for {old!r}, got {got}"
    return text.replace(old, new)


def gsq(nb, value: float) -> None:
    cell = nb["cells"][59]
    src = patch("".join(cell["source"]), GSQ_OLD, f"CONFIG['gsq_share'] = {value!r}")
    cell["source"] = src.splitlines(keepends=True)
    ens = nb["cells"][49]
    src = patch("".join(ens["source"]), GSQ_ASSERT_OLD,
                f"assert config['gsq_share'] == {value!r}, 'Unexpected GSQ share'")
    ens["source"] = src.splitlines(keepends=True)


def wemm(nb, value: float) -> None:
    cell = nb["cells"][59]
    src = patch("".join(cell["source"]), WEMM_OLD,
                f"'share': {value!r}, 'deadline_seconds': 30240")
    cell["source"] = src.splitlines(keepends=True)


def sinkhorn(nb, value: float) -> None:
    cell = nb["cells"][13]
    src = patch("".join(cell["source"]), SINK_OLD,
                f"def balance_logits(P, train_labels, strength={value!r}, target_mix=.35):")
    cell["source"] = src.splitlines(keepends=True)


VARIANTS = [
    ("kernel_v39", "ntl_v39_gsq12.ipynb", "sanjay0614/ntl-v39-gsq12", "NTL v39 gsq12",
     [("gsq_share 0.06->0.12", lambda nb: gsq(nb, 0.12))]),
    ("kernel_v40", "ntl_v40_wemm14.ipynb", "sanjay0614/ntl-v40-wemm14", "NTL v40 wemm14",
     [("wemm share 0.07->0.14", lambda nb: wemm(nb, 0.14))]),
    ("kernel_v41", "ntl_v41_sinkhorn45.ipynb", "sanjay0614/ntl-v41-sinkhorn45", "NTL v41 sinkhorn45",
     [("sinkhorn strength .30->.45", lambda nb: sinkhorn(nb, 0.45))]),
    ("kernel_v42", "ntl_v42_gsq12_wemm14.ipynb", "sanjay0614/ntl-v42-gsq12-wemm14", "NTL v42 gsq12 wemm14",
     [("gsq_share 0.06->0.12", lambda nb: gsq(nb, 0.12)),
      ("wemm share 0.07->0.14", lambda nb: wemm(nb, 0.14))]),
]


def main() -> None:
    base = json.loads(SRC.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    for folder, code_file, kid, title, edits in VARIANTS:
        nb = json.loads(SRC.read_text(encoding="utf-8"))
        for label, fn in edits:
            fn(nb)
        # only the intended cells may differ from the source notebook
        changed = [i for i, (a, b) in enumerate(zip(base["cells"], nb["cells"]))
                   if a["source"] != b["source"]]
        assert len(base["cells"]) == len(nb["cells"]), "cell count changed"
        out_dir = HERE / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / code_file).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        m = dict(meta)
        m["id"] = kid
        m["title"] = title
        m["code_file"] = code_file
        (out_dir / "kernel-metadata.json").write_text(
            json.dumps(m, indent=2) + "\n", encoding="utf-8")
        print(f"{folder:11s} {code_file:26s} cells changed {changed} :: "
              + "; ".join(label for label, _ in edits))


if __name__ == "__main__":
    main()
