"""Print the real attachment lists of public NTL notebooks.

`kaggle kernels pull` only writes the .ipynb, so the dataset/model/kernel
sources are invisible locally. The pull API returns them in JSON, which is what
we need before re-pushing a foreign notebook under our own slug.
"""

import json
import os
import sys
from pathlib import Path

import requests

cfg = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
auth = (cfg["username"], cfg["key"])

KERNELS = [
    "kirilltym/lora-trained27-evidence-for-close-calls",
    "prvsiyan/the-prompt-is-not-the-author",
    "whitegg/ntl-fuse33-0-668-fork-response-corpus",
    "heliosli/name-that-llm-lv683-a15",
    "lbjyang/ntl-684-ozertuu-mirror",
]

out = {}
for kernel in KERNELS:
    url = f"https://www.kaggle.com/api/v1/kernels/pull/{kernel}"
    r = requests.get(url, auth=auth, timeout=120)
    if r.status_code != 200:
        print(f"== {kernel}: HTTP {r.status_code}")
        continue
    data = r.json()
    meta = data.get("metadata", {})
    keys = [
        "id", "title", "lastRunTime", "currentVersionNumber", "categoryIds",
        "datasetDataSources", "kernelDataSources", "modelDataSources",
        "competitionDataSources", "dockerImage", "machineShape", "enableGpu",
        "enableInternet", "isPrivate", "totalVotes", "language",
    ]
    print("=" * 25, kernel)
    for k in keys:
        v = meta.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            v = [x.get("source") or x.get("slug") or x for x in v]
        print(f"  {k}: {v}")
    out[kernel] = meta

(Path("public_new3") / "metadata.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nsaved public_new3/metadata.json")
