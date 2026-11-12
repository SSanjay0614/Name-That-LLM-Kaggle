"""Push, verify and submit the 0.684 NTL slate.

Each kernel is pushed one at a time (the account allows two concurrent batch GPU
sessions, but one is enough and keeps CASMI free), waited to COMPLETE, its output
downloaded, and its submission.parquet checked for exactly three rows and the
competition's two columns. Only then is a submission slot spent, through the API
call that works for this account (`competition_submit_code`); the CLI's
`competitions submit -k ... -f ...` returns an empty-body 400 here.

Run from the name_that_llm folder:  python tools/run_slate_684.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMP = "name-that-llm"
T0 = time.time()
DEADLINE = T0 + 8 * 3600
GPU_BLOCKED = "Maximum batch GPU session count"

QUEUE = [
    ("kernel_v43_684_verbatim", "sanjay0614/ntl-v43-684-verbatim",
     "v43: public 0.684 (ozertuu/lbjyang) verbatim, no edits"),
    ("kernel_v44_684_sinkhorn20", "sanjay0614/ntl-v44-684-sinkhorn20",
     "v44: 0.684 code with Sinkhorn strength 0.30->0.20 (forum-documented +0.003)"),
    ("kernel_v45_684_targetmix50", "sanjay0614/ntl-v45-684-targetmix50",
     "v45: 0.684 code with Sinkhorn target mix 0.35->0.50 (lv683-a15 v14)"),
    ("kernel_v46_684_punct", "sanjay0614/ntl-v46-684-punct",
     "v46: 0.684 ensemble + training-answer punctuation prior (kirill v19)"),
    ("kernel_v47_684_native45", "sanjay0614/ntl-v47-684-native45",
     "v47: 0.684 code with native Qwen3-Embedding share 0.40->0.45"),
]


def log(msg):
    print(f"[{round(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


def kaggle(*args, timeout=1800):
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", *args],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def remaining():
    out = kaggle("competitions", "submission-limits", COMP)
    for line in out.splitlines():
        if "Remaining today" in line:
            try:
                return int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                return None
    return None


def push(folder):
    while time.time() < DEADLINE:
        out = kaggle("kernels", "push", "-p", str(HERE / folder))
        if "successfully pushed" in out.lower():
            return None
        if GPU_BLOCKED in out:
            log(f"{folder}: GPU sessions busy, retrying in 5 min")
            time.sleep(300)
            continue
        return out[-1200:]
    return "deadline hit while waiting for a GPU session"


def status(slug):
    out = kaggle("kernels", "status", slug)
    for line in reversed(out.splitlines()):
        if "KernelWorkerStatus." in line:
            return line.split("KernelWorkerStatus.")[1].split('"')[0]
    return "UNKNOWN"


def wait_complete(slug, minutes=75):
    deadline = time.time() + minutes * 60
    last = None
    while time.time() < deadline:
        state = status(slug)
        if state != last:
            log(f"{slug}: {state}")
            last = state
        if state == "COMPLETE":
            return True
        if state in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            return False
        time.sleep(60)
    log(f"{slug}: still {last} after {minutes} min")
    return False


def verify(slug, folder):
    out_dir = HERE / folder / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", slug, "-p", str(out_dir))
    artifact = out_dir / "submission.parquet"
    if not artifact.is_file():
        log(f"{slug}: no submission.parquet in output - not submitting")
        return False
    try:
        import pandas as pd
        frame = pd.read_parquet(artifact)
    except Exception as exc:  # noqa: BLE001
        log(f"{slug}: unreadable parquet ({exc}) - not submitting")
        return False
    mode_file = out_dir / "submission.mode.json"
    mode = json.loads(mode_file.read_text()).get("mode") if mode_file.is_file() else "n/a"
    log(f"{slug}: artifact rows={len(frame)} cols={frame.columns.tolist()} mode={mode}")
    ok = len(frame) == 3 and frame.columns.tolist() == ["id", "model"]
    if not ok:
        log(f"{slug}: commit run did not hit the preview path - refusing to submit")
    return ok


def submit(slug, desc):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    res = api.competition_submit_code("submission.parquet", desc, COMP, slug, 1)
    # the client returns a typed response object, not a dict
    ref = res.get("ref") if isinstance(res, dict) else getattr(res, "ref", None)
    log(f"{slug}: submitted ref={ref}")
    return bool(ref)


def main():
    log(f"slate start; slots remaining today = {remaining()}")
    done = []
    for folder, slug, desc in QUEUE:
        if remaining() == 0:
            log("no submission slots left today; stopping")
            break
        log(f"--- {slug}")
        err = push(folder)
        if err:
            log(f"{slug}: push failed -> {err}")
            break
        log(f"{slug}: pushed")
        if not wait_complete(slug):
            log(f"{slug}: commit did not complete cleanly; stopping slate")
            break
        if not verify(slug, folder):
            break
        if not submit(slug, desc):
            log(f"{slug}: submit call failed; stopping slate")
            break
        done.append((slug, desc))
    log(f"submitted {len(done)} of {len(QUEUE)}; remaining today = {remaining()}")
    for slug, desc in done:
        log(f"   OK {slug}  {desc}")


if __name__ == "__main__":
    main()
