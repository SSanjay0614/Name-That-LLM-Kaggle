"""Push, verify and submit the five-kernel NTL slate, waiting out the GPU-session cap.

The account allows two concurrent batch GPU sessions and both are held by CASMI training
kernels, so the first push is expected to be refused. Only that specific refusal is retried;
any other push error aborts the run. A kernel is submitted only after its commit run is
COMPLETE *and* its output contains the three-row preview artifact, so a commit that silently
fell through to the full pipeline can never consume a submission slot.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMP = "name-that-llm"
QUEUE = [
    ("kernel_v38", "sanjay0614/ntl-v38-0683-verbatim",
     "v38: 0.683 verbatim (V28 Sovereign Titan Apex Fusion)"),
    ("kernel_v39", "sanjay0614/ntl-v39-gsq12",
     "v39: v38 + GSQ share 0.12 (0.682 recipe's setting)"),
    ("kernel_v40", "sanjay0614/ntl-v40-wemm14",
     "v40: v38 + WeMM expert share 0.14 (was 0.07)"),
    ("kernel_v41", "sanjay0614/ntl-v41-sinkhorn45",
     "v41: v38 + sinkhorn strength .30->.45"),
    ("kernel_v42", "sanjay0614/ntl-v42-gsq12-wemm14",
     "v42: v38 + GSQ .12 + WeMM .14 together"),
]
T0 = time.time()
DEADLINE = T0 + 14 * 3600
GPU_BLOCKED = "Maximum batch GPU session count"


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
            return
        if GPU_BLOCKED in out:
            log(f"{folder}: GPU sessions busy, retrying in 5 min")
            time.sleep(300)
            continue
        raise RuntimeError(f"push failed for {folder}:\n{out[-1500:]}")
    raise RuntimeError("deadline hit while waiting for a GPU session")


def status(slug):
    out = kaggle("kernels", "status", slug)
    for line in reversed(out.splitlines()):
        if "KernelWorkerStatus." in line:
            return line.split("KernelWorkerStatus.")[1].split('"')[0]
    return "UNKNOWN"


def wait_complete(slug, minutes=60):
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
        time.sleep(30)
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
    mode = out_dir / "submission.mode.json"
    mode_info = json.loads(mode.read_text()) if mode.is_file() else {}
    log(f"{slug}: artifact rows={len(frame)} cols={frame.columns.tolist()} "
        f"mode={mode_info.get('mode')}")
    ok = (len(frame) == 3 and frame.columns.tolist() == ["id", "model"]
          and mode_info.get("mode") == "exact_public_format_preview")
    if not ok:
        log(f"{slug}: commit run is not the preview path - refusing to submit")
    return ok


def submit(slug, desc):
    out = kaggle("competitions", "submit", "-c", COMP, "-k", slug, "-v", "1",
                 "-f", "submission.parquet", "-m", desc)
    log("submit -> " + " | ".join(line.strip() for line in out.strip().splitlines()[-2:]))
    return "Submission ref:" in out


def main():
    log(f"slate start; slots remaining today = {remaining()}")
    done = []
    for folder, slug, desc in QUEUE:
        if remaining() == 0:
            log("no submission slots left today; stopping")
            break
        log(f"--- {slug}")
        try:
            push(folder)
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
        except Exception as exc:  # noqa: BLE001
            log(f"{slug}: aborted -> {exc}")
            break
    log(f"submitted {len(done)} of {len(QUEUE)}")
    for slug, desc in done:
        log(f"   OK {slug}  {desc}")


if __name__ == "__main__":
    main()
