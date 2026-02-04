import os
import shutil
import subprocess
import time

import modal

app = modal.App("versa-gpu-sleep")


def _gpu_config():
    gt = (os.getenv("VERSA_MODAL_GPU_TYPE") or "").strip()
    if not gt:
        return None
    try:
        gc = int((os.getenv("VERSA_MODAL_GPU_COUNT") or "1").strip() or "1")
    except Exception:
        gc = 1

    gt_u = gt.upper()
    mapping = {
        "B200": "B200",
        "H200": "H200",
        "H100": "H100",
        "A100": "A100",
        "L40S": "L40S",
        "A10": "A10G",
        "A10G": "A10G",
        "L4": "L4",
        "T4": "T4",
    }
    cls_name = mapping.get(gt_u, gt)
    if not hasattr(modal.gpu, cls_name):
        raise RuntimeError(f"Unsupported Modal GPU type: {gt!r} (class {cls_name!r} missing)")
    cls = getattr(modal.gpu, cls_name)

    # NOTE: Some Modal versions prefer gpu=\"H100\". Using gpu=modal.gpu.H100(...)
    # still works, but may show a deprecation warning. We keep count support.
    try:
        return cls(count=gc)
    except TypeError:
        return cls()


GPU = _gpu_config()


@app.function(gpu=GPU)
def sleeper(seconds: int = 600) -> str:
    nvsmi = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        out = subprocess.check_output([nvsmi, "-L"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        out = f"nvidia_smi_failed:{type(e).__name__}:{e}"
    print(out)

    seconds = int(seconds or 0)
    for i in range(seconds):
        time.sleep(1)
        if i and (i % 30) == 0:
            print(f"tick {i}/{seconds}")
    return "done"


@app.local_entrypoint()
def main() -> None:
    try:
        seconds = int((os.getenv("VERSA_SLEEP_SECONDS") or "600").strip() or "600")
    except Exception:
        seconds = 600

    gt = (os.getenv("VERSA_MODAL_GPU_TYPE") or "").strip()
    gc = (os.getenv("VERSA_MODAL_GPU_COUNT") or "").strip()
    print(f"requested_gpu_type={gt!r} requested_gpu_count={gc!r} seconds={seconds}")
    sleeper.remote(seconds)

