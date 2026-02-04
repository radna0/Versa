import os
import shutil
import subprocess

import modal

app = modal.App("versa-validate-gpu-smoke")


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

    try:
        return cls(count=gc)
    except TypeError:
        # Some configs might not accept count.
        return cls()


GPU = _gpu_config()


@app.function(gpu=GPU)
def probe() -> str:
    nvsmi = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        out = subprocess.check_output([nvsmi, "-L"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        out = f"nvidia_smi_failed:{type(e).__name__}:{e}"
    print(out)
    return out


@app.local_entrypoint()
def main() -> None:
    gt = (os.getenv("VERSA_MODAL_GPU_TYPE") or "").strip()
    gc = (os.getenv("VERSA_MODAL_GPU_COUNT") or "").strip()
    print(f"requested_gpu_type={gt!r} requested_gpu_count={gc!r}")
    out = probe.remote()
    print("probe_done")
    print(out)

