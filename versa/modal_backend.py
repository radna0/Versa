from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from versa.util import ensure_dir, new_id, utc_timestamp


@dataclass(frozen=True)
class ModalRunResult:
    versa_job_id: str
    pid: int
    log_path: str
    command: list[str]


def modal_run(
    *,
    target: str,
    extra_args: list[str],
    log_dir: str = "logs",
    log_path: str | None = None,
) -> ModalRunResult:
    ensure_dir(log_dir)
    versa_job_id = new_id()
    ts = utc_timestamp()
    log_path = log_path or os.path.join(log_dir, f"versa_modal_run_{ts}_{versa_job_id}.log")

    cmd = ["modal", "run", target, *extra_args]
    with open(log_path, "ab", buffering=0) as f:
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return ModalRunResult(versa_job_id=versa_job_id, pid=int(p.pid), log_path=log_path, command=cmd)


def modal_kill(pid: int, sig: int = 15) -> None:
    os.killpg(pid, sig)

