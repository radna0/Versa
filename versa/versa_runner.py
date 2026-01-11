from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

from versa.modal_backend import ModalRunResult, modal_run
from versa.remote_jupyter_backend import (
    JupyterJob,
    job_start_async,
    normalize_server_url,
    run_commands_async,
    sync_dir_async,
)
from versa.util import ensure_dir


Backend = Literal["modal", "jupyter"]


@dataclass(frozen=True)
class VersaJobPublic:
    versa_job_id: str
    backend: Backend
    log_path: str
    details: dict[str, Any]


def _public(job: VersaJobPublic) -> dict[str, Any]:
    return asdict(job)


def versa_run_modal(
    *,
    target: str,
    extra_args: list[str],
    log_dir: str = "logs",
    log_path: str | None = None,
) -> VersaJobPublic:
    r: ModalRunResult = modal_run(target=target, extra_args=extra_args, log_dir=log_dir, log_path=log_path)
    return VersaJobPublic(
        versa_job_id=r.versa_job_id,
        backend="modal",
        log_path=r.log_path,
        details={"pid": r.pid, "command": r.command},
    )


async def versa_run_jupyter(
    *,
    url: str,
    token: str = "",
    kernel_id: str | None = None,
    sync: dict[str, Any] | list[dict[str, Any]] | None = None,
    bootstrap: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
) -> VersaJobPublic:
    if run is None:
        raise ValueError("run is required for jupyter backend")

    server_url = normalize_server_url(url)
    token = token or ""

    # 1) Sync (optional)
    sync_ops: list[dict[str, Any]] = []
    if isinstance(sync, dict):
        sync_ops = [sync]
    elif isinstance(sync, list):
        sync_ops = sync

    for op in sync_ops:
        await sync_dir_async(
            server_url=server_url,
            token=token,
            local_dir=str(op["local_dir"]),
            remote_dir=str(op["remote_dir"]),
            exclude_globs_csv=str(op.get("exclude_globs_csv") or ""),
            clean_remote=bool(op.get("clean_remote", True)),
            keep_archive=bool(op.get("keep_archive", False)),
            max_bytes=int(op.get("max_bytes", 200 * 1024 * 1024)),
            kernel_id=kernel_id,
            timeout_s=float(op.get("timeout_s", 600.0)),
        )

    # 2) Bootstrap (optional)
    if bootstrap is not None:
        cmds = bootstrap.get("commands") or []
        if not isinstance(cmds, list) or not all(isinstance(c, str) for c in cmds):
            raise TypeError("bootstrap.commands must be a list[str]")
        await run_commands_async(
            server_url=server_url,
            token=token,
            commands=cmds,
            cwd=str(bootstrap.get("cwd", "")),
            env_overrides=bootstrap.get("env_overrides") or None,
            timeout_s_per_command=float(bootstrap.get("timeout_s_per_command", 600.0)),
            fail_fast=bool(bootstrap.get("fail_fast", True)),
            max_output_chars=int(bootstrap.get("max_output_chars", 20000)),
            kernel_id=kernel_id,
        )

    # 3) Run (required)
    cmd = run.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("run.command is required")

    job: JupyterJob = await job_start_async(
        server_url=server_url,
        token=token,
        command=cmd,
        cwd=str(run.get("cwd", "")),
        env_overrides=run.get("env_overrides") or None,
        log_path=run.get("log_path") or None,
        kernel_id=kernel_id,
        timeout_s=float(run.get("timeout_s", 60.0)),
    )

    return VersaJobPublic(
        versa_job_id=job.job_id,
        backend="jupyter",
        log_path=job.log_path,
        details={
            "pid": job.pid,
            "start_time_ticks": job.start_time_ticks,
            "server_url": job.server_url,
            "kernel_id": job.kernel_id,
            "command": job.command,
            "cwd": job.cwd,
            "env_overrides": job.env_overrides,
        },
    )


def ensure_versa_log_dir(path: str = "logs") -> None:
    ensure_dir(path)
    # Create a marker to make it easy to glob.
    marker = os.path.join(path, ".versa")
    if not os.path.exists(marker):
        open(marker, "a", encoding="utf-8").close()
