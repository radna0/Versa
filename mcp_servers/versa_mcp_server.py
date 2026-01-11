#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp.server import FastMCP

from versa.modal_file_runner import RUNNER_SOURCE
from versa.modal_backend import modal_kill, modal_run
from versa.remote_jupyter_backend import (
    DEFAULT_SYNC_EXCLUDES,
    download_dir_to_local_async,
    download_to_local_async,
    job_kill_async,
    job_start_async,
    job_status_async,
    job_tail_async,
    normalize_server_url,
    run_commands_async,
    sync_dir_async,
    upload_local_file_async,
)
from versa.util import ensure_dir


Backend = Literal["modal", "jupyter"]


_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _store(job_id: str, record: dict[str, Any]) -> None:
    with _LOCK:
        _JOBS[job_id] = record


def _get(job_id: str) -> dict[str, Any]:
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            raise KeyError(f"Unknown versa_job_id: {job_id}")
        return dict(rec)


def _mask_secrets(record: dict[str, Any]) -> dict[str, Any]:
    rec = dict(record)
    if "token" in rec:
        rec["token"] = "<redacted>"
    if "server_url" in rec and isinstance(rec["server_url"], str) and rec["server_url"]:
        try:
            from urllib.parse import urlsplit

            p = urlsplit(rec["server_url"])
            rec["server_url"] = f"{p.scheme}://{p.netloc}/<redacted>"
        except Exception:
            rec["server_url"] = "<redacted>"
    return rec


mcp = FastMCP(name="Versa MCP")


@mcp.tool()
async def versa_run(
    backend: Annotated[str, Field(description="Backend: modal|jupyter")] = "modal",
    target: Annotated[str, Field(description="Target to run (modal target or script/command).")] = "",
    args: Annotated[list[str], Field(description="Extra args to pass to target.")] = (),
    # Modal options
    modal_log_dir: Annotated[str, Field(description="Local log dir for modal backend.")] = "logs",
    modal_log_path: Annotated[str | None, Field(description="Local log path override (optional).")] = None,
    # Jupyter options
    url: Annotated[str | None, Field(description="Remote Jupyter base URL (e.g. https://.../proxy).")] = None,
    token: Annotated[str, Field(description="Remote Jupyter token (optional).")] = "",
    kernel_id: Annotated[str | None, Field(description="Existing kernel id (optional).")] = None,
    cwd: Annotated[str, Field(description="Remote working directory (jupyter).")] = "",
    env_overrides: Annotated[dict[str, str] | None, Field(description="Env overrides (jupyter).")] = None,
    # Jupyter sync/bootstrap
    sync_local_dir: Annotated[str | None, Field(description="Local dir to sync (optional).")] = None,
    sync_remote_dir: Annotated[str | None, Field(description="Remote dir to sync into (optional).")] = None,
    sync_exclude_globs_csv: Annotated[str, Field(description="Exclude globs CSV for sync.")] = DEFAULT_SYNC_EXCLUDES,
    sync_clean_remote: Annotated[bool, Field(description="Delete remote dir before sync.")] = True,
    bootstrap_commands: Annotated[list[str] | None, Field(description="Bootstrap commands (optional).")] = None,
    # Jupyter run mode
    command: Annotated[str | None, Field(description="Explicit shell command (optional).")] = None,
    script_path: Annotated[str | None, Field(description="Local python script to upload+run (optional).")] = None,
    remote_script_path: Annotated[str | None, Field(description="Remote path to upload script into (optional).")] = None,
    log_path: Annotated[str | None, Field(description="Remote log path (jupyter).")] = None,
    artifacts: Annotated[list[Any] | None, Field(description="Artifacts to collect later.")] = None,
) -> dict[str, Any]:
    """
    Modal-like versatile run:
    - backend=modal => starts `modal run <target> ...` locally and logs to file
    - backend=jupyter => sync/bootstrap (optional), upload script or run command, start background job with log
    """
    backend = (backend or "modal").strip().lower()
    args = list(args or [])

    if backend == "modal":
        if not target:
            raise ValueError("target is required for backend=modal")
        ensure_dir(modal_log_dir)
        r = modal_run(target=target, extra_args=args, log_dir=modal_log_dir, log_path=modal_log_path)
        record = {
            "versa_job_id": r.versa_job_id,
            "backend": "modal",
            "pid": r.pid,
            "log_path": r.log_path,
            "command": r.command,
        }
        _store(r.versa_job_id, record)
        return record

    if backend != "jupyter":
        raise ValueError("backend must be 'modal' or 'jupyter'")

    if not url:
        raise ValueError("url is required for backend=jupyter")

    server_url = normalize_server_url(url)
    env_overrides = env_overrides or {}
    artifacts = artifacts or []

    owns_kernel = False
    if kernel_id is None:
        import requests

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.post(f"{server_url}/api/kernels", json={"name": "python3"}, headers=headers, timeout=30)
        r.raise_for_status()
        kernel_id = (r.json() or {}).get("id")
        if not kernel_id:
            raise RuntimeError("Failed to create remote kernel")
        owns_kernel = True

    # Sync (optional)
    if sync_local_dir and sync_remote_dir:
        await sync_dir_async(
            server_url=server_url,
            token=token,
            local_dir=sync_local_dir,
            remote_dir=sync_remote_dir,
            exclude_globs_csv=sync_exclude_globs_csv,
            clean_remote=sync_clean_remote,
            keep_archive=False,
            max_bytes=200 * 1024 * 1024,
            kernel_id=kernel_id,
            timeout_s=600.0,
        )

    # Bootstrap (optional)
    if bootstrap_commands:
        await run_commands_async(
            server_url=server_url,
            token=token,
            commands=list(bootstrap_commands),
            cwd=cwd,
            env_overrides=env_overrides,
            timeout_s_per_command=600.0,
            fail_fast=True,
            max_output_chars=20000,
            kernel_id=kernel_id,
        )

    # Decide command to run
    if command and script_path:
        raise ValueError("Specify either command or script_path, not both.")

    run_cmd: str
    modal_local_path: Path | None = None
    modal_func: str | None = None
    if target and "::" in target:
        p, f = target.split("::", 1)
        modal_local_path = Path(p)
        modal_func = f.strip() or None
    elif target:
        p = Path(target)
        if p.exists() and p.is_file() and p.suffix == ".py":
            modal_local_path = p

    if command:
        run_cmd = command
        if args:
            run_cmd += " " + " ".join(shlex.quote(a) for a in args)
    elif script_path:
        local = script_path
        # Ensure remote directory exists if script is nested
        if remote_script_path:
            remote = remote_script_path
        else:
            remote = os.path.join(cwd, os.path.basename(local)) if cwd else os.path.basename(local)
        await run_commands_async(
            server_url=server_url,
            token=token,
            commands=[f"mkdir -p {shlex.quote(os.path.dirname(remote) or '.')}"],
            cwd="",
            env_overrides={},
            timeout_s_per_command=60.0,
            fail_fast=True,
            max_output_chars=20000,
            kernel_id=kernel_id,
        )
        await upload_local_file_async(server_url=server_url, token=token, local_path=local, remote_path=remote)
        run_cmd = "python " + shlex.quote(remote) + (" " + " ".join(shlex.quote(a) for a in args) if args else "")
    else:
        if not target:
            raise ValueError("Either target, command, or script_path is required for backend=jupyter")
        is_modal_style = False
        if modal_local_path is not None and modal_local_path.exists() and modal_local_path.suffix == ".py":
            try:
                is_modal_style = "import modal" in modal_local_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                is_modal_style = False

        if is_modal_style and modal_local_path is not None:
            repo_root = Path(__file__).resolve().parents[1]
            try:
                rel_file = str(modal_local_path.resolve().relative_to(repo_root)).replace(os.sep, "/")
            except Exception as e:
                raise ValueError(f"Modal file must be inside repo root ({repo_root}): {e}") from e

            wedlm_dir = repo_root / "external" / "WeDLM"

            # Upload the Modal file itself (syncing all of external/TPU-dlm is typically too large).
            remote_parent = os.path.dirname(rel_file).rstrip("/")
            if remote_parent:
                await run_commands_async(
                    server_url=server_url,
                    token=token,
                    commands=[f"mkdir -p {shlex.quote(remote_parent)}"],
                    cwd="",
                    env_overrides={},
                    timeout_s_per_command=60.0,
                    fail_fast=True,
                    max_output_chars=20000,
                    kernel_id=kernel_id,
                )
            await upload_local_file_async(
                server_url=server_url,
                token=token,
                local_path=str(modal_local_path),
                remote_path=rel_file,
            )

            if wedlm_dir.exists():
                await sync_dir_async(
                    server_url=server_url,
                    token=token,
                    local_dir=str(wedlm_dir),
                    remote_dir="external/WeDLM",
                    kernel_id=kernel_id,
                    timeout_s=900.0,
                )

            # Ensure runner exists on remote.
            await run_commands_async(
                server_url=server_url,
                token=token,
                commands=["mkdir -p .versa"],
                cwd="",
                env_overrides={},
                timeout_s_per_command=60.0,
                fail_fast=True,
                max_output_chars=20000,
                kernel_id=kernel_id,
            )
            with tempfile.NamedTemporaryFile("w", prefix="versa-", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(RUNNER_SOURCE)
                runner_path = f.name
            try:
                await upload_local_file_async(
                    server_url=server_url,
                    token=token,
                    local_path=runner_path,
                    remote_path=".versa/modal_run.py",
                )
            finally:
                try:
                    os.unlink(runner_path)
                except Exception:
                    pass

            # Best-effort: make WeDLM importable (optional).
            await run_commands_async(
                server_url=server_url,
                token=token,
                commands=["python -m pip install -e external/WeDLM --no-deps || true"],
                cwd="",
                env_overrides=env_overrides,
                timeout_s_per_command=1800.0,
                fail_fast=False,
                max_output_chars=20000,
                kernel_id=kernel_id,
            )

            run_cmd = (
                "python -u .versa/modal_run.py "
                + shlex.quote(f"--file={rel_file}")
                + (" " + shlex.quote(f"--func={modal_func}") if modal_func else "")
                + (" -- " + " ".join(shlex.quote(a) for a in args) if args else "")
            )
        else:
            run_cmd = target
            if args:
                run_cmd += " " + " ".join(shlex.quote(a) for a in args)

    job = await job_start_async(
        server_url=server_url,
        token=token,
        command=run_cmd,
        cwd=cwd,
        env_overrides=env_overrides,
        log_path=log_path,
        kernel_id=kernel_id,
        timeout_s=60.0,
    )

    versa_job_id = job.job_id
    record = {
        "versa_job_id": versa_job_id,
        "backend": "jupyter",
        "pid": job.pid,
        "start_time_ticks": job.start_time_ticks,
        "log_path": job.log_path,
        "server_url": server_url,
        "kernel_id": kernel_id,
        "cwd": cwd,
        "command": run_cmd,
        "env_overrides": env_overrides,
        "artifacts": artifacts,
        "token": token,  # stored but redacted on read
        "owns_kernel": owns_kernel,
    }
    _store(versa_job_id, record)
    return _mask_secrets(record)


@mcp.tool()
async def versa_status(
    versa_job_id: Annotated[str, Field(description="Versa job id returned by versa_run.")],
) -> dict[str, Any]:
    rec = _get(versa_job_id)
    backend: Backend = rec["backend"]
    if backend == "modal":
        pid = int(rec["pid"])
        return {"versa_job_id": versa_job_id, "backend": "modal", "pid": pid, "alive": os.path.exists(f"/proc/{pid}")}

    token = rec.get("token", "")
    st = await job_status_async(
        server_url=rec["server_url"],
        token=token,
        pid=int(rec["pid"]),
        start_time_ticks=(int(rec["start_time_ticks"]) if rec.get("start_time_ticks") is not None else None),
        kernel_id=rec.get("kernel_id"),
    )
    return {"versa_job_id": versa_job_id, "backend": "jupyter", **st}


@mcp.tool()
async def versa_tail(
    versa_job_id: Annotated[str, Field(description="Versa job id returned by versa_run.")],
    lines: Annotated[int, Field(description="Lines from end of log.")] = 200,
) -> str:
    rec = _get(versa_job_id)
    backend: Backend = rec["backend"]
    if backend == "modal":
        log_path = rec["log_path"]
        out = subprocess.check_output(["tail", "-n", str(int(lines)), log_path], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")

    token = rec.get("token", "")
    return await job_tail_async(
        server_url=rec["server_url"],
        token=token,
        log_path=rec["log_path"],
        lines=int(lines),
        kernel_id=rec.get("kernel_id"),
    )


@mcp.tool()
async def versa_kill(
    versa_job_id: Annotated[str, Field(description="Versa job id returned by versa_run.")],
    sig: Annotated[int, Field(description="Signal number (15=TERM,9=KILL).")] = 15,
) -> dict[str, Any]:
    rec = _get(versa_job_id)
    backend: Backend = rec["backend"]
    if backend == "modal":
        modal_kill(int(rec["pid"]), sig=int(sig))
        return {"versa_job_id": versa_job_id, "backend": "modal", "pid": int(rec["pid"]), "sig": int(sig)}

    token = rec.get("token", "")
    reply = await job_kill_async(
        server_url=rec["server_url"],
        token=token,
        pid=int(rec["pid"]),
        sig=int(sig),
        kernel_id=rec.get("kernel_id"),
    )
    if rec.get("owns_kernel") and rec.get("kernel_id"):
        import requests

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            requests.delete(
                f"{rec['server_url']}/api/kernels/{rec['kernel_id']}",
                headers=headers,
                timeout=30,
            )
        except Exception:
            pass
    return {"versa_job_id": versa_job_id, "backend": "jupyter", **reply}


@mcp.tool()
async def versa_collect(
    versa_job_id: Annotated[str, Field(description="Versa job id returned by versa_run.")],
    local_dir: Annotated[str, Field(description="Local directory to download artifacts into.")] = "versa_artifacts",
    require_done: Annotated[bool, Field(description="Require job to be done before collecting.")] = True,
) -> dict[str, Any]:
    rec = _get(versa_job_id)
    backend: Backend = rec["backend"]
    ensure_dir(local_dir)

    if backend == "modal":
        # Modal runs already write logs locally; treat log as the artifact.
        return {"versa_job_id": versa_job_id, "backend": "modal", "log_path": rec["log_path"], "local_dir": local_dir}

    token = rec.get("token", "")
    st = await job_status_async(
        server_url=rec["server_url"],
        token=token,
        pid=int(rec["pid"]),
        start_time_ticks=(int(rec["start_time_ticks"]) if rec.get("start_time_ticks") is not None else None),
        kernel_id=rec.get("kernel_id"),
    )
    if require_done and st.get("alive"):
        raise RuntimeError("Job still alive; set require_done=false to collect anyway.")

    downloads: list[dict[str, Any]] = []
    for a in rec.get("artifacts") or []:
        if isinstance(a, str):
            remote_path = a
            local_path = os.path.join(local_dir, os.path.basename(remote_path.rstrip("/")))
            downloads.append(
                await download_to_local_async(
                    server_url=rec["server_url"],
                    token=token,
                    remote_path=remote_path,
                    local_path=local_path,
                )
            )
        elif isinstance(a, dict) and "remote_dir" in a:
            downloads.append(
                await download_dir_to_local_async(
                    server_url=rec["server_url"],
                    token=token,
                    remote_dir=str(a["remote_dir"]),
                    local_dir=os.path.join(local_dir, str(a.get("local_subdir", "dir"))),
                    archive_remote_path=a.get("archive_remote_path"),
                    kernel_id=rec.get("kernel_id"),
                )
            )
        elif isinstance(a, dict) and "remote_path" in a:
            remote_path = str(a["remote_path"])
            local_path = str(a.get("local_path") or os.path.join(local_dir, os.path.basename(remote_path)))
            downloads.append(
                await download_to_local_async(
                    server_url=rec["server_url"], token=token, remote_path=remote_path, local_path=local_path
                )
            )
        else:
            raise TypeError("artifact must be string, {remote_path,...}, or {remote_dir,...}")

    return {"versa_job_id": versa_job_id, "backend": "jupyter", "status": st, "downloads": downloads}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Versa MCP server (Streamable HTTP).")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8010")))
    args = parser.parse_args(argv)

    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
