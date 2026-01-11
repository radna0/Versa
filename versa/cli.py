#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import typer

from versa.modal_file_runner import RUNNER_SOURCE
from versa.remote_jupyter_backend import (
    job_status_async,
    normalize_server_url,
    read_text_chunk_async,
    run_commands_async,
    sync_dir_async,
    upload_local_file_async,
)
from versa.versa_runner import versa_run_jupyter, versa_run_modal


app = typer.Typer(
    add_completion=False,
    help="Versa: Modal-like versatile runner.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Versa CLI."""
    return None


def _parse_env_kv(items: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid env override {item!r}; expected KEY=VALUE")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def _json_out(obj: Any) -> None:
    def _redact(v: Any) -> Any:
        if isinstance(v, dict):
            out: dict[str, Any] = {}
            for k, vv in v.items():
                if k in {"server_url", "url"} and isinstance(vv, str) and vv:
                    try:
                        p = urlsplit(vv)
                        out[k] = f"{p.scheme}://{p.netloc}/<redacted>"
                    except Exception:
                        out[k] = "<redacted>"
                else:
                    out[k] = _redact(vv)
            return out
        if isinstance(v, list):
            return [_redact(x) for x in v]
        return v

    typer.echo(json.dumps(_redact(obj), indent=2, sort_keys=True))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    target: str = typer.Argument(..., help="Modal target or local script/command."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Backend: auto|modal|jupyter (auto selects jupyter when --url is set).",
    ),
    # Jupyter backend options
    url: str = typer.Option("", "--url", help="Remote Jupyter base URL (e.g. https://.../proxy)."),
    token: str = typer.Option("", "--token", help="Remote Jupyter token (optional)."),
    kernel_id: str = typer.Option("", "--kernel-id", help="Existing kernel id (optional)."),
    cwd: str = typer.Option("", "--cwd", help="Remote working directory for the job (jupyter)."),
    env: list[str] = typer.Option(None, "--env", help="Env override KEY=VALUE (repeatable)."),
    sync_local_dir: str = typer.Option("", "--sync-local-dir", help="Local directory to sync (jupyter)."),
    sync_remote_dir: str = typer.Option("", "--sync-remote-dir", help="Remote dir to sync into (jupyter)."),
    bootstrap_cmd: list[str] = typer.Option(None, "--bootstrap-cmd", help="Bootstrap command (repeatable)."),
    log_path: str = typer.Option("", "--log-path", help="Log file path (backend-specific)."),
    detach: bool = typer.Option(False, "--detach", help="Detach (do not stream logs)."),
    # Modal backend options
    log_dir: str = typer.Option("logs", "--log-dir", help="Local log directory (modal backend)."),
):
    """
    Modal-like runner:
    - modal backend: `python -m versa run modal/file.py::entry -- --args...`
    - jupyter backend: `python -m versa run --url ... <script.py|command> -- --args...`
    """
    backend = backend.strip().lower()
    extra = list(ctx.args)

    if backend == "auto":
        backend = "jupyter" if url else "modal"

    if backend == "modal":
        result = versa_run_modal(
            target=target,
            extra_args=extra,
            log_dir=log_dir,
            log_path=(log_path or None),
        )
        _json_out(result.__dict__)
        return

    if backend != "jupyter":
        raise typer.BadParameter("backend must be 'modal' or 'jupyter'")

    if not url:
        raise typer.BadParameter("--url is required for jupyter backend")

    jupyter_url = normalize_server_url(url)
    kernel = kernel_id.strip() or None
    env_overrides = _parse_env_kv(env)

    async def _run() -> None:
        remote_command: str
        local_path: Path | None = None
        func_name: str | None = None
        is_modal_style = False
        created_kernel_id: str | None = None

        if "::" in target:
            path_part, func_part = target.split("::", 1)
            local_path = Path(path_part)
            func_name = func_part.strip()
        else:
            p = Path(target)
            if p.exists() and p.is_file() and p.suffix == ".py":
                local_path = p

        # Create a reusable kernel (so sync/bootstrap/tail don't start new kernels repeatedly).
        if kernel is None:
            import requests

            r = requests.post(f"{jupyter_url}/api/kernels", json={"name": "python3"}, timeout=30)
            r.raise_for_status()
            created_kernel_id = r.json().get("id")
            if not created_kernel_id:
                raise RuntimeError("Failed to create remote kernel")
            nonlocal_kernel = created_kernel_id
        else:
            nonlocal_kernel = kernel

        # Detect a Modal file (imports `modal`) so we can emulate `modal run file.py` semantics.
        if local_path is not None and local_path.exists() and local_path.suffix == ".py":
            try:
                is_modal_style = "import modal" in local_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                is_modal_style = False

        if local_path is not None and is_modal_style:
            if not local_path.exists():
                raise typer.BadParameter(f"Local file not found: {str(local_path)!r}")

            repo_root = Path(__file__).resolve().parents[1]
            wedlm_dir = repo_root / "external" / "WeDLM"

            # Upload the Modal file itself (syncing all of external/TPU-dlm is typically too large).
            try:
                rel_file = str(local_path.resolve().relative_to(repo_root)).replace(os.sep, "/")
            except Exception as e:
                raise typer.BadParameter(f"Modal file must be inside repo root ({repo_root}): {e}") from e
            remote_parent = os.path.dirname(rel_file).rstrip("/")
            if remote_parent:
                await run_commands_async(
                    server_url=jupyter_url,
                    token=token,
                    commands=[f"mkdir -p {shlex.quote(remote_parent)}"],
                    cwd="",
                    env_overrides={},
                    timeout_s_per_command=60.0,
                    fail_fast=True,
                    max_output_chars=20000,
                    kernel_id=nonlocal_kernel,
                )
            await upload_local_file_async(
                server_url=jupyter_url,
                token=token,
                local_path=str(local_path),
                remote_path=rel_file,
            )

            if wedlm_dir.exists():
                await sync_dir_async(
                    server_url=jupyter_url,
                    token=token,
                    local_dir=str(wedlm_dir),
                    remote_dir="external/WeDLM",
                    kernel_id=nonlocal_kernel,
                    timeout_s=900.0,
                )

            await run_commands_async(
                server_url=jupyter_url,
                token=token,
                commands=["mkdir -p .versa"],
                cwd="",
                env_overrides={},
                timeout_s_per_command=60.0,
                fail_fast=True,
                max_output_chars=20000,
                kernel_id=nonlocal_kernel,
            )
            runner_tmp = _write_temp(RUNNER_SOURCE)
            try:
                await upload_local_file_async(
                    server_url=jupyter_url,
                    token=token,
                    local_path=runner_tmp,
                    remote_path=".versa/modal_run.py",
                )
            finally:
                try:
                    os.unlink(runner_tmp)
                except Exception:
                    pass

            # Best-effort: make WeDLM importable.
            await run_commands_async(
                server_url=jupyter_url,
                token=token,
                commands=["python -m pip install -e external/WeDLM --no-deps || true"],
                cwd="",
                env_overrides=env_overrides,
                timeout_s_per_command=1800.0,
                fail_fast=False,
                max_output_chars=20000,
                kernel_id=nonlocal_kernel,
            )

            remote_command = (
                "python -u .versa/modal_run.py "
                + shlex.quote(f"--file={rel_file}")
                + (" " + shlex.quote(f"--func={func_name}") if func_name else "")
                + (" -- " + " ".join(shlex.quote(a) for a in extra) if extra else "")
            )
        elif local_path is not None:
            if not local_path.exists():
                raise typer.BadParameter(f"Local file not found: {str(local_path)!r}")

            remote_name = local_path.name
            remote_path = f"{cwd.rstrip('/')}/{remote_name}" if cwd else remote_name
            if cwd:
                await run_commands_async(
                    server_url=jupyter_url,
                    token=token,
                    commands=[f"mkdir -p {shlex.quote(cwd)}"],
                    cwd="",
                    env_overrides={},
                    timeout_s_per_command=60.0,
                    fail_fast=True,
                    max_output_chars=20000,
                    kernel_id=nonlocal_kernel,
                )
            await upload_local_file_async(
                server_url=jupyter_url, token=token, local_path=str(local_path), remote_path=remote_path
            )

            if func_name:
                stub_name = f"._versa_entry_{os.getpid()}_{local_path.stem}.py"
                stub_code = (
                    "import runpy, sys\n"
                    f"ns = runpy.run_path({remote_path!r})\n"
                    f"fn = ns.get({func_name!r})\n"
                    "if fn is None:\n"
                    f"  raise SystemExit('Missing function: {func_name}')\n"
                    "ret = fn(*sys.argv[1:])\n"
                    "raise SystemExit(0 if ret is None else int(ret))\n"
                )
                temp_path = _write_temp(stub_code)
                try:
                    await upload_local_file_async(
                        server_url=jupyter_url,
                        token=token,
                        local_path=temp_path,
                        remote_path=(f"{cwd.rstrip('/')}/{stub_name}" if cwd else stub_name),
                    )
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                remote_stub = f"{cwd.rstrip('/')}/{stub_name}" if cwd else stub_name
                remote_command = "python " + shlex.quote(remote_stub) + (
                    " " + " ".join(shlex.quote(a) for a in extra) if extra else ""
                )
            else:
                remote_command = "python " + shlex.quote(remote_path) + (
                    " " + " ".join(shlex.quote(a) for a in extra) if extra else ""
                )
        else:
            remote_command = target + (" " + " ".join(shlex.quote(a) for a in extra) if extra else "")

        sync_spec = None
        if sync_local_dir and sync_remote_dir:
            sync_spec = {"local_dir": sync_local_dir, "remote_dir": sync_remote_dir, "clean_remote": True}

        bootstrap_spec = None
        if bootstrap_cmd:
            bootstrap_spec = {
                "cwd": cwd,
                "commands": list(bootstrap_cmd),
                "fail_fast": True,
                "env_overrides": env_overrides,
            }

        run_spec = {
            "cwd": cwd,
            "command": remote_command,
            "env_overrides": env_overrides,
            "log_path": (log_path or None),
        }

        job = await versa_run_jupyter(
            url=jupyter_url,
            token=token,
            kernel_id=nonlocal_kernel,
            sync=sync_spec,
            bootstrap=bootstrap_spec,
            run=run_spec,
        )
        _json_out(job.__dict__)

        if detach:
            if created_kernel_id:
                import requests

                try:
                    requests.delete(f"{jupyter_url}/api/kernels/{created_kernel_id}", timeout=30)
                except Exception:
                    pass
            return

        offset = 0
        while True:
            chunk = await read_text_chunk_async(
                server_url=jupyter_url,
                token=token,
                path=job.log_path,
                offset=offset,
                max_chars=65536,
                kernel_id=nonlocal_kernel,
                timeout_s=30.0,
            )
            text = str(chunk.get("text") or "")
            if text:
                typer.echo(text, nl=False)
            offset = int(chunk.get("next_offset") or offset)

            st = await job_status_async(
                server_url=jupyter_url,
                token=token,
                pid=int(job.details.get("pid") or 0),
                start_time_ticks=(
                    int(job.details.get("start_time_ticks"))
                    if job.details.get("start_time_ticks") is not None
                    else None
                ),
                kernel_id=nonlocal_kernel,
                timeout_s=30.0,
            )
            if not st.get("alive"):
                chunk = await read_text_chunk_async(
                    server_url=jupyter_url,
                    token=token,
                    path=job.log_path,
                    offset=offset,
                    max_chars=65536,
                    kernel_id=nonlocal_kernel,
                    timeout_s=30.0,
                )
                text = str(chunk.get("text") or "")
                if text:
                    typer.echo(text, nl=False)
                break
            await asyncio.sleep(2.0)

        if created_kernel_id:
            import requests

            try:
                requests.delete(f"{jupyter_url}/api/kernels/{created_kernel_id}", timeout=30)
            except Exception:
                pass

    asyncio.run(_run())


def _write_temp(text: str) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(prefix="versa-", suffix=".py")
    os.close(fd)
    Path(path).write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    app()
