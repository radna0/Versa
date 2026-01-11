#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import threading
import time
import tarfile
import tempfile
import uuid
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

import requests
from pydantic import Field

try:
    from jupyter_kernel_client import KernelClient
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: jupyter_kernel_client\n"
        "Install with: python -m pip install --user jupyter_kernel_client\n"
        f"Original error: {e}"
    )

from mcp.server import FastMCP


DEFAULT_TIMEOUT_S = 30.0
DEFAULT_JOB_LOG_DIR = "logs"
DEFAULT_SYNC_EXCLUDES = (
    ".git/**,"
    ".hg/**,"
    ".svn/**,"
    "**/__pycache__/**,"
    "**/*.pyc,"
    ".venv/**,"
    "venv/**,"
    "build/**,"
    "dist/**,"
    "*.egg-info/**,"
    ".mypy_cache/**,"
    ".pytest_cache/**,"
    ".ruff_cache/**"
)

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _normalize_server_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("Empty server URL")

    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"server_url must include scheme and host (got: {raw_url!r})")

    clean_path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, clean_path, "", ""))


def _default_server_url() -> str:
    url = os.environ.get("REMOTE_JUPYTER_URL", "").strip()
    if not url:
        raise RuntimeError("Set REMOTE_JUPYTER_URL (e.g. https://.../proxy)")
    return _normalize_server_url(url)


def _default_token() -> str:
    # Empty token is valid for some deployments (e.g. Kaggle proxy URLs).
    return os.environ.get("REMOTE_JUPYTER_TOKEN", "")


def _auth_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    # Matches jupyter-kernel-client's default HTTP auth scheme.
    return {"Authorization": f"Bearer {token}"}


def _requests_json(
    *,
    method: str,
    url: str,
    token: str,
    timeout_s: float,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    r = requests.request(
        method=method,
        url=url,
        headers=_auth_headers(token),
        json=json_body,
        timeout=timeout_s,
    )
    r.raise_for_status()
    if not r.content:
        return {}
    return r.json()


async def _requests_json_async(
    *,
    method: str,
    url: str,
    token: str,
    timeout_s: float,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    return await asyncio.to_thread(
        _requests_json,
        method=method,
        url=url,
        token=token,
        timeout_s=timeout_s,
        json_body=json_body,
    )


def _kernel_execute(
    *,
    server_url: str,
    token: str,
    kernel_id: str | None,
    code: str,
    timeout_s: float,
) -> dict[str, Any]:
    with KernelClient(server_url=server_url, token=token, kernel_id=kernel_id) as kernel:
        return kernel.execute(code, timeout=float(timeout_s))


async def _kernel_execute_async(
    *,
    server_url: str,
    token: str,
    kernel_id: str | None,
    code: str,
    timeout_s: float,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _kernel_execute,
        server_url=server_url,
        token=token,
        kernel_id=kernel_id,
        code=code,
        timeout_s=timeout_s,
    )


mcp = FastMCP(name="Remote Jupyter MCP (bridge)")


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _matches_any_glob(rel_posix_path: str, globs: list[str]) -> bool:
    import re

    for g in globs:
        gg = (g or "").strip()
        if not gg:
            continue
        gg = gg.replace("\\", "/").lstrip("/")
        # If the pattern has no path separators, treat it as matching any basename.
        if "/" not in gg:
            gg = f"**/{gg}"

        # Translate a small glob subset to regex:
        # - `**/` matches zero or more path segments (including empty at start)
        # - `**` matches any characters (including '/')
        # - `*` matches any characters except '/'
        # - `?` matches a single character except '/'
        # - character classes [] are passed through
        regex = ["^"]
        i = 0
        while i < len(gg):
            if gg.startswith("**/", i):
                regex.append("(?:.*/)?")
                i += 3
                continue
            if gg.startswith("**", i):
                regex.append(".*")
                i += 2
                continue
            c = gg[i]
            if c == "*":
                regex.append("[^/]*")
            elif c == "?":
                regex.append("[^/]")
            elif c == "[":
                j = gg.find("]", i + 1)
                if j == -1:
                    regex.append(re.escape(c))
                else:
                    regex.append(gg[i : j + 1])
                    i = j
            else:
                regex.append(re.escape(c))
            i += 1
        regex.append("$")

        try:
            if re.match("".join(regex), rel_posix_path.lstrip("/")):
                return True
        except re.error:
            continue
    return False


def _create_tar_gz(
    local_dir: str,
    exclude_globs: list[str],
    *,
    max_bytes: int,
) -> dict[str, Any]:
    base = os.path.abspath(local_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"local_dir is not a directory: {local_dir}")

    included = 0
    excluded = 0

    with tempfile.NamedTemporaryFile(prefix="remote-jupyter-sync-", suffix=".tar.gz", delete=False) as f:
        tmp_path = f.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tf:
            for root, dirs, files in os.walk(base):
                rel_root = os.path.relpath(root, base)
                rel_root_posix = "" if rel_root == "." else rel_root.replace(os.sep, "/")

                # Prune excluded directories early.
                pruned_dirs: list[str] = []
                for d in list(dirs):
                    rel_dir = f"{rel_root_posix}/{d}" if rel_root_posix else d
                    if _matches_any_glob(rel_dir, exclude_globs) or _matches_any_glob(
                        rel_dir + "/**", exclude_globs
                    ):
                        pruned_dirs.append(d)
                for d in pruned_dirs:
                    dirs.remove(d)
                    excluded += 1

                for name in files:
                    rel_file = f"{rel_root_posix}/{name}" if rel_root_posix else name
                    if _matches_any_glob(rel_file, exclude_globs):
                        excluded += 1
                        continue
                    full_path = os.path.join(root, name)
                    # Store relative path in archive.
                    tf.add(full_path, arcname=rel_file, recursive=False)
                    included += 1

        size = os.path.getsize(tmp_path)
        if size > max_bytes:
            raise ValueError(f"Sync tarball too large: {size} bytes > max_bytes={max_bytes}")
        with open(tmp_path, "rb") as rf:
            data = rf.read()
        return {"bytes": data, "size_bytes": size, "included": included, "excluded": excluded}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@mcp.tool()
async def jupyter_env_snapshot(
    kernel_id: Annotated[
        str | None,
        Field(description="Existing kernel id to attach to (optional)."),
    ] = None,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="Timeout seconds for the snapshot command."),
    ] = 120.0,
) -> dict[str, Any]:
    """Collect a reproducibility snapshot (GPU/CUDA/Torch/pip) from the remote environment."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    code = r"""
import json, os, platform, subprocess, sys

def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"cmd": cmd, "error": repr(e)}

snapshot = {
    "python": {
        "executable": sys.executable,
        "version": sys.version,
    },
    "platform": {
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
    },
    "cwd": os.getcwd(),
    "env": {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    },
    "commands": {
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvidia_smi_L": _run(["nvidia-smi", "-L"]),
        "nvcc_version": _run(["bash", "-lc", "command -v nvcc >/dev/null 2>&1 && nvcc --version || true"]),
        "pip_freeze": _run(["python", "-m", "pip", "freeze"]),
    },
}

try:
    import torch
    snapshot["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda": getattr(torch.version, "cuda", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
    }
except Exception as e:
    snapshot["torch_error"] = repr(e)

print(json.dumps(snapshot))
"""

    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )

    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") != "stream" or out.get("name") != "stdout":
            continue
        for line in str(out.get("text") or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = __import__("json").loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict) and "python" in parsed and "commands" in parsed:
                return parsed

    raise RuntimeError(f"Failed to parse env snapshot from kernel reply: {reply!r}")


@mcp.tool()
async def jupyter_sync_dir(
    local_dir: Annotated[str, Field(description="Local directory to sync (on MCP server machine).")],
    remote_dir: Annotated[str, Field(description="Remote destination directory (Jupyter root-relative).")],
    exclude_globs_csv: Annotated[
        str,
        Field(description="Comma-separated glob patterns to exclude.", default=DEFAULT_SYNC_EXCLUDES),
    ] = DEFAULT_SYNC_EXCLUDES,
    clean_remote: Annotated[
        bool,
        Field(description="If true, delete remote_dir before extracting (more deterministic)."),
    ] = True,
    keep_archive: Annotated[
        bool,
        Field(description="If true, keep the uploaded tar.gz after extraction."),
    ] = False,
    max_bytes: Annotated[
        int,
        Field(description="Maximum allowed tar.gz size in bytes (upload safety)."),
    ] = 200 * 1024 * 1024,
    kernel_id: Annotated[
        str | None,
        Field(description="Kernel id to use for extraction (optional)."),
    ] = None,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="Timeout seconds for upload/extract operations."),
    ] = 600.0,
) -> dict[str, Any]:
    """Sync a local directory to the remote Jupyter filesystem (Modal-like add_local_dir(copy=True))."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    exclude_globs = _split_csv(exclude_globs_csv)
    archive = _create_tar_gz(local_dir, exclude_globs, max_bytes=int(max_bytes))
    archive_b64 = base64.b64encode(archive["bytes"]).decode("ascii")

    sync_id = str(uuid.uuid4())
    remote_archive = f".agent_sync/{sync_id}.tar.gz"
    safe_archive = remote_archive.lstrip("/")
    safe_remote_dir = remote_dir.lstrip("/")

    upload = await _requests_json_async(
        method="PUT",
        url=f"{base}/api/contents/{safe_archive}",
        token=token,
        timeout_s=timeout_s,
        json_body={"type": "file", "format": "base64", "content": archive_b64},
    )

    extract_code = f"""
import os, shutil, tarfile, json
from pathlib import Path

archive_path = {remote_archive!r}
dest_dir = {safe_remote_dir!r}
clean_remote = {bool(clean_remote)!r}
keep_archive = {bool(keep_archive)!r}

dest = Path(dest_dir)
if clean_remote and dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)

def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory) + os.sep) or target == directory
    except Exception:
        return False

def safe_extract(tf: tarfile.TarFile, path: Path) -> None:
    for member in tf.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f\"Unsafe path in tar: {{member.name}}\")  # nosec
    tf.extractall(path)

with tarfile.open(archive_path, 'r:gz') as tf:
    safe_extract(tf, dest)

if not keep_archive:
    try:
        os.remove(archive_path)
    except Exception:
        pass

print(json.dumps({{'dest_dir': str(dest), 'archive_path': archive_path, 'keep_archive': keep_archive}}))
"""

    extract_reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=extract_code, timeout_s=timeout_s
    )

    return {
        "server_url": base,
        "remote_dir": safe_remote_dir,
        "remote_archive": remote_archive,
        "upload": upload,
        "extract_reply": extract_reply,
        "local_stats": {
            "tar_size_bytes": archive["size_bytes"],
            "included": archive["included"],
            "excluded": archive["excluded"],
            "exclude_globs": exclude_globs,
        },
    }


@mcp.tool()
async def jupyter_ping(
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="HTTP timeout seconds."),
    ] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Check the remote Jupyter server is reachable."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    api = await _requests_json_async(method="GET", url=f"{base}/api", token=token, timeout_s=timeout_s)
    status = await _requests_json_async(
        method="GET", url=f"{base}/api/status", token=token, timeout_s=timeout_s
    )
    return {"base_url": base, "api": api, "status": status}


@mcp.tool()
async def jupyter_list_kernels(
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """List running kernels on the remote server."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    kernels = await _requests_json_async(
        method="GET", url=f"{base}/api/kernels", token=token, timeout_s=timeout_s
    )
    if not isinstance(kernels, list):
        raise TypeError("Expected /api/kernels to return a JSON list")
    return [k for k in kernels if isinstance(k, dict)]


@mcp.tool()
async def jupyter_list_sessions(
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """List active sessions on the remote server."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    sessions = await _requests_json_async(
        method="GET", url=f"{base}/api/sessions", token=token, timeout_s=timeout_s
    )
    if not isinstance(sessions, list):
        raise TypeError("Expected /api/sessions to return a JSON list")
    return [s for s in sessions if isinstance(s, dict)]


@mcp.tool()
async def jupyter_start_kernel(
    kernel_name: Annotated[str, Field(description="Kernel spec name (e.g. python3).")] = "python3",
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Start a new kernel via the Jupyter REST API."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    kernel = await _requests_json_async(
        method="POST",
        url=f"{base}/api/kernels",
        token=token,
        timeout_s=timeout_s,
        json_body={"name": kernel_name},
    )
    if not isinstance(kernel, dict):
        raise TypeError("Expected /api/kernels POST to return a JSON object")
    return kernel


@mcp.tool()
async def jupyter_shutdown_kernel(
    kernel_id: Annotated[str, Field(description="Kernel id to shut down.")],
    now: Annotated[bool, Field(description="If true, use HTTP DELETE immediately.")] = True,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Shut down a kernel."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    if now:
        # /api/kernels/{id} commonly returns 204 No Content.
        def _delete_now() -> None:
            r = requests.delete(
                f"{base}/api/kernels/{kernel_id}",
                headers=_auth_headers(token),
                timeout=timeout_s,
            )
            r.raise_for_status()

        await asyncio.to_thread(_delete_now)
        return {"kernel_id": kernel_id, "shutdown": True, "mode": "http_delete"}

    # Graceful shutdown via kernel message (then HTTP if needed).
    def _shutdown_graceful() -> None:
        kernel = KernelClient(server_url=base, token=token, kernel_id=kernel_id)
        kernel.start()
        kernel.stop(shutdown_kernel=True, shutdown_now=False, timeout=float(timeout_s))

    await asyncio.to_thread(_shutdown_graceful)
    return {"kernel_id": kernel_id, "shutdown": True, "mode": "graceful"}


@mcp.tool()
async def jupyter_exec_python(
    code: Annotated[str, Field(description="Python code to execute in the remote kernel.")],
    kernel_id: Annotated[
        str | None,
        Field(description="Existing kernel id to attach to (optional)."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="Execution timeout seconds (kernel-side)."),
    ] = 120.0,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
) -> dict[str, Any]:
    """Execute Python in a remote Jupyter kernel and return notebook-style outputs."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    return await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )


@mcp.tool()
async def jupyter_list_dir(
    path: Annotated[str, Field(description="Directory path in Jupyter contents ('' means root).")] = "",
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """List a directory via the Jupyter contents API."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    safe_path = path.lstrip("/")
    return await _requests_json_async(
        method="GET", url=f"{base}/api/contents/{safe_path}", token=token, timeout_s=timeout_s
    )


@mcp.tool()
async def jupyter_read_text(
    path: Annotated[str, Field(description="File path to read (served from /files/<path>).")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> str:
    """Read a remote text file."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    def _read() -> str:
        r = requests.get(
            f"{base}/files/{path.lstrip('/')}",
            headers=_auth_headers(token),
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.text

    return await asyncio.to_thread(_read)


@mcp.tool()
async def jupyter_write_text(
    path: Annotated[str, Field(description="Remote path to write via /api/contents/<path>.")],
    content: Annotated[str, Field(description="Text content to write.")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Write a remote text file via the Jupyter contents API."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    safe_path = path.lstrip("/")
    return await _requests_json_async(
        method="PUT",
        url=f"{base}/api/contents/{safe_path}",
        token=token,
        timeout_s=timeout_s,
        json_body={"type": "file", "format": "text", "content": content},
    )


@mcp.tool()
async def jupyter_upload_base64(
    path: Annotated[str, Field(description="Remote path to write via /api/contents/<path>.")],
    content_base64: Annotated[str, Field(description="Base64-encoded file content.")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = 60.0,
) -> dict[str, Any]:
    """Upload a (possibly binary) file via the Jupyter contents API."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()
    safe_path = path.lstrip("/")
    # Validate base64 early to give clearer errors.
    base64.b64decode(content_base64.encode("ascii"), validate=True)
    return await _requests_json_async(
        method="PUT",
        url=f"{base}/api/contents/{safe_path}",
        token=token,
        timeout_s=timeout_s,
        json_body={"type": "file", "format": "base64", "content": content_base64},
    )


@mcp.tool()
async def jupyter_upload_local_file(
    local_path: Annotated[str, Field(description="Local filesystem path (on the MCP server machine).")],
    remote_path: Annotated[str, Field(description="Remote path to write via /api/contents/<path>.")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = 120.0,
) -> dict[str, Any]:
    """Upload a local file to the remote Jupyter server."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    with open(local_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    safe_path = remote_path.lstrip("/")
    return await _requests_json_async(
        method="PUT",
        url=f"{base}/api/contents/{safe_path}",
        token=token,
        timeout_s=timeout_s,
        json_body={"type": "file", "format": "base64", "content": data},
    )


@mcp.tool()
async def jupyter_download_base64(
    remote_path: Annotated[str, Field(description="Remote file path to download (from /files/<path>).")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = 120.0,
) -> dict[str, Any]:
    """Download a remote file and return base64 content (for artifacts)."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    def _get() -> tuple[str, str]:
        r = requests.get(
            f"{base}/files/{remote_path.lstrip('/')}",
            headers=_auth_headers(token),
            timeout=timeout_s,
        )
        r.raise_for_status()
        return base64.b64encode(r.content).decode("ascii"), r.headers.get("content-type", "")

    content_b64, content_type = await asyncio.to_thread(_get)
    return {"remote_path": remote_path, "content_type": content_type, "content_base64": content_b64}


@mcp.tool()
async def jupyter_download_to_local(
    remote_path: Annotated[str, Field(description="Remote file path to download (from /files/<path>).")],
    local_path: Annotated[str, Field(description="Local filesystem path to write (on MCP server machine).")],
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="HTTP timeout seconds.")] = 120.0,
) -> dict[str, Any]:
    """Download a remote file to the local filesystem (artifact pull)."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    def _download() -> dict[str, Any]:
        r = requests.get(
            f"{base}/files/{remote_path.lstrip('/')}",
            headers=_auth_headers(token),
            timeout=timeout_s,
        )
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return {
            "remote_path": remote_path,
            "local_path": local_path,
            "bytes": len(r.content),
            "content_type": r.headers.get("content-type", ""),
        }

    return await asyncio.to_thread(_download)


@mcp.tool()
async def jupyter_download_dir_to_local(
    remote_dir: Annotated[str, Field(description="Remote directory path (Jupyter root-relative).")],
    local_dir: Annotated[str, Field(description="Local directory to extract into (on MCP server machine).")],
    archive_remote_path: Annotated[
        str | None,
        Field(description="Remote tar.gz path to create (default: .agent_artifacts/<uuid>.tar.gz)."),
    ] = None,
    kernel_id: Annotated[
        str | None,
        Field(description="Kernel id to use for packing (optional)."),
    ] = None,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
    timeout_s: Annotated[float, Field(description="Timeout seconds for packing/downloading.")] = 600.0,
) -> dict[str, Any]:
    """Pack a remote directory to a tar.gz under Jupyter root, download it, and extract locally."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    pack_id = str(uuid.uuid4())
    remote_archive = archive_remote_path or f".agent_artifacts/{pack_id}.tar.gz"
    safe_remote_dir = remote_dir.lstrip("/")
    safe_remote_archive = remote_archive.lstrip("/")

    pack_code = f"""
import os, tarfile, json
from pathlib import Path

src_dir = Path({safe_remote_dir!r})
dst_archive = Path({safe_remote_archive!r})
dst_archive.parent.mkdir(parents=True, exist_ok=True)

with tarfile.open(dst_archive, 'w:gz') as tf:
    tf.add(src_dir, arcname='.', recursive=True)

print(json.dumps({{'remote_dir': str(src_dir), 'remote_archive': str(dst_archive), 'size_bytes': dst_archive.stat().st_size}}))
"""

    pack_reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=pack_code, timeout_s=timeout_s
    )

    # Always download to a temp tarball locally, then extract.
    os.makedirs(local_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="remote-jupyter-artifact-", suffix=".tar.gz", delete=False) as f:
        local_archive = f.name

    try:
        download = await jupyter_download_to_local(
            remote_path=safe_remote_archive,
            local_path=local_archive,
            server_url=base,
            timeout_s=timeout_s,
        )
        with tarfile.open(local_archive, "r:gz") as tf:
            tf.extractall(path=local_dir)
    finally:
        try:
            os.unlink(local_archive)
        except Exception:
            pass

    return {
        "pack_reply": pack_reply,
        "download": download,
        "local_dir": local_dir,
        "remote_archive": safe_remote_archive,
    }


@mcp.tool()
async def jupyter_job_start(
    command: Annotated[str, Field(description="Shell command to run (remote).")],
    kernel_id: Annotated[
        str | None,
        Field(description="Kernel id to use for launching the job (optional)."),
    ] = None,
    cwd: Annotated[
        str,
        Field(description="Remote working directory (relative to Jupyter root unless absolute)."),
    ] = "",
    env_overrides: Annotated[
        dict[str, str] | None,
        Field(description="Environment variable overrides for the job."),
    ] = None,
    log_path: Annotated[
        str | None,
        Field(description="Remote log file path (default: logs/agent_<ts>_<id>.log)."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="Timeout for the *launch* RPC only (job runs in background)."),
    ] = 60.0,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
) -> dict[str, Any]:
    """Start a long-running remote shell job and stream logs via jupyter_job_tail."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    job_id = str(uuid.uuid4())
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    remote_log_path = log_path or f"{DEFAULT_JOB_LOG_DIR}/agent_{ts}_{job_id}.log"

    env_overrides = env_overrides or {}

    launch_code = f"""
import json, os, subprocess, time
from pathlib import Path

command = {command!r}
cwd = {cwd!r}
log_path = {remote_log_path!r}
env_overrides = {env_overrides!r}

Path(log_path).parent.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
for k, v in (env_overrides or {{}}).items():
    env[str(k)] = str(v)

log_f = open(log_path, 'a', buffering=1)
p = subprocess.Popen(
    command,
    shell=True,
    cwd=(cwd or None),
    env=env,
    stdout=log_f,
    stderr=log_f,
    start_new_session=True,
    text=True,
)
print(json.dumps({{'pid': int(p.pid), 'log_path': log_path, 'started_unix_s': time.time()}}))
"""

    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=launch_code, timeout_s=timeout_s
    )

    pid: int | None = None
    started_unix_s: float | None = None

    # Extract the JSON line from stdout.
    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") != "stream" or out.get("name") != "stdout":
            continue
        text = out.get("text") or ""
        for line in str(text).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = __import__("json").loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and "pid" in payload:
                pid = int(payload["pid"])
                started_unix_s = float(payload.get("started_unix_s") or 0.0)
                remote_log_path = str(payload.get("log_path") or remote_log_path)

    if pid is None:
        raise RuntimeError(f"Failed to start job; could not parse pid from kernel reply: {reply!r}")

    record = {
        "job_id": job_id,
        "pid": pid,
        "log_path": remote_log_path,
        "kernel_id": kernel_id,
        "server_url": base,
        "started_unix_s": started_unix_s,
        "command": command,
        "cwd": cwd,
        "env_overrides": env_overrides,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = record
    return record


@mcp.tool()
async def jupyter_run_commands(
    commands: Annotated[list[str], Field(description="Commands to run sequentially (shell).")],
    cwd: Annotated[
        str,
        Field(description="Remote working directory for commands."),
    ] = "",
    env_overrides: Annotated[
        dict[str, str] | None,
        Field(description="Environment variable overrides."),
    ] = None,
    timeout_s_per_command: Annotated[
        float,
        Field(description="Timeout seconds per command."),
    ] = 600.0,
    fail_fast: Annotated[
        bool,
        Field(description="Stop after first non-zero return code."),
    ] = True,
    max_output_chars: Annotated[
        int,
        Field(description="Max characters to return for stdout/stderr per command."),
    ] = 20000,
    kernel_id: Annotated[
        str | None,
        Field(description="Kernel id to use for running the commands (optional)."),
    ] = None,
    server_url: Annotated[
        str | None,
        Field(description="Remote Jupyter base URL (defaults to REMOTE_JUPYTER_URL)."),
    ] = None,
) -> dict[str, Any]:
    """Run shell commands sequentially (Modal-like run_commands) and return outputs."""
    base = _normalize_server_url(server_url) if server_url else _default_server_url()
    token = _default_token()

    env_overrides = env_overrides or {}
    code = f"""
import json, os, subprocess, time

commands = {list(commands)!r}
cwd = {cwd!r}
env_overrides = {env_overrides!r}
timeout_s = float({float(timeout_s_per_command)!r})
fail_fast = bool({bool(fail_fast)!r})
max_chars = int({int(max_output_chars)!r})

env = os.environ.copy()
for k, v in (env_overrides or {{}}).items():
    env[str(k)] = str(v)

results = []
for cmd in commands:
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            cwd=(cwd or None),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc = int(p.returncode)
        out = p.stdout or ''
        err = p.stderr or ''
        if len(out) > max_chars:
            out = out[:max_chars] + f\"\\n... [truncated to {max_chars} chars]\\n\"
        if len(err) > max_chars:
            err = err[:max_chars] + f\"\\n... [truncated to {max_chars} chars]\\n\"
        results.append({{
            'command': cmd,
            'returncode': rc,
            'stdout': out,
            'stderr': err,
            'duration_s': time.time() - t0,
        }})
        if fail_fast and rc != 0:
            break
    except subprocess.TimeoutExpired as e:
        results.append({{
            'command': cmd,
            'returncode': 124,
            'stdout': (e.stdout or '')[:max_chars],
            'stderr': (e.stderr or '')[:max_chars],
            'duration_s': time.time() - t0,
            'error': 'timeout',
        }})
        if fail_fast:
            break
    except Exception as e:
        results.append({{
            'command': cmd,
            'returncode': 1,
            'stdout': '',
            'stderr': '',
            'duration_s': time.time() - t0,
            'error': repr(e),
        }})
        if fail_fast:
            break

print(json.dumps({{'cwd': cwd, 'env_overrides': env_overrides, 'results': results}}))
"""

    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=float(timeout_s_per_command) + 30.0
    )

    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") != "stream" or out.get("name") != "stdout":
            continue
        for line in str(out.get("text") or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = __import__("json").loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed

    raise RuntimeError(f"Failed to parse run_commands output from kernel reply: {reply!r}")


@mcp.tool()
async def jupyter_job_status(
    job_id: Annotated[str, Field(description="Job id returned by jupyter_job_start.")],
    timeout_s: Annotated[
        float,
        Field(description="Timeout seconds for the status check."),
    ] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Check whether a started job pid is still alive."""
    with _JOBS_LOCK:
        record = dict(_JOBS.get(job_id) or {})
    if not record:
        raise KeyError(f"Unknown job_id: {job_id}")

    base = record["server_url"]
    token = _default_token()
    pid = int(record["pid"])
    kernel_id = record.get("kernel_id")

    code = f"""
import os, json
pid = {pid}
alive = os.path.exists(f"/proc/{{pid}}")
print(json.dumps({{'pid': pid, 'alive': bool(alive)}}))
"""
    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )

    alive: bool | None = None
    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") != "stream" or out.get("name") != "stdout":
            continue
        for line in str(out.get("text") or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = __import__("json").loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("pid") == pid:
                alive = bool(payload.get("alive"))

    return {**record, "alive": alive, "status_reply": reply}


@mcp.tool()
async def jupyter_job_tail(
    job_id: Annotated[str, Field(description="Job id returned by jupyter_job_start.")],
    lines: Annotated[int, Field(description="Number of lines from the end of the log file.")] = 200,
    timeout_s: Annotated[float, Field(description="Timeout seconds for the tail command.")] = 30.0,
) -> str:
    """Return the last N lines of a job log file (Modal-like `tail -f`)."""
    with _JOBS_LOCK:
        record = dict(_JOBS.get(job_id) or {})
    if not record:
        raise KeyError(f"Unknown job_id: {job_id}")

    base = record["server_url"]
    token = _default_token()
    kernel_id = record.get("kernel_id")
    log_path = record["log_path"]

    code = f"""
import subprocess
out = subprocess.check_output(['tail','-n',str({int(lines)}),{log_path!r}], stderr=subprocess.STDOUT)
print(out.decode('utf-8', errors='replace'))
"""
    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )

    chunks: list[str] = []
    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            chunks.append(str(out.get("text") or ""))
    return "".join(chunks)


@mcp.tool()
async def jupyter_job_kill(
    job_id: Annotated[str, Field(description="Job id returned by jupyter_job_start.")],
    sig: Annotated[int, Field(description="Signal number (e.g. 15=TERM, 9=KILL).")] = 15,
    timeout_s: Annotated[float, Field(description="Timeout seconds for the kill command.")] = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Send a signal to a job's process group."""
    with _JOBS_LOCK:
        record = dict(_JOBS.get(job_id) or {})
    if not record:
        raise KeyError(f"Unknown job_id: {job_id}")

    base = record["server_url"]
    token = _default_token()
    kernel_id = record.get("kernel_id")
    pid = int(record["pid"])

    code = f"""
import os, signal, json
pid = {pid}
sig = {int(sig)}
try:
    os.killpg(pid, sig)
    ok = True
    mode = 'killpg'
except Exception:
    os.kill(pid, sig)
    ok = True
    mode = 'kill'
print(json.dumps({{'pid': pid, 'sig': sig, 'ok': ok, 'mode': mode}}))
"""
    reply = await _kernel_execute_async(
        server_url=base, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )
    return {**record, "kill_reply": reply}


@mcp.tool()
async def jupyter_job_submit(
    spec: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Declarative job spec (Modal-like): "
                "{sync:[...], bootstrap:{...}, run:{...}, artifacts:[...]}."
            )
        ),
    ],
) -> dict[str, Any]:
    """
    Submit a Modal-like job to a remote Jupyter server:
    - optional `sync` (local->remote)
    - optional `bootstrap` (run commands)
    - required `run` (background job + logs)
    """
    server_url = spec.get("server_url")
    kernel_id = spec.get("kernel_id")

    sync_ops = spec.get("sync") or []
    bootstrap = spec.get("bootstrap") or None
    run = spec.get("run") or None
    artifacts = spec.get("artifacts") or []

    if not isinstance(sync_ops, list):
        raise TypeError("spec.sync must be a list")
    if bootstrap is not None and not isinstance(bootstrap, dict):
        raise TypeError("spec.bootstrap must be an object")
    if not isinstance(run, dict):
        raise TypeError("spec.run is required and must be an object")

    # 1) Sync sources
    sync_results: list[dict[str, Any]] = []
    for op in sync_ops:
        if not isinstance(op, dict):
            raise TypeError("Each sync op must be an object")
        sync_results.append(
            await jupyter_sync_dir(
                local_dir=str(op["local_dir"]),
                remote_dir=str(op["remote_dir"]),
                exclude_globs_csv=str(op.get("exclude_globs_csv", DEFAULT_SYNC_EXCLUDES)),
                clean_remote=bool(op.get("clean_remote", True)),
                keep_archive=bool(op.get("keep_archive", False)),
                max_bytes=int(op.get("max_bytes", 200 * 1024 * 1024)),
                kernel_id=kernel_id,
                server_url=server_url,
                timeout_s=float(op.get("timeout_s", 600.0)),
            )
        )

    # 2) Bootstrap
    bootstrap_result: dict[str, Any] | None = None
    if bootstrap is not None:
        cmds = bootstrap.get("commands") or []
        if not isinstance(cmds, list) or not all(isinstance(c, str) for c in cmds):
            raise TypeError("bootstrap.commands must be a list[str]")
        bootstrap_result = await jupyter_run_commands(
            commands=cmds,
            cwd=str(bootstrap.get("cwd", "")),
            env_overrides=bootstrap.get("env_overrides") or None,
            timeout_s_per_command=float(bootstrap.get("timeout_s_per_command", 600.0)),
            fail_fast=bool(bootstrap.get("fail_fast", True)),
            max_output_chars=int(bootstrap.get("max_output_chars", 20000)),
            kernel_id=kernel_id,
            server_url=server_url,
        )

    # 3) Run (background job)
    cmd = run.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("run.command is required")
    job = await jupyter_job_start(
        command=cmd,
        kernel_id=kernel_id,
        cwd=str(run.get("cwd", "")),
        env_overrides=run.get("env_overrides") or None,
        log_path=run.get("log_path") or None,
        timeout_s=float(run.get("timeout_s", 60.0)),
        server_url=server_url,
    )

    # Persist artifacts in the in-memory job store for later collection.
    with _JOBS_LOCK:
        if job["job_id"] in _JOBS:
            _JOBS[job["job_id"]]["artifacts"] = artifacts

    return {
        "job": job,
        "sync": sync_results,
        "bootstrap": bootstrap_result,
        "artifacts": artifacts,
    }


@mcp.tool()
async def jupyter_job_collect(
    job_id: Annotated[str, Field(description="Job id returned by jupyter_job_start/job_submit.")],
    local_dir: Annotated[
        str,
        Field(description="Local directory to download artifacts into (on MCP server machine)."),
    ],
    require_done: Annotated[
        bool,
        Field(description="If true, require the job to be not alive before downloading."),
    ] = True,
    timeout_s: Annotated[float, Field(description="Timeout seconds per download.")] = 120.0,
) -> dict[str, Any]:
    """Download job artifacts to the local filesystem."""
    os.makedirs(local_dir, exist_ok=True)

    with _JOBS_LOCK:
        record = dict(_JOBS.get(job_id) or {})
    if not record:
        raise KeyError(f"Unknown job_id: {job_id}")

    status = await jupyter_job_status(job_id=job_id, timeout_s=DEFAULT_TIMEOUT_S)
    if require_done and status.get("alive"):
        raise RuntimeError("Job still alive; set require_done=false to collect anyway.")

    artifacts = record.get("artifacts") or []
    downloads: list[dict[str, Any]] = []

    for a in artifacts:
        if isinstance(a, str):
            remote_path = a
            local_path = os.path.join(local_dir, os.path.basename(remote_path.rstrip("/")))
            downloads.append(
                await jupyter_download_to_local(
                    remote_path=remote_path,
                    local_path=local_path,
                    server_url=record.get("server_url"),
                    timeout_s=timeout_s,
                )
            )
        elif isinstance(a, dict):
            if "remote_dir" in a:
                downloads.append(
                    await jupyter_download_dir_to_local(
                        remote_dir=str(a["remote_dir"]),
                        local_dir=os.path.join(local_dir, str(a.get("local_subdir", "artifact_dir"))),
                        archive_remote_path=a.get("archive_remote_path"),
                        kernel_id=record.get("kernel_id"),
                        server_url=record.get("server_url"),
                        timeout_s=float(a.get("timeout_s", timeout_s)),
                    )
                )
            else:
                remote_path = str(a["remote_path"])
                local_path = str(a.get("local_path") or os.path.join(local_dir, os.path.basename(remote_path)))
                downloads.append(
                    await jupyter_download_to_local(
                        remote_path=remote_path,
                        local_path=local_path,
                        server_url=record.get("server_url"),
                        timeout_s=float(a.get("timeout_s", timeout_s)),
                    )
                )
        else:
            raise TypeError("artifact must be a string path or an object")

    return {"job_id": job_id, "status": status, "downloads": downloads, "local_dir": local_dir}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Remote Jupyter MCP server (Streamable HTTP).")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8009")))
    args = parser.parse_args(argv)

    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
