from __future__ import annotations

import asyncio
import base64
import os
import re
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    from jupyter_kernel_client import KernelClient
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: jupyter_kernel_client. Install with:\n"
        "  python -m pip install --user jupyter_kernel_client\n"
        f"Original error: {e}"
    )


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


def normalize_server_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("Empty server URL")
    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"server_url must include scheme and host (got: {raw_url!r})")
    clean_path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, clean_path, "", ""))


def auth_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def redact_url(raw_url: str) -> str:
    try:
        p = urlsplit(raw_url)
        return urlunsplit((p.scheme, p.netloc, "/<redacted>", "", ""))
    except Exception:
        return "<redacted>"


async def requests_json_async(
    *,
    method: str,
    url: str,
    token: str,
    timeout_s: float,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    def _do() -> dict[str, Any] | list[Any]:
        r: requests.Response | None = None
        try:
            r = requests.request(
                method=method,
                url=url,
                headers=auth_headers(token),
                json=json_body,
                timeout=timeout_s,
            )
            r.raise_for_status()
            if not r.content:
                return {}
            return r.json()
        except requests.RequestException as e:
            status = getattr(r, "status_code", None)
            raise RuntimeError(f"HTTP request failed: {method} {redact_url(url)} status={status}") from e

    return await asyncio.to_thread(_do)


async def kernel_execute_async(
    *,
    server_url: str,
    token: str,
    kernel_id: str | None,
    code: str,
    timeout_s: float,
) -> dict[str, Any]:
    def _do() -> dict[str, Any]:
        with KernelClient(server_url=server_url, token=token, kernel_id=kernel_id) as kernel:
            return kernel.execute(code, timeout=float(timeout_s))

    return await asyncio.to_thread(_do)


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def matches_any_glob(rel_posix_path: str, globs: list[str]) -> bool:
    p = rel_posix_path.lstrip("/")
    for g in globs:
        gg = (g or "").strip()
        if not gg:
            continue
        gg = gg.replace("\\", "/").lstrip("/")
        if "/" not in gg:
            gg = f"**/{gg}"

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
            if re.match("".join(regex), p):
                return True
        except re.error:
            continue
    return False


def create_tar_gz(local_dir: str, exclude_globs: list[str], *, max_bytes: int) -> dict[str, Any]:
    base = os.path.abspath(local_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"local_dir is not a directory: {local_dir}")

    included = 0
    excluded = 0

    with tempfile.NamedTemporaryFile(prefix="versa-sync-", suffix=".tar.gz", delete=False) as f:
        tmp_path = f.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tf:
            for root, dirs, files in os.walk(base):
                rel_root = os.path.relpath(root, base)
                rel_root_posix = "" if rel_root == "." else rel_root.replace(os.sep, "/")

                pruned_dirs: list[str] = []
                for d in list(dirs):
                    rel_dir = f"{rel_root_posix}/{d}" if rel_root_posix else d
                    if matches_any_glob(rel_dir, exclude_globs) or matches_any_glob(
                        rel_dir + "/**", exclude_globs
                    ):
                        pruned_dirs.append(d)
                for d in pruned_dirs:
                    dirs.remove(d)
                    excluded += 1

                for name in files:
                    rel_file = f"{rel_root_posix}/{name}" if rel_root_posix else name
                    if matches_any_glob(rel_file, exclude_globs):
                        excluded += 1
                        continue
                    full_path = os.path.join(root, name)
                    tf.add(full_path, arcname=rel_file, recursive=False)
                    included += 1

        size = os.path.getsize(tmp_path)
        if size > max_bytes:
            raise ValueError(f"Sync tarball too large: {size} bytes > max_bytes={max_bytes}")
        with open(tmp_path, "rb") as f:
            data = f.read()
        return {"bytes": data, "size_bytes": size, "included": included, "excluded": excluded}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def upload_base64_file_async(
    *,
    server_url: str,
    token: str,
    remote_path: str,
    content_base64: str,
    timeout_s: float,
) -> dict[str, Any]:
    safe_path = remote_path.lstrip("/")
    base64.b64decode(content_base64.encode("ascii"), validate=True)
    resp = await requests_json_async(
        method="PUT",
        url=f"{server_url}/api/contents/{safe_path}",
        token=token,
        timeout_s=timeout_s,
        json_body={"type": "file", "format": "base64", "content": content_base64},
    )
    if not isinstance(resp, dict):
        raise TypeError("Expected upload response to be a JSON object")
    return resp


async def upload_local_file_async(
    *,
    server_url: str,
    token: str,
    local_path: str,
    remote_path: str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    return await upload_base64_file_async(
        server_url=server_url,
        token=token,
        remote_path=remote_path,
        content_base64=content_b64,
        timeout_s=timeout_s,
    )


async def sync_dir_async(
    *,
    server_url: str,
    token: str,
    local_dir: str,
    remote_dir: str,
    exclude_globs_csv: str = DEFAULT_SYNC_EXCLUDES,
    clean_remote: bool = True,
    keep_archive: bool = False,
    max_bytes: int = 200 * 1024 * 1024,
    kernel_id: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    remote_dir = remote_dir.lstrip("/")
    exclude_globs = split_csv(exclude_globs_csv)
    archive = create_tar_gz(local_dir, exclude_globs, max_bytes=int(max_bytes))
    archive_b64 = base64.b64encode(archive["bytes"]).decode("ascii")

    sync_id = str(uuid.uuid4())
    remote_archive = f".agent_sync/{sync_id}.tar.gz"
    try:
        upload = await upload_base64_file_async(
            server_url=server_url,
            token=token,
            remote_path=remote_archive,
            content_base64=archive_b64,
            timeout_s=timeout_s,
        )
    except Exception:
        if kernel_id is not None:
            return await sync_dir_kernel_async(
                server_url=server_url,
                token=token,
                kernel_id=kernel_id,
                local_archive_bytes=archive["bytes"],
                remote_archive=remote_archive,
                remote_dir=remote_dir,
                exclude_globs=exclude_globs,
                clean_remote=clean_remote,
                keep_archive=keep_archive,
                timeout_s=timeout_s,
                local_stats={
                    "tar_size_bytes": archive["size_bytes"],
                    "included": archive["included"],
                    "excluded": archive["excluded"],
                    "exclude_globs": exclude_globs,
                },
            )
        return await sync_dir_files_async(
            server_url=server_url,
            token=token,
            local_dir=local_dir,
            remote_dir=remote_dir,
            exclude_globs_csv=exclude_globs_csv,
            clean_remote=clean_remote,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
        )

    extract_code = f"""
import os, shutil, tarfile, json
from pathlib import Path

archive_path = {remote_archive!r}
dest_dir = {remote_dir!r}
clean_remote = {bool(clean_remote)!r}
keep_archive = {bool(keep_archive)!r}

dest = Path(dest_dir)
if clean_remote and dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)

def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory) + os.sep) or target == directory

def safe_extract(tf: tarfile.TarFile, path: Path) -> None:
    for member in tf.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Unsafe path in tar: {{member.name}}")
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

    extract_reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=extract_code, timeout_s=timeout_s
    )

    return {
        "remote_dir": remote_dir,
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


async def _upload_bytes_via_kernel_async(
    *,
    server_url: str,
    token: str,
    kernel_id: str,
    remote_path: str,
    data: bytes,
    timeout_s: float,
    chunk_chars: int = 1_000_000,  # must be divisible by 4 for base64
) -> dict[str, Any]:
    if chunk_chars % 4 != 0:
        raise ValueError("chunk_chars must be divisible by 4")

    remote_path = remote_path.lstrip("/")
    b64 = base64.b64encode(data).decode("ascii")

    init_code = f"""
from pathlib import Path
p = {remote_path!r}
Path(p).parent.mkdir(parents=True, exist_ok=True)
open(p, 'wb').close()
print('ok')
"""
    await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=init_code, timeout_s=float(timeout_s)
    )

    written = 0
    for i in range(0, len(b64), int(chunk_chars)):
        chunk = b64[i : i + int(chunk_chars)]
        code = f"""
import base64
p = {remote_path!r}
chunk = {chunk!r}
with open(p, 'ab') as f:
    f.write(base64.b64decode(chunk.encode('ascii')))
print(len(chunk))
"""
        await kernel_execute_async(
            server_url=server_url,
            token=token,
            kernel_id=kernel_id,
            code=code,
            timeout_s=float(timeout_s),
        )
        written += len(chunk)

    return {"remote_path": remote_path, "base64_chars": written}


async def sync_dir_kernel_async(
    *,
    server_url: str,
    token: str,
    kernel_id: str,
    local_archive_bytes: bytes,
    remote_archive: str,
    remote_dir: str,
    exclude_globs: list[str],
    clean_remote: bool,
    keep_archive: bool,
    timeout_s: float,
    local_stats: dict[str, Any],
) -> dict[str, Any]:
    upload = await _upload_bytes_via_kernel_async(
        server_url=server_url,
        token=token,
        kernel_id=kernel_id,
        remote_path=remote_archive,
        data=local_archive_bytes,
        timeout_s=float(timeout_s),
    )

    extract_code = f"""
import os, shutil, tarfile, json
from pathlib import Path

archive_path = {remote_archive!r}
dest_dir = {remote_dir!r}
clean_remote = {bool(clean_remote)!r}
keep_archive = {bool(keep_archive)!r}

dest = Path(dest_dir)
if clean_remote and dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)

def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory) + os.sep) or target == directory

def safe_extract(tf: tarfile.TarFile, path: Path) -> None:
    for member in tf.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Unsafe path in tar: {{member.name}}")
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
    extract_reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=extract_code, timeout_s=float(timeout_s)
    )

    return {
        "remote_dir": remote_dir,
        "remote_archive": remote_archive,
        "upload": upload,
        "extract_reply": extract_reply,
        "local_stats": local_stats,
        "mode": "kernel",
    }


async def sync_dir_files_async(
    *,
    server_url: str,
    token: str,
    local_dir: str,
    remote_dir: str,
    exclude_globs_csv: str = DEFAULT_SYNC_EXCLUDES,
    clean_remote: bool = True,
    max_bytes: int = 200 * 1024 * 1024,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    remote_dir = remote_dir.lstrip("/")
    exclude_globs = split_csv(exclude_globs_csv)
    base = os.path.abspath(local_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"local_dir is not a directory: {local_dir}")

    if clean_remote and remote_dir:
        try:
            await requests_json_async(
                method="DELETE",
                url=f"{server_url}/api/contents/{remote_dir}",
                token=token,
                timeout_s=float(timeout_s),
            )
        except Exception:
            pass

    ensured_dirs: set[str] = set()

    async def _ensure_remote_dir(path: str) -> None:
        path = path.strip("/").replace("//", "/")
        if not path:
            return
        parts = path.split("/")
        cur = ""
        for part in parts:
            cur = f"{cur}/{part}" if cur else part
            if cur in ensured_dirs:
                continue
            try:
                await requests_json_async(
                    method="PUT",
                    url=f"{server_url}/api/contents/{cur}",
                    token=token,
                    timeout_s=float(timeout_s),
                    json_body={"type": "directory"},
                )
            except Exception:
                pass
            ensured_dirs.add(cur)

    included = 0
    excluded = 0
    total_bytes = 0

    for root, _dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, base)
        rel_root_posix = "" if rel_root == "." else rel_root.replace(os.sep, "/")
        remote_root = f"{remote_dir}/{rel_root_posix}".strip("/") if remote_dir else rel_root_posix.strip("/")
        if remote_root:
            await _ensure_remote_dir(remote_root)

        for name in files:
            rel_file = f"{rel_root_posix}/{name}" if rel_root_posix else name
            if matches_any_glob(rel_file, exclude_globs):
                excluded += 1
                continue
            full_path = os.path.join(root, name)
            size = os.path.getsize(full_path)
            total_bytes += int(size)
            if total_bytes > int(max_bytes):
                raise ValueError(f"Sync too large: {total_bytes} bytes > max_bytes={max_bytes}")

            with open(full_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("ascii")

            remote_path = f"{remote_dir}/{rel_file}".strip("/") if remote_dir else rel_file
            await upload_base64_file_async(
                server_url=server_url,
                token=token,
                remote_path=remote_path,
                content_base64=content_b64,
                timeout_s=float(timeout_s),
            )
            included += 1

    return {
        "remote_dir": remote_dir,
        "mode": "files",
        "local_stats": {
            "included": included,
            "excluded": excluded,
            "total_bytes": total_bytes,
            "exclude_globs": exclude_globs,
        },
    }


async def run_commands_async(
    *,
    server_url: str,
    token: str,
    commands: list[str],
    cwd: str = "",
    env_overrides: dict[str, str] | None = None,
    timeout_s_per_command: float = 600.0,
    fail_fast: bool = True,
    max_output_chars: int = 20000,
    kernel_id: str | None = None,
) -> dict[str, Any]:
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
            out = out[:max_chars] + "\\n... [truncated to %d chars]\\n" % (max_chars,)
        if len(err) > max_chars:
            err = err[:max_chars] + "\\n... [truncated to %d chars]\\n" % (max_chars,)
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

    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=code, timeout_s=float(timeout_s_per_command) + 30.0
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


@dataclass(frozen=True)
class JupyterJob:
    job_id: str
    pid: int
    start_time_ticks: int | None
    log_path: str
    server_url: str
    kernel_id: str | None
    command: str
    cwd: str
    env_overrides: dict[str, str]


async def job_start_async(
    *,
    server_url: str,
    token: str,
    command: str,
    cwd: str = "",
    env_overrides: dict[str, str] | None = None,
    log_path: str | None = None,
    kernel_id: str | None = None,
    timeout_s: float = 60.0,
) -> JupyterJob:
    job_id = str(uuid.uuid4())
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    remote_log_path = log_path or f"{DEFAULT_JOB_LOG_DIR}/agent_{ts}_{job_id}.log"
    env_overrides = env_overrides or {}

    launch_code = f"""
import json, os, subprocess, time, threading
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
def _wait_and_reap(proc, lp):  # noqa: ANN001
    try:
        rc = proc.wait()
    except Exception:
        rc = None
    try:
        with open(lp, 'a', buffering=1) as f:
            f.write(f\"\\n[versa] exitcode={{rc}}\\n\")
    except Exception:
        pass
threading.Thread(target=_wait_and_reap, args=(p, log_path), daemon=True).start()
start_ticks = None
try:
    stat = open("/proc/%d/stat" % (p.pid,), "r").read().strip()
    # Field 2 is comm in parens and can contain spaces; starttime is field 22.
    rparen = stat.rfind(")")
    after = stat[rparen + 2 :] if rparen != -1 else ""
    fields = after.split()
    if len(fields) >= 20:
        start_ticks = int(fields[19])
except Exception:
    start_ticks = None

print(json.dumps({{'pid': int(p.pid), 'start_time_ticks': start_ticks, 'log_path': log_path, 'started_unix_s': time.time()}}))
"""

    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=launch_code, timeout_s=timeout_s
    )

    pid: int | None = None
    start_time_ticks: int | None = None
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
            if isinstance(payload, dict) and "pid" in payload:
                pid = int(payload["pid"])
                st = payload.get("start_time_ticks")
                start_time_ticks = int(st) if isinstance(st, int) else None
                remote_log_path = str(payload.get("log_path") or remote_log_path)

    if pid is None:
        raise RuntimeError(f"Failed to start job; could not parse pid from reply: {reply!r}")

    return JupyterJob(
        job_id=job_id,
        pid=pid,
        start_time_ticks=start_time_ticks,
        log_path=remote_log_path,
        server_url=server_url,
        kernel_id=kernel_id,
        command=command,
        cwd=cwd,
        env_overrides=env_overrides,
    )


async def job_tail_async(
    *,
    server_url: str,
    token: str,
    log_path: str,
    lines: int = 200,
    kernel_id: str | None = None,
    timeout_s: float = 30.0,
) -> str:
    code = f"""
import subprocess
out = subprocess.check_output(['tail','-n',str({int(lines)}),{log_path!r}], stderr=subprocess.STDOUT)
print(out.decode('utf-8', errors='replace'))
"""
    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )
    chunks: list[str] = []
    for out in reply.get("outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            chunks.append(str(out.get("text") or ""))
    return "".join(chunks)


async def read_text_chunk_async(
    *,
    server_url: str,
    token: str,
    path: str,
    offset: int,
    max_chars: int = 65536,
    kernel_id: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    Read a text file chunk from a remote path (relative to the remote FS),
    starting at a byte offset, returning decoded text and next offset.

    This is used to stream job logs in "foreground" mode.
    """
    code = f"""
import json
path = {path!r}
offset = int({int(offset)})
max_chars = int({int(max_chars)})
with open(path, 'rb') as f:
    f.seek(offset)
    data = f.read(max_chars)
text = data.decode('utf-8', errors='replace')
next_offset = offset + len(data)
print(json.dumps({{'offset': offset, 'next_offset': next_offset, 'text': text}}))
"""
    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
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
            if isinstance(parsed, dict) and "next_offset" in parsed:
                return parsed
    raise RuntimeError(f"Failed to parse read_text_chunk output: {reply!r}")


async def job_status_async(
    *,
    server_url: str,
    token: str,
    pid: int,
    start_time_ticks: int | None = None,
    kernel_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    code = f"""
import os, json
pid = {int(pid)}
expected = {start_time_ticks!r}
alive = os.path.exists(f"/proc/{{pid}}")
start_ticks = None
reused = False
zombie = False
if alive:
    try:
        stat = open(f"/proc/{pid}/stat", "r").read().strip()
        rparen = stat.rfind(")")
        after = stat[rparen + 2 :] if rparen != -1 else ""
        fields = after.split()
        if fields and fields[0] == "Z":
            zombie = True
            alive = False
        if len(fields) >= 20:
            start_ticks = int(fields[19])
    except Exception:
        start_ticks = None
if expected is not None and start_ticks is not None and int(expected) != int(start_ticks):
    reused = True
    alive = False
print(json.dumps({{'pid': pid, 'alive': bool(alive), 'start_time_ticks': start_ticks, 'reused': bool(reused), 'zombie': bool(zombie)}}))
"""
    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
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
            if isinstance(payload, dict) and int(payload.get("pid", -1)) == int(pid):
                alive = bool(payload.get("alive"))
    return {"pid": pid, "alive": alive, "status_reply": reply}


async def job_kill_async(
    *,
    server_url: str,
    token: str,
    pid: int,
    sig: int = 15,
    kernel_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    code = f"""
import os, signal, json
pid = {int(pid)}
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
    reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=code, timeout_s=timeout_s
    )
    return {"pid": pid, "sig": sig, "reply": reply}


async def download_to_local_async(
    *,
    server_url: str,
    token: str,
    remote_path: str,
    local_path: str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    def _download() -> dict[str, Any]:
        r = requests.get(
            f"{server_url}/files/{remote_path.lstrip('/')}",
            headers=auth_headers(token),
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


async def download_dir_to_local_async(
    *,
    server_url: str,
    token: str,
    remote_dir: str,
    local_dir: str,
    archive_remote_path: str | None = None,
    kernel_id: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    pack_id = str(uuid.uuid4())
    remote_archive = (archive_remote_path or f".agent_artifacts/{pack_id}.tar.gz").lstrip("/")
    remote_dir = remote_dir.lstrip("/")

    pack_code = f"""
import tarfile, json
from pathlib import Path
src_dir = Path({remote_dir!r})
dst_archive = Path({remote_archive!r})
dst_archive.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(dst_archive, 'w:gz') as tf:
    tf.add(src_dir, arcname='.', recursive=True)
print(json.dumps({{'remote_dir': str(src_dir), 'remote_archive': str(dst_archive), 'size_bytes': dst_archive.stat().st_size}}))
"""
    pack_reply = await kernel_execute_async(
        server_url=server_url, token=token, kernel_id=kernel_id, code=pack_code, timeout_s=timeout_s
    )

    os.makedirs(local_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="versa-artifact-", suffix=".tar.gz", delete=False) as f:
        local_archive = f.name
    try:
        download = await download_to_local_async(
            server_url=server_url, token=token, remote_path=remote_archive, local_path=local_archive, timeout_s=timeout_s
        )
        with tarfile.open(local_archive, "r:gz") as tf:
            tf.extractall(path=local_dir)
    finally:
        try:
            os.unlink(local_archive)
        except Exception:
            pass

    return {"pack_reply": pack_reply, "download": download, "local_dir": local_dir, "remote_archive": remote_archive}
