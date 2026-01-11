#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _normalize_server_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("Empty --url")

    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"--url must include scheme and host (got: {raw_url!r})")

    # Drop query/fragment and strip trailing slash for Jupyter REST paths.
    clean_path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, clean_path, "", ""))


def _require_deps() -> None:
    try:
        import requests  # noqa: F401
        from jupyter_kernel_client import KernelClient  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Install with:\n"
            "  python -m pip install --user jupyter_kernel_client requests\n"
            f"Original error: {e}"
        )


def _auth_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    # Matches jupyter-kernel-client's default HTTP auth scheme.
    return {"Authorization": f"Bearer {token}"}


def _json_print(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def cmd_list_kernels(server_url: str, token: str) -> int:
    import requests

    r = requests.get(
        f"{server_url}/api/kernels",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    _json_print(r.json())
    return 0


def cmd_list_sessions(server_url: str, token: str) -> int:
    import requests

    r = requests.get(
        f"{server_url}/api/sessions",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    _json_print(r.json())
    return 0


def cmd_exec(server_url: str, token: str, kernel_id: str | None, code: str) -> int:
    reply = _exec_code(server_url, token, kernel_id, code)
    _json_print(reply)
    return 0


def cmd_upload(server_url: str, token: str, local_path: Path, remote_path: str) -> int:
    response = _upload_file(server_url, token, local_path, remote_path)
    _json_print(response)
    return 0


def cmd_run_file(
    server_url: str,
    token: str,
    kernel_id: str | None,
    local_path: Path,
    remote_path: str,
) -> int:
    # Upload the script first, then run it as __main__.
    upload_response = _upload_file(server_url, token, local_path, remote_path)
    code = f"import runpy; _ = runpy.run_path({remote_path!r}, run_name='__main__')"
    exec_reply = _exec_code(server_url, token, kernel_id, code)
    _json_print({"upload": upload_response, "exec": exec_reply})
    return 0


def _exec_code(server_url: str, token: str, kernel_id: str | None, code: str) -> dict[str, Any]:
    from jupyter_kernel_client import KernelClient

    with KernelClient(server_url=server_url, token=token, kernel_id=kernel_id) as kernel:
        return kernel.execute(code)


def _upload_file(server_url: str, token: str, local_path: Path, remote_path: str) -> dict[str, Any]:
    import base64
    import requests

    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    r = requests.put(
        f"{server_url}/api/contents/{remote_path}",
        headers=_auth_headers(token),
        json={"type": "file", "format": "base64", "content": content_b64},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Remote Jupyter execution helper (works with Kaggle Jupyter proxy URLs)."
    )
    p.add_argument(
        "--url",
        required=True,
        help="Jupyter server base URL (e.g. https://.../proxy or http://localhost:8888).",
    )
    p.add_argument(
        "--token",
        default="",
        help="Jupyter token (Kaggle proxy usually works with empty token).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("kernels", help="List running kernels.")
    sub.add_parser("sessions", help="List active sessions.")

    exec_p = sub.add_parser("exec", help="Execute code in a kernel.")
    exec_p.add_argument("--kernel-id", default=None, help="Attach to an existing kernel id.")
    exec_src = exec_p.add_mutually_exclusive_group(required=True)
    exec_src.add_argument("--code", help="Code string to execute.")
    exec_src.add_argument("--file", type=Path, help="Local file to execute by sending its text.")

    up_p = sub.add_parser("upload", help="Upload a local file via Jupyter contents API.")
    up_p.add_argument("--local", type=Path, required=True, help="Local file path.")
    up_p.add_argument(
        "--remote",
        required=True,
        help="Remote path relative to the Jupyter contents root (e.g. foo.py).",
    )

    run_p = sub.add_parser("run-file", help="Upload then run a local Python file.")
    run_p.add_argument("--kernel-id", default=None, help="Attach to an existing kernel id.")
    run_p.add_argument("--local", type=Path, required=True, help="Local Python file path.")
    run_p.add_argument(
        "--remote",
        default=None,
        help="Remote path to upload to (default: basename of --local).",
    )

    return p


def main(argv: list[str]) -> int:
    _require_deps()
    args = build_parser().parse_args(argv)

    server_url = _normalize_server_url(args.url)
    token = args.token or ""

    if args.cmd == "kernels":
        return cmd_list_kernels(server_url, token)
    if args.cmd == "sessions":
        return cmd_list_sessions(server_url, token)
    if args.cmd == "exec":
        code = args.code if args.code is not None else args.file.read_text()
        return cmd_exec(server_url, token, args.kernel_id, code)
    if args.cmd == "upload":
        return cmd_upload(server_url, token, args.local, args.remote)
    if args.cmd == "run-file":
        remote_path = args.remote or args.local.name
        return cmd_run_file(server_url, token, args.kernel_id, args.local, remote_path)

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
