#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name) or default).strip()


def _versa_base() -> str:
    return _env("VERSA_API_BASE", "https://versa.iseki.cloud").rstrip("/")


def _admin_key() -> str:
    return _env("VERSA_ADMIN_KEY") or _env("DEEPGHS_ADMIN_API_KEY")


def _headers() -> dict[str, str]:
    k = _admin_key()
    if not k:
        raise RuntimeError("Missing DEEPGHS_ADMIN_API_KEY (or VERSA_ADMIN_KEY)")
    return {"Authorization": f"Bearer {k}", "Accept": "application/json"}


def _safe_get_account_id(obj: dict[str, Any]) -> str | None:
    t = obj.get("tokens")
    if isinstance(t, dict):
        v = t.get("account_id")
        if v:
            return str(v)
    return None


def main() -> int:
    auth_path = Path(_env("CODEX_AUTH_JSON", str(Path.home() / ".codex" / "auth.json"))).expanduser()
    interval_s = float(_env("KODO_WATCH_INTERVAL_S", "2.0"))

    base = _versa_base()
    headers = _headers()

    last_mtime_ns: int | None = None
    last_account_id: str | None = None

    print(f"[kodo-watch] watching: {auth_path}")
    print(f"[kodo-watch] versa: {base}")

    while True:
        try:
            st = auth_path.stat()
        except FileNotFoundError:
            time.sleep(interval_s)
            continue

        if last_mtime_ns is not None and st.st_mtime_ns == last_mtime_ns:
            time.sleep(interval_s)
            continue

        try:
            raw = auth_path.read_text(encoding="utf-8")
            obj = json.loads(raw)
        except Exception:
            time.sleep(interval_s)
            continue

        last_mtime_ns = st.st_mtime_ns
        account_id = _safe_get_account_id(obj)
        if not account_id:
            time.sleep(interval_s)
            continue

        if account_id != last_account_id:
            print(f"[kodo-watch] account_id changed: {account_id}")
            last_account_id = account_id

        # Upsert into Versa SSOT (encrypted in R2). Never prints secrets.
        payload = {
            "max_concurrency_default": 1,
            "tokens": [
                {
                    "account_id": account_id,
                    "label": account_id,
                    "secret": obj,
                    "max_concurrency": 1,
                }
            ],
        }

        try:
            r = requests.post(
                f"{base}/admin/codex/tokens/import",
                headers={**headers, "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            r.raise_for_status()
            print("[kodo-watch] synced")
        except Exception as e:
            print(f"[kodo-watch] sync failed: {e}")

        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())

