#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import typer

# Windows consoles often default to legacy code pages (e.g. cp1252) and can
# crash when printing Unicode (Modal emits "✓" and box-drawing characters).
if os.name == "nt":  # pragma: no cover
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Allow running from source checkout without installing the package:
#   python apps/versa-cli/versa/cli.py ...
if __package__ is None:  # pragma: no cover
    pkg_root = Path(__file__).resolve().parents[1]  # apps/versa-cli
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

from versa.modal_file_runner import RUNNER_SOURCE
from versa.modal_info import get_modal_gpu_catalog
from versa.remote_jupyter_backend import (
    job_status_async,
    normalize_server_url,
    read_text_chunk_async,
    run_commands_async,
    sync_dir_async,
    upload_local_file_async,
)
from versa.versa_runner import versa_run_jupyter, versa_run_modal
from versa.util import ensure_dir, utc_timestamp
from versa.config import (
    VersaLocalConfig,
    clear_local_config,
    config_dir,
    config_path_for_actor,
    list_actor_config_paths,
    read_local_config,
    read_local_config_from,
    set_active_actor,
    write_local_config,
    write_local_config_to,
)


app = typer.Typer(
    add_completion=False,
    help="Versa: Modal-like versatile runner.",
    no_args_is_help=True,
)

cloud_app = typer.Typer(add_completion=False, help="Iseki Cloud control plane (versa.iseki.cloud).")
app.add_typer(cloud_app, name="cloud")

modal_app = typer.Typer(add_completion=False, help="Local Modal utilities.")
app.add_typer(modal_app, name="modal")

# Cloud sub-groups (non-breaking; legacy flat commands remain available).
cloud_modal_app = typer.Typer(add_completion=False, help="Modal provider: tokens + runs.")
cloud_app.add_typer(cloud_modal_app, name="modal")

cloud_modal_profile_app = typer.Typer(add_completion=False, help="Modal profiles stored in Versa.")
cloud_modal_app.add_typer(cloud_modal_profile_app, name="profile")

cloud_kaggle_app = typer.Typer(add_completion=False, help="Kaggle provider: tokens + runs.")
cloud_app.add_typer(cloud_kaggle_app, name="kaggle")

cloud_kaggle_profile_app = typer.Typer(add_completion=False, help="Kaggle profiles (usernames) stored in Versa.")
cloud_kaggle_app.add_typer(cloud_kaggle_profile_app, name="profile")

cloud_codex_app = typer.Typer(add_completion=False, help="Codex provider: tokens + refresh.")
cloud_app.add_typer(cloud_codex_app, name="codex")

cloud_kodo_app = typer.Typer(add_completion=False, help="KODO (Codex lease coordinator).")
cloud_app.add_typer(cloud_kodo_app, name="kodo")

actor_app = typer.Typer(add_completion=False, help="Local actor identities (isolation across engineers/codebases).")
app.add_typer(actor_app, name="actor")


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
                lk = str(k).lower()
                if k in {"server_url", "url"} and isinstance(vv, str) and vv:
                    try:
                        p = urlsplit(vv)
                        out[k] = f"{p.scheme}://{p.netloc}/<redacted>"
                    except Exception:
                        out[k] = "<redacted>"
                elif lk in {"access_token", "refresh_token", "id_token", "token_secret", "api_key"}:
                    out[k] = "<redacted>"
                else:
                    out[k] = _redact(vv)
            return out
        if isinstance(v, list):
            return [_redact(x) for x in v]
        return v

    typer.echo(json.dumps(_redact(obj), indent=2, sort_keys=True))


def _compact_id(s: str, *, head: int = 6, tail: int = 4) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}...{s[-tail:]}"


def _print_table(headers: list[str], rows: list[list[str]], *, right_align: set[int] | None = None) -> None:
    """Very small, dependency-free table printer (ASCII, stable on Windows)."""
    right_align = right_align or set()
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_cell(i: int, v: str) -> str:
        v = str(v)
        w = widths[i]
        if i in right_align:
            return v.rjust(w)
        return v.ljust(w)

    typer.echo("  ".join(_fmt_cell(i, h) for i, h in enumerate(headers)))
    typer.echo("  ".join(("-" * w) for w in widths))
    for r in rows:
        typer.echo("  ".join(_fmt_cell(i, v) for i, v in enumerate(r)))


def _versa_base_url(base_url: str) -> str:
    cfg = read_local_config()
    b = (
        base_url
        or os.getenv("VERSA_API_BASE")
        or (cfg.api_base or "")
        or "https://versa.iseki.cloud"
    ).strip().rstrip("/")
    return b


def _versa_admin_key(admin_key: str) -> str:
    cfg = read_local_config()
    k = (
        admin_key
        or os.getenv("VERSA_ADMIN_KEY")
        or (cfg.admin_key or "")
        or os.getenv("DEEPGHS_ADMIN_API_KEY")
        or ""
    ).strip()
    if not k:
        raise typer.BadParameter("Missing admin key. Set VERSA_ADMIN_KEY or pass --admin-key.")
    return k


def _versa_headers(
    admin_key: str,
    *,
    admin_override: bool = False,
    actor_id: str = "",
    actor_name: str = "",
) -> dict[str, str]:
    h: dict[str, str] = {"Authorization": f"Bearer {admin_key}", "Accept": "application/json"}

    # Always send actor identity so the server can enforce owner-only kill/finish/log access.
    # This is also what prevents accidental cross-engineer collisions on shared machines.
    try:
        h["X-Versa-Actor-Id"] = _versa_actor_id(actor_id)
        h["X-Versa-Actor-Name"] = _versa_actor_name(actor_name)
    except Exception:
        # Some commands (e.g. `versa login`) run before config is initialized.
        pass

    if admin_override:
        h["X-Versa-Admin-Override"] = "1"

    return h


def _versa_actor_name(actor_name: str = "") -> str:
    cfg0 = read_local_config()
    n = (actor_name or os.getenv("VERSA_ACTOR_NAME") or (cfg0.actor_name or "")).strip()
    if not n:
        raise typer.BadParameter("Missing actor name. Run: versa login --actor-name <name> (or set VERSA_ACTOR_NAME).")
    return n


def _versa_actor_id(actor_id: str = "") -> str:
    """Stable per-user id used for lease ownership (prevents account collisions)."""
    cfg0 = read_local_config()
    _ = _versa_actor_name("")  # enforce explicit actor labeling on shared machines
    a = (actor_id or os.getenv("VERSA_ACTOR_ID") or (cfg0.actor_id or "")).strip()
    if a:
        return a

    # Self-heal: generate a persistent actor id if the user hasn't re-run `versa login`.
    a = f"va_{uuid.uuid4()}"
    try:
        write_local_config(
            VersaLocalConfig(api_base=cfg0.api_base, admin_key=cfg0.admin_key, actor_id=a, actor_name=cfg0.actor_name),
        )
    except Exception:
        pass
    return a


@actor_app.command("current")
def actor_current() -> None:
    """Show the currently active actor identity on this machine."""
    cfg = read_local_config()
    n = (cfg.actor_name or "").strip() or "(unset)"
    a = (cfg.actor_id or "").strip() or "(unset)"
    typer.echo(f"actor_name={n}")
    typer.echo(f"actor_id={a}")
    typer.echo(f"config_dir={config_dir()}")


@actor_app.command("list")
def actor_list() -> None:
    """List local actor configs under `~/.versa/actors/*.json`."""
    paths = list_actor_config_paths()
    active = ""
    try:
        p = config_dir() / "active_actor.txt"
        if p.exists():
            active = p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        active = ""

    if not paths:
        typer.echo("No actors yet. Create one with: versa login --actor-name <name>")
        return

    rows: list[list[str]] = []
    for p in paths:
        cfg = read_local_config_from(p)
        name = str(cfg.actor_name or p.stem)
        actor_id = str(cfg.actor_id or "")
        mark = "*" if active and active == name else ""
        rows.append([mark, name, _compact_id(actor_id), str(p)])

    _print_table(["", "actor_name", "actor_id", "config_path"], rows)
    if active:
        typer.echo(f"Active actor: {active}")


@actor_app.command("use")
def actor_use(
    actor_name: str = typer.Argument(..., help="Actor name to activate (e.g. alice)."),
    clone: bool = typer.Option(True, "--clone/--no-clone", help="If missing, create config by cloning admin_key/base_url from current actor."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing actor config when cloning."),
) -> None:
    """Switch the active actor (affects subsequent `versa cloud ...` commands)."""
    name = actor_name.strip()
    if not name:
        raise typer.BadParameter("missing_actor_name")

    p = config_path_for_actor(name)
    if (not p.exists()) and clone:
        cur = read_local_config()
        if not cur.admin_key or not cur.api_base:
            raise typer.BadParameter("missing_current_login (run: versa login --actor-name <name>)")
        if p.exists() and (not overwrite):
            raise typer.BadParameter("actor_exists (use --overwrite)")
        new_cfg = VersaLocalConfig(api_base=cur.api_base, admin_key=cur.admin_key, actor_id=f"va_{uuid.uuid4()}", actor_name=name)
        write_local_config_to(p, new_cfg)

    set_active_actor(name)
    typer.echo(f"OK: active_actor={name} ({p})")


@app.command("login")
def login(
    admin_key: str = typer.Argument("", help="Admin API key for Versa (stored locally)."),
    base_url: str = typer.Option("", "--base-url", help="Default Versa API base URL (stored locally)."),
    prompt: bool = typer.Option(False, "--prompt", help="Prompt for the key (avoids putting it in shell history)."),
    actor_name: str = typer.Option("", "--actor-name", help="Actor name for isolation on shared machines (e.g. alice)."),
    set_active: bool = typer.Option(True, "--set-active/--no-set-active", help="Make this actor the default on this machine."),
) -> None:
    """Store your Versa admin key + default API base URL in `~/.versa/config.json`."""
    cfg0 = read_local_config()
    k = str(admin_key or "").strip()
    if prompt or not k:
        k = typer.prompt("Versa admin key", hide_input=True).strip()
    if not k:
        raise typer.BadParameter("missing_admin_key")

    b = (base_url or cfg0.api_base or os.getenv("VERSA_API_BASE") or "https://versa.iseki.cloud").strip().rstrip("/")
    a = (cfg0.actor_id or os.getenv("VERSA_ACTOR_ID") or "").strip()
    if not a:
        a = f"va_{uuid.uuid4()}"

    n = (actor_name or os.getenv("VERSA_ACTOR_NAME") or (cfg0.actor_name or "")).strip()
    if not n:
        raise typer.BadParameter("missing_actor_name (use --actor-name)")

    cfg = VersaLocalConfig(api_base=b, admin_key=k, actor_id=a, actor_name=n)
    p = write_local_config(cfg)
    if set_active:
        try:
            ap = set_active_actor(n)
            typer.echo(f"OK: active_actor={n} ({ap})")
        except Exception:
            pass
    typer.echo(f"OK: saved config to {p} (actor_name={n} actor_id={a})")


@app.command("logout")
def logout() -> None:
    """Remove local `~/.versa/config.json` (does not delete server-side tokens)."""
    clear_local_config()
    typer.echo("OK: local config cleared")


@modal_app.command("gpus")
def modal_gpus(
    json_out: bool = typer.Option(True, "--json/--text", help="Print JSON (default) or a human list."),
) -> None:
    """List GPU types supported by the installed Modal SDK (local)."""
    cat = get_modal_gpu_catalog()
    if json_out:
        _json_out(cat)
        return

    typer.echo(f"modal_sdk_version={cat.get('modal_sdk_version')}")
    typer.echo("supported:")
    for t in (cat.get("supported") or []):
        typer.echo(f"  - {t}")
    missing = cat.get("missing_from_sdk") or []
    if missing:
        typer.echo("missing_from_sdk:")
        for t in missing:
            typer.echo(f"  - {t}")
    aliases = cat.get("aliases") or {}
    if aliases:
        typer.echo("aliases:")
        for k, v in aliases.items():
            typer.echo(f"  - {k} -> {v}")
    hint = cat.get("hint")
    if hint:
        typer.echo(f"hint: {hint}")


@cloud_app.command("modal-summary")
def cloud_modal_summary(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL (default: https://versa.iseki.cloud)."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Show Modal token summary from Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/modal/summary", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("codex-tokens")
def cloud_codex_tokens(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List Codex accounts stored in Versa SSOT (secrets never returned)."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/codex/tokens", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("codex-import")
def cloud_codex_import(
    codex_dir: str = typer.Option(
        str(Path.home() / ".codex" / "CODEX_ACCOUNTS"),
        "--dir",
        help="Directory containing Codex account JSONs (CODEX_ACCOUNTS).",
    ),
    max_concurrency: int = typer.Option(1, "--max-concurrency", help="Max concurrent leases per account."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Import local Codex account JSONs into Versa (D1+R2)."""
    import base64
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    p = Path(codex_dir).expanduser()
    if not p.exists() or not p.is_dir():
        raise typer.BadParameter(f"codex_dir not found: {p}")

    def _jwt_email(id_token: str) -> str | None:
        try:
            parts = id_token.split(".")
            if len(parts) < 2:
                return None
            s = parts[1].replace("-", "+").replace("_", "/")
            s += "=" * ((4 - (len(s) % 4)) % 4)
            payload = json.loads(base64.b64decode(s.encode("utf-8")).decode("utf-8"))
            e = payload.get("email")
            return str(e) if e else None
        except Exception:
            return None

    items: list[dict[str, Any]] = []
    for fp in sorted(p.glob("*.json")):
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            parsed = json.loads(raw)
            account_id = str((parsed.get("tokens") or {}).get("account_id") or "").strip()
            if not account_id:
                continue
            id_token = str((parsed.get("tokens") or {}).get("id_token") or "")
            email = _jwt_email(id_token) if id_token else None
            items.append(
                {
                    "account_id": account_id,
                    "email": email,
                    "label": email or account_id,
                    "max_concurrency": int(max(1, max_concurrency)),
                    "secret": parsed,
                }
            )
        except Exception:
            continue

    payload = {"max_concurrency_default": int(max(1, max_concurrency)), "tokens": items}
    r = requests.post(
        f"{b}/admin/codex/tokens/import",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("kodo-acquire")
def cloud_kodo_acquire(
    ttl_ms: int = typer.Option(600_000, "--ttl-ms", help="Lease TTL in ms (default: 10m)."),
    out_json: str = typer.Option("", "--out-json", help="Write leased secret JSON to a file (dangerous)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Acquire a Codex account lease (returns secret material; admin-only)."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.post(
        f"{b}/admin/kodo/lease/acquire",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps({"ttl_ms": int(ttl_ms)}),
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if out_json.strip():
        Path(out_json).expanduser().write_text(json.dumps(data, indent=2), encoding="utf-8")
        typer.echo(f"Wrote lease to {out_json}")
    _json_out(data)


@cloud_app.command("kodo-release")
def cloud_kodo_release(
    lease_id: str = typer.Argument(..., help="Lease id to release."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Release a Codex account lease."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.post(
        f"{b}/admin/kodo/lease/release",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps({"lease_id": lease_id.strip()}),
        timeout=30,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("kodo-leases")
def cloud_kodo_leases(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List active Codex leases."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/kodo/leases", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("codex-refresh")
def cloud_codex_refresh(
    account_id: str = typer.Argument(..., help="Codex account_id to refresh."),
    min_validity_ms: int = typer.Option(600_000, "--min-validity-ms", help="Only refresh if expiring within this window."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Refresh a Codex token using its stored refresh_token (server-side)."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.post(
        f"{b}/admin/codex/tokens/{account_id.strip()}/refresh",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps({"min_validity_ms": int(min_validity_ms)}),
        timeout=60,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("modal-tokens")
def cloud_modal_tokens(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List Modal tokens stored in Versa Worker (token secrets never returned)."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/modal/tokens", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("modal-import")
def cloud_modal_import(
    profile: str = typer.Argument(..., help="Modal profile name (display name)."),
    token_id: str = typer.Argument(..., help="Modal token id."),
    token_secret: str = typer.Argument(..., help="Modal token secret."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Upsert a Modal token into Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    payload = {
        "upsert": True,
        "tokens": [
            {
                "profile": profile.strip(),
                "token_id": token_id.strip(),
                "token_secret": token_secret.strip(),
            }
        ],
    }
    r = requests.post(
        f"{b}/admin/modal/tokens/import",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("modal-import-b3")
def cloud_modal_import_b3(
    file: str = typer.Argument(..., help="Path to B3.txt (email|token_id|token_secret|...)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Bulk import Modal tokens from B3.txt format into Versa Worker."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    p = Path(file).expanduser()
    if not p.exists():
        raise typer.BadParameter(f"file not found: {p}")

    items = _parse_modal_profiles_b3(p)
    if not items:
        raise typer.BadParameter(f"No valid rows found in: {p}")

    payload = {"upsert": True, "tokens": items}
    r = requests.post(
        f"{b}/admin/modal/tokens/import",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("modal-delete")
def cloud_modal_delete(
    token_row_id: str = typer.Argument(..., help="Token row id to delete (Versa internal id)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Delete a Modal token from Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.delete(f"{b}/admin/modal/tokens/{token_row_id}", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


def _parse_modal_profiles_tsv(path: Path) -> list[dict[str, Any]]:
    """
    TSV format (one per line):
      profile<TAB>token_id<TAB>token_secret[<TAB>budget_usd]
    """
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        profile, token_id, token_secret = parts[0].strip(), parts[1].strip(), parts[2].strip()
        # Some TSV exports can include a UTF-8 BOM on the first field.
        profile = profile.lstrip("\ufeff")
        if not profile or not token_id or not token_secret:
            continue
        budget_usd: float | None = None
        if len(parts) >= 4 and parts[3].strip():
            try:
                budget_usd = float(parts[3].strip())
            except Exception:
                budget_usd = None
        rows.append(
            {
                "profile": profile,
                "token_id": token_id,
                "token_secret": token_secret,
                "budget_usd": budget_usd,
            }
        )
    return rows


def _parse_modal_profiles_b3(path: Path) -> list[dict[str, Any]]:
    """
    B3 format (one per line):
      email|token_id|token_secret|<ignored>|<ignored>

    We treat `email` as the Modal `profile` label in Versa.
    """
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        profile = parts[0].strip().lstrip("\ufeff")
        token_id = parts[1].strip()
        token_secret = parts[2].strip()
        if not profile or not token_id or not token_secret:
            continue
        rows.append(
            {
                "profile": profile,
                "token_id": token_id,
                "token_secret": token_secret,
                "budget_usd": None,
            }
        )
    return rows


def _cloud_modal_probe_impl(
    *,
    profiles_file: str,
    source: str,
    concurrency: int,
    base_url: str,
    admin_key: str,
    dry_run: bool,
    include_disabled: bool,
) -> None:
    """
    Probe Modal tokens locally (auth + optional billing report) and write status back to Versa Worker.

    Notes:
    - Modal does NOT currently expose an 'exact credits remaining' API for all workspaces.
    - If billing report is not enabled (PermissionDenied), we record billing_enabled=false but still treat the token as valid.
    """
    import requests

    try:
        import modal.config as modal_config  # type: ignore
        import modal._billing as modal_billing  # type: ignore
        from modal.client import _Client  # type: ignore
        from modal.exception import PermissionDeniedError  # type: ignore
        from modal_proto import api_pb2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"Missing Modal SDK; install requirements.txt (modal) first. ({e})")

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    src = (source or "").strip().lower()
    if src not in {"versa", "file"}:
        raise typer.BadParameter("--source must be 'versa' or 'file'")

    items: list[dict[str, Any]]
    if src == "versa":
        # Pull profiles + budgets from Versa SSOT (D1). Secrets are fetched per-profile via /admin/modal/credentials.
        j = _cloud_get_json(b, k, "/admin/modal/tokens", timeout_s=60.0)
        toks = j.get("tokens") or []
        items = []
        for t in toks:
            profile = str(t.get("profile") or t.get("account_ref") or "").strip()
            if not profile:
                continue
            if not include_disabled and bool(t.get("disabled")):
                continue
            items.append(
                {
                    "profile": profile,
                    "token_id": None,
                    "token_secret": None,
                    "budget_usd": t.get("budget_usd", t.get("budgetUsd")),
                }
            )
        if not items:
            raise typer.BadParameter("No Modal profiles found in Versa. Import tokens first.")
    else:
        pf = profiles_file.strip()
        if not pf:
            pf = str(Path.home() / ".versa" / "modal_profiles_import.txt")
        p = Path(pf).expanduser()
        if not p.exists():
            raise typer.BadParameter(f"profiles_file not found: {p}")
        items = _parse_modal_profiles_tsv(p)
        if not items:
            raise typer.BadParameter(f"No profiles found in: {p}")

    server_url = str(getattr(modal_config, "config", {}).get("server_url") or "https://api.modal.com")
    now = datetime.datetime.now(datetime.timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _probe_one(it: dict[str, Any]) -> dict[str, Any]:
        profile = str(it["profile"])
        budget_usd = it.get("budget_usd")

        # Resolve secrets either from local TSV or from Versa Worker (admin-only).
        token_id = it.get("token_id")
        token_secret = it.get("token_secret")
        if not token_id or not token_secret:
            try:
                from urllib.parse import quote

                qprof = quote(profile, safe="")
                creds = await asyncio.to_thread(
                    _cloud_get_json,
                    b,
                    k,
                    f"/admin/modal/credentials?profile={qprof}",
                    30.0,
                )
                token_id = creds.get("token_id")
                token_secret = creds.get("token_secret")
            except Exception as e:
                token_id = None
                token_secret = None

        token_id = str(token_id or "").strip()
        token_secret = str(token_secret or "").strip()

        valid: bool | None = None
        has_credit: bool | None = None
        last_error: str | None = None
        quota: dict[str, Any] = {
            "probe_version": 1,
            "server_url": server_url,
            "profile": profile,
            "workspace": None,
            "billing_enabled": None,
            "mtd_cost_usd": None,
            "budget_usd": budget_usd,
            "budget_remaining_usd": None,
        }

        async with sem:
            try:
                if not token_id or not token_secret:
                    raise RuntimeError("missing_credentials")
                resp = await modal_config._lookup_workspace(server_url, token_id, token_secret)
                quota["workspace"] = getattr(resp, "workspace_name", None) or None
                valid = True
            except Exception as e:
                # Treat transient upstream errors as "unknown" instead of flipping a token to invalid.
                # This avoids poisoning the pool during Modal API incidents.
                msg = str(e)
                transient = any(
                    s in msg
                    for s in (
                        "Received :status = '502'",
                        "Received :status = '503'",
                        "Received :status = '504'",
                        "Deadline exceeded",
                        "timed out",
                        "Temporary failure",
                    )
                )
                if transient:
                    valid = None
                    last_error = f"auth_transient:{type(e).__name__}:{e}"
                else:
                    valid = False
                    last_error = f"auth_failed:{type(e).__name__}:{e}"

            if valid:
                try:
                    async with _Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, (token_id, token_secret)) as client:
                        rows = await modal_billing._workspace_billing_report(
                            start=month_start,
                            end=now,
                            resolution="d",
                            tag_names=None,
                            client=client,
                        )
                    quota["billing_enabled"] = True
                    cost = 0.0
                    for r in rows or []:
                        try:
                            cost += float(r.get("cost", 0) or 0)
                        except Exception:
                            continue
                    quota["mtd_cost_usd"] = cost
                    if isinstance(budget_usd, (int, float)):
                        rem = float(budget_usd) - float(cost)
                        quota["budget_remaining_usd"] = rem
                        has_credit = rem > 0.0
                except PermissionDeniedError as e:
                    # Common on many workspaces: billing report disabled.
                    msg = str(e)
                    if "Billing API is not enabled" in msg:
                        quota["billing_enabled"] = False
                        last_error = None
                    else:
                        quota["billing_enabled"] = None
                        last_error = f"billing_denied:{msg}"
                except Exception as e:
                    quota["billing_enabled"] = None
                    last_error = f"billing_error:{type(e).__name__}:{e}"

            # If the token is valid but we don't have a deterministic credit signal, keep it unknown.
            # This avoids overstating available credits; schedulers may still choose to use these tokens
            # (e.g. for low-cost operations) depending on policy.
            if valid and has_credit is None:
                quota["credit_status"] = "unknown"

        return {
            "profile": profile,
            "last_checked_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
            "valid": valid,
            "has_credit": has_credit,
            "last_error": last_error,
            "quota_json": json.dumps(quota, ensure_ascii=False),
        }

    async def _run() -> list[dict[str, Any]]:
        tasks = [_probe_one(it) for it in items]
        return await asyncio.gather(*tasks)

    updates = asyncio.run(_run())

    if dry_run:
        _json_out({"ok": True, "dry_run": True, "updates": updates})
        return

    # Post in chunks so big files don't hit request limits.
    chunk_size = 50
    sent = 0
    for i in range(0, len(updates), chunk_size):
        chunk = updates[i : i + chunk_size]
        payload = {"updates": chunk}
        r = requests.post(
            f"{b}/admin/modal/tokens/update_status",
            headers={**_versa_headers(k), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        r.raise_for_status()
        sent += len(chunk)

    _json_out({"ok": True, "updated": sent, "profiles": len(items)})


@cloud_app.command("modal-probe")
def cloud_modal_probe(
    profiles_file: str = typer.Option(
        "",
        "--profiles-file",
        help="TSV file with Modal profiles (default: ~/.versa/modal_profiles_import.txt).",
    ),
    source: str = typer.Option(
        "versa",
        "--source",
        help="Where to read Modal profiles from: versa (D1+R2) | file (TSV).",
    ),
    concurrency: int = typer.Option(10, "--concurrency", help="Concurrent probes to run."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not POST updates to Versa; just print JSON."),
    include_disabled: bool = typer.Option(False, "--include-disabled", help="Also probe disabled/frozen profiles."),
) -> None:
    _cloud_modal_probe_impl(
        profiles_file=profiles_file,
        source=source,
        concurrency=concurrency,
        base_url=base_url,
        admin_key=admin_key,
        dry_run=dry_run,
        include_disabled=include_disabled,
    )


def _normalize_cookie_for_modal(cookie: str) -> str:
    c = (cookie or "").strip()
    if not c:
        return ""
    if "modal-session=" in c:
        return c
    # allow passing just the modal-session value
    return f"modal-session={c}"


@cloud_app.command("modal-probe-web")
def cloud_modal_probe_web(
    profiles_file: str = typer.Option(
        "",
        "--profiles-file",
        help="TSV file with Modal profiles (default: ~/.versa/modal_profiles_import.txt).",
    ),
    modal_session: str = typer.Option(
        "",
        "--modal-session",
        help="Modal web session cookie value (or full Cookie header). Prefer env MODAL_SESSION to avoid shell history.",
    ),
    concurrency: int = typer.Option(10, "--concurrency", help="Concurrent probes to run."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not POST updates to Versa; just print JSON."),
) -> None:
    """
    Probe Modal **web** APIs to extract spend limit + cycle credits, matching what the Modal website shows.

    Security:
    - This uses your local `modal-session` cookie (web login). Versa NEVER stores the cookie.
    - Only derived numeric fields are written back to Versa (`quota_json`).

    Caveat:
    - These endpoints are not part of the public Modal SDK contract and could change.
    """
    import requests

    try:
        import modal.config as modal_config  # type: ignore
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"Missing Modal SDK; install requirements.txt (modal) first. ({e})")

    cookie = _normalize_cookie_for_modal(modal_session or os.getenv("MODAL_SESSION") or "")
    if not cookie:
        raise typer.BadParameter("Missing modal-session cookie. Pass --modal-session or set MODAL_SESSION.")

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    pf = profiles_file.strip()
    if not pf:
        pf = str(Path.home() / ".versa" / "modal_profiles_import.txt")
    p = Path(pf).expanduser()
    if not p.exists():
        raise typer.BadParameter(f"profiles_file not found: {p}")

    items = _parse_modal_profiles_tsv(p)
    if not items:
        raise typer.BadParameter(f"No profiles found in: {p}")

    server_url = str(getattr(modal_config, "config", {}).get("server_url") or "https://api.modal.com")
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    def _get_json(url: str) -> Any:
        r = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Cookie": cookie,
                "User-Agent": "Versa-CLI/1 (modal-probe-web)",
            },
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"http_{r.status_code}:{r.text[:200]}")
        return r.json()

    def _first_number(d: Any, *keys: str) -> float | None:
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    def _find_number_recursive(d: Any, wanted_keys: set[str], max_depth: int = 4) -> dict[str, float]:
        """
        Best-effort recursive extractor for numeric values in unknown JSON shapes.
        Returns {key: value} for keys that match wanted_keys.
        """
        found: dict[str, float] = {}

        def _walk(x: Any, depth: int) -> None:
            if depth > max_depth:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in wanted_keys and v is not None:
                        try:
                            found[k] = float(v)
                        except Exception:
                            pass
                    _walk(v, depth + 1)
            elif isinstance(x, list):
                for v in x:
                    _walk(v, depth + 1)

        _walk(d, 0)
        return found

    async def _probe_one(it: dict[str, Any]) -> dict[str, Any]:
        profile = str(it["profile"])
        token_id = str(it["token_id"])
        token_secret = str(it["token_secret"])
        budget_usd = it.get("budget_usd")

        valid: bool | None = None
        has_credit: bool | None = None
        last_error: str | None = None
        quota: dict[str, Any] = {
            "probe_version": 2,
            "server_url": server_url,
            "profile": profile,
            "workspace": None,
            "billing_enabled": None,
            "mtd_cost_usd": None,
            "budget_usd": budget_usd,
            "budget_remaining_usd": None,
            # Web-derived fields (best-effort)
            "spend_limit_usd": None,
            "spend_limit_remaining_usd": None,
            "cycle_credit_usd": None,
            "cycle_credit_remaining_usd": None,
            "effective_remaining_usd": None,
        }

        async with sem:
            # First: workspace lookup via official token.
            try:
                resp = await modal_config._lookup_workspace(server_url, token_id, token_secret)
                ws = getattr(resp, "workspace_name", None) or None
                quota["workspace"] = ws
                valid = True
            except Exception as e:
                valid = False
                last_error = f"auth_failed:{type(e).__name__}:{e}"

            if valid and quota["workspace"]:
                ws = str(quota["workspace"])
                try:
                    # These are the same endpoints the modal.com UI calls (best-effort).
                    ws_info = await asyncio.to_thread(_get_json, f"https://modal.com/api/workspaces/{ws}")
                    credit = await asyncio.to_thread(_get_json, f"https://modal.com/api/workspaces/{ws}/credit-summary")
                    quota["billing_enabled"] = True

                    # Attempt to extract spend limit + remaining.
                    # Structure may change; we keep it defensive.
                    spend_limit = (
                        _first_number(ws_info, "spend_limit_usd", "spendLimitUsd", "spend_limit", "spendLimit")
                        or _first_number(credit, "spend_limit_usd", "spendLimitUsd", "spend_limit", "spendLimit", "spend_limit_dollars", "spendLimitDollars")
                    )
                    spend_remaining = (
                        _first_number(credit, "spend_limit_remaining_usd", "spendLimitRemainingUsd", "spendLimitRemaining", "spend_limit_remaining")
                        or _first_number(ws_info, "spend_limit_remaining_usd", "spendLimitRemainingUsd", "spendLimitRemaining")
                    )

                    # Some versions may provide "spent" instead of remaining.
                    if spend_limit is not None and spend_remaining is None:
                        spent = (
                            _first_number(credit, "spent_usd", "spentUsd", "spent", "used_usd", "usedUsd", "used")
                            or _first_number(ws_info, "spent_usd", "spentUsd", "spent", "used_usd", "usedUsd", "used")
                        )
                        if spent is not None:
                            spend_remaining = spend_limit - spent

                    quota["spend_limit_usd"] = spend_limit
                    quota["spend_limit_remaining_usd"] = spend_remaining

                    # Cycle credits (optional extra endpoint).
                    try:
                        cycles = await asyncio.to_thread(_get_json, f"https://modal.com/api/workspaces/{ws}/billing-cycles")
                        # Try to find the "current" cycle object.
                        if isinstance(cycles, dict):
                            items2 = cycles.get("cycles") if isinstance(cycles.get("cycles"), list) else None
                            if items2 is None and isinstance(cycles.get("items"), list):
                                items2 = cycles.get("items")
                        elif isinstance(cycles, list):
                            items2 = cycles
                        else:
                            items2 = None
                        cur = items2[0] if items2 else None
                        if isinstance(cur, dict):
                            for k1 in ("credit_usd", "credit", "cycle_credit_usd", "cycleCredit"):
                                if k1 in cur and cur[k1] is not None:
                                    quota["cycle_credit_usd"] = float(cur[k1])
                                    break
                            for k1 in ("credit_remaining_usd", "credit_remaining", "remaining_credit_usd", "cycleCreditRemaining"):
                                if k1 in cur and cur[k1] is not None:
                                    quota["cycle_credit_remaining_usd"] = float(cur[k1])
                                    break
                    except Exception:
                        pass

                    # Effective remaining: you can spend up to the spend limit, not just the cycle credit.
                    quota["effective_remaining_usd"] = quota["spend_limit_remaining_usd"]

                    # Best-effort: also record any additional numeric fields we might care about in the future
                    # without breaking if the schema changes.
                    extra_nums = {}
                    extra_nums.update(
                        _find_number_recursive(
                            ws_info,
                            {
                                "spend_limit_usd",
                                "spend_limit_remaining_usd",
                                "cycle_credit_usd",
                                "cycle_credit_remaining_usd",
                                "credit_usd",
                                "credit_remaining_usd",
                            },
                            max_depth=3,
                        )
                    )
                    extra_nums.update(
                        _find_number_recursive(
                            credit,
                            {
                                "spend_limit_usd",
                                "spend_limit_remaining_usd",
                                "cycle_credit_usd",
                                "cycle_credit_remaining_usd",
                                "credit_usd",
                                "credit_remaining_usd",
                            },
                            max_depth=3,
                        )
                    )
                    if extra_nums:
                        quota["web_numbers_debug"] = extra_nums

                    if isinstance(budget_usd, (int, float)) and quota.get("mtd_cost_usd") is not None:
                        quota["budget_remaining_usd"] = float(budget_usd) - float(quota["mtd_cost_usd"])
                        has_credit = float(quota["budget_remaining_usd"]) > 0.0
                except Exception as e:
                    quota["billing_enabled"] = None
                    last_error = f"web_probe_failed:{type(e).__name__}:{e}"

        return {
            "profile": profile,
            "last_checked_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
            "valid": valid,
            "has_credit": has_credit,
            "last_error": last_error,
            "quota_json": json.dumps(quota, ensure_ascii=False),
        }

    async def _run() -> list[dict[str, Any]]:
        tasks = [_probe_one(it) for it in items]
        return await asyncio.gather(*tasks)

    updates = asyncio.run(_run())

    if dry_run:
        _json_out({"ok": True, "dry_run": True, "updates": updates})
        return

    chunk_size = 25
    sent = 0
    for i in range(0, len(updates), chunk_size):
        chunk = updates[i : i + chunk_size]
        payload = {"updates": chunk}
        r = requests.post(
            f"{b}/admin/modal/tokens/update_status",
            headers={**_versa_headers(k), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        r.raise_for_status()
        sent += len(chunk)

    _json_out({"ok": True, "updated": sent, "profiles": len(items)})
@cloud_app.command("kaggle-summary")
def cloud_kaggle_summary(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL (default: https://versa.iseki.cloud)."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Show Kaggle token summary from Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/kaggle/summary", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("kaggle-tokens")
def cloud_kaggle_tokens(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List Kaggle tokens stored in Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.get(f"{b}/admin/kaggle/tokens", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("kaggle-import")
def cloud_kaggle_import(
    username: str = typer.Argument(..., help="Kaggle username."),
    key: str = typer.Argument(..., help="Kaggle API key."),
    label: str = typer.Option("", "--label", help="Display label (defaults to username)."),
    max_concurrency: int = typer.Option(5, "--max-concurrency", help="Max concurrent jobs for this token."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Upsert a Kaggle token into Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    payload = {
        "upsert": True,
        "tokens": [
            {
                "label": (label or username).strip(),
                "username": username.strip(),
                "key": key.strip(),
                "max_concurrency": int(max_concurrency),
            }
        ],
    }
    r = requests.post(
        f"{b}/admin/kaggle/tokens/import",
        headers={**_versa_headers(k), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    r.raise_for_status()
    _json_out(r.json())


@cloud_app.command("kaggle-delete")
def cloud_kaggle_delete(
    token_id: str = typer.Argument(..., help="Token id to delete."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Delete a Kaggle token from Versa Worker."""
    import requests

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    r = requests.delete(f"{b}/admin/kaggle/tokens/{token_id}", headers=_versa_headers(k), timeout=30)
    r.raise_for_status()
    _json_out(r.json())


def _cloud_post_json(
    b: str,
    k: str,
    path: str,
    payload: dict[str, Any],
    timeout_s: float = 60.0,
    *,
    admin_override: bool = False,
) -> dict[str, Any]:
    r = requests.post(
        f"{b}{path}",
        headers={**_versa_headers(k, admin_override=admin_override), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def _cloud_get_json(
    b: str,
    k: str,
    path: str,
    timeout_s: float = 30.0,
    *,
    admin_override: bool = False,
) -> dict[str, Any]:
    r = requests.get(f"{b}{path}", headers=_versa_headers(k, admin_override=admin_override), timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def _cloud_try_json(resp: "requests.Response") -> dict[str, Any] | None:
    try:
        j = resp.json()
        return j if isinstance(j, dict) else None
    except Exception:
        return None


def _is_terminal_status(status: str) -> bool:
    st = (status or "").strip().lower()
    return st in {"succeeded", "failed", "canceled", "killed"}


def _cloud_list_active_runs_for_actor(
    b: str,
    k: str,
    *,
    provider: str,
    actor_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    j = _cloud_get_json(b, k, f"/api/runs?provider={provider}&actor_id={actor_id}&limit={int(limit)}", timeout_s=30.0)
    runs = j.get("runs") if isinstance(j, dict) else None
    if not isinstance(runs, list):
        return []
    out: list[dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        st = str(r.get("status") or "")
        if _is_terminal_status(st):
            continue
        out.append(r)
    return out


def _cloud_force_finish_run(b: str, k: str, run_id: str, *, status: str = "canceled") -> dict[str, Any]:
    return _cloud_post_json(b, k, f"/api/runs/{run_id}/finished", {"status": status}, timeout_s=30.0)


def _cloud_kill_run(b: str, k: str, run_id: str) -> dict[str, Any]:
    return _cloud_post_json(b, k, f"/api/runs/{run_id}/kill", {}, timeout_s=30.0)


@cloud_app.command("release")
def cloud_release(
    provider: str = typer.Argument(..., help="Provider to release: modal|kaggle."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force-release by marking runs finished immediately (releases the lease even if the runner is dead).",
    ),
    wait_s: float = typer.Option(
        120.0,
        "--wait-s",
        help="How long to wait for graceful kill to become terminal before failing (ignored with --force).",
    ),
    poll_s: float = typer.Option(1.0, "--poll-s", help="Poll interval while waiting."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """
    Release your currently-held account/profile for a provider.

    Mechanism:
    - Versa holds accounts via per-run leases.
    - Releasing means making the run terminal, which releases its lease in the coordinator.

    Graceful (default): send kill to your active runs and wait for them to finish.
    Force: mark your active runs as finished immediately (releases the lease even if runners are dead).
    """
    p = provider.strip().lower()
    if p not in {"modal", "kaggle"}:
        raise typer.BadParameter("provider must be modal|kaggle")

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")

    active = _cloud_list_active_runs_for_actor(b, k, provider=p, actor_id=actor_id, limit=500)
    if not active:
        typer.echo(f"OK: no active runs for actor_id={actor_id} provider={p}")
        return

    run_ids = [str(r.get("runId") or r.get("run_id") or "") for r in active]
    run_ids = [x for x in run_ids if x]
    if not run_ids:
        typer.echo(f"OK: no active run ids for actor_id={actor_id} provider={p}")
        return

    typer.echo(f"[versa] releasing provider={p} actor_id={actor_id} active_runs={len(run_ids)}")
    if force:
        for rid in run_ids:
            _cloud_force_finish_run(b, k, rid, status="canceled")
        typer.echo("OK: force-released (runs marked canceled; leases released)")
        return

    for rid in run_ids:
        _cloud_kill_run(b, k, rid)

    deadline = time.time() + max(1.0, float(wait_s))
    remaining = set(run_ids)
    while remaining and time.time() < deadline:
        done: set[str] = set()
        for rid in list(remaining):
            flag = _cloud_kill_flag(b, k, rid)
            if bool(flag.get("terminal")):
                done.add(rid)
        remaining -= done
        if remaining:
            time.sleep(max(0.2, float(poll_s)))

    if remaining:
        raise typer.BadParameter(
            f"release_timeout: {len(remaining)} run(s) still non-terminal. "
            f"Try `versa cloud release {p} --force` to release immediately."
        )
    typer.echo("OK: released (all active runs are terminal)")


def _cloud_put_logs(b: str, k: str, run_id: str, chunk: str) -> None:
    _cloud_post_json(b, k, f"/api/runs/{run_id}/logs", {"chunk": chunk}, timeout_s=60.0)


def _cloud_kill_flag(b: str, k: str, run_id: str, *, admin_override: bool = False) -> dict[str, Any]:
    return _cloud_get_json(b, k, f"/api/runs/{run_id}/kill-flag", timeout_s=30.0, admin_override=admin_override)


def _cloud_heartbeat(b: str, k: str, run_id: str) -> None:
    # Best-effort lease heartbeat to prevent long runs from expiring their slot reservations.
    try:
        _cloud_post_json(b, k, f"/api/runs/{run_id}/heartbeat", {}, timeout_s=30.0)
    except Exception:
        pass


def _terminate_process_tree(pid: int) -> None:
    # Cross-platform best-effort process tree termination.
    try:
        if os.name != "nt":
            os.killpg(pid, 15)
            return
    except Exception:
        pass
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
    except Exception:
        pass


_MODAL_APP_URL_RE = re.compile(r"https?://modal\.com/apps/[^\s/]+/main/(ap-[A-Za-z0-9]+)")


def _extract_modal_app_id(text: str) -> str | None:
    m = _MODAL_APP_URL_RE.search(text or "")
    if not m:
        return None
    return str(m.group(1)).strip() or None


def _modal_app_stop(app_id: str, env: dict[str, str], *, timeout_s: float = 90.0) -> str | None:
    """Best-effort: stop a Modal app run by id (ap-...). Returns combined stdout/stderr text."""
    app_id = str(app_id or "").strip()
    if not app_id:
        return None
    try:
        r = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", app_id],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5.0, float(timeout_s)),
        )
        out = ""
        if r.stdout:
            out += r.stdout
        if r.stderr:
            if out and not out.endswith("\n"):
                out += "\n"
            out += r.stderr
        out = out.strip()
        # Always return a marker; Modal prints Unicode sometimes and may write to stderr.
        return f"rc={int(r.returncode)} {out}".strip()
    except Exception as e:
        return f"modal_app_stop_failed:{type(e).__name__}:{e}"


@cloud_app.command("usage")
def cloud_usage(
    summary: bool = typer.Option(True, "--summary/--json", help="Print a small summary instead of full JSON."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Show current token + lease utilization (single-call endpoint used by the UI)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    j = _cloud_get_json(b, k, "/admin/usage", timeout_s=60.0)
    if not summary:
        _json_out(j)
        return

    kag = (j.get("kaggle") or {}).get("accounts") or []
    mod = (j.get("modal") or {}).get("accounts") or []

    k_caps = ((j.get("kaggle") or {}).get("caps") or {}).get("per_account") or {}
    k_total_per = int(k_caps.get("total") or 11)

    m_caps = ((j.get("modal") or {}).get("caps") or {}).get("per_profile") or {}
    m_gpu_max = int(m_caps.get("gpu_slots_max") or 10)
    m_ctr_max = int(m_caps.get("container_slots_max") or 100)

    kag_valid = [x for x in kag if not bool(x.get("disabled"))]
    # "Valid only" view (matches UI): require an explicit successful probe for Modal profiles.
    mod_valid = [x for x in mod if (not bool(x.get("disabled"))) and (x.get("valid") is True) and (x.get("has_credit") is not False)]

    kag_used = 0
    for x in kag_valid:
        kag_used += int(x.get("bg_total") or 0) + int(x.get("active_total") or 0)
    kag_cap = len(kag_valid) * k_total_per

    mod_gpu_used = 0
    mod_ctr_used = 0
    for x in mod_valid:
        mod_gpu_used += int(x.get("gpu_slots") or 0)
        mod_ctr_used += int(x.get("container_slots") or 0)

    typer.echo(f"Kaggle: tokens={len(kag_valid)} usage={kag_used}/{kag_cap}")
    typer.echo(
        f"Modal: profiles={len(mod_valid)} gpu_slots={mod_gpu_used}/{len(mod_valid)*m_gpu_max} containers={mod_ctr_used}/{len(mod_valid)*m_ctr_max}"
    )


@cloud_app.command("runs")
def cloud_runs(
    provider: str = typer.Option("", "--provider", help="Filter by provider: modal|kaggle."),
    status: str = typer.Option("", "--status", help="Filter by status: queued|assigned|running|succeeded|failed|killed..."),
    account_ref: str = typer.Option("", "--account-ref", "--profile", help="Filter by account/profile (kaggle username or modal profile)."),
    actor_id: str = typer.Option("", "--actor-id", help="Filter by actor_id (owner/client id)."),
    limit: int = typer.Option(50, "--limit", help="Max runs to list (default 50)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List recent runs from the coordinator (admin-only)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    qs: list[str] = [f"limit={int(limit)}"]
    if provider.strip():
        qs.append(f"provider={provider.strip()}")
    if status.strip():
        qs.append(f"status={status.strip()}")
    if account_ref.strip():
        qs.append(f"account_ref={account_ref.strip()}")
    if actor_id.strip():
        qs.append(f"actor_id={actor_id.strip()}")

    path = "/api/runs"
    if qs:
        path += "?" + "&".join(qs)

    j = _cloud_get_json(b, k, path, timeout_s=30.0)
    _json_out(j)


@cloud_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cloud_run(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider: modal (kaggle not wired yet here)."),
    target: str = typer.Argument(..., help="Provider target (modal: file.py::main)."),
    name: str = typer.Option("", "--name", help="Versa run display name."),
    gpu_type: str = typer.Option("", "--gpu-type", help="Modal GPU type (e.g. H100, A100, L4). Empty => CPU-only."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="GPU count (0 => CPU-only)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    poll_kill_s: float = typer.Option(2.0, "--poll-kill-s", help="Kill-flag poll interval."),
    flush_logs_s: float = typer.Option(1.0, "--flush-logs-s", help="Log flush interval."),
    heartbeat_s: float = typer.Option(30.0, "--heartbeat-s", help="Lease heartbeat interval."),
    log_dir: str = typer.Option("logs", "--log-dir", help="Local log directory."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream logs to stdout while running."),
) -> None:
    """
    Provider-agnostic entrypoint for agents.

    Examples:
      python -m versa.cli cloud run modal my_app.py::main -- --arg1 foo
    """
    p = provider.strip().lower()
    extra_args = list(ctx.args)
    if p == "modal":
        cloud_run_modal(
            target=target,
            name=name,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            base_url=base_url,
            admin_key=admin_key,
            poll_kill_s=poll_kill_s,
            flush_logs_s=flush_logs_s,
            heartbeat_s=heartbeat_s,
            log_dir=log_dir,
            stream=stream,
            extra_args=extra_args,
        )
        return
    raise typer.BadParameter("Only provider=modal is supported right now (kaggle runner not implemented in Versa-CLI yet).")


@cloud_app.command("run-modal")
def cloud_run_modal(
    target: str = typer.Argument(..., help="Modal target, e.g. file.py::main or file.py"),
    name: str = typer.Option("", "--name", help="Versa run display name."),
    gpu_type: str = typer.Option("", "--gpu-type", help="Modal GPU type (e.g. H100, A100, L4). Empty => CPU-only."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="GPU count (0 => CPU-only)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    requested_profile: str = typer.Option("", "--profile", help="Request a specific Modal profile (must exist in Versa)."),
    no_preempt: bool = typer.Option(False, "--no-preempt", help="If switching profiles would be required, fail instead of prompting."),
    assume_yes: bool = typer.Option(False, "--yes", help="Non-interactive: auto-switch profiles if needed (may release your current lease)."),
    kill_after_s: float = typer.Option(
        0.0,
        "--kill-after-s",
        help="Auto-request kill after N seconds (0 disables). Useful for cancel tests.",
    ),
    kill_after_ready_s: float = typer.Option(
        0.0,
        "--kill-after-ready-s",
        help="After Modal prints the app URL, auto-request kill after N seconds (0 disables).",
    ),
    poll_kill_s: float = typer.Option(2.0, "--poll-kill-s", help="Kill-flag poll interval."),
    flush_logs_s: float = typer.Option(1.0, "--flush-logs-s", help="Log flush interval."),
    heartbeat_s: float = typer.Option(30.0, "--heartbeat-s", help="Lease heartbeat interval."),
    log_dir: str = typer.Option("logs", "--log-dir", help="Local log directory."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream logs to stdout while running."),
    sync_quota: bool = typer.Option(True, "--sync-quota/--no-sync-quota", help="After the run, sync Modal billing signals into Versa."),
    extra_args: list[str] = typer.Argument(None, help="Args passed through to `modal run` (use `--` before args)."),
) -> None:
    """
    Create a Versa run (provider=modal), claim the assigned Modal profile, execute `modal run`,
    stream logs into Versa Worker, and honor `versa kill` via kill-flag polling.

    This is the canonical runner loop for Modal in Versa.
    """
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    ensure_dir(log_dir)

    def _resolve_modal_target_ref(ref: str) -> str:
        """
        Allow running built-in scripts from any CWD.

        If user passes `scripts/foo.py::main` but their CWD doesn't contain that path,
        resolve it to the package's `apps/versa-cli/scripts/foo.py`.
        """
        ref = str(ref or "").strip()
        if not ref:
            return ref

        suffix = ""
        file_part = ref
        if "::" in ref:
            file_part, rest = ref.split("::", 1)
            suffix = f"::{rest}"

        try:
            p = Path(file_part)
        except Exception:
            return ref

        if p.is_absolute():
            return ref

        # If it already exists relative to CWD, keep it.
        try:
            if p.exists():
                return ref
        except Exception:
            pass

        norm = file_part.replace("\\", "/")
        if norm.startswith("scripts/"):
            try:
                rel = Path(norm).relative_to("scripts")
            except Exception:
                rel = Path(norm).name
            scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
            cand = scripts_dir / rel
            if cand.exists():
                return f"{cand.as_posix()}{suffix}"

        return ref

    run_name = name.strip() or target
    resources: dict[str, Any] = {"network": {"internet": True}}
    gt = gpu_type.strip()
    gc = int(gpu_count or 0)
    if gt or gc > 0:
        resources["gpu"] = {"type": gt or None, "count": (gc if gc > 0 else 1)}
    spec = {
        "entrypoint": {"kind": "modal_python", "ref": target},
        "resources": resources,
        "tags": ["versa", "modal"],
    }
    payload: dict[str, Any] = {
        "provider": "modal",
        "mode": "background",
        "name": run_name,
        "spec": spec,
        "actor_id": _versa_actor_id(""),
    }
    rp = requested_profile.strip()

    modal_tokens_cache: list[dict[str, Any]] | None = None

    def _get_modal_tokens() -> list[dict[str, Any]]:
        nonlocal modal_tokens_cache
        if modal_tokens_cache is None:
            j = _cloud_get_json(b, k, "/admin/modal/tokens", timeout_s=30.0)
            items = j.get("tokens") if isinstance(j, dict) else None
            modal_tokens_cache = items if isinstance(items, list) else []
        return modal_tokens_cache

    def _profile_disabled_state(profile: str) -> bool | None:
        """Return True if disabled, False if enabled, None if unknown profile."""
        p = str(profile or "").strip()
        if not p:
            return None
        target = p.lower()
        has_any = False
        has_enabled = False
        for it in _get_modal_tokens():
            if not isinstance(it, dict):
                continue
            name = str(it.get("profile") or "").strip()
            if not name or name.lower() != target:
                continue
            has_any = True
            if not bool(it.get("disabled")):
                has_enabled = True
        if not has_any:
            return None
        return (not has_enabled)

    if rp:
        st = _profile_disabled_state(rp)
        if st is None:
            raise typer.BadParameter(f"requested_profile_not_found: {rp!r}")
        if st is True:
            raise typer.BadParameter(f"requested_profile_disabled: {rp!r}")
        payload["account_ref"] = rp
    import requests

    def _create_run() -> dict[str, Any]:
        r = requests.post(
            f"{b}/api/runs",
            headers={**_versa_headers(k), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        if r.status_code >= 400:
            j = _cloud_try_json(r)
            if j is None:
                raise typer.BadParameter(f"upstream_non_json: HTTP {r.status_code}")
            # Only prompt on an intentional profile switch attempt.
            if (
                r.status_code == 409
                and str(j.get("error") or "") == "actor_already_holds_account"
                and rp
                and str(j.get("held_account_ref") or "").strip()
                and str(j.get("requested_account_ref") or "").strip()
            ):
                held = str(j.get("held_account_ref") or "").strip()
                reqd = str(j.get("requested_account_ref") or "").strip()
                typer.echo(f"[versa] You already hold Modal profile {held!r}. Requested {reqd!r}.")
                if no_preempt:
                    raise typer.BadParameter("no_preempt: actor_already_holds_account (release first or use a different actor)")
                do_switch = True if assume_yes else typer.confirm("Switch profiles now? (This will release your current lease)", default=False)
                if do_switch:
                    # Preferred: graceful kill then retry. If that times out, offer force-release.
                    try:
                        cloud_release("modal", force=False, wait_s=120.0, poll_s=poll_kill_s, base_url=b, admin_key=k)  # type: ignore[arg-type]
                    except Exception:
                        if assume_yes or typer.confirm("Graceful release timed out. Force-release (mark canceled immediately)?", default=False):
                            cloud_release("modal", force=True, wait_s=0.0, poll_s=poll_kill_s, base_url=b, admin_key=k)  # type: ignore[arg-type]
                        else:
                            raise typer.BadParameter("switch_aborted")
                    # Retry after release
                    r2 = requests.post(
                        f"{b}/api/runs",
                        headers={**_versa_headers(k), "Content-Type": "application/json"},
                        data=json.dumps(payload),
                        timeout=60,
                    )
                    if r2.status_code >= 400:
                        j2 = _cloud_try_json(r2)
                        raise typer.BadParameter(f"run_create_failed: HTTP {r2.status_code}: {j2 or '<non-json>'}")
                    j = _cloud_try_json(r2) or {}
                    return j
                raise typer.BadParameter("actor_already_holds_account")

            raise typer.BadParameter(f"run_create_failed: HTTP {r.status_code}: {j}")

        j = _cloud_try_json(r)
        if j is None:
            raise typer.BadParameter("upstream_non_json")
        return j

    created = _create_run()
    run = created.get("run") or {}
    run_id = str(run.get("runId") or run.get("run_id") or "")
    profile = str(run.get("accountRef") or run.get("account_ref") or "")
    if not run_id:
        raise typer.BadParameter(f"run_create_failed: {created}")
    if not profile:
        raise typer.BadParameter("No Modal profile available (import tokens + probe status first).")

    # Safety: never proceed if the server assigns a disabled/frozen profile.
    # If this happens, cancel/kill immediately so we don't burn spend on a frozen account.
    assigned_disabled = _profile_disabled_state(profile)
    if assigned_disabled is True:
        try:
            _cloud_force_finish_run(b, k, run_id, status="canceled")
        except Exception:
            pass
        try:
            _cloud_kill_run(b, k, run_id)
        except Exception:
            pass
        raise typer.BadParameter(f"assigned_profile_disabled: {profile!r}")
    if rp and profile != rp:
        raise typer.BadParameter(f"requested_profile_mismatch: requested={rp!r} assigned={profile!r}")

    ts = utc_timestamp()
    local_log = os.path.join(log_dir, f"versa_modal_{ts}_{run_id}.log")

    typer.echo(f"[versa] run_id={run_id} provider=modal profile={profile}")
    typer.echo(f"[versa] local_log={local_log}")

    creds = _cloud_get_json(b, k, f"/api/runs/{run_id}/credentials", timeout_s=30.0)
    token_id = str(creds.get("token_id") or "").strip()
    token_secret = str(creds.get("token_secret") or "").strip()
    if not token_id or not token_secret:
        raise typer.BadParameter("Modal credentials missing (import Modal profiles into Versa first).")

    env = dict(os.environ)
    # Do NOT rely on a local .modal.toml (agents should be zero-config).
    # Modal honors env tokens and they take precedence over config profiles.
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret
    # Keep profile as a human-readable label for logs/debugging only.
    env["MODAL_PROFILE"] = profile
    # Windows: avoid UnicodeEncodeError when Modal prints fancy symbols (e.g. checkmarks).
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Best-effort: expose requested GPU to scripts that follow Versa conventions.
    # Note: Modal GPU provisioning is defined in the Python code (decorators), not via CLI flags.
    if gt or gc > 0:
        env["VERSA_MODAL_GPU_TYPE"] = gt
        env["VERSA_MODAL_GPU_COUNT"] = str(gc if gc > 0 else 1)

    resolved_target = _resolve_modal_target_ref(target)

    # Use the current Python environment's Modal module (more reliable than relying on a `modal` shim in PATH).
    cmd = [sys.executable, "-m", "modal", "run", resolved_target, *(extra_args or [])]
    with open(local_log, "ab", buffering=0) as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    _cloud_post_json(
        b,
        k,
        f"/api/runs/{run_id}/started",
        {"provider_run_id": f"pid:{p.pid}", "started_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)},
        timeout_s=30.0,
    )
    _cloud_heartbeat(b, k, run_id)

    # Optional: schedule a kill request (for deterministic cancel tests).
    try:
        ka = float(kill_after_s or 0.0)
    except Exception:
        ka = 0.0
    if ka > 0:
        def _auto_kill() -> None:
            time.sleep(max(0.0, ka))
            try:
                _cloud_post_json(b, k, f"/api/runs/{run_id}/kill", {}, timeout_s=30.0)
            except Exception:
                pass

        t = threading.Thread(target=_auto_kill, daemon=True)
        t.start()

    # Tail-and-forward loop
    last_off = 0
    last_flush = 0.0
    last_kill_poll = 0.0
    last_heartbeat = 0.0
    killed = False
    modal_app_id: str | None = None
    ready_kill_started = False

    start = time.time()
    while True:
        rc = p.poll()

        now = time.time()
        if now - last_flush >= flush_logs_s:
            last_flush = now
            try:
                with open(local_log, "rb") as rf:
                    rf.seek(last_off)
                    data = rf.read()
                    last_off += len(data)
                if data:
                    txt = data.decode("utf-8", errors="replace")
                    if modal_app_id is None:
                        modal_app_id = _extract_modal_app_id(txt)
                    if (not ready_kill_started) and modal_app_id:
                        try:
                            ka2 = float(kill_after_ready_s or 0.0)
                        except Exception:
                            ka2 = 0.0
                        if ka2 > 0:
                            ready_kill_started = True

                            def _auto_kill_ready() -> None:
                                time.sleep(max(0.0, ka2))
                                try:
                                    _cloud_post_json(b, k, f"/api/runs/{run_id}/kill", {}, timeout_s=30.0)
                                except Exception:
                                    pass

                            threading.Thread(target=_auto_kill_ready, daemon=True).start()
                    _cloud_put_logs(b, k, run_id, txt)
                    if stream:
                        try:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                        except Exception:
                            pass
            except Exception:
                # best-effort; keep the runner alive
                pass

        if now - last_kill_poll >= poll_kill_s:
            last_kill_poll = now
            try:
                flag = _cloud_kill_flag(b, k, run_id)
                if bool(flag.get("kill")) and rc is None:
                    killed = True
                    _terminate_process_tree(int(p.pid))
                    # Also stop the Modal app run (best-effort) so remote work doesn't continue.
                    if modal_app_id is None:
                        try:
                            modal_app_id = _extract_modal_app_id(Path(local_log).read_text(encoding="utf-8", errors="ignore"))
                        except Exception:
                            modal_app_id = None
                    if modal_app_id:
                        out = _modal_app_stop(modal_app_id, env)
                        if out:
                            msg = f"\n[versa] modal app stop {modal_app_id}: {out}\n"
                            try:
                                with open(local_log, "ab", buffering=0) as wf:
                                    wf.write(msg.encode("utf-8", errors="replace"))
                            except Exception:
                                pass
                            try:
                                _cloud_put_logs(b, k, run_id, msg)
                            except Exception:
                                pass
            except Exception:
                pass

        if heartbeat_s > 0 and (now - last_heartbeat) >= heartbeat_s:
            last_heartbeat = now
            _cloud_heartbeat(b, k, run_id)

        if rc is not None:
            break

        time.sleep(0.1)

    status = "killed" if killed else ("succeeded" if int(p.returncode or 0) == 0 else "failed")
    _cloud_post_json(
        b,
        k,
        f"/api/runs/{run_id}/finished",
        {"status": status, "ended_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)},
        timeout_s=30.0,
    )

    # Always update basic validity from the actual run outcome (even when --no-sync-quota).
    # This ensures “can I run compute?” is reflected in the shared pool immediately.
    try:
        valid_from_run: bool | None = None
        last_error_from_run: str | None = None
        skip_status_post = False

        if status == "succeeded":
            valid_from_run = True
            last_error_from_run = ""
        elif status == "failed":
            # Try to derive a stable reason from the local log tail.
            txt = ""
            try:
                txt = Path(local_log).read_text(encoding="utf-8", errors="ignore")[-4000:]
            except Exception:
                txt = ""
            t = txt.lower()
            if "filenotfounderror" in t or "no such file or directory" in t:
                # Local script/module resolution error; do not poison token validity.
                skip_status_post = True
            if "token id is malformed" in t:
                valid_from_run = False
                last_error_from_run = "token_id_malformed"
            elif "token secret is malformed" in t:
                valid_from_run = False
                last_error_from_run = "token_secret_malformed"
            elif "unauthorized" in t or "http 401" in t:
                valid_from_run = False
                last_error_from_run = "auth_401"
            elif "forbidden" in t or "http 403" in t:
                valid_from_run = False
                last_error_from_run = "auth_403"
            elif "requested_profile_invalid" in t:
                valid_from_run = False
                last_error_from_run = "profile_invalid"
            elif "requested_profile_disabled" in t or "account_disabled" in t or "frozen" in t:
                valid_from_run = None
                last_error_from_run = "profile_disabled_or_frozen"
            elif "timed out" in t or "timeout" in t or "upstream_non_json" in t or "error code: 522" in t:
                valid_from_run = None
                last_error_from_run = "auth_transient:timeout_or_upstream"
            else:
                valid_from_run = False
                last_error_from_run = "run_failed"

        if (not skip_status_post) and (valid_from_run is not None or last_error_from_run is not None):
            _cloud_post_json(
                b,
                k,
                "/admin/modal/tokens/update_status",
                {
                    "updates": [
                        {
                            "profile": profile,
                            "last_checked_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
                            "valid": valid_from_run,
                            "last_error": last_error_from_run,
                        }
                    ]
                },
                timeout_s=30.0,
            )
    except Exception:
        pass

    # Best-effort: keep Modal quota/billing signals fresh for schedulers + UI.
    if sync_quota:
        try:
            import modal.config as modal_config  # type: ignore
            import modal._billing as modal_billing  # type: ignore
            from modal.client import _Client  # type: ignore
            from modal.exception import PermissionDeniedError  # type: ignore
            from modal_proto import api_pb2  # type: ignore

            server_url = str(getattr(modal_config, "config", {}).get("server_url") or "https://api.modal.com")
            now = datetime.datetime.now(datetime.timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Fetch optional per-profile budget (if configured in Versa).
            budget_usd = None
            try:
                toks = _cloud_get_json(b, k, "/admin/modal/tokens", timeout_s=30.0).get("tokens") or []
                for t in toks:
                    if str(t.get("profile") or "").strip() == profile:
                        budget_usd = t.get("budget_usd", t.get("budgetUsd"))
                        break
            except Exception:
                budget_usd = None

            async def _sync_one() -> dict[str, Any]:
                valid: bool | None = None
                has_credit: bool | None = None
                last_error: str | None = None
                quota: dict[str, Any] = {
                    "probe_version": 1,
                    "server_url": server_url,
                    "profile": profile,
                    "workspace": None,
                    "billing_enabled": None,
                    "mtd_cost_usd": None,
                    "budget_usd": budget_usd,
                    "budget_remaining_usd": None,
                }

                try:
                    resp = await modal_config._lookup_workspace(server_url, token_id, token_secret)
                    quota["workspace"] = getattr(resp, "workspace_name", None) or None
                    valid = True
                except Exception as e:
                    valid = False
                    last_error = f"auth_failed:{type(e).__name__}:{e}"

                if valid:
                    try:
                        async with _Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, (token_id, token_secret)) as client:
                            rows = await modal_billing._workspace_billing_report(
                                start=month_start,
                                end=now,
                                resolution="d",
                                tag_names=None,
                                client=client,
                            )
                        quota["billing_enabled"] = True
                        cost = 0.0
                        for r in rows or []:
                            try:
                                cost += float(r.get("cost", 0) or 0)
                            except Exception:
                                continue
                        quota["mtd_cost_usd"] = cost
                        if isinstance(budget_usd, (int, float)):
                            rem = float(budget_usd) - float(cost)
                            quota["budget_remaining_usd"] = rem
                            has_credit = rem > 0.0
                    except PermissionDeniedError as e:
                        msg = str(e)
                        if "Billing API is not enabled" in msg:
                            quota["billing_enabled"] = False
                            last_error = None
                        else:
                            quota["billing_enabled"] = None
                            last_error = f"billing_denied:{msg}"
                    except Exception as e:
                        quota["billing_enabled"] = None
                        last_error = f"billing_error:{type(e).__name__}:{e}"

                if valid and has_credit is None:
                    quota["credit_status"] = "unknown"

                return {
                    "profile": profile,
                    "last_checked_at_ms": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
                    "valid": valid,
                    "has_credit": has_credit,
                    "last_error": last_error,
                    "quota_json": json.dumps(quota, ensure_ascii=False),
                }

            update = asyncio.run(_sync_one())
            _cloud_post_json(b, k, "/admin/modal/tokens/update_status", {"updates": [update]}, timeout_s=60.0)
        except Exception:
            # Never fail the runner on metering sync issues.
            pass

    _json_out(
        {
            "ok": True,
            "run_id": run_id,
            "profile": profile,
            "status": status,
            "returncode": int(p.returncode or 0),
            "local_log": local_log,
            "elapsed_s": round(time.time() - start, 3),
        }
    )


@cloud_app.command("modal-validate")
def cloud_modal_validate(
    include_disabled: bool = typer.Option(False, "--include-disabled", help="Also validate disabled profiles (default false)."),
    only_valid: bool = typer.Option(False, "--only-valid", help="Only validate profiles currently marked valid in Versa."),
    only_unknown: bool = typer.Option(False, "--only-unknown", help="Only validate profiles with valid=null or transient probe errors."),
    only_invalid: bool = typer.Option(False, "--only-invalid", help="Only validate profiles currently marked invalid (valid=false)."),
    limit: int = typer.Option(0, "--limit", help="Limit number of profiles to validate (0 = all)."),
    concurrency: int = typer.Option(3, "--concurrency", help="Parallelism (profiles validated concurrently)."),
    cpu: bool = typer.Option(True, "--cpu/--no-cpu", help="Run CPU smoke test (square)."),
    cancel: bool = typer.Option(True, "--cancel/--no-cancel", help="Run cancel smoke test (sleep + kill)."),
    gpu: bool = typer.Option(False, "--gpu/--no-gpu", help="Run GPU smoke test (nvidia-smi)."),
    gpu_type: str = typer.Option("L4", "--gpu-type", help="GPU type for GPU smoke test."),
    gpu_count: int = typer.Option(1, "--gpu-count", help="GPU count for GPU smoke test."),
    kill_after_s: float = typer.Option(15.0, "--kill-after-s", help="For cancel test, request kill after N seconds."),
    kill_after_ready_s: float = typer.Option(5.0, "--kill-after-ready-s", help="For cancel test, request kill N seconds after Modal URL is observed."),
    sleep_seconds: int = typer.Option(600, "--sleep-seconds", help="Cancel test sleep duration (remote)."),
    max_run_s: float = typer.Option(300.0, "--max-run-s", help="Safety max runtime for CPU/GPU tests (kills if exceeded)."),
    log_dir: str = typer.Option("logs", "--log-dir", help="Local log directory for runs."),
    out: str = typer.Option("", "--out", help="Write full JSON results to a file (optional)."),
    summary_only: bool = typer.Option(False, "--summary-only", help="Print summary only (still writes --out if set)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """
    Validate Modal profiles end-to-end via Versa:
    - CPU smoke: runs scripts/modal_square.py::main
    - Cancel smoke: runs scripts/modal_sleep.py::main and auto-kills
    - GPU smoke (optional): runs scripts/modal_gpu_smoke.py::main

    This uses `versa cloud run-modal` under the hood so it also validates:
    - profile routing (account_ref)
    - run creation + logs streaming + kill path
    """
    import concurrent.futures

    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    ensure_dir(log_dir)

    toks = _cloud_get_json(b, k, "/admin/modal/tokens", timeout_s=60.0).get("tokens") or []
    profiles: list[dict[str, Any]] = []
    for t in toks:
        prof = str(t.get("profile") or "").strip()
        if not prof:
            continue
        disabled = bool(t.get("disabled"))
        if disabled and not include_disabled:
            continue
        if only_valid and t.get("valid") is not True:
            continue
        if only_invalid and t.get("valid") is not False:
            continue
        if only_unknown:
            v = t.get("valid")
            le = str(t.get("lastError") or t.get("last_error") or "")
            is_unknown = (v is None) or le.startswith("auth_transient:")
            if not is_unknown:
                continue
        profiles.append(
            {
                "profile": prof,
                "disabled": disabled,
                "valid": t.get("valid"),
                "has_credit": t.get("has_credit"),
                "last_error": t.get("last_error"),
            }
        )

    if limit and int(limit) > 0:
        profiles = profiles[: int(limit)]

    def _run_one(args: list[str], env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
        cmd = [sys.executable, "-m", "versa.cli", "cloud", *args]
        env = dict(os.environ)
        if env_overrides:
            env.update(env_overrides)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        if p.returncode != 0:
            return {"ok": False, "returncode": int(p.returncode), "output": out[-2000:]}
        i = out.find("{")
        if i < 0:
            return {"ok": False, "returncode": int(p.returncode), "output": out[-2000:], "error": "no_json_output"}
        try:
            return json.loads(out[i:])
        except Exception as e:
            return {"ok": False, "returncode": int(p.returncode), "output": out[i:][-2000:], "error": f"json_parse_failed:{type(e).__name__}:{e}"}

    # Use absolute paths so callers can run this command from any CWD.
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    square_target = f"{(scripts_dir / 'modal_square.py').as_posix()}::main"
    sleep_target = f"{(scripts_dir / 'modal_sleep.py').as_posix()}::main"
    gpu_target = f"{(scripts_dir / 'modal_gpu_smoke.py').as_posix()}::main"

    def _validate_profile(row: dict[str, Any]) -> dict[str, Any]:
        prof = str(row["profile"])
        res: dict[str, Any] = {"profile": prof, "meta": row, "cpu": None, "cancel": None, "gpu": None}
        # Isolation: each validated profile uses a distinct actor_id so parallel validation
        # doesn't collide with "actor already holds account" safety checks.
        actor_id = f"va_{uuid.uuid4()}"
        base_env = {"VERSA_API_BASE": b, "VERSA_ADMIN_KEY": k, "VERSA_ACTOR_ID": actor_id}

        if cpu:
            res["cpu"] = _run_one(
                [
                    "run-modal",
                    square_target,
                    "--profile",
                    prof,
                    "--yes",
                    "--log-dir",
                    log_dir,
                    "--no-stream",
                    "--no-sync-quota",
                    "--kill-after-s",
                    str(float(max_run_s)),
                ],
                env_overrides=base_env,
            )

        cpu_ok = bool((res.get("cpu") or {}).get("ok")) and str((res.get("cpu") or {}).get("status")) == "succeeded"

        if cancel and cpu_ok:
            res["cancel"] = _run_one(
                [
                    "run-modal",
                    sleep_target,
                    "--profile",
                    prof,
                    "--yes",
                    "--log-dir",
                    log_dir,
                    "--no-stream",
                    "--no-sync-quota",
                    "--kill-after-s",
                    str(float(kill_after_s)),
                    "--kill-after-ready-s",
                    str(float(kill_after_ready_s)),
                ],
                env_overrides={**base_env, "VERSA_SLEEP_SECONDS": str(int(sleep_seconds))},
            )
            # Extra assertion: confirm the runner logged a Modal app stop attempt.
            try:
                c = res.get("cancel") or {}
                lp = str(c.get("local_log") or "").strip()
                if lp and Path(lp).exists():
                    txt = Path(lp).read_text(encoding="utf-8", errors="ignore")
                    c["modal_app_stop_logged"] = ("[versa] modal app stop ap-" in txt)
                else:
                    c["modal_app_stop_logged"] = None
            except Exception:
                try:
                    (res.get("cancel") or {})["modal_app_stop_logged"] = None
                except Exception:
                    pass

        if gpu and cpu_ok:
            res["gpu"] = _run_one(
                [
                    "run-modal",
                    gpu_target,
                    "--profile",
                    prof,
                    "--yes",
                    "--log-dir",
                    log_dir,
                    "--no-stream",
                    "--no-sync-quota",
                    "--kill-after-s",
                    str(float(max_run_s)),
                    "--gpu-type",
                    str(gpu_type),
                    "--gpu-count",
                    str(int(gpu_count)),
                ],
                env_overrides=base_env,
            )

        return res

    max_workers = max(1, int(concurrency or 1))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_validate_profile, r) for r in profiles]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"ok": False, "error": f"validate_profile_failed:{type(e).__name__}:{e}"})

    def _count_ok(k: str, want_status: str | None = None) -> int:
        n = 0
        for r in results:
            x = r.get(k) or {}
            if not isinstance(x, dict):
                continue
            if not bool(x.get("ok")):
                continue
            if want_status is not None and str(x.get("status")) != want_status:
                continue
            n += 1
        return n

    summary = {
        "profiles_total": len(profiles),
        "cpu_succeeded": _count_ok("cpu", "succeeded") if cpu else 0,
        "cancel_killed": _count_ok("cancel", "killed") if cancel else 0,
        "cancel_stop_logged": sum(
            1
            for r in results
            if isinstance(r.get("cancel"), dict) and (r.get("cancel") or {}).get("modal_app_stop_logged") is True
        )
        if cancel
        else 0,
        "gpu_succeeded": _count_ok("gpu", "succeeded") if gpu else 0,
    }

    # Best-effort: push validation results back to Versa so the UI/API reflects
    # real, end-to-end smoke test outcomes (not just whoami probes).
    try:
        now_ms = int(time.time() * 1000)
        updates: list[dict[str, Any]] = []

        def _classify_valid_and_error(cpu_res: dict[str, Any] | None) -> tuple[bool | None, str | None]:
            if not isinstance(cpu_res, dict):
                return None, "smoke_missing_result"

            status = str(cpu_res.get("status") or "")
            if bool(cpu_res.get("ok")) and status == "succeeded":
                return True, ""

            # Prefer parsing local log for human-readable errors.
            txt = ""
            lp = str(cpu_res.get("local_log") or "").strip()
            if lp and Path(lp).exists():
                try:
                    txt = Path(lp).read_text(encoding="utf-8", errors="ignore")[-4000:]
                except Exception:
                    txt = ""
            if not txt:
                txt = str(cpu_res.get("output") or "")[-4000:]

            t = txt.lower()
            if "requested_profile_disabled" in t or "account_disabled" in t or "frozen" in t:
                return None, "profile_disabled_or_frozen"
            if "requested_profile_invalid" in t:
                return False, "profile_invalid"
            if "token id is malformed" in t:
                return False, "token_id_malformed"
            if "token secret is malformed" in t:
                return False, "token_secret_malformed"
            if "http 401" in t or "unauthorized" in t:
                return False, "auth_401"
            if "http 403" in t or "forbidden" in t:
                return False, "auth_403"
            if "timed out" in t or "timeout" in t or "522" in t or "upstream_non_json" in t:
                return None, "auth_transient:timeout_or_upstream"

            if status:
                return False, f"smoke_failed:{status}"
            return False, "smoke_failed"

        for r in results:
            prof = str(r.get("profile") or "").strip()
            if not prof:
                continue
            v, le = _classify_valid_and_error(r.get("cpu") if isinstance(r, dict) else None)
            updates.append(
                {
                    "profile": prof,
                    "last_checked_at_ms": now_ms,
                    "valid": v,
                    "last_error": le,
                }
            )

        # Chunk to avoid request size limits.
        chunk_size = 50
        for i in range(0, len(updates), chunk_size):
            chunk = updates[i : i + chunk_size]
            _cloud_post_json(b, k, "/admin/modal/tokens/update_status", {"updates": chunk}, timeout_s=60.0)
    except Exception:
        # Never fail the command on status sync issues.
        pass

    payload = {"ok": True, "summary": summary, "results": results}
    out_p = (out or "").strip()
    if out_p:
        try:
            out_path = Path(out_p)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            typer.echo(f"[versa] failed to write --out {out_p!r}: {type(e).__name__}: {e}")

    if summary_only:
        _json_out({"ok": True, "summary": summary, "out": out_p or None})
    else:
        _json_out(payload)


@cloud_app.command("logs")
def cloud_logs(
    run_id: str = typer.Argument(..., help="Versa run id (vr_...)."),
    cursor: int = typer.Option(0, "--cursor", help="Start cursor (seq)."),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Follow until terminal."),
    poll_s: float = typer.Option(1.0, "--poll-s", help="Poll interval when following."),
    admin: bool = typer.Option(False, "--admin", help="Admin override: allow reading logs for runs you don't own."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Fetch run logs (and optionally follow)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    cur = int(cursor)
    while True:
        j = _cloud_get_json(b, k, f"/api/runs/{run_id}/logs?cursor={cur}&limit=200", timeout_s=60.0, admin_override=admin)
        lines = j.get("lines") or []
        for ln in lines:
            print(str(ln), end="" if str(ln).endswith("\n") else "\n")
        cur = int(j.get("cursor") or cur)

        if not follow:
            break

        flag = _cloud_kill_flag(b, k, run_id, admin_override=admin)
        if bool(flag.get("terminal")):
            break
        time.sleep(max(0.2, float(poll_s)))


@cloud_app.command("kill")
def cloud_kill(
    run_id: str = typer.Argument(..., help="Versa run id (vr_...)."),
    wait: bool = typer.Option(False, "--wait", help="Wait until the run becomes terminal."),
    poll_s: float = typer.Option(1.0, "--poll-s", help="Poll interval for --wait."),
    admin: bool = typer.Option(False, "--admin", help="Admin override: allow killing runs you don't own."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Request kill for a run. Runners should honor kill via `/api/runs/:id/kill-flag`."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    j = _cloud_post_json(b, k, f"/api/runs/{run_id}/kill", {}, timeout_s=30.0, admin_override=admin)
    if not wait:
        _json_out(j)
        return
    while True:
        flag = _cloud_kill_flag(b, k, run_id, admin_override=admin)
        if bool(flag.get("terminal")):
            _json_out({"ok": True, "run_id": run_id, "terminal": True, "status": flag.get("status")})
            return
        time.sleep(max(0.2, float(poll_s)))


@cloud_app.command("reserve")
def cloud_reserve(
    provider: str = typer.Argument(..., help="Provider: modal|kaggle."),
    account_ref: str = typer.Argument(..., help="Account/profile to reserve (Modal profile or Kaggle username)."),
    label: str = typer.Option("", "--label", help="Optional label (project/codebase) for this reservation."),
    ttl_hours: int = typer.Option(24, "--ttl-hours", help="Reservation TTL in hours (default 24)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Reserve (lock) a specific provider account/profile without launching compute."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    p = provider.strip().lower()
    if p not in ("modal", "kaggle"):
        raise typer.BadParameter("provider must be modal|kaggle")

    actor_id = _versa_actor_id("")
    ttl_ms = int(max(1, int(ttl_hours)) * 60 * 60 * 1000)
    nm = f"{p}:reserve:{label.strip()}" if label.strip() else f"{p}:reserve"
    payload = {"provider": p, "actor_id": actor_id, "account_ref": account_ref.strip(), "name": nm, "lease_ttl_ms": ttl_ms}
    j = _cloud_post_json(b, k, "/api/reservations", payload, timeout_s=60.0)
    run = j.get("run") if isinstance(j, dict) else None
    if not isinstance(run, dict) or not run.get("runId") or not run.get("accountRef"):
        typer.echo(json.dumps(j, indent=2))
        raise typer.Exit(1)
    typer.echo(f"OK: reserved provider={p} account_ref={run.get('accountRef')} run_id={run.get('runId')} actor_id={actor_id}")


# -----------------------------------------------------------------------------
# Structured cloud subcommands (preferred UX). Legacy flat commands remain too.
# -----------------------------------------------------------------------------

@cloud_modal_app.command("gpus")
def cloud_modal_gpus_cmd(
    json_out: bool = typer.Option(True, "--json/--text", help="Print JSON (default) or a human list."),
) -> None:
    """List GPU types supported by the installed Modal SDK (local)."""
    modal_gpus(json_out=json_out)


@cloud_modal_app.command("summary")
def cloud_modal_summary_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_summary(base_url=base_url, admin_key=admin_key)


@cloud_modal_app.command("tokens")
def cloud_modal_tokens_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_tokens(base_url=base_url, admin_key=admin_key)


@cloud_modal_profile_app.command("list")
def cloud_modal_profile_list_cmd(
    details: bool = typer.Option(False, "--details", help="Show a small table (status/valid/credit/id) instead of names-only."),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Only show enabled profiles."),
    valid_only: bool = typer.Option(False, "--valid-only", help="Only show profiles with valid=true."),
    credit_only: bool = typer.Option(False, "--credit-only", help="Only show profiles with hasCredit=true."),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Names-only mode: collapse duplicates by profile name."),
    show_ids: bool = typer.Option(False, "--show-ids", help="Show full Versa token row ids (details mode)."),
    limit: int = typer.Option(0, "--limit", help="Limit number of profiles printed (0 => all)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List Modal profile names in a readable format (no JSON)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    j = _cloud_get_json(b, k, "/admin/modal/tokens", timeout_s=60.0)
    items = j.get("tokens") if isinstance(j, dict) else None
    if not isinstance(items, list):
        items = []

    inv_enabled = 0
    inv_disabled = 0
    filtered: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        profile = str(it.get("profile") or "").strip()
        if not profile:
            continue
        is_disabled = bool(it.get("disabled"))
        if is_disabled:
            inv_disabled += 1
            if enabled_only:
                continue
        else:
            inv_enabled += 1

        if valid_only and it.get("valid") is not True:
            continue
        if credit_only and it.get("hasCredit") is not True:
            continue

        filtered.append(it)

    # Stable ordering for copy/paste and diffs.
    filtered.sort(key=lambda x: str(x.get("profile") or "").lower())
    if limit and limit > 0:
        filtered = filtered[: int(limit)]

    inv_total = inv_enabled + inv_disabled if not enabled_only else inv_enabled
    typer.echo(f"Modal profiles: {len(filtered)}/{inv_total} (enabled {inv_enabled}, disabled {inv_disabled})")

    if not details:
        profiles: list[str] = []
        if dedupe:
            seen: set[str] = set()
            for it in filtered:
                p = str(it.get("profile") or "").strip()
                if not p:
                    continue
                lp = p.lower()
                if lp in seen:
                    continue
                seen.add(lp)
                profiles.append(p)
        else:
            profiles = [str(it.get("profile") or "").strip() for it in filtered if str(it.get("profile") or "").strip()]

        if limit and limit > 0:
            profiles = profiles[: int(limit)]

        for i, p in enumerate(profiles, start=1):
            typer.echo(f"{i:4d}. {p}")
        return

    def _tri(v: Any) -> str:
        if v is True:
            return "yes"
        if v is False:
            return "no"
        return "?"

    table_rows: list[list[str]] = []
    for it in filtered:
        profile = str(it.get("profile") or "").strip()
        status = "disabled" if bool(it.get("disabled")) else "enabled"
        valid_s = _tri(it.get("valid"))
        credit_s = _tri(it.get("hasCredit"))
        row_id = str(it.get("id") or "").strip()
        row_id_s = row_id if show_ids else _compact_id(row_id)
        table_rows.append([profile, status, valid_s, credit_s, row_id_s])

    _print_table(["profile", "status", "valid", "credit", "row_id"], table_rows)


@cloud_modal_profile_app.command("lock")
def cloud_modal_profile_lock_cmd(
    profile: str = typer.Argument(..., help="Modal profile name to lock (reserve) for this actor."),
    label: str = typer.Option("", "--label", help="Optional label (project/codebase) for this lock."),
    ttl_hours: int = typer.Option(24, "--ttl-hours", help="Lock TTL in hours (default 24)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Reserve (lock) a specific Modal profile without launching compute."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")
    ttl_ms = int(max(1, ttl_hours) * 60 * 60 * 1000)

    name = f"modal:lock:{label.strip()}" if label.strip() else "modal:lock"
    payload = {
        "provider": "modal",
        "actor_id": actor_id,
        "account_ref": profile.strip(),
        "name": name,
        "lease_ttl_ms": ttl_ms,
    }
    j = _cloud_post_json(b, k, "/api/reservations", payload, timeout_s=60.0)
    run = j.get("run") if isinstance(j, dict) else None
    if not isinstance(run, dict) or not run.get("runId") or not run.get("accountRef"):
        typer.echo(json.dumps(j, indent=2))
        raise typer.Exit(1)

    typer.echo(f"OK: locked modal profile={run.get('accountRef')} run_id={run.get('runId')} actor_id={actor_id}")


@cloud_modal_profile_app.command("locks")
def cloud_modal_profile_locks_cmd(
    limit: int = typer.Option(200, "--limit", help="Max locks to show."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List your active Modal profile locks (reservations)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")

    j = _cloud_get_json(b, k, f"/api/runs?provider=modal&kind=reservation&actor_id={actor_id}&limit={int(limit)}", timeout_s=30.0)
    runs = j.get("runs") if isinstance(j, dict) else None
    if not isinstance(runs, list):
        runs = []

    active: list[dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        st = str(r.get("status") or "")
        terminal = st in ("succeeded", "failed", "canceled", "killed")
        if terminal:
            continue
        active.append(r)

    active.sort(key=lambda x: int(x.get("createdAtMs") or 0), reverse=True)
    typer.echo(f"Modal locks: {len(active)} (actor_id={actor_id})")
    for i, r in enumerate(active, start=1):
        rid = str(r.get("runId") or "")
        prof = str(r.get("accountRef") or "")
        name = str(r.get("name") or "")
        st = str(r.get("status") or "")
        typer.echo(f"{i:3d}. {prof}  {st:<9}  {rid}  {name}")


@cloud_modal_profile_app.command("unlock")
def cloud_modal_profile_unlock_cmd(
    run_id: str = typer.Option("", "--run-id", help="Reservation run_id to unlock."),
    profile: str = typer.Option("", "--profile", help="Unlock the latest lock for this profile."),
    all: bool = typer.Option(False, "--all", help="Unlock ALL active locks for this actor."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Release a Modal profile lock (reservation) by marking it canceled."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")

    if not run_id and not profile and not all:
        raise typer.BadParameter("Provide --run-id, --profile, or --all.")

    targets: list[str] = []
    if run_id.strip():
        targets = [run_id.strip()]
    else:
        j = _cloud_get_json(b, k, f"/api/runs?provider=modal&kind=reservation&actor_id={actor_id}&limit=500", timeout_s=30.0)
        runs = j.get("runs") if isinstance(j, dict) else None
        if not isinstance(runs, list):
            runs = []

        def _is_active(r: dict[str, Any]) -> bool:
            st = str(r.get("status") or "")
            return st not in ("succeeded", "failed", "canceled", "killed")

        active = [r for r in runs if isinstance(r, dict) and _is_active(r)]
        if all:
            targets = [str(r.get("runId") or "").strip() for r in active if str(r.get("runId") or "").strip()]
        else:
            p = profile.strip()
            cand = [r for r in active if str(r.get("accountRef") or "").strip() == p]
            cand.sort(key=lambda x: int(x.get("createdAtMs") or 0), reverse=True)
            if not cand:
                typer.echo(f"OK: no active lock found for profile={p}")
                return
            targets = [str(cand[0].get("runId") or "").strip()]

    if not targets:
        typer.echo("OK: nothing to unlock")
        return

    for rid in targets:
        _cloud_post_json(b, k, f"/api/runs/{rid}/finished", {"status": "canceled"}, timeout_s=30.0)

    typer.echo(f"OK: unlocked {len(targets)} lock(s) (actor_id={actor_id})")


@cloud_modal_profile_app.command("freeze")
def cloud_modal_profile_freeze_cmd(
    profile: str = typer.Argument(..., help="Modal profile name to freeze (disable scheduling)."),
    until_ms: int = typer.Option(0, "--until-ms", help="Disable until UTC ms since epoch (0 => end of current UTC month)."),
    reason: str = typer.Option("billing_limit", "--reason", help="Reason to record (default: billing_limit)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Freeze a Modal profile globally (prevents scheduling) until end-of-month by default."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    payload: dict[str, Any] = {"profile": profile.strip(), "reason": reason}
    if until_ms and until_ms > 0:
        payload["until_ms"] = int(until_ms)

    j = _cloud_post_json(b, k, "/admin/modal/tokens/freeze", payload, timeout_s=30.0)
    if not isinstance(j, dict) or not j.get("ok"):
        typer.echo(json.dumps(j, indent=2))
        raise typer.Exit(1)

    dm = j.get("disabledUntilMs")
    typer.echo(f"OK: frozen modal profile={profile.strip()} disabled_until_ms={dm} reason={j.get('disabledReason')}")


@cloud_modal_profile_app.command("unfreeze")
def cloud_modal_profile_unfreeze_cmd(
    profile: str = typer.Argument(..., help="Modal profile name to unfreeze (re-enable scheduling)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Unfreeze a Modal profile globally (clears disabled_until_ms/disabled_reason)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    j = _cloud_post_json(b, k, "/admin/modal/tokens/unfreeze", {"profile": profile.strip()}, timeout_s=30.0)
    if not isinstance(j, dict) or not j.get("ok"):
        typer.echo(json.dumps(j, indent=2))
        raise typer.Exit(1)

    typer.echo(f"OK: unfrozen modal profile={profile.strip()}")


@cloud_modal_app.command("import")
def cloud_modal_import_cmd(
    profile: str = typer.Argument(..., help="Modal profile name (display name)."),
    token_id: str = typer.Argument(..., help="Modal token id."),
    token_secret: str = typer.Argument(..., help="Modal token secret."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_import(profile=profile, token_id=token_id, token_secret=token_secret, base_url=base_url, admin_key=admin_key)


@cloud_modal_app.command("import-b3")
def cloud_modal_import_b3_cmd(
    file: str = typer.Argument(..., help="Path to B3.txt (email|token_id|token_secret|...)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_import_b3(file=file, base_url=base_url, admin_key=admin_key)


@cloud_modal_app.command("delete")
def cloud_modal_delete_cmd(
    token_row_id: str = typer.Argument(..., help="Token row id to delete (Versa internal id)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_delete(token_row_id=token_row_id, base_url=base_url, admin_key=admin_key)


@cloud_modal_app.command("probe")
def cloud_modal_probe_cmd(
    profiles_file: str = typer.Option("", "--profiles-file", help="TSV: profile<TAB>token_id<TAB>token_secret<TAB>budget_usd (optional)."),
    source: str = typer.Option("versa", "--source", help="Source of profiles: versa (D1+R2) | file (TSV)."),
    concurrency: int = typer.Option(8, "--concurrency", help="Concurrent probes."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not POST updates to Versa; just print JSON."),
    include_disabled: bool = typer.Option(False, "--include-disabled", help="Also probe disabled/frozen profiles."),
) -> None:
    _cloud_modal_probe_impl(
        profiles_file=profiles_file,
        source=source,
        concurrency=concurrency,
        base_url=base_url,
        admin_key=admin_key,
        dry_run=dry_run,
        include_disabled=include_disabled,
    )


@cloud_modal_app.command("probe-web")
def cloud_modal_probe_web_cmd(
    profiles_file: str = typer.Option("", "--profiles-file", help="TSV: profile<TAB>token_id<TAB>token_secret<TAB>budget_usd (optional)."),
    modal_session: str = typer.Option("", "--modal-session", help="modal-session cookie value."),
    concurrency: int = typer.Option(8, "--concurrency", help="Concurrent probes."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_probe_web(
        profiles_file=profiles_file,
        modal_session=modal_session,
        concurrency=concurrency,
        base_url=base_url,
        admin_key=admin_key,
    )


@cloud_modal_app.command("validate")
def cloud_modal_validate_cmd(
    only_valid: bool = typer.Option(False, "--only-valid", help="Only validate profiles marked valid."),
    gpu: bool = typer.Option(False, "--gpu", help="Include GPU smoke test."),
    concurrency: int = typer.Option(4, "--concurrency", help="Concurrent validations."),
    out: str = typer.Option("", "--out", help="Write JSON results to a file."),
    summary_only: bool = typer.Option(False, "--summary-only", help="Print only summary counters."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_modal_validate(
        only_valid=only_valid,
        gpu=gpu,
        concurrency=concurrency,
        out=out,
        summary_only=summary_only,
        base_url=base_url,
        admin_key=admin_key,
    )


@cloud_modal_app.command("run")
def cloud_modal_run_cmd(
    target: str = typer.Argument(..., help="Modal target, e.g. file.py::main or file.py"),
    name: str = typer.Option("", "--name", help="Versa run display name."),
    gpu_type: str = typer.Option("", "--gpu-type", help="Modal GPU type (e.g. H100, A100, L4). Empty => CPU-only."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="GPU count (0 => CPU-only)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
    requested_profile: str = typer.Option("", "--profile", help="Request a specific Modal profile (must exist in Versa)."),
    no_preempt: bool = typer.Option(False, "--no-preempt", help="If switching profiles would be required, fail instead of prompting."),
    kill_after_s: float = typer.Option(0.0, "--kill-after-s", help="Auto-request kill after N seconds (0 disables)."),
    kill_after_ready_s: float = typer.Option(0.0, "--kill-after-ready-s", help="After Modal prints the app URL, auto-request kill after N seconds."),
    poll_kill_s: float = typer.Option(2.0, "--poll-kill-s", help="Kill-flag poll interval."),
    flush_logs_s: float = typer.Option(1.0, "--flush-logs-s", help="Log flush interval."),
    heartbeat_s: float = typer.Option(30.0, "--heartbeat-s", help="Lease heartbeat interval."),
    log_dir: str = typer.Option("logs", "--log-dir", help="Local log directory."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream logs to stdout while running."),
    sync_quota: bool = typer.Option(True, "--sync-quota/--no-sync-quota", help="After the run, sync Modal billing signals into Versa."),
    extra_args: list[str] = typer.Argument(None, help="Args passed through to `modal run` (use `--` before args)."),
) -> None:
    cloud_run_modal(
        target=target,
        name=name,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        base_url=base_url,
        admin_key=admin_key,
        requested_profile=requested_profile,
        no_preempt=no_preempt,
        kill_after_s=kill_after_s,
        kill_after_ready_s=kill_after_ready_s,
        poll_kill_s=poll_kill_s,
        flush_logs_s=flush_logs_s,
        heartbeat_s=heartbeat_s,
        log_dir=log_dir,
        stream=stream,
        sync_quota=sync_quota,
        extra_args=extra_args,
    )


@cloud_kaggle_app.command("summary")
def cloud_kaggle_summary_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kaggle_summary(base_url=base_url, admin_key=admin_key)


@cloud_kaggle_app.command("tokens")
def cloud_kaggle_tokens_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kaggle_tokens(base_url=base_url, admin_key=admin_key)


@cloud_kaggle_app.command("import")
def cloud_kaggle_import_cmd(
    username: str = typer.Argument(..., help="Kaggle username."),
    key: str = typer.Argument(..., help="Kaggle key JSON string (or a path; see docs)."),
    label: str = typer.Option("", "--label", help="Human label for this token."),
    max_concurrency: int = typer.Option(5, "--max-concurrency", help="Max concurrent leases per token."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kaggle_import(
        username=username,
        key=key,
        label=label,
        max_concurrency=max_concurrency,
        base_url=base_url,
        admin_key=admin_key,
    )


@cloud_kaggle_app.command("delete")
def cloud_kaggle_delete_cmd(
    token_id: str = typer.Argument(..., help="Token id to delete."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kaggle_delete(token_id=token_id, base_url=base_url, admin_key=admin_key)


@cloud_kaggle_profile_app.command("list")
def cloud_kaggle_profile_list_cmd(
    details: bool = typer.Option(False, "--details", help="Show a small table (status/maxConcurrency/id) instead of names-only."),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Only show enabled profiles."),
    show_ids: bool = typer.Option(False, "--show-ids", help="Show full token UUIDs (details mode)."),
    limit: int = typer.Option(0, "--limit", help="Limit number of profiles printed (0 => all)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List Kaggle usernames in a readable format (no JSON)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)

    j = _cloud_get_json(b, k, "/admin/kaggle/tokens", timeout_s=60.0)
    items = j.get("tokens") if isinstance(j, dict) else None
    if not isinstance(items, list):
        items = []

    rows: list[dict[str, Any]] = []
    enabled = 0
    disabled = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        username = str(it.get("username") or "").strip()
        if not username:
            continue
        is_disabled = bool(it.get("disabled"))
        if is_disabled:
            disabled += 1
            if enabled_only:
                continue
        else:
            enabled += 1
        rows.append(it)

    # Stable ordering for copy/paste and diffs.
    rows.sort(key=lambda x: str(x.get("username") or "").lower())
    if limit and limit > 0:
        rows = rows[: int(limit)]

    total = enabled + disabled if not enabled_only else enabled
    typer.echo(f"Kaggle profiles: {total} (enabled {enabled}, disabled {disabled})")

    if not details:
        for i, it in enumerate(rows, start=1):
            username = str(it.get("username") or "").strip()
            typer.echo(f"{i:4d}. {username}")
        return

    table_rows: list[list[str]] = []
    for it in rows:
        username = str(it.get("username") or "").strip()
        label = str(it.get("label") or "").strip()
        max_c = it.get("maxConcurrency")
        if max_c is None:
            max_c = it.get("max_concurrency")
        try:
            max_c_s = str(int(max_c or 0))
        except Exception:
            max_c_s = str(max_c or "")
        status = "disabled" if bool(it.get("disabled")) else "enabled"
        token_id = str(it.get("id") or "").strip()
        token_id_s = token_id if show_ids else _compact_id(token_id)
        label_s = label if (label and label != username) else ""
        table_rows.append([username, label_s, status, max_c_s, token_id_s])

    _print_table(["username", "label", "status", "max", "token_id"], table_rows, right_align={3})


@cloud_kaggle_profile_app.command("lock")
def cloud_kaggle_profile_lock_cmd(
    username: str = typer.Argument(..., help="Kaggle username to lock (reserve) for this actor."),
    label: str = typer.Option("", "--label", help="Optional label (project/codebase) for this lock."),
    ttl_hours: int = typer.Option(24, "--ttl-hours", help="Lock TTL in hours (default 24)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Reserve (lock) a specific Kaggle account without launching compute."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")
    ttl_ms = int(max(1, ttl_hours) * 60 * 60 * 1000)

    name = f"kaggle:lock:{label.strip()}" if label.strip() else "kaggle:lock"
    payload = {
        "provider": "kaggle",
        "actor_id": actor_id,
        "account_ref": username.strip(),
        "name": name,
        "lease_ttl_ms": ttl_ms,
    }
    j = _cloud_post_json(b, k, "/api/reservations", payload, timeout_s=60.0)
    run = j.get("run") if isinstance(j, dict) else None
    if not isinstance(run, dict) or not run.get("runId") or not run.get("accountRef"):
        typer.echo(json.dumps(j, indent=2))
        raise typer.Exit(1)

    typer.echo(f"OK: locked kaggle username={run.get('accountRef')} run_id={run.get('runId')} actor_id={actor_id}")


@cloud_kaggle_profile_app.command("locks")
def cloud_kaggle_profile_locks_cmd(
    limit: int = typer.Option(200, "--limit", help="Max locks to show."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """List your active Kaggle locks (reservations)."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")

    j = _cloud_get_json(b, k, f"/api/runs?provider=kaggle&kind=reservation&actor_id={actor_id}&limit={int(limit)}", timeout_s=30.0)
    runs = j.get("runs") if isinstance(j, dict) else None
    if not isinstance(runs, list):
        runs = []

    active: list[dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        st = str(r.get("status") or "")
        terminal = st in ("succeeded", "failed", "canceled", "killed")
        if terminal:
            continue
        active.append(r)

    active.sort(key=lambda x: int(x.get("createdAtMs") or 0), reverse=True)
    typer.echo(f"Kaggle locks: {len(active)} (actor_id={actor_id})")
    for i, r in enumerate(active, start=1):
        rid = str(r.get("runId") or "")
        user = str(r.get("accountRef") or "")
        name = str(r.get("name") or "")
        st = str(r.get("status") or "")
        typer.echo(f"{i:3d}. {user}  {st:<9}  {rid}  {name}")


@cloud_kaggle_profile_app.command("unlock")
def cloud_kaggle_profile_unlock_cmd(
    run_id: str = typer.Option("", "--run-id", help="Reservation run_id to unlock."),
    username: str = typer.Option("", "--username", help="Unlock the latest lock for this username."),
    all: bool = typer.Option(False, "--all", help="Unlock ALL active locks for this actor."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    """Release a Kaggle account lock (reservation) by marking it canceled."""
    b = _versa_base_url(base_url)
    k = _versa_admin_key(admin_key)
    actor_id = _versa_actor_id("")

    if not run_id and not username and not all:
        raise typer.BadParameter("Provide --run-id, --username, or --all.")

    targets: list[str] = []
    if run_id.strip():
        targets = [run_id.strip()]
    else:
        j = _cloud_get_json(b, k, f"/api/runs?provider=kaggle&kind=reservation&actor_id={actor_id}&limit=500", timeout_s=30.0)
        runs = j.get("runs") if isinstance(j, dict) else None
        if not isinstance(runs, list):
            runs = []

        def _is_active(r: dict[str, Any]) -> bool:
            st = str(r.get("status") or "")
            return st not in ("succeeded", "failed", "canceled", "killed")

        active = [r for r in runs if isinstance(r, dict) and _is_active(r)]
        if all:
            targets = [str(r.get("runId") or "").strip() for r in active if str(r.get("runId") or "").strip()]
        else:
            u = username.strip()
            cand = [r for r in active if str(r.get("accountRef") or "").strip() == u]
            cand.sort(key=lambda x: int(x.get("createdAtMs") or 0), reverse=True)
            if not cand:
                typer.echo(f"OK: no active lock found for username={u}")
                return
            targets = [str(cand[0].get("runId") or "").strip()]

    if not targets:
        typer.echo("OK: nothing to unlock")
        return

    for rid in targets:
        _cloud_post_json(b, k, f"/api/runs/{rid}/finished", {"status": "canceled"}, timeout_s=30.0)

    typer.echo(f"OK: unlocked {len(targets)} lock(s) (actor_id={actor_id})")


@cloud_codex_app.command("tokens")
def cloud_codex_tokens_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_codex_tokens(base_url=base_url, admin_key=admin_key)


@cloud_codex_app.command("import")
def cloud_codex_import_cmd(
    codex_dir: str = typer.Option(str(Path.home() / ".codex" / "CODEX_ACCOUNTS"), "--dir", help="Directory containing Codex account JSONs."),
    max_concurrency: int = typer.Option(1, "--max-concurrency", help="Max concurrent leases per account."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_codex_import(codex_dir=codex_dir, max_concurrency=max_concurrency, base_url=base_url, admin_key=admin_key)


@cloud_codex_app.command("refresh")
def cloud_codex_refresh_cmd(
    account_id: str = typer.Argument(..., help="Codex account_id."),
    min_validity_ms: int = typer.Option(600_000, "--min-validity-ms", help="Min validity window before refresh."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_codex_refresh(account_id=account_id, min_validity_ms=min_validity_ms, base_url=base_url, admin_key=admin_key)


@cloud_kodo_app.command("acquire")
def cloud_kodo_acquire_cmd(
    ttl_ms: int = typer.Option(600_000, "--ttl-ms", help="Lease TTL in ms (default: 10m)."),
    out_json: str = typer.Option("", "--out-json", help="Write leased secret JSON to a file (dangerous)."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kodo_acquire(ttl_ms=ttl_ms, out_json=out_json, base_url=base_url, admin_key=admin_key)


@cloud_kodo_app.command("release")
def cloud_kodo_release_cmd(
    lease_id: str = typer.Argument(..., help="Lease id to release."),
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kodo_release(lease_id=lease_id, base_url=base_url, admin_key=admin_key)


@cloud_kodo_app.command("leases")
def cloud_kodo_leases_cmd(
    base_url: str = typer.Option("", "--base-url", help="Versa API base URL."),
    admin_key: str = typer.Option("", "--admin-key", help="Admin API key (Bearer token)."),
) -> None:
    cloud_kodo_leases(base_url=base_url, admin_key=admin_key)


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
