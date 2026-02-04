from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _default_config_dir() -> Path:
    # Shared Versa state directory (used for token TSVs, etc.)
    return Path.home() / ".versa"


def config_dir() -> Path:
    override = (os.getenv("VERSA_CONFIG_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return _default_config_dir()


def config_path() -> Path:
    override = (os.getenv("VERSA_CONFIG_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    return config_dir() / "config.json"


@dataclass(frozen=True)
class VersaLocalConfig:
    api_base: str | None = None
    admin_key: str | None = None
    # Stable per-user identifier for account/lease ownership.
    # This is how Versa prevents two users from colliding on the same provider profile.
    actor_id: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VersaLocalConfig":
        api_base = d.get("api_base")
        admin_key = d.get("admin_key")
        actor_id = d.get("actor_id") or d.get("actorId") or d.get("client_id") or d.get("clientId")
        return VersaLocalConfig(
            api_base=str(api_base).strip() if isinstance(api_base, str) and api_base.strip() else None,
            admin_key=str(admin_key).strip() if isinstance(admin_key, str) and admin_key.strip() else None,
            actor_id=str(actor_id).strip() if isinstance(actor_id, str) and actor_id.strip() else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "admin_key": self.admin_key,
            "actor_id": self.actor_id,
        }


def read_local_config() -> VersaLocalConfig:
    p = config_path()
    try:
        if not p.exists():
            return VersaLocalConfig()
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            return VersaLocalConfig()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return VersaLocalConfig()
        return VersaLocalConfig.from_dict(data)
    except Exception:
        return VersaLocalConfig()


def write_local_config(cfg: VersaLocalConfig) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = config_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def clear_local_config() -> None:
    p = config_path()
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass
