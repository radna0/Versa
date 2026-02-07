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


def _safe_actor_filename(actor: str) -> str:
    a = (actor or "").strip()
    safe = "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in a])
    return safe or "default"


def config_path_for_actor(actor: str) -> Path:
    return config_dir() / "actors" / f"{_safe_actor_filename(actor)}.json"


def _active_actor_name_from_file() -> str | None:
    try:
        p = config_dir() / "active_actor.txt"
        if not p.exists():
            return None
        name = p.read_text(encoding="utf-8", errors="ignore").strip()
        return name if name else None
    except Exception:
        return None


def config_path() -> Path:
    override = (os.getenv("VERSA_CONFIG_PATH") or "").strip()
    if override:
        return Path(override).expanduser()

    actor = (os.getenv("VERSA_ACTOR_NAME") or "").strip() or (_active_actor_name_from_file() or "")
    if actor:
        return config_path_for_actor(actor)

    return config_dir() / "config.json"


def set_active_actor(name: str) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / "active_actor.txt"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(str(name or "").strip(), encoding="utf-8")
    tmp.replace(p)
    return p


@dataclass(frozen=True)
class VersaLocalConfig:
    api_base: str | None = None
    admin_key: str | None = None
    # Stable per-user identifier for account/lease ownership.
    # This is how Versa prevents two users from colliding on the same provider profile.
    actor_id: str | None = None
    # Human label for the actor (e.g. alice, bob). Used for isolation on shared machines.
    actor_name: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VersaLocalConfig":
        api_base = d.get("api_base")
        admin_key = d.get("admin_key")
        actor_id = d.get("actor_id") or d.get("actorId") or d.get("client_id") or d.get("clientId")
        actor_name = d.get("actor_name") or d.get("actorName") or d.get("owner") or d.get("ownerName")
        return VersaLocalConfig(
            api_base=str(api_base).strip() if isinstance(api_base, str) and api_base.strip() else None,
            admin_key=str(admin_key).strip() if isinstance(admin_key, str) and admin_key.strip() else None,
            actor_id=str(actor_id).strip() if isinstance(actor_id, str) and actor_id.strip() else None,
            actor_name=str(actor_name).strip() if isinstance(actor_name, str) and actor_name.strip() else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "admin_key": self.admin_key,
            "actor_id": self.actor_id,
            "actor_name": self.actor_name,
        }


def read_local_config() -> VersaLocalConfig:
    return read_local_config_from(config_path())


def read_local_config_from(path: Path) -> VersaLocalConfig:
    try:
        if not path.exists():
            return VersaLocalConfig()
        raw = path.read_text(encoding="utf-8", errors="ignore")
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
    write_local_config_to(p, cfg)
    return p


def write_local_config_to(path: Path, cfg: VersaLocalConfig) -> None:
    d = path.parent
    d.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def list_actor_config_paths() -> list[Path]:
    try:
        d = config_dir() / "actors"
        if not d.exists():
            return []
        return sorted([p for p in d.glob("*.json") if p.is_file()], key=lambda p: p.name.lower())
    except Exception:
        return []


def clear_local_config() -> None:
    p = config_path()
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass
