from __future__ import annotations

import importlib.metadata
import inspect
from typing import Any


def modal_sdk_version() -> str | None:
    try:
        return importlib.metadata.version("modal")
    except Exception:
        return None


def list_modal_gpu_types() -> list[str]:
    """List GPU types supported by the *installed* Modal Python SDK.

    This is the ground truth for what `modal.gpu.<TYPE>` exists locally.
    """
    try:
        import modal  # type: ignore
    except Exception:
        return []

    names: list[str] = []
    for name in dir(modal.gpu):
        if name.startswith("_"):
            continue
        obj = getattr(modal.gpu, name, None)
        if not inspect.isclass(obj):
            continue
        # Filter out helper types.
        if name in {"Any", "InvalidError"}:
            continue
        # Only keep GPU-ish names (uppercase / digits).
        if not all((c.isupper() or c.isdigit()) for c in name):
            continue
        names.append(name)

    # Stable UX order.
    return sorted(set(names))


def versa_modal_gpu_aliases() -> dict[str, str]:
    # User-facing aliases that our scripts accept / normalize.
    return {
        "A10": "A10G",
    }


def versa_modal_gpu_known_types() -> list[str]:
    # "Known" list for UI/help. Some may not exist in older Modal SDKs.
    # We intentionally keep this superset so the CLI can tell you if you need to upgrade `modal`.
    return [
        "B200",
        "H200",
        "H100",
        "A100",
        "L40S",
        "A10G",
        "L4",
        "T4",
    ]


def get_modal_gpu_catalog() -> dict[str, Any]:
    supported = list_modal_gpu_types()
    known = versa_modal_gpu_known_types()
    supported_set = set(supported)
    missing = [t for t in known if t not in supported_set]
    return {
        "ok": True,
        "modal_sdk_version": modal_sdk_version(),
        "supported": supported,
        "known": known,
        "missing_from_sdk": missing,
        "aliases": versa_modal_gpu_aliases(),
        "hint": "Use `pip install -U modal` if you need GPU types that are missing from your local SDK.",
    }

