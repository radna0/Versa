from __future__ import annotations

# Source code uploaded to remote Jupyter to emulate `modal run file.py[::func] --args`.
# This is intentionally self-contained (no local imports).

RUNNER_SOURCE = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any


def _install_modal_stub() -> None:
    if "modal" in sys.modules:
        return

    modal = types.ModuleType("modal")

    class _Chain:
        def __getattr__(self, _name: str):
            return self

        def __call__(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self

    class Volume:
        @classmethod
        def from_name(cls, name: str, create_if_missing: bool = False):  # noqa: FBT001,FBT002
            return cls()

        def commit(self) -> None:
            return None

    class Image(_Chain):
        @classmethod
        def from_registry(cls, *args, **kwargs):  # noqa: ANN002,ANN003
            return cls()

        def apt_install(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self

        def pip_install(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self

        def add_local_dir(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self

        def run_commands(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self

    class FunctionWrapper:
        def __init__(self, fn):
            self._versa_fn = fn
            self.__name__ = getattr(fn, "__name__", "wrapped")
            self.__doc__ = getattr(fn, "__doc__", None)

        def __call__(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self._versa_fn(*args, **kwargs)

        def remote(self, *args, **kwargs):  # noqa: ANN002,ANN003
            return self._versa_fn(*args, **kwargs)

    class App:
        def __init__(self, *args, **kwargs):  # noqa: ANN002,ANN003
            self._versa_local_entrypoints = []

        def function(self, *dargs, **dkwargs):  # noqa: ANN002,ANN003
            def _decorator(fn):
                return FunctionWrapper(fn)

            return _decorator

        def local_entrypoint(self, *dargs, **dkwargs):  # noqa: ANN002,ANN003
            def _decorator(fn):
                setattr(fn, "_versa_local_entrypoint", True)
                return fn

            return _decorator

    modal.Volume = Volume
    modal.Image = Image
    modal.App = App

    sys.modules["modal"] = modal


def _select_runnable(ns: dict[str, Any], func_name: str | None) -> Any:
    if func_name:
        obj = ns.get(func_name)
        if obj is None:
            raise SystemExit(f"Function not found: {func_name}")
        return obj

    # Auto-select a local entrypoint like `modal run file.py`
    entrypoints = []
    for name, obj in ns.items():
        if callable(obj) and getattr(obj, "_versa_local_entrypoint", False):
            entrypoints.append((name, obj))
    if len(entrypoints) == 1:
        return entrypoints[0][1]
    if len(entrypoints) > 1:
        names = ", ".join(n for n, _ in entrypoints)
        raise SystemExit(f"Multiple local entrypoints found; specify --func. Candidates: {names}")
    raise SystemExit("No local entrypoint found; specify --func")


def _truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _unwrap_callable(obj: Any) -> Any:
    return getattr(obj, "_versa_fn", obj)


def _pick_writable_dir(preferred: str, fallback: str) -> str:
    for base in [preferred, fallback]:
        try:
            p = Path(base)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".versa_write_probe"
            probe.write_text("1", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return str(p)
        except Exception:
            continue
    return fallback


def _patch_download_model(ns: dict[str, Any]) -> None:
    '''
    Best-effort compatibility for Modal files that assume a Volume mounted at /models.

    We patch `download_model(model_name, cache_dir)` to:
    - prefer `cache_dir` if writable
    - otherwise fall back to `VERSA_MODELS_DIR` (or `.versa/models`)
    - set `HF_HOME` to a writable location for huggingface_hub internals
    '''
    orig = ns.get("download_model")
    if not callable(orig):
        return

    def download_model(model_name: str, cache_dir: str) -> str:
        fallback = os.environ.get("VERSA_MODELS_DIR") or ".versa/models"
        base_dir = _pick_writable_dir(cache_dir, fallback)

        # If the requested cache_dir is writable, keep original behavior.
        if os.path.abspath(base_dir) == os.path.abspath(cache_dir):
            return orig(model_name, cache_dir)

        hf_home = str(Path(base_dir) / "hf_cache")
        try:
            Path(hf_home).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        os.environ["HF_HOME"] = hf_home

        from huggingface_hub import snapshot_download

        local_dir = Path(base_dir) / model_name.replace("/", "--")
        if not local_dir.exists():
            print(f"Downloading {model_name} to {local_dir}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        return str(local_dir)

    ns["download_model"] = download_model


def _annotation_base(ann: Any) -> Any:
    # Best-effort: handle Optional[T] / Union[T, None]
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", None) or ()
    if origin is None:
        return ann
    if origin is list:
        return list
    if origin is dict:
        return dict
    if origin is tuple:
        return tuple
    if origin is type(None):  # noqa: E721
        return type(None)
    if str(origin).endswith("typing.Union"):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0]
    return ann


def _coerce(val: str, ann: Any, default: Any) -> Any:
    if ann is inspect._empty:
        if default is not inspect._empty and default is not None:
            ann = type(default)
        else:
            return val
    ann = _annotation_base(ann)
    if ann is bool:
        return _truthy(val)
    if ann is int:
        return int(val)
    if ann is float:
        return float(val)
    if ann is str:
        return str(val)
    return val


def _parse_kwargs(fn: Any, argv: list[str]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]

    by_flag: dict[str, inspect.Parameter] = {}
    for p in params:
        by_flag["--" + p.name.replace("_", "-")] = p

    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise SystemExit(f"Unexpected argument (expected flag): {tok}")

        key, eq, val = tok.partition("=")
        param = by_flag.get(key)
        if param is None:
            raise SystemExit(f"Unknown flag: {key}")

        ann = param.annotation
        default = param.default
        is_bool = (ann is bool) or (default is True) or (default is False)

        if is_bool and eq == "":
            # Treat as a boolean flag.
            out[param.name] = True
            i += 1
            continue
        if is_bool and eq == "=":
            out[param.name] = _coerce(val, bool, default)
            i += 1
            continue

        if eq == "=":
            out[param.name] = _coerce(val, ann, default)
            i += 1
            continue

        if i + 1 >= len(argv):
            raise SystemExit(f"Missing value for flag: {key}")
        out[param.name] = _coerce(argv[i + 1], ann, default)
        i += 2

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Versa modal-file runner (remote).")
    p.add_argument("--file", required=True, help="Path to python file (relative to CWD).")
    p.add_argument("--func", default=None, help="Function/local entrypoint to run.")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the function after `--`.")
    ns = p.parse_args()

    _install_modal_stub()

    # `argparse.REMAINDER` keeps the leading `--` if present.
    raw = list(ns.args)
    if raw and raw[0] == "--":
        raw = raw[1:]

    module_ns = {}
    try:
        module_ns = __import__("runpy").run_path(ns.file)
    except Exception as e:
        raise SystemExit(f"Failed to import {ns.file}: {e}")

    _patch_download_model(module_ns)

    runnable = _select_runnable(module_ns, ns.func)
    target = _unwrap_callable(runnable)
    kwargs = _parse_kwargs(target, raw)

    try:
        if hasattr(runnable, "remote") and callable(getattr(runnable, "remote")):
            result = runnable.remote(**kwargs)
        else:
            result = target(**kwargs)
    except TypeError:
        # Retry positional if the function expects no keywords (best-effort).
        result = target(**kwargs)

    print(json.dumps({"ok": True, "result": result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
