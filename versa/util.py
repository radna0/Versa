from __future__ import annotations

import os
import time
import uuid


def utc_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def new_id() -> str:
    return str(uuid.uuid4())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

