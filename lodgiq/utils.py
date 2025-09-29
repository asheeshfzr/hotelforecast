import os
import json
import hashlib
from datetime import datetime
from typing import Any


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def compute_md5(path: str, block_size: int = 2**20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def timestamp_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(obj: Any, path: str, logger=None):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(obj, f, default=str, indent=2)
        if logger:
            logger.info("Saved JSON to %s", path)
    except Exception as e:
        if logger:
            logger.exception("Failed to save JSON to %s: %s", path, e)
        else:
            raise
