"""Persistent cache for the per-row data used by the Fig. 06 renderer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "all_pos", "all_neg", "gen_pos", "gen_neg", "non_gen_pos", "non_gen_neg",
    "decay_pos", "decay_neg", "spindle_pos", "spindle_neg",
)
FIT_KEYS = ("alpha", "beta", "gamma", "sigma2", "min_loss")
CACHE_VERSION = "fig6-row-cache-v2-truncated-spindle"


def input_signature(config, roots):
    """Hash configuration and metadata for all files below the input roots."""
    digest = hashlib.sha256(CACHE_VERSION.encode("ascii"))
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    for root in sorted({Path(root) for root in roots}, key=str):
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load(cache_path, signature):
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        if str(cached["signature"]) != signature:
            return None
        data = {key: cached[key].copy() for key in ARRAY_KEYS}
        data["fit"] = json.loads(str(cached["fit_json"]))
        return data


def save(cache_path, signature, data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fit = {key: float(data["fit"][key]) for key in FIT_KEYS}
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        fit_json=np.asarray(json.dumps(fit)),
        **{key: data[key] for key in ARRAY_KEYS},
    )
