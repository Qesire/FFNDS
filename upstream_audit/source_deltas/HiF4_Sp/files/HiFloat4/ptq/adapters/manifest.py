from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from .module_utils import build_linear_manifest


def manifest_hash(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_linear_manifest(
    path: str | Path,
    model: nn.Module,
    module_names: list[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_linear_manifest(model, module_names)
    payload = {
        "__meta__": {**(metadata or {}), "manifest_sha256": manifest_hash(manifest)},
        "modules": manifest,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_linear_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    modules = list(payload.get("modules", []))
    expected = payload.get("__meta__", {}).get("manifest_sha256")
    actual = manifest_hash(modules)
    if expected and expected != actual:
        raise ValueError("module manifest hash mismatch")
    return dict(payload.get("__meta__", {})), modules
