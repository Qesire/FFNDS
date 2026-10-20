from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn


def get_module(root: nn.Module, name: str) -> nn.Module:
    current = root
    for part in name.split(".") if name else []:
        current = getattr(current, part)
    return current


def get_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    if not parts or not parts[-1]:
        raise ValueError("module name cannot be empty")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, child = get_parent(root, name)
    setattr(parent, child, module)


def build_linear_manifest(
    model: nn.Module,
    module_names: Iterable[str],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        manifest.append(
            {
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "dtype": str(module.weight.dtype),
                "device": str(module.weight.device),
            }
        )
    return manifest
