"""Reversible hook and monkey-patch helpers for trace scripts."""

from __future__ import annotations

import types
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch.nn as nn


class HookScope(AbstractContextManager["HookScope"]):
    """Collect removable hook handles and remove them reliably."""

    def __init__(self) -> None:
        self.handles: List[Any] = []

    def add(self, handle: Any) -> Any:
        self.handles.append(handle)
        return handle

    def extend(self, handles: Iterable[Any]) -> None:
        self.handles.extend(handles)

    def close(self) -> None:
        for handle in reversed(self.handles):
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def __enter__(self) -> "HookScope":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class PatchScope(AbstractContextManager["PatchScope"]):
    """Reversible attribute patch scope.

    Use :meth:`patch_method` for instance methods so Python binds `self`
    correctly and :meth:`patch_attr` for ordinary attributes.
    """

    def __init__(self) -> None:
        self._originals: List[Tuple[Any, str, Any]] = []

    def patch_attr(self, obj: Any, name: str, value: Any) -> None:
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def patch_method(self, obj: Any, name: str, fn: Callable[..., Any]) -> None:
        self.patch_attr(obj, name, types.MethodType(fn, obj))

    def close(self) -> None:
        for obj, name, value in reversed(self._originals):
            setattr(obj, name, value)
        self._originals.clear()

    def __enter__(self) -> "PatchScope":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def find_module(root: nn.Module, full_name: str) -> nn.Module:
    cur: Any = root
    if not full_name:
        return root
    for part in full_name.split("."):
        cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Object at {full_name} is not a module: {type(cur).__name__}")
    return cur


def named_modules_matching(
    root: nn.Module,
    *,
    module_type: Optional[type | Tuple[type, ...]] = None,
    name_prefixes: Sequence[str] = (),
    name_suffixes: Sequence[str] = (),
    contains: Sequence[str] = (),
) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, module in root.named_modules():
        if module_type is not None and not isinstance(module, module_type):
            continue
        if name_prefixes and not any(name.startswith(p) for p in name_prefixes):
            continue
        if name_suffixes and not any(name.endswith(s) for s in name_suffixes):
            continue
        if contains and not all(s in name for s in contains):
            continue
        out.append((name, module))
    return out


def parse_block_idx(module_name: str) -> Optional[int]:
    parts = module_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None
