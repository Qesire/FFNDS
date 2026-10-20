"""Shared trace context and JSONL writer for Wan quantization probes."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def stable_text_hash(text: str, length: int = 16) -> str:
    """Return a stable short hash for prompts or sample identifiers."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[: int(length)]


def tensor_shape_dtype(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype).replace("torch.", ""),
    }


@dataclass
class TraceContext:
    """Mutable context shared by hook callbacks.

    Hook bodies should call :meth:`base_record` and then add reduced statistics.
    The context intentionally stores metadata only, never activation tensors.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sample_id: str = ""
    prompt_hash: str = ""
    branch: str = "single"
    expert: str = ""
    step_idx: Optional[int] = None
    timestep: Optional[float] = None
    timestep_bin: str = ""
    model_name: str = ""

    def begin_sample(self, sample_id: str, prompt: str = "") -> None:
        self.sample_id = str(sample_id)
        self.prompt_hash = stable_text_hash(prompt) if prompt else ""

    def begin_step(
        self,
        *,
        step_idx: int,
        timestep: float,
        expert: str,
        model_name: Optional[str] = None,
        timestep_bin: str = "",
    ) -> None:
        self.step_idx = int(step_idx)
        self.timestep = float(timestep)
        self.expert = str(expert)
        self.model_name = str(model_name) if model_name is not None else self.model_name
        self.timestep_bin = str(timestep_bin)

    def begin_branch(self, branch: str) -> None:
        self.branch = str(branch)

    def base_record(
        self,
        *,
        point: str,
        module_name: str = "",
        block_idx: Optional[int] = None,
        tensor: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "prompt_hash": self.prompt_hash,
            "branch": self.branch,
            "expert": self.expert,
            "step_idx": self.step_idx,
            "timestep": self.timestep,
            "timestep_bin": self.timestep_bin,
            "model_name": self.model_name,
            "block_idx": None if block_idx is None else int(block_idx),
            "module_name": module_name,
            "point": point,
        }
        if tensor is not None:
            rec.update(tensor_shape_dtype(tensor))
        if extra:
            rec.update(extra)
        return rec


class JsonlTraceWriter:
    """Small JSONL writer for reduced trace records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def write_metadata(path: str | Path, payload: Dict[str, Any]) -> None:
    out = dict(payload)
    out.setdefault("created_at_unix", time.time())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj
