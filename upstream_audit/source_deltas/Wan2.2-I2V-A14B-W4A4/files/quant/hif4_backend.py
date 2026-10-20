from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class HiF4Stats:
    weight_qdq_calls: int = 0
    activation_qdq_calls: int = 0
    weight_qdq_tensors: int = 0
    activation_qdq_tensors: int = 0
    skipped_tensors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GLOBAL_HIF4_STATS = HiF4Stats()


def add_hif4_repo_to_path(hif4_repo: str | None = None) -> None:
    hif4_repo = hif4_repo or os.environ.get("HIF4_MAIN_REPO") or os.environ.get("HIF4_REPO")
    if not hif4_repo:
        return

    root = Path(hif4_repo).expanduser().resolve()
    for p in [
        root,
        root / "HiFloat4",
        root / "HiFloat4" / "hif4_gpu",
    ]:
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def load_hif4_api():
    add_hif4_repo_to_path()

    candidates = [
        "HiFloat4.main",
        "HiFloat4.hif4_gpu.quant_cy",
        "HiFloat4.hif4_gpu.quant_cy.quant",
        "hif4_gpu.quant_cy",
        "hif4_gpu.quant_cy.quant",
        "quant_cy",
        "quant_cy.quant",
    ]

    errors = []
    for name in candidates:
        try:
            m = importlib.import_module(name)
            if hasattr(m, "quant_dequant_float") and hasattr(m, "QType"):
                return m.quant_dequant_float, m.QType, name
        except Exception as e:
            errors.append((name, repr(e)))

    raise ImportError("Cannot import HiF4 quant_dequant_float/QType. Tried: " + repr(errors))


_QUANT_DEQUANT_FLOAT = None
_QTYPE = None
_API_NAME = None


def get_hif4_api():
    global _QUANT_DEQUANT_FLOAT, _QTYPE, _API_NAME
    if _QUANT_DEQUANT_FLOAT is None or _QTYPE is None:
        _QUANT_DEQUANT_FLOAT, _QTYPE, _API_NAME = load_hif4_api()
    return _QUANT_DEQUANT_FLOAT, _QTYPE, _API_NAME


def hif4_qdq_tensor(
    x: torch.Tensor,
    *,
    dim: int = -1,
    force_fp32: bool = True,
    kind: str = "activation",
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError(f"HiF4 QDQ requires CUDA tensor, got device={x.device}")

    quant_dequant_float, QType, _ = get_hif4_api()
    y = quant_dequant_float(x, QType("hifx4").dim(dim), force_fp32=force_fp32)

    if kind == "weight":
        GLOBAL_HIF4_STATS.weight_qdq_calls += 1
        GLOBAL_HIF4_STATS.weight_qdq_tensors += int(x.numel())
    elif kind == "activation":
        GLOBAL_HIF4_STATS.activation_qdq_calls += 1
        GLOBAL_HIF4_STATS.activation_qdq_tensors += int(x.numel())

    return y


@torch.no_grad()
def hif4_rtn_qdq_weight(
    weight: torch.Tensor,
    *,
    dim: int = -1,
    force_fp32: bool = True,
    quant_device: str = "cuda",
    return_device: torch.device | None = None,
) -> torch.Tensor:
    orig_dtype = weight.dtype
    orig_device = weight.device
    return_device = return_device or orig_device

    if not torch.cuda.is_available():
        raise RuntimeError("HiF4 weight QDQ requires CUDA")

    w_cuda = weight.detach().to(device=quant_device)
    y_cuda = hif4_qdq_tensor(w_cuda, dim=dim, force_fp32=force_fp32, kind="weight")
    y = y_cuda.to(device=return_device, dtype=orig_dtype)

    del w_cuda, y_cuda
    torch.cuda.empty_cache()

    return y


def reset_hif4_stats() -> None:
    global GLOBAL_HIF4_STATS
    GLOBAL_HIF4_STATS = HiF4Stats()


def get_hif4_stats() -> dict[str, Any]:
    return GLOBAL_HIF4_STATS.to_dict()


def hif4_backend_metadata() -> dict[str, Any]:
    _, _, api_name = get_hif4_api()
    return {
        "quant_scheme": "hif4",
        "code_name": "hifx4",
        "weight_method": "rtn_qdq",
        "activation_method": "online_qdq",
        "group_size": 64,
        "weight_storage": "float_qdq_weight",
        "activation_storage": "float_runtime_qdq",
        "compute_backend": "bf16_gemm",
        "api_module": api_name,
        "hif4_repo": os.environ.get("HIF4_MAIN_REPO") or os.environ.get("HIF4_REPO") or "",
    }
