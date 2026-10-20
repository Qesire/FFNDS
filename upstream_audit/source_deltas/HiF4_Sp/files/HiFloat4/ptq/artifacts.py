from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .state import SCHEMA_VERSION, ModuleQuantizedState, QuantizedWeightState

ARTIFACT_KIND = "hif4_ptq_quantized_weights"


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def build_quantization_payload(
    modules: dict[str, ModuleQuantizedState],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, module in modules.items():
        module.validate()
        state = module.weight
        records[name] = {
            "module_name": name,
            "in_features": int(state.dequantized_weight.shape[1]),
            "out_features": int(state.dequantized_weight.shape[0]),
            "weight_qdq": state.dequantized_weight.detach().cpu().contiguous(),
            "bias": None if module.bias is None else module.bias.detach().cpu().contiguous(),
            "smooth_scale": (
                None if state.smooth_scale is None
                else state.smooth_scale.detach().cpu().float().contiguous()
            ),
            "format_name": state.format_name,
            "algorithm": state.algorithm,
            "transforms": list(state.transforms),
            "metadata": _to_cpu(state.metadata),
            "report": _to_cpu(state.report),
            "format_state": _to_cpu(state.format_state),
        }
    return {
        "__meta__": {
            "kind": ARTIFACT_KIND,
            "schema_version": SCHEMA_VERSION,
            **(metadata or {}),
        },
        "modules": records,
    }


def parse_quantization_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, ModuleQuantizedState]]:
    meta = dict(payload.get("__meta__", {}))
    if meta.get("kind") != ARTIFACT_KIND:
        raise ValueError(f"unexpected artifact kind={meta.get('kind')!r}")
    schema = int(meta.get("schema_version", -1))
    if schema not in {1, 2}:
        raise ValueError("unsupported quantization artifact schema")

    output: dict[str, ModuleQuantizedState] = {}
    for name, record in payload.get("modules", {}).items():
        state = QuantizedWeightState(
            dequantized_weight=record["weight_qdq"].detach().cpu().contiguous(),
            format_name=str(record["format_name"]),
            algorithm=str(record["algorithm"]),
            smooth_scale=(
                None if record.get("smooth_scale") is None
                else record["smooth_scale"].detach().cpu().float().contiguous()
            ),
            transforms=list(record.get("transforms", [])),
            metadata=dict(record.get("metadata", {})),
            report=dict(record.get("report", {})),
            format_state=record.get("format_state"),
        )
        output[name] = ModuleQuantizedState(
            module_name=name,
            weight=state,
            bias=None if record.get("bias") is None else record["bias"].detach().cpu(),
            in_features=int(record["in_features"]),
            out_features=int(record["out_features"]),
        )
        output[name].validate()
    return meta, output


def save_quantization_artifact(
    path: str | Path,
    modules: dict[str, ModuleQuantizedState],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_quantization_payload(modules, metadata=metadata), target)


def load_quantization_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, ModuleQuantizedState]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("quantization artifact must contain a dict payload")
    return parse_quantization_payload(payload)
