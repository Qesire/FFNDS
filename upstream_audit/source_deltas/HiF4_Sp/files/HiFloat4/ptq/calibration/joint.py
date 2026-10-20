from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from ..state import ActivationStats, CurvatureStats, ModuleCalibrationStats
from .curvature import CurvatureCollector


def _find_module(root: nn.Module, name: str) -> nn.Module:
    current: nn.Module = root
    for part in name.split(".") if name else []:
        current = getattr(current, part)
    return current


class LinearCalibrationCollector:
    """一次 forward 同时收集 activation absmax 与 curvature sum。"""

    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        *,
        curvature_device: str | torch.device = "cpu",
        sample_tokens: int | None = None,
        max_tokens: int | None = None,
        keep_activation_on_gpu: bool = False,
        random_seed: int = 0,
    ) -> None:
        self.model = model
        self.module_names = list(dict.fromkeys(str(name) for name in module_names))
        self.curvature_device = torch.device(curvature_device)
        self.keep_activation_on_gpu = bool(keep_activation_on_gpu)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activation: dict[str, dict[str, Any]] = {}
        self.curvature: dict[str, CurvatureCollector] = {}

        for index, name in enumerate(self.module_names):
            module = _find_module(model, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target {name!r} is {type(module).__name__}, expected nn.Linear")
            self.activation[name] = {"absmax": None, "num_calls": 0}
            self.curvature[name] = CurvatureCollector(
                module.in_features,
                device=self.curvature_device,
                sample_tokens=sample_tokens,
                max_tokens=max_tokens,
                module_name=name,
                random_seed=random_seed + index * 1000003,
            )

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            assert isinstance(module, nn.Linear)
            x = inputs[0].detach()
            if x.shape[-1] != module.in_features:
                raise ValueError(f"{name}: invalid input shape={tuple(x.shape)}")

            reduce_dims = tuple(range(x.ndim - 1))
            current = x.abs().float()
            current = current if not reduce_dims else current.amax(dim=reduce_dims)
            if not self.keep_activation_on_gpu:
                current = current.cpu()
            record = self.activation[name]
            record["absmax"] = (
                current if record["absmax"] is None else torch.maximum(record["absmax"], current)
            )
            record["num_calls"] += 1
            self.curvature[name].add_batch(x)

        return hook

    def register(self) -> None:
        if self.handles:
            raise RuntimeError("collector hooks are already registered")
        for name in self.module_names:
            module = _find_module(self.model, name)
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def export(self, *, curvature_normalize: bool = False) -> dict[str, ModuleCalibrationStats]:
        output: dict[str, ModuleCalibrationStats] = {}
        for name in self.module_names:
            module = _find_module(self.model, name)
            assert isinstance(module, nn.Linear)
            record = self.activation[name]
            absmax = record["absmax"]
            if absmax is None:
                continue
            activation = ActivationStats(
                absmax=absmax.detach().cpu().float().contiguous(),
                num_calls=int(record["num_calls"]),
                channels_dim=-1,
                module_name=name,
            )
            curvature = self.curvature[name].export(normalize=curvature_normalize)
            output[name] = ModuleCalibrationStats(
                module_name=name,
                in_features=module.in_features,
                activation=activation,
                curvature=curvature,
            )
        return output
