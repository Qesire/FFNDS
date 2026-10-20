from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

from ..state import ActivationStats


def _find_module(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    if not name:
        return module
    for part in name.split("."):
        module = getattr(module, part)
    return module


class LinearActivationAbsmaxCollector:
    """通过 forward_pre_hook 收集 Linear 输入的逐通道绝对最大值。

    收集器不保存原始激活，仅保留长度为 in_features 的向量。
    模型适配器负责提供 module_names，本类不包含任何模型结构假设。
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        *,
        channels_dim: int = -1,
        keep_stats_on_gpu: bool = False,
    ) -> None:
        self.model = model
        self.module_names = list(dict.fromkeys(str(x) for x in module_names))
        self.channels_dim = int(channels_dim)
        self.keep_stats_on_gpu = bool(keep_stats_on_gpu)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: dict[str, dict[str, Any]] = {
            name: {"absmax": None, "num_calls": 0} for name in self.module_names
        }

    def _make_hook(self, module_name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target {module_name!r} is not nn.Linear")
            if not inputs or not torch.is_tensor(inputs[0]):
                return

            x = inputs[0].detach()
            if x.ndim == 0:
                return

            cdim = self.channels_dim if self.channels_dim >= 0 else x.ndim + self.channels_dim
            if cdim < 0 or cdim >= x.ndim:
                raise ValueError(
                    f"invalid channels_dim={self.channels_dim} for input shape={tuple(x.shape)}"
                )
            if cdim != x.ndim - 1:
                x = x.movedim(cdim, -1)
            if x.shape[-1] != module.in_features:
                raise ValueError(
                    f"{module_name}: input last dim={x.shape[-1]}, expected={module.in_features}"
                )

            reduce_dims = tuple(range(x.ndim - 1))
            current = x.abs().float()
            current = current if not reduce_dims else current.amax(dim=reduce_dims)
            if not self.keep_stats_on_gpu:
                current = current.cpu()

            record = self._stats[module_name]
            record["absmax"] = (
                current if record["absmax"] is None else torch.maximum(record["absmax"], current)
            )
            record["num_calls"] += 1

        return hook

    def register(self) -> None:
        if self._handles:
            raise RuntimeError("collector hooks are already registered")
        for name in self.module_names:
            module = _find_module(self.model, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target {name!r} is {type(module).__name__}, expected nn.Linear")
            self._handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def export(self, *, dtype: torch.dtype = torch.float32) -> dict[str, ActivationStats]:
        output: dict[str, ActivationStats] = {}
        for name, record in self._stats.items():
            value = record["absmax"]
            if value is None:
                continue
            output[name] = ActivationStats(
                absmax=value.detach().to(device="cpu", dtype=dtype).contiguous(),
                num_calls=int(record["num_calls"]),
                channels_dim=self.channels_dim,
                module_name=name,
            )
        return output

    def state_dict(self) -> dict[str, Mapping[str, Any]]:
        return {
            name: {
                "absmax": None if rec["absmax"] is None else rec["absmax"].detach().cpu(),
                "num_calls": int(rec["num_calls"]),
                "channels_dim": self.channels_dim,
            }
            for name, rec in self._stats.items()
        }
