from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import torch.nn as nn


class ModelAdapter(ABC):
    """模型差异边界；PTQ 数学实现不得依赖具体 adapter。"""

    @abstractmethod
    def iter_layers(self, model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
        raise NotImplementedError

    @abstractmethod
    def target_linear_names(self, model: nn.Module) -> Sequence[str]:
        raise NotImplementedError

    def quantization_groups(self, layer_name: str, layer: nn.Module) -> Sequence[Sequence[str]]:
        del layer_name
        return [[name] for name, module in layer.named_modules() if isinstance(module, nn.Linear)]

    @abstractmethod
    def run_layer(
        self,
        layer: nn.Module,
        hidden_states: Any,
        context: dict[str, Any],
    ) -> Any:
        raise NotImplementedError
