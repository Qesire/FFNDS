from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch.nn as nn

from .base import ModelAdapter


class DecoderOnlyAdapter(ModelAdapter):
    """对 model.model.layers 结构的轻量 adapter；不包含 GPTQ 数学。"""

    def iter_layers(self, model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("model does not expose model.layers")
        for index, layer in enumerate(model.model.layers):
            yield f"model.layers.{index}", layer

    def target_linear_names(self, model: nn.Module) -> Sequence[str]:
        output: list[str] = []
        for layer_name, layer in self.iter_layers(model):
            for local_name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    output.append(f"{layer_name}.{local_name}" if local_name else layer_name)
        return output

    def quantization_groups(self, layer_name: str, layer: nn.Module) -> Sequence[Sequence[str]]:
        names = {name for name, module in layer.named_modules() if isinstance(module, nn.Linear)}
        candidates = [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]
        groups = [[name for name in group if name in names] for group in candidates]
        covered = {name for group in groups for name in group}
        groups.extend([[name] for name in sorted(names - covered)])
        return [group for group in groups if group]

    def run_layer(
        self,
        layer: nn.Module,
        hidden_states: Any,
        context: dict[str, Any],
    ) -> Any:
        output = layer(hidden_states, **context)
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
