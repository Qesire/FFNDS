"""Compatibility path: the default GPTQ is the preserved repository implementation."""

from .gptq_original import (
    GPTQConfig,
    OriginalGPTQConfig,
    quantize_gptq,
    quantize_gptq_original,
)

__all__ = [
    "GPTQConfig",
    "OriginalGPTQConfig",
    "quantize_gptq",
    "quantize_gptq_original",
]
