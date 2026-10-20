"""Backward-compatible import path for the preserved repository GPTQ."""

from .gptq_original import GPTQConfig, OriginalGPTQConfig, quantize_gptq, quantize_gptq_original

__all__ = ["GPTQConfig", "OriginalGPTQConfig", "quantize_gptq", "quantize_gptq_original"]
