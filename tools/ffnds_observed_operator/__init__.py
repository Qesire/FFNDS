"""NumPy FP64 reference for observed FFN operators; no native model binding."""

from .core import (
    ModeAnchor,
    ObservedAnchor,
    compile_native_modes,
    dense_ffn,
    fit_mode_coefficients,
    fit_observed_readout,
    gelu_tanh,
    make_mode_anchor,
    make_observed_anchor,
    mode_hit,
    native_mode_templates,
    observed_hit,
    paired_features,
    prepare_support,
    select_probes,
)

__all__ = [name for name in globals() if not name.startswith("_")]
