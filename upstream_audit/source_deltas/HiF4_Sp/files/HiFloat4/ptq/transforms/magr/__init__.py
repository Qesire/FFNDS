from .config import MagRConfig, MagRMode
from .core import apply_magr, magr_regularizer, prox_hif4_tree
from .linf_prox import project_onto_l1_ball, prox_group_linf, prox_linf

__all__ = [
    "MagRConfig",
    "MagRMode",
    "apply_magr",
    "magr_regularizer",
    "project_onto_l1_ball",
    "prox_group_linf",
    "prox_hif4_tree",
    "prox_linf",
]
