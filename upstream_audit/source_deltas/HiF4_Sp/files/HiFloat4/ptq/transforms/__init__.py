"""HiF4 PTQ transforms package."""

from .magr import MagRConfig, apply_magr
from .smooth import SmoothConfig, apply_smooth

__all__ = ["MagRConfig", "SmoothConfig", "apply_magr", "apply_smooth"]
