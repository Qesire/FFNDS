"""通用 hifx4 PTQ 核心；CUDA 格式扩展按需延迟加载。"""

from .algorithms.gptq_original import (
    GPTQConfig,
    OriginalGPTQConfig,
    quantize_gptq,
    quantize_gptq_original,
)
from .algorithms.rtn import RTNConfig, quantize_rtn
from .experiments.two_stage_hifx4 import (
    HiF4Stage1Config,
    HiF4Stage2Config,
    TwoStageHiF4GPTQConfig,
    quantize_gptq_two_stage,
)
from .transforms.magr import MagRConfig, apply_magr
from .state import (
    ActivationStats,
    CurvatureStats,
    DeviationStats,
    GroupLayout,
    MagRResult,
    ModuleCalibrationStats,
    PreparedWeightState,
    QuantizedWeightState,
)

__all__ = [
    "ActivationStats",
    "CurvatureStats",
    "DeviationStats",
    "GPTQConfig",
    "OriginalGPTQConfig",
    "HiF4Stage1Config",
    "HiF4Stage2Config",
    "TwoStageHiF4GPTQConfig",
    "GroupLayout",
    "MagRConfig",
    "MagRResult",
    "ModuleCalibrationStats",
    "PreparedWeightState",
    "QuantizedWeightState",
    "RTNConfig",
    "quantize_gptq",
    "quantize_gptq_original",
    "quantize_gptq_two_stage",
    "quantize_rtn",
    "apply_magr",
]
