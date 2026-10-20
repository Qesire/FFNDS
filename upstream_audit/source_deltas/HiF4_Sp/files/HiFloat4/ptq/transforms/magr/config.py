from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal


MagRMode = Literal["group64", "leaf4", "hif4_tree"]
SpectralMethod = Literal["power", "exact"]


@dataclass(frozen=True)
class MagRConfig:
    """AozhongZhang/MagR 风格的 hifx4 权重预处理配置。

    默认先将 H=X^T X 谱归一化，再使用步长 1 的 proximal gradient；
    这与原 MagR 仓库对 X 做完整 SVD 并令最大奇异值为 1 等价。
    """

    mode: MagRMode = "hif4_tree"
    alpha: float = 1e-4
    outer_iters: int = 200
    outer_tol: float = 0.0
    step_scale: float = 1.0

    normalize_curvature: bool = True
    spectral_method: SpectralMethod = "exact"
    power_iters: int = 32
    power_tol: float = 1e-6
    random_seed: int = 0
    symmetrize_curvature: bool = True
    eps: float = 1e-8

    dykstra_iters: int = 32
    dykstra_tol: float = 1e-6

    # 来自层级误差 surrogate 的理论初值；仅用于 hif4_tree。
    e6m2_rounding_factor: float = 1.0
    lambda64: float | None = None
    lambda8: float | None = None
    lambda4: float | None = None
    # v4.8 method-only controls. Keep False for faithful group64/legacy behavior.
    normalize_tree_group_count: bool = False
    normalize_tree_lambdas: bool = False

    allow_partial_groups: bool = True
    # 限制排序投影/Dykstra 的临时显存；None 表示一次处理全部输出行。
    row_chunk_size: int | None = 128
    checkpoint_iterations: tuple[int, ...] = (0, 10, 25, 50, 100, 150, 200)
    compute_device: str | None = None

    def validate(self) -> None:
        if self.mode not in {"group64", "leaf4", "hif4_tree"}:
            raise ValueError(f"unsupported MagR mode={self.mode!r}")
        if self.alpha < 0:
            raise ValueError("alpha cannot be negative")
        if self.outer_iters < 0:
            raise ValueError("outer_iters cannot be negative")
        if self.outer_tol < 0:
            raise ValueError("outer_tol cannot be negative")
        if not 0 < self.step_scale <= 1.0:
            raise ValueError("step_scale must be in (0, 1]")
        if self.spectral_method not in {"power", "exact"}:
            raise ValueError("spectral_method must be 'power' or 'exact'")
        if self.power_iters <= 0:
            raise ValueError("power_iters must be positive")
        if self.power_tol < 0:
            raise ValueError("power_tol cannot be negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.dykstra_iters <= 0:
            raise ValueError("dykstra_iters must be positive")
        if self.dykstra_tol < 0:
            raise ValueError("dykstra_tol cannot be negative")
        if self.e6m2_rounding_factor <= 0:
            raise ValueError("e6m2_rounding_factor must be positive")
        for name, value in (
            ("lambda64", self.lambda64),
            ("lambda8", self.lambda8),
            ("lambda4", self.lambda4),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.row_chunk_size is not None and self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive or None")
        if any(int(i) < 0 for i in self.checkpoint_iterations):
            raise ValueError("checkpoint iterations cannot be negative")
        if self.compute_device is not None and not str(self.compute_device).strip():
            raise ValueError("compute_device cannot be empty")

    def resolved_level_weights(self) -> tuple[float, float, float]:
        self.validate()
        lambda64 = (
            self.e6m2_rounding_factor / 7.0
            if self.lambda64 is None
            else float(self.lambda64)
        )
        lambda8 = 1.0 / sqrt(32.0) if self.lambda8 is None else float(self.lambda8)
        lambda4 = 0.25 if self.lambda4 is None else float(self.lambda4)
        values = (float(lambda64), float(lambda8), float(lambda4))
        if self.normalize_tree_lambdas:
            total = sum(values)
            if total <= 0:
                raise ValueError("tree lambda weights must have positive sum")
            values = tuple(v / total for v in values)
        return values
