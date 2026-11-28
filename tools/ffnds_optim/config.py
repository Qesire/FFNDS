"""Explicit, composable numerical choices; importing this file needs no GPU."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class OptimizationConfig:
    name: str = "frozen"
    tile_rows: int = 8192
    ffn_staged: bool = False
    anchor_partition: str = "native"  # native, torch reference, triton
    attention_streaming: bool = False
    block_streaming: bool = False
    selected_backend: str = "triton"
    projection_precompose: bool = False
    high_output_fold: bool = False
    attention_basis_dir: str | None = None
    attention_code_dtype: str = "bfloat16"
    joint_readout: str = "direct"
    ffn_factor_dir: str | None = None
    ffn_gauge_dir: str | None = None
    ffn_projection_dtype: str = "bfloat16"
    ffn_backend: str = "native"
    preserve_ffn_probe_u: bool = True
    anchor_direct: bool = True

    def __post_init__(self):
        if not self.name or type(self.tile_rows) is not int or self.tile_rows < 1:
            raise ValueError("A profile name and positive tile_rows are required")
        choices = {
            "anchor_partition": ("native", "torch", "triton"),
            "selected_backend": ("torch", "triton"),
            "joint_readout": ("direct", "source_order"),
            "attention_code_dtype": ("bfloat16", "float32"),
            "ffn_projection_dtype": ("bfloat16", "float32"),
            "ffn_backend": ("native", "torch"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unsupported {name}: {getattr(self, name)}")
        if self.high_output_fold and not self.block_streaming:
            raise ValueError("high_output_fold is connected through block_streaming")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        # Resolve external assets before the archived runner changes cwd.
        for key in ("attention_basis_dir", "ffn_factor_dir", "ffn_gauge_dir"):
            if data.get(key):
                value = Path(data[key])
                data[key] = str((value if value.is_absolute() else path.parent/value).resolve())
        return cls(**data)

    @property
    def needs_custom_ffn(self):
        return bool(self.ffn_staged or self.block_streaming or self.ffn_factor_dir or self.ffn_gauge_dir or not self.anchor_direct)

    @property
    def uses_optimized_attention(self):
        return self.anchor_partition != "native" or self.uses_new_attention_readout

    @property
    def uses_new_attention_readout(self):
        return bool(self.attention_streaming or self.block_streaming or self.projection_precompose or self.attention_basis_dir)

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def semantics(self):
        """Classify operations, not an unverified whole-model equivalence claim."""
        rows = []
        if self.ffn_staged:
            rows.append(dict(option="ffn_staged", kind="preserved_operation_order", scope="original tile, difference/LUT/base/readout casts"))
        if self.anchor_direct and self.needs_custom_ffn:
            rows.append(dict(option="anchor_direct", kind="preserved_operation_order", scope="typed native addmm writes y0 directly; native fallback counted"))
        for name, enabled in (("anchor_partition", self.anchor_partition != "native"),
                              ("attention_streaming", self.attention_streaming),
                              ("block_streaming", self.block_streaming),
                              ("projection_precompose", self.projection_precompose),
                              ("high_output_fold", self.high_output_fold),
                              ("ffn_gauge", bool(self.ffn_gauge_dir))):
            if enabled:
                rows.append(dict(option=name, kind="real_arithmetic_equivalent", scope="floating-point operation shapes/order may change"))
        for name, enabled in (("ffn_factor", bool(self.ffn_factor_dir)), ("attention_joint_history", bool(self.attention_basis_dir))):
            if enabled:
                rows.append(dict(option=name, kind="controlled_approximation", scope="explicit external factors/bases and unchanged native P64 teacher"))
        if not rows:
            rows.append(dict(option="frozen", kind="frozen_reference", scope="existing centered plus wide8192"))
        return rows

    def assert_generation_backends(self):
        """The Torch reference kernels remain useful for local numerical checks."""
        if self.anchor_partition == "torch" or (self.needs_custom_ffn and self.ffn_backend == "torch"):
            raise ValueError("Torch reference backends are component oracles; choose native/Triton for the full video runner")
        if (self.attention_streaming or self.block_streaming) and self.selected_backend == "torch":
            raise ValueError("Full-field generation requires the Triton selected kernel for streamed attention")
