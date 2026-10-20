from __future__ import annotations

"""LightX2V-native Wan2.2 I2V distill calibration/runtime bridge.

The bridge keeps LightX2V's ``wan2.2_moe_distill`` runner and
``Wan22StepDistillScheduler`` unchanged.  It instruments the default
``MMWeight.apply`` path to collect calibration statistics or to execute an
existing HiF4 fake-QDQ cache.

The cache runtime is intentionally a correctness/reference implementation:
weights remain float-QDQ tensors and are moved one block at a time.  It does
not claim packed 4-bit storage or a fused GEMM kernel.
"""

import contextvars
import copy
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from .wan_i2v import native_hif4_activation_qdq

LOGGER = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).with_name("lightx2v_wan_distill_contract.json")


def load_distill_runtime_contract() -> dict[str, Any]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version", "artifact", "model_cls", "task", "scheduler",
        "infer_steps", "target_video_length", "target_height", "target_width",
        "fps", "attention", "sample_shift", "sample_guide_scale", "enable_cfg",
        "seq_parallel", "parallel", "dim", "num_heads",
        "denoising_step_list", "boundary_step_index", "boundary",
        "aspect_ratio", "rope_type", "resize_mode", "audio_fps",
        "double_precision_rope", "norm_modulate_backend",
        "distilled_sigma_values", "cpu_offload", "offload_granularity",
        "t5_cpu_offload", "image_encoder_offload", "vae_cpu_offload",
        "high_noise_filename", "low_noise_filename",
        "high_noise_sha256", "low_noise_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"distill runtime contract missing keys: {missing}")
    if payload["model_cls"] != "wan2.2_moe_distill":
        raise ValueError(payload["model_cls"])
    if payload["scheduler"] != "Wan22StepDistillScheduler":
        raise ValueError(payload["scheduler"])
    if int(payload["infer_steps"]) != len(payload["denoising_step_list"]):
        raise ValueError("infer_steps must equal len(denoising_step_list)")
    if list(payload["denoising_step_list"]) != [1000, 750, 500, 250]:
        raise ValueError(payload["denoising_step_list"])
    if int(payload["boundary_step_index"]) != 2:
        raise ValueError(payload["boundary_step_index"])
    if int(payload["dim"]) % int(payload["num_heads"]) != 0:
        raise ValueError("dim must be divisible by num_heads")
    return payload


DISTILL_RUNTIME_CONTRACT = load_distill_runtime_contract()
MODEL_CLASS = str(DISTILL_RUNTIME_CONTRACT["model_cls"])
SCHEDULER_CLASS = str(DISTILL_RUNTIME_CONTRACT["scheduler"])
DENOISING_STEP_LIST = tuple(int(x) for x in DISTILL_RUNTIME_CONTRACT["denoising_step_list"])
BOUNDARY_STEP_INDEX = int(DISTILL_RUNTIME_CONTRACT["boundary_step_index"])


def build_scheduler_config_from_contract() -> dict[str, Any]:
    contract = load_distill_runtime_contract()
    keys = (
        "infer_steps", "target_video_length", "sample_shift", "seq_parallel",
        "sample_guide_scale", "dim", "num_heads", "denoising_step_list",
        "boundary_step_index",
    )
    return {key: copy.deepcopy(contract[key]) for key in keys}


def create_generator_kwargs_from_contract() -> dict[str, Any]:
    c = load_distill_runtime_contract()
    return {
        "attn_mode": str(c["attention"]),
        "infer_steps": int(c["infer_steps"]),
        "num_frames": int(c["target_video_length"]),
        "height": int(c["target_height"]),
        "width": int(c["target_width"]),
        "guidance_scale": float(c["sample_guide_scale"]),
        "sample_shift": float(c["sample_shift"]),
        "fps": int(c["fps"]),
        "aspect_ratio": str(c["aspect_ratio"]),
        "boundary": float(c["boundary"]),
        "boundary_step_index": int(c["boundary_step_index"]),
        "denoising_step_list": list(c["denoising_step_list"]),
        "config_json": None,
        "resize_mode": c["resize_mode"],
        "rope_type": str(c["rope_type"]),
        "audio_fps": int(c["audio_fps"]),
        "double_precision_rope": bool(c["double_precision_rope"]),
        "norm_modulate_backend": str(c["norm_modulate_backend"]),
        "distilled_sigma_values": c["distilled_sigma_values"],
    }


def build_pipeline_runtime_overlay_from_contract() -> dict[str, Any]:
    """Return the full runtime overlay applied after upstream model config merging.

    LightX2V's current ``set_args2config`` removes every key present in the
    union of all request InputInfo dataclasses.  That union includes
    ``target_video_length``, ``resize_mode`` and ``aspect_ratio`` even though
    those are generator configuration fields.  This adapter therefore builds
    the effective config without that lossy filter and reapplies the frozen
    experiment contract after loading the base-model config.
    """
    c = load_distill_runtime_contract()
    keys = (
        "model_cls", "task", "infer_steps", "target_video_length",
        "target_height", "target_width", "fps", "sample_shift",
        "sample_guide_scale", "enable_cfg", "seq_parallel", "parallel",
        "dim", "num_heads", "in_dim", "ffn_dim", "freq_dim",
        "num_layers", "out_dim", "eps", "model_type", "text_len",
        "use_image_encoder", "denoising_step_list", "boundary_step_index",
        "boundary", "aspect_ratio", "rope_type", "resize_mode",
        "audio_fps", "double_precision_rope", "norm_modulate_backend",
        "distilled_sigma_values", "cpu_offload", "offload_granularity",
        "t5_cpu_offload", "vae_cpu_offload",
    )
    overlay = {key: copy.deepcopy(c[key]) for key in keys}
    overlay.update({
        "self_attn_1_type": str(c["attention"]),
        "cross_attn_1_type": str(c["attention"]),
        "cross_attn_2_type": str(c["attention"]),
        "clip_encoder_offload": bool(c["image_encoder_offload"]),
    })
    return overlay


def _set_pipeline_infer_config_from_contract(pipe: Any) -> None:
    kw = create_generator_kwargs_from_contract()
    pipe.resize_mode = kw["resize_mode"]
    pipe.set_infer_config(
        attn_mode=kw["attn_mode"],
        rope_type=kw["rope_type"],
        infer_steps=kw["infer_steps"],
        num_frames=kw["num_frames"],
        height=kw["height"],
        width=kw["width"],
        guidance_scale=kw["guidance_scale"],
        sample_shift=kw["sample_shift"],
        fps=kw["fps"],
        aspect_ratio=kw["aspect_ratio"],
        boundary=kw["boundary"],
        boundary_step_index=kw["boundary_step_index"],
        denoising_step_list=kw["denoising_step_list"],
        audio_fps=kw["audio_fps"],
        double_precision_rope=kw["double_precision_rope"],
        norm_modulate_backend=kw["norm_modulate_backend"],
        distilled_sigma_values=kw["distilled_sigma_values"],
    )


def validate_effective_pipeline_config_against_contract(config: Mapping[str, Any]) -> None:
    overlay = build_pipeline_runtime_overlay_from_contract()
    missing = sorted(key for key in overlay if key not in config)
    if missing:
        raise KeyError(f"effective LightX2V pipeline config missing keys: {missing}")
    mismatches = []
    for key, expected in overlay.items():
        actual = config[key]
        if actual != expected:
            mismatches.append(f"{key}: expected={expected!r} actual={actual!r}")
    if mismatches:
        raise ValueError("effective LightX2V pipeline config drift: " + "; ".join(mismatches))
    stride = tuple(int(x) for x in config["vae_stride"])
    if len(stride) != 3 or (int(config["target_video_length"]) - 1) % stride[0] != 0:
        raise ValueError(
            f"invalid target_video_length/vae_stride: "
            f"frames={config['target_video_length']} stride={stride}"
        )
    if int(config["dim"]) // int(config["num_heads"]) != 128:
        raise ValueError("unexpected Wan2.2 attention head size")


def build_effective_lightx2v_pipeline_config(pipe: Any) -> MutableMapping[str, Any]:
    """Build the exact config passed to the LightX2V runner, without loading it."""
    from lightx2v.utils.set_config import auto_calc_config, get_default_config

    _set_pipeline_infer_config_from_contract(pipe)
    config = get_default_config()
    # Deliberately preserve all generator-level pipeline attributes.  Do not use
    # upstream set_args2config(), whose ALL_INPUT_INFO_KEYS filter removes
    # target_video_length/resize_mode/aspect_ratio in this LightX2V revision.
    config.update({
        key: copy.deepcopy(value)
        for key, value in vars(pipe).items()
        if value is not None
    })
    # Apply once before auto-calc so mandatory shape fields are present when
    # LightX2V validates VAE stride, then again after model-config merging so
    # the frozen experiment contract wins over upstream defaults.
    overlay = build_pipeline_runtime_overlay_from_contract()
    config.update(copy.deepcopy(overlay))
    config = auto_calc_config(config)
    config.update(copy.deepcopy(overlay))
    validate_effective_pipeline_config_against_contract(config)
    return config


def initialize_lightx2v_generator_from_contract(pipe: Any) -> MutableMapping[str, Any]:
    """Initialize the production runner from the same effective config dry-run uses."""
    from lightx2v.utils.set_config import set_parallel_config
    from lightx2v.utils.utils import validate_config_paths
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    config = build_effective_lightx2v_pipeline_config(pipe)
    validate_config_paths(config)
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        if platform_device is None:
            raise RuntimeError(f"unregistered LightX2V platform: {os.getenv('PLATFORM', 'cuda')}")
        platform_device.init_parallel_env()
        set_parallel_config(config)
    pipe.runner = pipe._init_runner(config)
    if not hasattr(pipe.runner, "config"):
        raise RuntimeError("LightX2V runner did not expose effective config")
    validate_effective_pipeline_config_against_contract(pipe.runner.config)
    return config

def validate_pipeline_arguments_against_contract(
    *, height: int, width: int, num_frames: int, fps: int, attention: str,
    cpu_offload: bool, offload_granularity: str, t5_cpu_offload: bool,
    vae_cpu_offload: bool, image_encoder_offload: bool = False,
) -> dict[str, Any]:
    c = load_distill_runtime_contract()
    actual = {
        "target_height": int(height), "target_width": int(width),
        "target_video_length": int(num_frames), "fps": int(fps),
        "attention": str(attention), "cpu_offload": bool(cpu_offload),
        "offload_granularity": str(offload_granularity),
        "t5_cpu_offload": bool(t5_cpu_offload),
        "image_encoder_offload": bool(image_encoder_offload),
        "vae_cpu_offload": bool(vae_cpu_offload),
    }
    for key, value in actual.items():
        if value != c[key]:
            raise ValueError(
                f"LightX2V runtime argument violates frozen contract: "
                f"{key} expected={c[key]!r} actual={value!r}"
            )
    return c
DEFAULT_HIGH_NOISE_FILENAME = str(DISTILL_RUNTIME_CONTRACT["high_noise_filename"])
DEFAULT_LOW_NOISE_FILENAME = str(DISTILL_RUNTIME_CONTRACT["low_noise_filename"])
OFFICIAL_HIGH_NOISE_SHA256 = str(DISTILL_RUNTIME_CONTRACT["high_noise_sha256"])
OFFICIAL_LOW_NOISE_SHA256 = str(DISTILL_RUNTIME_CONTRACT["low_noise_sha256"])
EXPERTS = ("high_noise_model", "low_noise_model")

_TARGET_RE = re.compile(
    r"^blocks\.(\d+)\.(?:self_attn|cross_attn)\.(?:q|k|v|o)$"
    r"|^blocks\.(\d+)\.ffn\.(?:0|2)$"
)
_BLOCK_RE = re.compile(r"^(blocks\.)\d+(\.)")

_expert_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lightx2v_hif4_expert", default=None
)
_step_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "lightx2v_hif4_step", default=None
)
_block_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "lightx2v_hif4_block", default=None
)

_PATCHED = False
_ORIGINALS: dict[str, Any] = {}
_ACTIVE_OBSERVER: "WeightInputObserver | None" = None
_ACTIVE_RUNTIME: "HIF4CacheRuntime | None" = None


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_distill_weight_layout(
    *,
    base_model_dir: str | Path,
    high_noise_ckpt: str | Path,
    low_noise_ckpt: str | Path,
    check_hash: bool = False,
    verify_official_hash: bool = False,
    expected_high_sha256: str = OFFICIAL_HIGH_NOISE_SHA256,
    expected_low_sha256: str = OFFICIAL_LOW_NOISE_SHA256,
) -> dict[str, Any]:
    base = Path(base_model_dir).expanduser().resolve()
    high = Path(high_noise_ckpt).expanduser().resolve()
    low = Path(low_noise_ckpt).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(base)
    if not high.is_file():
        raise FileNotFoundError(high)
    if not low.is_file():
        raise FileNotFoundError(low)
    if high.name != DEFAULT_HIGH_NOISE_FILENAME:
        raise ValueError(
            f"high-noise filename must be {DEFAULT_HIGH_NOISE_FILENAME!r}, got {high.name!r}"
        )
    if low.name != DEFAULT_LOW_NOISE_FILENAME:
        raise ValueError(
            f"low-noise filename must be {DEFAULT_LOW_NOISE_FILENAME!r}, got {low.name!r}"
        )
    if high.parent.name != "high_noise_model":
        raise ValueError(f"high-noise weight must be under a high_noise_model directory: {high}")
    if low.parent.name != "low_noise_model":
        raise ValueError(f"low-noise weight must be under a low_noise_model directory: {low}")
    result: dict[str, Any] = {
        "artifact": "lightx2v_wan22_i2v_distill_weight_layout",
        "base_model_dir": str(base),
        "high_noise_ckpt": str(high),
        "low_noise_ckpt": str(low),
        "high_noise_filename": high.name,
        "low_noise_filename": low.name,
        "high_noise_bytes": high.stat().st_size,
        "low_noise_bytes": low.stat().st_size,
        "model_cls": MODEL_CLASS,
        "scheduler": SCHEDULER_CLASS,
        "denoising_step_list": list(DENOISING_STEP_LIST),
        "boundary_step_index": BOUNDARY_STEP_INDEX,
        "expected_high_noise_sha256": str(expected_high_sha256),
        "expected_low_noise_sha256": str(expected_low_sha256),
    }
    if check_hash or verify_official_hash:
        high_hash = sha256_file(high)
        low_hash = sha256_file(low)
        result["high_noise_sha256"] = high_hash
        result["low_noise_sha256"] = low_hash
        result["hash_source"] = "computed"
        if verify_official_hash:
            if high_hash.lower() != str(expected_high_sha256).lower():
                raise ValueError(
                    f"high-noise SHA256 mismatch: expected={expected_high_sha256} actual={high_hash}"
                )
            if low_hash.lower() != str(expected_low_sha256).lower():
                raise ValueError(
                    f"low-noise SHA256 mismatch: expected={expected_low_sha256} actual={low_hash}"
                )
            result["official_hash_verified"] = True
    else:
        result["high_noise_sha256"] = str(expected_high_sha256)
        result["low_noise_sha256"] = str(expected_low_sha256)
        result["hash_source"] = "expected_not_recomputed"
    return result


def normalize_module_name(weight_name: str, block_index: int | None = None) -> str:
    name = str(weight_name)
    for prefix in ("diffusion_model.", "model.diffusion_model.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if block_index is not None and name.startswith("blocks."):
        name = _BLOCK_RE.sub(rf"\g<1>{int(block_index)}\g<2>", name, count=1)
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name


def is_target_module(module_name: str) -> bool:
    return _TARGET_RE.fullmatch(module_name) is not None


def module_block_index(module_name: str) -> int:
    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] != "blocks":
        raise ValueError(f"not a block module: {module_name}")
    return int(parts[1])


class WeightInputObserver(Protocol):
    def observe(
        self,
        *,
        expert: str,
        step_index: int,
        block_index: int,
        module_name: str,
        activation: torch.Tensor,
    ) -> None: ...


@dataclass
class ActivationAbsmaxObserver:
    module_name_filter: str = ""
    keep_stats_on_gpu: bool = False

    def __post_init__(self) -> None:
        self._filter = re.compile(self.module_name_filter) if self.module_name_filter else None
        self.stats: dict[str, dict[str, dict[str, Any]]] = {
            "high_noise_model": {},
            "low_noise_model": {},
        }
        self.step_calls: dict[str, dict[str, int]] = {
            "high_noise_model": {},
            "low_noise_model": {},
        }

    def observe(
        self,
        *,
        expert: str,
        step_index: int,
        block_index: int,
        module_name: str,
        activation: torch.Tensor,
    ) -> None:
        if expert not in self.stats or not is_target_module(module_name):
            return
        if self._filter is not None and self._filter.search(module_name) is None:
            return
        x = activation.detach()
        if x.ndim == 0:
            return
        x2 = x.reshape(-1, x.shape[-1])
        current = x2.abs().amax(dim=0).to(torch.float32)
        if not self.keep_stats_on_gpu:
            current = current.cpu()
        rec = self.stats[expert].setdefault(
            module_name,
            {"s": None, "channels_dim": -1, "num_calls": 0, "step_calls": {}},
        )
        rec["s"] = current if rec["s"] is None else torch.maximum(rec["s"], current)
        rec["num_calls"] = int(rec["num_calls"]) + 1
        key = str(int(step_index))
        rec["step_calls"][key] = int(rec["step_calls"].get(key, 0)) + 1
        self.step_calls[expert][key] = int(self.step_calls[expert].get(key, 0)) + 1

    def export(self, *, save_dtype: torch.dtype = torch.float32) -> dict[str, Any]:
        output: dict[str, Any] = {
            "__meta__": {
                "version": 2,
                "model": "wan22_i2v_lightx2v_distill",
                "model_cls": MODEL_CLASS,
                "scheduler": SCHEDULER_CLASS,
                "stat_type": "per_channel_absmax",
                "channels_dim": -1,
                "sampling_steps": 4,
                "denoising_step_list": list(DENOISING_STEP_LIST),
                "boundary_step_index": BOUNDARY_STEP_INDEX,
                "step_calls": self.step_calls,
            },
            "high_noise_model": {},
            "low_noise_model": {},
        }
        for expert in ("high_noise_model", "low_noise_model"):
            for name, rec in self.stats[expert].items():
                s = rec["s"]
                if s is None:
                    continue
                output[expert][name] = {
                    "s": s.detach().to(device="cpu", dtype=save_dtype).contiguous(),
                    "channels_dim": -1,
                    "num_calls": int(rec["num_calls"]),
                    "step_calls": dict(rec["step_calls"]),
                }
        return output


@dataclass
class HessianObserver:
    target_expert: str
    target_block: int
    smooth_records: Mapping[str, Mapping[str, Any]]
    activation_domain: str = "smooth_bf16_noqdq"
    sample_tokens: int = 1024
    hessian_device: str = "cpu"
    seed: int = 20260717

    def __post_init__(self) -> None:
        if self.target_expert not in ("high_noise_model", "low_noise_model"):
            raise ValueError(self.target_expert)
        if self.activation_domain != "smooth_bf16_noqdq":
            raise ValueError(f"only smooth_bf16_noqdq is supported, got {self.activation_domain}")
        self.device = torch.device(self.hessian_device)
        self.records: dict[str, dict[str, Any]] = {}
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(self.seed))

    def _sample_rows(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])
        limit = int(self.sample_tokens)
        if limit <= 0 or rows.shape[0] <= limit:
            return rows
        # Deterministic stratified selection avoids a prefix-only temporal bias.
        positions = torch.linspace(0, rows.shape[0] - 1, steps=limit, device=rows.device)
        return rows.index_select(0, positions.round().long())

    def observe(
        self,
        *,
        expert: str,
        step_index: int,
        block_index: int,
        module_name: str,
        activation: torch.Tensor,
    ) -> None:
        if expert != self.target_expert or int(block_index) != int(self.target_block):
            return
        if not is_target_module(module_name):
            return
        expert_records = self.smooth_records.get(expert, {})
        smooth = expert_records.get(module_name)
        if not isinstance(smooth, Mapping):
            raise KeyError(f"missing smooth record for {expert}.{module_name}")
        scale = smooth.get("scale", smooth.get("smooth_scale"))
        if not torch.is_tensor(scale):
            raise TypeError(f"missing smooth scale for {expert}.{module_name}")
        x = activation.detach()
        scale_dev = scale.to(device=x.device, dtype=torch.float32)
        x_s = (x.to(torch.float32) / scale_dev.view(*([1] * (x.ndim - 1)), -1)).to(x.dtype)
        rows = self._sample_rows(x_s).to(device=self.device, dtype=torch.float32)
        rec = self.records.get(module_name)
        if rec is None:
            n = int(rows.shape[-1])
            rec = {
                "hessian": torch.zeros((n, n), device=self.device, dtype=torch.float32),
                "nsamples": 0,
                "num_calls": 0,
                "step_calls": {},
                "in_features": n,
            }
            self.records[module_name] = rec
        rec["hessian"].addmm_(rows.t(), rows)
        rec["nsamples"] = int(rec["nsamples"]) + int(rows.shape[0])
        rec["num_calls"] = int(rec["num_calls"]) + 1
        key = str(int(step_index))
        rec["step_calls"][key] = int(rec["step_calls"].get(key, 0)) + 1

    def export(self, *, save_dtype: torch.dtype = torch.float32) -> dict[str, Any]:
        domain = "smooth_bf16_noqdq"
        payload: dict[str, Any] = {
            "__meta__": {
                "artifact": "wan22_lightx2v_distill_hessian_shard",
                "schema_version": 490,
                "model_cls": MODEL_CLASS,
                "scheduler": SCHEDULER_CLASS,
                "expert": self.target_expert,
                "block": int(self.target_block),
                "domain": domain,
                "main_input": "bf16_teacher_then_smooth_no_activation_qdq",
                "teacher_trajectory": "lightx2v_distill_bf16_4step",
                "sampling_steps": 4,
                "denoising_step_list": list(DENOISING_STEP_LIST),
                "boundary_step_index": BOUNDARY_STEP_INDEX,
                "sample_tokens": int(self.sample_tokens),
            },
            "high_noise_model": {},
            "low_noise_model": {},
        }
        for name, rec in self.records.items():
            nsamples = int(rec["nsamples"])
            if nsamples <= 0:
                continue
            matrix = (rec["hessian"] / float(nsamples)).to(device="cpu", dtype=save_dtype)
            payload[self.target_expert][name] = {
                "hessian": matrix.contiguous(),
                "nsamples": nsamples,
                "num_calls": int(rec["num_calls"]),
                "step_calls": dict(rec["step_calls"]),
                "in_features": int(rec["in_features"]),
                "normalization": "mean_xtx",
                "domain": domain,
            }
        return payload


class HIF4CacheRuntime:
    def __init__(
        self,
        cache_path: str | Path,
        *,
        activation_qdq_override: bool | None = None,
        keep_fp_regex: str = "",
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"cache must contain dict: {self.cache_path}")
        self.meta = dict(payload.get("__meta__", {}))
        self.records: dict[str, Mapping[str, Any]] = {}
        for expert in ("high_noise_model", "low_noise_model"):
            records = payload.get(expert, {})
            if not isinstance(records, Mapping):
                raise TypeError(f"cache[{expert}] must be mapping")
            self.records[expert] = records
        self.activation_qdq_override = activation_qdq_override
        self.keep_fp = re.compile(keep_fp_regex) if keep_fp_regex else None
        self._resident_key: tuple[str, int, str, str] | None = None
        self._resident_weights: dict[str, torch.Tensor] = {}
        self.counters = {
            "linear_forward_calls": 0,
            "activation_qdq_calls": 0,
            "weight_transfers": 0,
            "cache_misses": 0,
        }

    def has_record(self, expert: str, module_name: str) -> bool:
        return module_name in self.records.get(expert, {})

    def _record(self, expert: str, module_name: str) -> Mapping[str, Any] | None:
        if self.keep_fp is not None and self.keep_fp.search(f"{expert}.{module_name}"):
            return None
        rec = self.records.get(expert, {}).get(module_name)
        if rec is None:
            self.counters["cache_misses"] += 1
        return rec

    def _weight(
        self,
        *,
        expert: str,
        block_index: int,
        module_name: str,
        record: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (expert, int(block_index), str(device), str(dtype))
        if key != self._resident_key:
            self._resident_weights.clear()
            self._resident_key = key
            if device.type == "cuda":
                torch.cuda.empty_cache()
        tensor = self._resident_weights.get(module_name)
        if tensor is None:
            source = record.get("weight_qdq")
            if not torch.is_tensor(source):
                raise TypeError(f"cache record missing weight_qdq: {expert}.{module_name}")
            tensor = source.to(device=device, dtype=dtype, non_blocking=False).contiguous()
            self._resident_weights[module_name] = tensor
            self.counters["weight_transfers"] += 1
        return tensor

    def apply(
        self,
        mm_weight: Any,
        input_tensor: torch.Tensor,
        *,
        expert: str,
        step_index: int,
        block_index: int,
        module_name: str,
    ) -> torch.Tensor | None:
        record = self._record(expert, module_name)
        if record is None:
            return None
        x = input_tensor
        scale = record.get("smooth_scale")
        if torch.is_tensor(scale):
            scale_dev = scale.to(device=x.device, dtype=torch.float32)
            x = (x.to(torch.float32) / scale_dev.view(*([1] * (x.ndim - 1)), -1)).to(x.dtype)
        use_a4 = bool(record.get("activation_qdq", True))
        if self.activation_qdq_override is not None:
            use_a4 = bool(self.activation_qdq_override)
        if use_a4:
            x = native_hif4_activation_qdq(x, force_fp32=bool(record.get("force_fp32", True))).to(x.dtype)
            self.counters["activation_qdq_calls"] += 1
        weight = self._weight(
            expert=expert,
            block_index=block_index,
            module_name=module_name,
            record=record,
            device=x.device,
            dtype=x.dtype,
        )
        bias = None
        if hasattr(mm_weight, "_get_actual_bias"):
            bias = mm_weight._get_actual_bias()
        elif hasattr(mm_weight, "bias"):
            bias = mm_weight.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        self.counters["linear_forward_calls"] += 1
        return F.linear(x, weight, bias)

    def report(self) -> dict[str, Any]:
        return {
            "__meta__": {
                "artifact": "lightx2v_hif4_runtime_report",
                "cache_path": str(self.cache_path),
                "cache_meta": self.meta,
                "model_cls": MODEL_CLASS,
                "scheduler": SCHEDULER_CLASS,
                "denoising_step_list": list(DENOISING_STEP_LIST),
                "boundary_step_index": BOUNDARY_STEP_INDEX,
            },
            **self.counters,
        }

    def write_report(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.report(), indent=2, ensure_ascii=False), encoding="utf-8")


def _patch_lightx2v() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct

    _ORIGINALS["mm_apply"] = MMWeight.apply
    _ORIGINALS["infer_block"] = WanTransformerInfer.infer_block
    _ORIGINALS["multi_infer"] = MultiDistillModelStruct.infer

    def patched_multi_infer(self: Any, inputs: Any) -> Any:
        step = int(self.scheduler.step_index)
        expert = "high_noise_model" if step < int(self.boundary_step_index) else "low_noise_model"
        token_e = _expert_ctx.set(expert)
        token_s = _step_ctx.set(step)
        try:
            return _ORIGINALS["multi_infer"](self, inputs)
        finally:
            _step_ctx.reset(token_s)
            _expert_ctx.reset(token_e)

    def patched_infer_block(self: Any, block: Any, x: Any, pre_infer_out: Any) -> Any:
        index = int(getattr(self, "block_idx", 0))
        token = _block_ctx.set(index)
        try:
            return _ORIGINALS["infer_block"](self, block, x, pre_infer_out)
        finally:
            _block_ctx.reset(token)

    def patched_mm_apply(self: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        expert = _expert_ctx.get()
        step = _step_ctx.get()
        block = _block_ctx.get()
        if expert is None or step is None or block is None:
            return _ORIGINALS["mm_apply"](self, input_tensor)
        name = normalize_module_name(getattr(self, "weight_name", ""), block)
        if not is_target_module(name):
            return _ORIGINALS["mm_apply"](self, input_tensor)
        if _ACTIVE_OBSERVER is not None:
            _ACTIVE_OBSERVER.observe(
                expert=expert,
                step_index=step,
                block_index=block,
                module_name=name,
                activation=input_tensor,
            )
        if _ACTIVE_RUNTIME is not None:
            result = _ACTIVE_RUNTIME.apply(
                self,
                input_tensor,
                expert=expert,
                step_index=step,
                block_index=block,
                module_name=name,
            )
            if result is not None:
                return result
        return _ORIGINALS["mm_apply"](self, input_tensor)

    MultiDistillModelStruct.infer = patched_multi_infer
    WanTransformerInfer.infer_block = patched_infer_block
    MMWeight.apply = patched_mm_apply
    _PATCHED = True


def activate_lightx2v_hooks(
    *,
    observer: WeightInputObserver | None = None,
    runtime: HIF4CacheRuntime | None = None,
) -> None:
    global _ACTIVE_OBSERVER, _ACTIVE_RUNTIME
    _patch_lightx2v()
    _ACTIVE_OBSERVER = observer
    _ACTIVE_RUNTIME = runtime


def deactivate_lightx2v_hooks() -> None:
    global _ACTIVE_OBSERVER, _ACTIVE_RUNTIME
    _ACTIVE_OBSERVER = None
    _ACTIVE_RUNTIME = None


def build_lightx2v_distill_pipeline(
    *,
    base_model_dir: str | Path,
    high_noise_ckpt: str | Path,
    low_noise_ckpt: str | Path,
    device_id: int = 0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    fps: int = 16,
    attention: str = "torch_sdpa",
    cpu_offload: bool = True,
    offload_granularity: str = "block",
    t5_cpu_offload: bool = True,
    vae_cpu_offload: bool = False,
) -> Any:
    contract = validate_pipeline_arguments_against_contract(
        height=height, width=width, num_frames=num_frames, fps=fps,
        attention=attention, cpu_offload=cpu_offload,
        offload_granularity=offload_granularity,
        t5_cpu_offload=t5_cpu_offload, vae_cpu_offload=vae_cpu_offload,
        image_encoder_offload=False,
    )
    validate_distill_weight_layout(
        base_model_dir=base_model_dir,
        high_noise_ckpt=high_noise_ckpt,
        low_noise_ckpt=low_noise_ckpt,
        check_hash=False,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_id))
    from lightx2v import LightX2VPipeline

    pipe = LightX2VPipeline(
        model_path=str(Path(base_model_dir).resolve()),
        model_cls=str(contract["model_cls"]),
        task=str(contract["task"]),
        high_noise_original_ckpt=str(Path(high_noise_ckpt).resolve()),
        low_noise_original_ckpt=str(Path(low_noise_ckpt).resolve()),
    )
    pipe.enable_offload(
        cpu_offload=bool(contract["cpu_offload"]),
        offload_granularity=str(contract["offload_granularity"]),
        text_encoder_offload=bool(contract["t5_cpu_offload"]),
        image_encoder_offload=False,
        vae_offload=bool(contract["vae_cpu_offload"]),
    )
    initialize_lightx2v_generator_from_contract(pipe)
    scheduler = pipe.runner.scheduler
    if scheduler.__class__.__name__ != SCHEDULER_CLASS:
        raise RuntimeError(
            f"expected scheduler {SCHEDULER_CLASS}, got {scheduler.__class__.__name__}"
        )
    if tuple(int(x) for x in scheduler.denoising_step_list) != DENOISING_STEP_LIST:
        raise RuntimeError(f"unexpected denoising steps: {scheduler.denoising_step_list}")
    if int(scheduler.boundary_step_index) != BOUNDARY_STEP_INDEX:
        raise RuntimeError(f"unexpected boundary_step_index={scheduler.boundary_step_index}")
    return pipe


def load_canonical_samples(path: str | Path, *, start: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"canonical manifest must be list: {source}")
    selected = data[int(start) :]
    if limit is not None:
        selected = selected[: int(limit)]
    output: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=int(start)):
        if not isinstance(item, Mapping):
            raise TypeError(f"manifest[{index}] must be object")
        prompt = str(item.get("prompt", "")).strip()
        paths = item.get("img_paths")
        if not prompt or not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
            raise ValueError(f"invalid canonical sample at index {index}: {item}")
        image = Path(str(paths[0])).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        output.append(
            {
                "videoid": str(item.get("videoid", index)),
                "prompt": prompt,
                "image_path": str(image),
                "source_index": index,
            }
        )
    return output


def run_calibration_samples(
    pipe: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    scratch_dir: str | Path | None = None,
) -> None:
    owned_tmp = None
    if scratch_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="lightx2v_calib_")
        root = Path(owned_tmp.name)
    else:
        root = Path(scratch_dir)
        root.mkdir(parents=True, exist_ok=True)
    try:
        for index, sample in enumerate(samples):
            save_path = root / f"{index:05d}_{sample['videoid']}.mp4"
            pipe.generate(
                seed=int(seed) + index,
                image_path=str(sample["image_path"]),
                prompt=str(sample["prompt"]),
                negative_prompt="",
                save_result_path=str(save_path),
            )
            try:
                save_path.unlink()
            except FileNotFoundError:
                pass
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()


__all__ = [
    "ActivationAbsmaxObserver",
    "BOUNDARY_STEP_INDEX",
    "DISTILL_RUNTIME_CONTRACT",
    "DEFAULT_HIGH_NOISE_FILENAME",
    "DEFAULT_LOW_NOISE_FILENAME",
    "OFFICIAL_HIGH_NOISE_SHA256",
    "OFFICIAL_LOW_NOISE_SHA256",
    "DENOISING_STEP_LIST",
    "EXPERTS",
    "HIF4CacheRuntime",
    "HessianObserver",
    "MODEL_CLASS",
    "SCHEDULER_CLASS",
    "activate_lightx2v_hooks",
    "build_lightx2v_distill_pipeline",
    "build_scheduler_config_from_contract",
    "create_generator_kwargs_from_contract",
    "deactivate_lightx2v_hooks",
    "is_target_module",
    "load_canonical_samples",
    "load_distill_runtime_contract",
    "module_block_index",
    "normalize_module_name",
    "run_calibration_samples",
    "sha256_file",
    "validate_distill_weight_layout",
    "validate_pipeline_arguments_against_contract",
]
