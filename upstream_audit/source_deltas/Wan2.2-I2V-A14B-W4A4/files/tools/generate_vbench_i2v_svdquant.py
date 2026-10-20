#!/usr/bin/env python3
"""Generate Wan I2V videos for VBench-I2V evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
# BEGIN HIFX4 WAN ADAPTER IMPORT
from HiFloat4.ptq.adapters.wan_i2v import (
    add_wan_hif4_preprocess_args,
    setup_wan_hif4_preprocess_from_args,
)
# END HIFX4 WAN ADAPTER IMPORT
from typing import Any, Dict, List, Set

def _wan_actstat_mark_module_tree(root, label):
    """Attach stable actstat root/path labels without touching nn.Module.__setattr__."""
    try:
        import torch.nn as nn
    except Exception:
        nn = None

    if root is None:
        return

    if nn is not None and not isinstance(root, nn.Module):
        return

    try:
        object.__setattr__(root, "_wan_actstat_root_label", str(label))
    except Exception:
        pass

    def rec(m):
        try:
            children = list(m.named_children())
        except Exception:
            children = []
        for name, child in children:
            try:
                object.__setattr__(child, "_wan_actstat_parent", m)
                object.__setattr__(child, "_wan_actstat_child_name", str(name))
            except Exception:
                pass
            rec(child)

    rec(root)


def _wan_actstat_label_pipeline(pipe):
    """Best-effort labels for Wan I2V components after pipeline construction."""
    if not bool(int(__import__("os").environ.get("WAN_ACTSTAT_ENABLE", "0"))):
        return

    candidates = [
        ("high_noise_model", ["high_noise_model", "high_noise_dit", "high_model"]),
        ("low_noise_model", ["low_noise_model", "low_noise_dit", "low_model"]),
        ("model", ["model", "transformer", "diffusion_model"]),
        ("vae", ["vae", "first_stage_model"]),
        ("text_encoder", ["text_encoder", "t5", "text_encoder_model"]),
    ]

    labeled = []
    for label, names in candidates:
        for name in names:
            if hasattr(pipe, name):
                obj = getattr(pipe, name)
                _wan_actstat_mark_module_tree(obj, label)
                labeled.append(f"{label}<={name}:{obj.__class__.__name__}")
                break

    print("[WAN_ACTSTAT] explicit pipeline labels: " + ", ".join(labeled), flush=True)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGGER = logging.getLogger("generate_vbench_i2v_svdquant")


DEFAULT_DIMENSIONS = [
    "i2v_subject",
    "i2v_background",
    "camera_motion",
]


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in ("yes", "true", "t", "1", "y"):
        return True
    if text in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate VBench-I2V samples with Wan I2V SVDQuant")
    p.add_argument("--full_info_json", type=str, default="VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json", help="VBench-I2V full info JSON.")
    p.add_argument("--vbench_root", type=str, default="VBench/vbench2_beta_i2v", help="VBench-I2V root directory.")
    p.add_argument("--ratio", type=str, default="16-9", help="VBench crop ratio folder, e.g. 1-1, 8-5, 7-4, 16-9.")
    p.add_argument("--image_folder", type=str, default="", help="Override image folder. Defaults to <vbench_root>/data/crop/<ratio>.")
    p.add_argument("--output_dir", type=str, default="outputs/vbench_quant", help="Directory for generated VBench videos.")
    p.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS, help="VBench dimensions to generate.")
    p.add_argument("--samples_per_prompt", type=int, default=5, help="Number of videos per prompt.")
    p.add_argument("--start", type=int, default=0, help="Start offset after filtering prompts.")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of filtered prompts to process.")
    p.add_argument("--seed", type=int, default=0, help="Base seed. Actual seed is seed + prompt_index * samples_per_prompt + sample_index.")
    p.add_argument("--overwrite", action="store_true", default=False, help="Regenerate existing mp4 files.")
    p.add_argument("--dry_run", action="store_true", default=False, help="List planned samples without loading Wan.")
    p.add_argument("--manifest", type=str, default="", help="JSONL manifest path. Defaults to <output_dir>/generation_manifest.jsonl.")

    p.add_argument("--mode", type=str, default="quant", choices=["bf16", "quant", "hif4", "mxfp4_rtn", "nvfp4_calibrate", "nvfp4_rtn"], help="Run full precision/bf16, SVDQuant PTQ, or HiF4 W4A4 RTN QDQ.")
    p.add_argument("--ckpt_dir", type=str, default="../Wan2.2-I2V-A14B-bf16", help="Wan checkpoint directory.")
    p.add_argument("--ptq_dir", type=str, default="/home/wjh/Wan2.2/outputs/gptq_ptq", help="PTQ artifact path for quant mode.")
    p.add_argument("--size", type=str, default="480*832", help="Wan size key.")
    p.add_argument("--frame_num", type=int, default=61, help="Number of video frames.")
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampler.")
    p.add_argument("--sample_steps", type=int, default=40, help="Sampling steps.")
    p.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    p.add_argument("--sample_guide_scale", type=float, default=None, help="CFG scale.")
    p.add_argument("--device_id", type=int, default=1, help="CUDA device id.")
    p.add_argument("--offload_model", type=_str2bool, default=True, help="Offload model between submodules.")
    p.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    p.add_argument("--convert_model_dtype", action="store_true", default=True, help="Convert model dtype like Wan defaults.")
    p.add_argument("--low_keep_fp_blocks", type=str, default="", help="Low expert block indices to keep BF16.")
    p.add_argument("--high_keep_fp_blocks", type=str, default="", help="High expert block indices to keep BF16.")
    p.add_argument("--low_keep_fp_modules", type=str, default="", help="Comma-separated low expert module names to keep BF16.")
    p.add_argument("--high_keep_fp_modules", type=str, default="", help="Comma-separated high expert module names to keep BF16.")
    p.add_argument("--keep_fp_module_regex", type=str, default="", help="Regex for modules to keep BF16 in both experts.")
    p.add_argument("--low_keep_fp_module_regex", type=str, default="", help="Regex for low expert modules to keep BF16.")
    p.add_argument("--high_keep_fp_module_regex", type=str, default="", help="Regex for high expert modules to keep BF16.")
    p.add_argument("--print_replaced_modules", action="store_true", default=False, help="Log final quant/BF16 module lists.")
    p.add_argument("--act_policy_json", type=str, default="outputs/act_policy.json", help="Optional MXFP4 activation policy JSON.")
    p.add_argument("--act_scale_method", type=str, default="", choices=["", "ocp_floor", "safe_ceil"], help="Override activation MXFP4 scale method.")
    p.add_argument("--act_exp_offset", type=int, default=0, help="Override activation exponent offset when nonzero.")
    p.add_argument("--act_policy_print", action="store_true", default=False, help="Print activation policy setup summary.")
    p.add_argument("--freeze_condition_latent", action="store_true", default=True, help="Use infer script experiment that resets condition latent only after the final step.")

    # HiF4 W4A4 RTN-QDQ path. This is fake-quant numeric simulation:
    #   BF16 checkpoint -> one-time HiF4 RTN QDQ weight -> online HiF4 activation QDQ -> BF16 F.linear.
    p.add_argument("--hif4_main_repo", type=str, default="", help="HiF4_Sp repo root containing HiFloat4.")
    p.add_argument("--hif4_include_regex", type=str, default="", help="Optional regex selecting Linear modules for HiF4.")
    p.add_argument("--hif4_exclude_regex", type=str, default="", help="Optional regex excluding Linear modules from HiF4.")
    p.add_argument("--hif4_no_activation_qdq", action="store_true", default=False, help="Disable online activation QDQ; gives HiF4 W4-only RTN path.")
    p.add_argument("--hif4_force_fp32", action="store_true", default=True, help="Call HiF4 QDQ with force_fp32=True.")
    p.add_argument("--hif4_dry_run", action="store_true", default=False, help="Load Wan, select/replace Linear modules, write report, then exit.")
    p.add_argument("--hif4_setup_only", action="store_true", default=False, help="Actually replace HiF4 Linear modules, write report, then exit before generation.")
    p.add_argument("--hif4_report_json", type=str, default="", help="Path to write HiF4 setup report JSON.")
    p.add_argument("--hif4_runtime_report_json", type=str, default="", help="Path to write HiF4 runtime report JSON after generation.")


    # MXFP4 RTN W4A4 direct fake-quant baseline.
    # BF16 checkpoint -> MXFP4 RTN weight QDQ + activation QDQ -> BF16 GEMM.
    p.add_argument("--mxfp4_rtn_include_regex", type=str, default=None)
    p.add_argument("--mxfp4_rtn_exclude_regex", type=str, default=None)
    p.add_argument("--mxfp4_rtn_no_activation_qdq", action="store_true")
    p.add_argument("--mxfp4_rtn_block_size", type=int, default=32)
    p.add_argument("--mxfp4_rtn_force_fp32", action="store_true", default=True)
    p.add_argument("--mxfp4_rtn_no_force_fp32", action="store_true")
    p.add_argument("--mxfp4_rtn_dry_run", action="store_true")
    p.add_argument("--mxfp4_rtn_report_json", type=str, default="")
    p.add_argument("--mxfp4_rtn_weight_scale_method", type=str, default="ocp_floor", choices=["ocp_floor", "safe_ceil"])
    p.add_argument("--mxfp4_rtn_act_scale_method", type=str, default="safe_ceil", choices=["ocp_floor", "safe_ceil"])
    p.add_argument("--mxfp4_rtn_runtime_report_json", type=str, default="")

    p.add_argument("--nvfp4_rtn_activation_scales", type=str, default="")
    p.add_argument("--nvfp4_rtn_include_regex", type=str, default=None)
    p.add_argument("--nvfp4_rtn_exclude_regex", type=str, default=None)
    p.add_argument("--nvfp4_rtn_no_activation_qdq", action="store_true")
    p.add_argument("--nvfp4_rtn_group_size", type=int, default=16)
    p.add_argument("--nvfp4_rtn_force_fp32", action="store_true", default=True)
    p.add_argument("--nvfp4_rtn_no_force_fp32", action="store_true")
    p.add_argument("--nvfp4_rtn_quant_device", type=str, default="cuda:0")
    p.add_argument("--nvfp4_rtn_chunk_rows", type=int, default=128)
    p.add_argument("--nvfp4_rtn_dry_run", action="store_true")
    p.add_argument("--nvfp4_rtn_setup_only", action="store_true")
    p.add_argument("--nvfp4_rtn_allow_missing_scales", action="store_true")
    p.add_argument("--nvfp4_rtn_collect_weight_error_stats", action="store_true")
    p.add_argument("--nvfp4_rtn_report_json", type=str, default="")
    p.add_argument("--nvfp4_rtn_runtime_report_json", type=str, default="")
    p.add_argument("--nvfp4_calibration_scales_output", type=str, default="")
    p.add_argument("--nvfp4_calibration_report_json", type=str, default="")
    p.add_argument("--nvfp4_calibration_include_regex", type=str, default=None)
    p.add_argument("--nvfp4_calibration_exclude_regex", type=str, default=None)
    p.add_argument("--nvfp4_calibration_margin", type=float, default=1.0)
    p.add_argument("--nvfp4_calibration_allow_unobserved", action="store_true")

    add_wan_hif4_preprocess_args(p)
    return p.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> None:
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-A14B"]
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def _load_vbench_items(path: Path, dimensions: Set[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    items = []
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"{path}[{idx}] must be dict, got {type(item).__name__}")
        item_dims = set(str(x) for x in item.get("dimension", []))
        if dimensions and not (item_dims & dimensions):
            continue
        prompt = str(item.get("prompt_en", "")).strip()
        image_name = str(item.get("image_name", "")).strip()
        if not prompt or not image_name:
            LOGGER.warning("Skipping invalid VBench item %d: prompt=%r image_name=%r", idx, prompt, image_name)
            continue
        key = (prompt, image_name)
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item["_source_index"] = idx
        items.append(item)
    return items


def _append_manifest(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _setup_quant_if_needed(pipe: Any, args: argparse.Namespace) -> None:
    if args.mode != "quant":
        return
    from quant.act_policy import (
        build_wan_i2v_timestep_schedule,
        load_act_policy_json,
        maybe_patch_model_forward_for_act_context,
        set_quant_modules_act_policy,
    )
    from tools.infer_wan_i2v_svdquant import (
        _apply_ptq_state_to_model,
        _load_i2v_ptq_states,
        _parse_keep_fp_blocks,
        _parse_keep_fp_modules,
        _print_quant_fp_module_summary,
        _regex_matched_modules,
        _restore_bf16_modules,
        _snapshot_bf16_modules,
    )

    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    low_keep_fp_blocks = _parse_keep_fp_blocks(args.low_keep_fp_blocks)
    high_keep_fp_blocks = _parse_keep_fp_blocks(args.high_keep_fp_blocks)
    low_keep_fp_modules = _parse_keep_fp_modules(args.low_keep_fp_modules, pipe.low_noise_model)
    high_keep_fp_modules = _parse_keep_fp_modules(args.high_keep_fp_modules, pipe.high_noise_model)
    low_keep_fp_modules.update(_regex_matched_modules(pipe.low_noise_model, args.keep_fp_module_regex))
    high_keep_fp_modules.update(_regex_matched_modules(pipe.high_noise_model, args.keep_fp_module_regex))
    low_keep_fp_modules.update(_regex_matched_modules(pipe.low_noise_model, args.low_keep_fp_module_regex))
    high_keep_fp_modules.update(_regex_matched_modules(pipe.high_noise_model, args.high_keep_fp_module_regex))
    LOGGER.info("low_noise_model keep-fp module requests expanded to %d modules", len(low_keep_fp_modules))
    LOGGER.info("high_noise_model keep-fp module requests expanded to %d modules", len(high_keep_fp_modules))

    low_bf16_snapshot = _snapshot_bf16_modules(pipe.low_noise_model, low_keep_fp_modules, "low_noise_model")
    high_bf16_snapshot = _snapshot_bf16_modules(pipe.high_noise_model, high_keep_fp_modules, "high_noise_model")
    _apply_ptq_state_to_model(
        pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=low_keep_fp_blocks,
    )
    _apply_ptq_state_to_model(
        pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=high_keep_fp_blocks,
    )
    _restore_bf16_modules(pipe.low_noise_model, low_bf16_snapshot, "low_noise_model")
    _restore_bf16_modules(pipe.high_noise_model, high_bf16_snapshot, "high_noise_model")

    act_policy = load_act_policy_json(args.act_policy_json) if args.act_policy_json else None
    low_policy_modules = set_quant_modules_act_policy(
        pipe.low_noise_model,
        "low_noise_model",
        act_policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    high_policy_modules = set_quant_modules_act_policy(
        pipe.high_noise_model,
        "high_noise_model",
        act_policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    if args.act_policy_print:
        LOGGER.info(
            "Activation policy configured: json=%s low_modules=%d high_modules=%d override_scale_method=%s exp_offset=%d",
            args.act_policy_json or "<none>",
            low_policy_modules,
            high_policy_modules,
            args.act_scale_method or "<state/default>",
            int(args.act_exp_offset),
        )

    timesteps, boundary_timestep, expert_timestep_schedule = build_wan_i2v_timestep_schedule(
        num_train_timesteps=pipe.num_train_timesteps,
        boundary=pipe.boundary,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        shift=args.sample_shift,
        device=pipe.device,
    )
    LOGGER.info(
        "Activation policy timestep schedule: solver=%s steps=%d boundary=%.3f high_steps=%d low_steps=%d first_t=%.3f last_t=%.3f",
        args.sample_solver,
        int(args.sample_steps),
        float(boundary_timestep),
        len(expert_timestep_schedule["high_timestep_list"]),
        len(expert_timestep_schedule["low_timestep_list"]),
        float(timesteps[0].detach().cpu().item()) if len(timesteps) else float("nan"),
        float(timesteps[-1].detach().cpu().item()) if len(timesteps) else float("nan"),
    )
    maybe_patch_model_forward_for_act_context(
        pipe.low_noise_model,
        "low",
        expert_timestep_schedule=expert_timestep_schedule,
    )
    maybe_patch_model_forward_for_act_context(
        pipe.high_noise_model,
        "high",
        expert_timestep_schedule=expert_timestep_schedule,
    )

    if args.print_replaced_modules:
        _print_quant_fp_module_summary(pipe.low_noise_model, "low_noise_model")
        _print_quant_fp_module_summary(pipe.high_noise_model, "high_noise_model")



def _setup_hif4_if_needed(pipe: Any, args: argparse.Namespace) -> None:
    if args.mode != "hif4":
        return

    import json
    import os
    import sys
    from pathlib import Path

    hif4_repo = args.hif4_main_repo or os.environ.get("HIF4_MAIN_REPO", "")
    if hif4_repo:
        hif4_root = Path(hif4_repo).expanduser().resolve()
        os.environ["HIF4_MAIN_REPO"] = str(hif4_root)
        for p in [
            hif4_root,
            hif4_root / "HiFloat4",
            hif4_root / "HiFloat4" / "hif4_gpu",
        ]:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    from quant.hif4_backend import (
        get_hif4_stats,
        hif4_backend_metadata,
        reset_hif4_stats,
    )
    from quant.hif4_linear import replace_linear_with_hif4

    reset_hif4_stats()

    LOGGER.info("Setting up HiF4 W4A4 RTN-QDQ path")
    LOGGER.info("HiF4 repo: %s", os.environ.get("HIF4_MAIN_REPO", "<not set>"))
    LOGGER.info("HiF4 include regex: %s", args.hif4_include_regex or "<default>")
    LOGGER.info("HiF4 exclude regex: %s", args.hif4_exclude_regex or "<default>")
    LOGGER.info("HiF4 activation QDQ: %s", not args.hif4_no_activation_qdq)
    LOGGER.info("HiF4 dry run: %s", args.hif4_dry_run)

    reports = {}

    for label, model in [
        ("low_noise_model", pipe.low_noise_model),
        ("high_noise_model", pipe.high_noise_model),
    ]:
        LOGGER.info("[%s] move to %s for one-time weight QDQ", label, pipe.device)
        model.to(pipe.device).eval().requires_grad_(False)

        report = replace_linear_with_hif4(
            model,
            include_regex=args.hif4_include_regex or None,
            exclude_regex=args.hif4_exclude_regex or None,
            activation_qdq=not args.hif4_no_activation_qdq,
            force_fp32=bool(args.hif4_force_fp32),
            dry_run=bool(args.hif4_dry_run and not args.hif4_setup_only),
            quant_device=str(pipe.device),
        )

        LOGGER.info(
            "[%s] HiF4 selected=%d skipped=%d dry_run=%s",
            label,
            report["selected_count"],
            report["skipped_count"],
            report["dry_run"],
        )

        reports[label] = {
            "selected_count": report["selected_count"],
            "skipped_count": report["skipped_count"],
            "selected": report["selected"],
            "skipped": report["skipped"],
        }

        if args.hif4_setup_only:
            import torch
            LOGGER.info("[%s] setup-only: offload model to CPU after replacement", label)
            model.cpu()
            torch.cuda.empty_cache()

    payload = {
        "mode": "hif4",
        "definition": "HiF4 W4A4 RTN-QDQ fake quant: BF16 checkpoint, one-time HiF4 weight QDQ, online HiF4 activation QDQ, BF16 F.linear",
        "metadata": hif4_backend_metadata(),
        "stats_after_setup": get_hif4_stats(),
        "reports": reports,
    }

    report_json = args.hif4_report_json
    if not report_json:
        report_json = str(Path(args.output_dir) / "hif4_setup_report.json")

    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("HiF4 setup report written to %s", report_json)
    LOGGER.info("HiF4 stats after setup: %s", get_hif4_stats())

    if args.hif4_dry_run or args.hif4_setup_only:
        if args.hif4_setup_only:
            LOGGER.info("HIF4_SETUP_ONLY_DONE")
        else:
            LOGGER.info("HIF4_DRY_RUN_DONE")
        raise SystemExit(0)


def _generate_one(pipe: Any, args: argparse.Namespace, image_path: Path, prompt: str, seed: int) -> Any:
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS

    img = Image.open(image_path).convert("RGB")
    if args.freeze_condition_latent:
        from tools.infer_wan_i2v_svdquant import _generate_i2v_freezing_condition_latent

        local_args = argparse.Namespace(**vars(args))
        local_args.prompt = prompt
        local_args.base_seed = seed
        return _generate_i2v_freezing_condition_latent(pipe, local_args, img)
    return pipe.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=seed,
        offload_model=args.offload_model,
    )



def _setup_mxfp4_rtn_if_needed(pipe: Any, args: argparse.Namespace) -> None:
    if getattr(args, "mode", None) != "mxfp4_rtn":
        return

    import json as _json
    import os as _os
    from pathlib import Path as _Path

    from quant.mxfp4_rtn_backend import get_mxfp4_rtn_stats, reset_mxfp4_rtn_stats
    from quant.mxfp4_rtn_linear import replace_linear_with_mxfp4_rtn

    reset_mxfp4_rtn_stats()

    runtime_report = getattr(args, "mxfp4_rtn_runtime_report_json", "")
    if runtime_report:
        _os.environ["MXFP4_RTN_RUNTIME_REPORT_JSON"] = str(runtime_report)

    include_regex = getattr(args, "mxfp4_rtn_include_regex", None)
    exclude_regex = getattr(args, "mxfp4_rtn_exclude_regex", None)
    activation_qdq = not bool(getattr(args, "mxfp4_rtn_no_activation_qdq", False))
    block_size = int(getattr(args, "mxfp4_rtn_block_size", 32))
    force_fp32 = bool(getattr(args, "mxfp4_rtn_force_fp32", True))
    if bool(getattr(args, "mxfp4_rtn_no_force_fp32", False)):
        force_fp32 = False
    weight_scale_method = getattr(args, "mxfp4_rtn_weight_scale_method", "ocp_floor")
    act_scale_method = getattr(args, "mxfp4_rtn_act_scale_method", "safe_ceil")

    report = {
        "mode": "mxfp4_rtn",
        "quant_scheme": "MXFP4_E2M1_E8M0_reference_rtn_qdq",
        "source_checkpoint": "bf16",
        "uses_ptq_stats": False,
        "uses_activation_policy": False,
        "uses_svd_branch": False,
        "uses_gptq": False,
        "compute_backend": "bf16_or_fp16_gemm_after_qdq",
        "activation_qdq": activation_qdq,
        "block_size": block_size,
        "force_fp32": force_fp32,
        "weight_scale_method": weight_scale_method,
        "act_scale_method": act_scale_method,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "experts": {},
    }

    for attr in ["low_noise_model", "high_noise_model"]:
        model = getattr(pipe, attr, None)
        if model is None:
            report["experts"][attr] = {"present": False}
            continue

        r = replace_linear_with_mxfp4_rtn(
            model,
            prefix=attr,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            activation_qdq=activation_qdq,
            block_size=block_size,
            force_fp32=force_fp32,
            weight_scale_method=weight_scale_method,
            act_scale_method=act_scale_method,
            dry_run=bool(getattr(args, "mxfp4_rtn_dry_run", False)),
        )
        r["present"] = True
        report["experts"][attr] = r

    report["stats_after_setup"] = get_mxfp4_rtn_stats()

    setup_report = getattr(args, "mxfp4_rtn_report_json", "")
    if setup_report:
        out = _Path(setup_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[MXFP4_RTN_SETUP]", _json.dumps(report["stats_after_setup"], ensure_ascii=False), flush=True)

    if bool(getattr(args, "mxfp4_rtn_dry_run", False)):
        raise SystemExit(0)



def _setup_nvfp4_calibration_if_needed(pipe: Any, args: argparse.Namespace) -> None:
    if getattr(args,"mode",None)!="nvfp4_calibrate": return
    from quant.nvfp4_calibration import NVFP4ActivationCalibrator
    scales=getattr(args,"nvfp4_calibration_scales_output","") or str(Path(args.output_dir)/"nvfp4_activation_scales.safetensors")
    report=getattr(args,"nvfp4_calibration_report_json","") or str(Path(args.output_dir)/"nvfp4_calibration_report.json")
    collector=NVFP4ActivationCalibrator(include_regex=getattr(args,"nvfp4_calibration_include_regex",None),
        exclude_regex=getattr(args,"nvfp4_calibration_exclude_regex",None),margin=float(getattr(args,"nvfp4_calibration_margin",1.0)))
    attaches={"low_noise_model":collector.attach(pipe.low_noise_model,prefix="low_noise_model"),
              "high_noise_model":collector.attach(pipe.high_noise_model,prefix="high_noise_model")}
    pipe._nvfp4_activation_calibrator=collector; pipe._nvfp4_calibration_scales_output=scales
    pipe._nvfp4_calibration_report_json=report; pipe._nvfp4_calibration_attach_reports=attaches
    LOGGER.info("NVFP4 calibration hooks attached: %s",attaches)


def _finalize_nvfp4_calibration_if_needed(pipe: Any,args: argparse.Namespace)->None:
    if getattr(args,"mode",None)!="nvfp4_calibrate": return
    collector=getattr(pipe,"_nvfp4_activation_calibrator",None)
    if collector is None: raise RuntimeError("NVFP4 calibration collector missing")
    report=collector.finalize(scales_output=pipe._nvfp4_calibration_scales_output,
        report_output=pipe._nvfp4_calibration_report_json,
        require_all_observed=not bool(getattr(args,"nvfp4_calibration_allow_unobserved",False)))
    report["attach_reports"]=getattr(pipe,"_nvfp4_calibration_attach_reports",{})
    Path(pipe._nvfp4_calibration_report_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    LOGGER.info("NVFP4_CALIBRATION_DONE selected=%d observed=%d scales=%d missing=%d",report["selected_count"],report["observed_count"],report["scale_count"],report["missing_count"])


def _setup_nvfp4_rtn_if_needed(pipe: Any,args: argparse.Namespace)->None:
    if getattr(args,"mode",None)!="nvfp4_rtn": return
    from quant.nvfp4_rtn_backend import get_nvfp4_rtn_stats,nvfp4_backend_metadata,reset_nvfp4_rtn_stats
    from quant.nvfp4_rtn_linear import load_activation_scale_map,replace_linear_with_nvfp4_rtn
    reset_nvfp4_rtn_stats(); activation_qdq=not bool(getattr(args,"nvfp4_rtn_no_activation_qdq",False))
    scales_path=getattr(args,"nvfp4_rtn_activation_scales","")
    if activation_qdq and not scales_path: raise ValueError("--nvfp4_rtn_activation_scales is required")
    scales=load_activation_scale_map(scales_path) if scales_path else {}
    runtime=getattr(args,"nvfp4_rtn_runtime_report_json","")
    if runtime: os.environ["NVFP4_RTN_RUNTIME_REPORT_JSON"]=runtime
    force=not bool(getattr(args,"nvfp4_rtn_no_force_fp32",False))
    report={"mode":"nvfp4_rtn","definition":"NVFP4 W4A4 RTN-QDQ accuracy baseline",
        "metadata":nvfp4_backend_metadata(),"source_checkpoint":"bf16","uses_ptq_optimization":False,
        "uses_gptq":False,"activation_qdq":activation_qdq,"activation_scales":scales_path,
        "activation_scale_count":len(scales),"selection_policy":"exact_match_to_successful_HiF4_selector",
        "group_size":int(args.nvfp4_rtn_group_size),"force_fp32":force,
        "quant_device":args.nvfp4_rtn_quant_device,"chunk_rows":int(args.nvfp4_rtn_chunk_rows),"experts":{}}
    for attr in ["low_noise_model","high_noise_model"]:
        r=replace_linear_with_nvfp4_rtn(getattr(pipe,attr),prefix=attr,activation_scales=scales,
            include_regex=args.nvfp4_rtn_include_regex,exclude_regex=args.nvfp4_rtn_exclude_regex,
            activation_qdq=activation_qdq,group_size=args.nvfp4_rtn_group_size,force_fp32=force,
            quant_device=args.nvfp4_rtn_quant_device,chunk_rows=args.nvfp4_rtn_chunk_rows,
            dry_run=args.nvfp4_rtn_dry_run,allow_missing_scales=args.nvfp4_rtn_allow_missing_scales,
            collect_error_stats=args.nvfp4_rtn_collect_weight_error_stats)
        report["experts"][attr]=r; LOGGER.info("[%s] NVFP4 selected=%d replaced=%d missing=%d",attr,r["selected"],r["replaced"],len(r["missing_activation_scales"]))
    report["stats_after_setup"]=get_nvfp4_rtn_stats()
    rp=args.nvfp4_rtn_report_json or str(Path(args.output_dir)/"nvfp4_rtn_setup_report.json")
    Path(rp).parent.mkdir(parents=True,exist_ok=True); Path(rp).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    if args.nvfp4_rtn_dry_run or args.nvfp4_rtn_setup_only:
        LOGGER.info("NVFP4_SETUP_ONLY_DONE"); raise SystemExit(0)


def _write_nvfp4_runtime_report_if_needed(args: argparse.Namespace,*,generated:int,skipped:int,failed:int,manifest_path:Path)->None:
    if getattr(args,"mode",None)!="nvfp4_rtn": return
    from quant.nvfp4_rtn_backend import get_nvfp4_rtn_stats,nvfp4_backend_metadata
    p=args.nvfp4_rtn_runtime_report_json or str(Path(args.output_dir)/"nvfp4_rtn_runtime_report.json")
    payload={"mode":"nvfp4_rtn","definition":"NVFP4 W4A4 RTN-QDQ fake-quant accuracy baseline",
        "metadata":nvfp4_backend_metadata(),"stats_after_generation":get_nvfp4_rtn_stats(),"generated":generated,
        "skipped":skipped,"failed":failed,"manifest":str(manifest_path),"output_dir":str(args.output_dir)}
    Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    LOGGER.info("NVFP4 runtime report written to %s",p)

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dimensions = set(str(x) for x in args.dimensions)
    full_info_path = Path(args.full_info_json)
    image_folder = Path(args.image_folder) if args.image_folder else Path(args.vbench_root) / "data" / "crop" / args.ratio
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "generation_manifest.jsonl"

    items = _load_vbench_items(full_info_path, dimensions)
    selected = items[int(args.start):]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    LOGGER.info(
        "Loaded VBench items: total_filtered=%d selected=%d dimensions=%s image_folder=%s output=%s",
        len(items),
        len(selected),
        sorted(dimensions),
        image_folder,
        output_dir,
    )
    if args.dry_run:
        for idx, item in enumerate(selected[:20]):
            LOGGER.info("dry-run[%d] image=%s prompt=%s", idx, image_folder / item["image_name"], item["prompt_en"])
        LOGGER.info("Dry run complete. Planned videos: %d", len(selected) * int(args.samples_per_prompt))
        return

    _resolve_defaults(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    _wan_actstat_label_pipeline(pipe)
    _setup_quant_if_needed(pipe, args)
    hif4_preprocess_result = setup_wan_hif4_preprocess_from_args(
        pipe, args, logger=LOGGER
    )
    if hif4_preprocess_result.get("handled", False):
        if bool(getattr(args, "hif4_setup_only", False)):
            LOGGER.info("HiF4 preprocessed setup-only completed; skipping video generation.")
            raise SystemExit(0)
    else:
        _setup_hif4_if_needed(pipe, args)
    _setup_mxfp4_rtn_if_needed(pipe, args)
    _setup_nvfp4_calibration_if_needed(pipe, args)
    _setup_nvfp4_rtn_if_needed(pipe, args)
    # Quant mode replaces Linear modules after pipeline construction.
    # Relabel again so newly inserted QuantLinear / QuantLinearWithBranch modules
    # get stable high_noise_model / low_noise_model paths for actstat.
    _wan_actstat_label_pipeline(pipe)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)

    generated = 0
    skipped = 0
    failed = 0
    for prompt_idx, item in enumerate(selected, start=int(args.start)):
        prompt = str(item["prompt_en"])
        image_path = image_folder / str(item["image_name"])
        if not image_path.exists():
            LOGGER.error("Missing VBench image: %s", image_path)
            failed += int(args.samples_per_prompt)
            continue
        for sample_idx in range(int(args.samples_per_prompt)):
            save_path = output_dir / f"{prompt}-{sample_idx}.mp4"
            seed = int(args.seed) + prompt_idx * int(args.samples_per_prompt) + sample_idx
            record: Dict[str, Any] = {
                "prompt": prompt,
                "sample_index": sample_idx,
                "seed": seed,
                "image_path": str(image_path),
                "image_name": item["image_name"],
                "dimensions": item.get("dimension", []),
                "image_type": item.get("image_type", ""),
                "source_index": item.get("_source_index"),
                "save_path": str(save_path),
                "mode": args.mode,
                "size": args.size,
                "frame_num": int(args.frame_num),
                "sample_steps": int(args.sample_steps),
                "sample_solver": args.sample_solver,
                "sample_shift": float(args.sample_shift),
                "sample_guide_scale": _jsonable(args.sample_guide_scale),
                "ratio": args.ratio,
            }
            if save_path.exists() and not args.overwrite:
                LOGGER.info("Skipping existing video: %s", save_path)
                record["status"] = "skipped_existing"
                _append_manifest(manifest_path, record)
                skipped += 1
                continue
            try:
                LOGGER.info(
                    "Generating [%d/%d] sample=%d seed=%d prompt=%s",
                    prompt_idx - int(args.start) + 1,
                    len(selected),
                    sample_idx,
                    seed,
                    prompt,
                )
                video = _generate_one(pipe, args, image_path, prompt, seed)
                if args.mode == "nvfp4_calibrate":
                    record["status"] = "calibrated"
                else:
                    save_video(
                        tensor=video[None],
                        save_file=str(save_path),
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    record["status"] = "ok"
                del video
                generated += 1
            except Exception as exc:
                LOGGER.exception("Failed generating prompt=%r sample=%d", prompt, sample_idx)
                record["status"] = "failed"
                record["error"] = repr(exc)
                failed += 1
            _append_manifest(manifest_path, record)

    LOGGER.info("Finished VBench generation: generated=%d skipped=%d failed=%d manifest=%s", generated, skipped, failed, manifest_path)

    _finalize_nvfp4_calibration_if_needed(pipe, args)
    _write_nvfp4_runtime_report_if_needed(args, generated=generated, skipped=skipped, failed=failed, manifest_path=manifest_path)

    if args.mode == "hif4":
        from quant.hif4_backend import get_hif4_stats, hif4_backend_metadata

        runtime_report_json = args.hif4_runtime_report_json
        if not runtime_report_json:
            runtime_report_json = str(Path(args.output_dir) / "hif4_runtime_report.json")

        payload = {
            "mode": "hif4",
            "definition": "HiF4 W4A4 RTN-QDQ fake quant runtime report: BF16 checkpoint, one-time HiF4 weight QDQ, online HiF4 activation QDQ, BF16 F.linear",
            "metadata": hif4_backend_metadata(),
            "stats_after_generation": get_hif4_stats(),
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "manifest": str(manifest_path),
            "output_dir": str(args.output_dir),
        }

        Path(runtime_report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(runtime_report_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        LOGGER.info("HiF4 runtime report written to %s", runtime_report_json)
        LOGGER.info("HiF4 runtime stats after generation: %s", get_hif4_stats())


if __name__ == "__main__":
    main()
