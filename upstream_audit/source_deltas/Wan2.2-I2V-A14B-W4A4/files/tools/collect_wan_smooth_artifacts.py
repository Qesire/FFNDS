#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import wan
from quant.wan_smooth import (
    WanActStatConfig,
    build_wan_smoothed_weights,
    calibrate_wan_acts,
    save_wan_act_stats,
    save_wan_smoothed_weights,
)
from tools.generate_opens2v_eval import load_eval_samples, select_samples
from tools.ptq_wan_i2v_opens2v_svdquant import _iter_opens2v_calib_items
from wan.configs import WAN_CONFIGS

LOGGER = logging.getLogger("collect_wan_smooth_artifacts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-json", nargs="+", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--target-frames", type=int, default=61)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    p.add_argument("--sampling-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--act-save-dtype", default="float32")
    p.add_argument("--wgt-save-dtype", default="bfloat16")
    p.add_argument("--module-name-filter", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eval_paths = [Path(x).resolve() for x in args.eval_json]
    samples = load_eval_samples(eval_paths, Path(args.image_root).resolve())
    samples = select_samples(samples, args.start, args.num_samples)
    if len(samples) != args.num_samples:
        raise RuntimeError(f"requested {args.num_samples} samples, got {len(samples)}")

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=args.device_id,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=True,
    )
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)

    calib_iter = _iter_opens2v_calib_items(
        pipe=pipe, samples=samples, target_frames=args.target_frames,
        target_size=(args.height, args.width),
    )
    stat_cfg = WanActStatConfig(
        max_calib_samples=args.num_samples,
        save_dtype=args.act_save_dtype,
        channels_dim=-1,
        module_name_filter=args.module_name_filter or None,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        progress_interval=1,
        show_timestep_progress=True,
    )
    stats = calibrate_wan_acts(pipe, calib_iter, config=stat_cfg)
    act_path = out / "act.pt"
    save_wan_act_stats(str(act_path), stats)
    wgt = build_wan_smoothed_weights(
        pipe=pipe, act_stats=stats, alpha=args.alpha, eps=args.eps,
        save_dtype=args.wgt_save_dtype,
    )
    wgt_path = out / "wgt.pt"
    save_wan_smoothed_weights(str(wgt_path), wgt)
    summary = {
        "act_path": str(act_path), "wgt_path": str(wgt_path),
        "num_samples": args.num_samples, "alpha": args.alpha,
        "high_modules": len(stats["high_noise_model"]),
        "low_modules": len(stats["low_noise_model"]),
        "act_bytes": act_path.stat().st_size,
        "wgt_bytes": wgt_path.stat().st_size,
    }
    (out / "smooth_collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["high_modules"] != 400 or summary["low_modules"] != 400:
        raise RuntimeError(f"expected 400 modules per expert, got {summary}")
    print("WAN_HIF4_SMOOTH_ARTIFACTS_OK")


if __name__ == "__main__":
    main()
