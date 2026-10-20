#!/usr/bin/env python3
from __future__ import annotations

"""Collect one Wan expert/block curvature shard.

v4.8 makes the activation domain, token sampling, and aggregation semantics
explicit so reproduction and method experiments cannot silently share Hessians.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from math import gcd
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wan
from quant.gptq import GPTQHessianCollector
from quant.wan_smooth import _build_schedule, _smooth_scale_from_act_and_weight, _weight_absmax_in_channel
from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
from tools.generate_opens2v_eval import load_eval_samples, select_samples
from tools.ptq_wan_i2v_opens2v_svdquant import _load_single_image_sample, _prep_calib_item
from tools.validate_opens2v_manifest import validate_manifest, sha256_file
from wan.configs import WAN_CONFIGS

LOGGER = logging.getLogger("collect_wan_hif4_hessian_shard")
SKIP = ("norm", "time", "embed", "embedding", "modulation", "modulate", "adaln", "head", "final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-json", nargs="+", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--act-path", required=True, help="Wan act.pt used to reproduce Smooth scales.")
    p.add_argument("--smooth-alpha", type=float, default=0.5)
    p.add_argument("--smooth-eps", type=float, default=1e-8)
    p.add_argument("--expert", choices=["low_noise_model", "high_noise_model"], required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--module-regex", default="")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--target-frames", type=int, default=61)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sampling-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    p.add_argument(
        "--activation-domain", choices=["a4", "a16"], default="a16",
        help="Generic GPTQ/MagR default is no activation QDQ; A14B W4A4 reproduction passes a4 explicitly.",
    )
    p.add_argument(
        "--aggregation",
        choices=["aggregate", "aggregate_samples_timesteps", "last_call", "single_sample_single_timestep"],
        default="aggregate_samples_timesteps",
    )
    p.add_argument("--reference-sample-index", type=int, default=0)
    p.add_argument("--reference-timestep", type=int, default=-1, help="<0 chooses by --reference-timestep-policy")
    p.add_argument("--reference-timestep-policy", choices=["middle", "first", "last"], default="middle")
    p.add_argument("--token-sampling", choices=["deterministic", "head", "random"], default="deterministic")
    p.add_argument("--sample-tokens", type=int, default=1024, help="0 keeps all tokens in every hook call")
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--hessian-device", default="cuda:0")
    p.add_argument("--save-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--max-hessian-gb", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hash-images", action="store_true")
    return p.parse_args()


def select_modules(model: nn.Module, block: int, module_regex: str) -> list[str]:
    prefix = f"blocks.{block}."
    pat = re.compile(module_regex) if module_regex else None
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not name.startswith(prefix):
            continue
        low = name.lower()
        if any(token in low for token in SKIP):
            continue
        if pat is not None and pat.search(name) is None:
            continue
        names.append(name)
    return sorted(names)


def find_module(root: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def save_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def deterministic_token_subset(x: torch.Tensor, count: int, key: str) -> torch.Tensor:
    x2 = x.detach().reshape(-1, int(x.shape[-1]))
    n = int(x2.shape[0])
    if count <= 0 or n <= count:
        return x2
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "little") % n
    stride = 1 + (int.from_bytes(digest[8:16], "little") % max(1, n - 1))
    while gcd(stride, n) != 1:
        stride += 1
        if stride >= n:
            stride = 1
    idx = (start + torch.arange(count, device=x2.device, dtype=torch.long) * stride) % n
    return x2.index_select(0, idx)


def choose_tokens(x: torch.Tensor, *, count: int, max_tokens: int, mode: str, key: str) -> torch.Tensor:
    x2 = x.detach().reshape(-1, int(x.shape[-1]))
    limit = count if count > 0 else 0
    if max_tokens > 0:
        limit = max_tokens if limit <= 0 else min(limit, max_tokens)
    if limit <= 0 or int(x2.shape[0]) <= limit:
        return x2
    if mode == "head":
        return x2[:limit]
    if mode == "random":
        idx = torch.randperm(x2.shape[0], device=x2.device)[:limit]
        return x2.index_select(0, idx)
    return deterministic_token_subset(x2, limit, key)


def main() -> None:
    args = parse_args()
    if args.aggregation == "aggregate":
        args.aggregation = "aggregate_samples_timesteps"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    eval_paths = [Path(x).resolve() for x in args.eval_json]
    image_root = Path(args.image_root).resolve()
    manifest = validate_manifest(
        eval_paths, image_root, expected=args.num_samples, start=args.start,
        hash_images=args.hash_images,
    )

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=args.device_id,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=True,
    )
    high = pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    low = pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    model = getattr(pipe, args.expert)
    names = select_modules(model, args.block, args.module_regex)
    if not names:
        raise RuntimeError(f"no Linear modules selected for {args.expert} block={args.block}")

    estimate = sum(int(find_module(model, n).in_features) ** 2 * 4 for n in names)
    estimate_gb = estimate / (1024 ** 3)
    LOGGER.info("selected modules=%d estimated fp32 Hessian memory=%.3f GiB", len(names), estimate_gb)
    if estimate_gb > args.max_hessian_gb:
        raise MemoryError(
            f"estimated Hessian memory {estimate_gb:.3f} GiB exceeds --max-hessian-gb={args.max_hessian_gb}"
        )

    act_path = Path(args.act_path).resolve()
    act_payload = torch.load(act_path, map_location="cpu", weights_only=False)
    if not isinstance(act_payload, dict) or args.expert not in act_payload:
        raise TypeError(f"invalid Wan act artifact: {act_path}")
    expert_act = act_payload[args.expert]

    collectors: dict[str, GPTQHessianCollector] = {}
    smooth_scales: dict[str, torch.Tensor] = {}
    hook_calls: dict[str, int] = {name: 0 for name in names}
    sampled_tokens: dict[str, int] = {name: 0 for name in names}
    runtime_context: dict[str, Any] = {"sample_id": "", "sample_index": -1, "timestep": -1, "forward_call": 0, "collect": True}
    handles = []

    for name in names:
        module = find_module(model, name)
        assert isinstance(module, nn.Linear)
        if name not in expert_act or "s" not in expert_act[name]:
            raise KeyError(f"act.pt missing {args.expert}.{name}")
        act_s = expert_act[name]["s"].detach().float()
        w_s = _weight_absmax_in_channel(module.weight.detach(), eps=args.smooth_eps)
        smooth_scales[name] = _smooth_scale_from_act_and_weight(
            act_s=act_s.to(w_s.device), w_s=w_s,
            alpha=args.smooth_alpha, eps=args.smooth_eps,
        ).detach().cpu().float().contiguous()
        collectors[name] = GPTQHessianCollector(
            in_features=module.in_features, device=args.hessian_device,
            sample_tokens=None, max_tokens=None,
        )

        def hook(_module: nn.Module, inp: tuple[Any, ...], module_name: str = name) -> None:
            if args.aggregation == "single_sample_single_timestep" and not runtime_context["collect"]:
                return
            x = next((obj for obj in inp if torch.is_tensor(obj)), None)
            if x is None:
                return
            scale = smooth_scales[module_name].to(device=x.device, dtype=x.dtype)
            view = [1] * x.ndim
            view[-1] = scale.numel()
            x_smooth = x / scale.view(*view)
            if args.activation_domain == "a4":
                x_main = native_hif4_activation_qdq(x_smooth, force_fp32=True).to(dtype=x.dtype)
            else:
                x_main = x_smooth
            key = (
                f"v48|seed={args.seed}|module={module_name}|sample={runtime_context['sample_id']}|"
                f"t={runtime_context['timestep']}|call={runtime_context['forward_call']}"
            )
            selected = choose_tokens(
                x_main, count=args.sample_tokens, max_tokens=args.max_tokens,
                mode=args.token_sampling, key=key,
            )
            collector = collectors[module_name]
            if args.aggregation == "last_call":
                collector.H.zero_()
                collector.nsamples = 0
            collector.add_batch(selected)
            hook_calls[module_name] += 1
            sampled_tokens[module_name] += int(selected.reshape(-1, selected.shape[-1]).shape[0])

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=False))

    samples = load_eval_samples(eval_paths, image_root)
    samples = select_samples(samples, args.start, args.num_samples)
    ordered_ids = [str(s.videoid) for s in samples]
    if ordered_ids != manifest["ordered_sample_ids"]:
        raise RuntimeError("manifest validation/order mismatch before calibration")

    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    reference_sample_index: int | None = None
    reference_sample_id: str | None = None
    reference_timestep: int | None = None
    reference_candidates: list[int] = []
    if args.aggregation == "single_sample_single_timestep":
        if not 0 <= args.reference_sample_index < len(samples):
            raise ValueError(
                f"--reference-sample-index={args.reference_sample_index} outside selected sample range 0..{len(samples)-1}"
            )
        reference_sample_index = int(args.reference_sample_index)
        reference_sample_id = ordered_ids[reference_sample_index]
        ref_scheduler, ref_timesteps = _build_schedule(
            sample_solver=args.sample_solver, sampling_steps=args.sampling_steps,
            shift=args.shift, num_train_timesteps=pipe.num_train_timesteps, device=pipe.device,
        )
        reference_candidates = [
            int(t.item()) for t in ref_timesteps
            if ((float(t.item()) >= boundary) == (args.expert == "high_noise_model"))
        ]
        del ref_scheduler, ref_timesteps
        if not reference_candidates:
            raise RuntimeError(f"no active timesteps for {args.expert}")
        if args.reference_timestep >= 0:
            reference_timestep = min(reference_candidates, key=lambda t: abs(t - int(args.reference_timestep)))
        elif args.reference_timestep_policy == "first":
            reference_timestep = reference_candidates[0]
        elif args.reference_timestep_policy == "last":
            reference_timestep = reference_candidates[-1]
        else:
            midpoint = 0.5 * (min(reference_candidates) + max(reference_candidates))
            reference_timestep = min(reference_candidates, key=lambda t: (abs(t - midpoint), reference_candidates.index(t)))
        LOGGER.info(
            "single-sample/single-timestep reference: expert=%s sample_index=%d sample_id=%s timestep=%d candidates=%s",
            args.expert, reference_sample_index, reference_sample_id, reference_timestep, reference_candidates,
        )

    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)
    completed = 0
    active_timesteps: list[int] = []
    active_forward_calls = 0
    try:
        for index, sample in enumerate(samples, start=1):
            if args.aggregation == "single_sample_single_timestep" and (index - 1) != reference_sample_index:
                continue
            LOGGER.info("sample %d/%d id=%s", index, len(samples), sample.videoid)
            raw = _load_single_image_sample(
                image_path=str(sample.used_image), prompt=sample.prompt,
                target_frames=args.target_frames, target_size=(args.height, args.width),
            )
            item = _prep_calib_item(pipe, raw)
            latent = item["latent"].to(pipe.device, dtype=pipe.param_dtype)
            y = item["y"].to(pipe.device, dtype=pipe.param_dtype)
            context = [t.to(pipe.device, dtype=pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver, sampling_steps=args.sampling_steps,
                shift=args.shift, num_train_timesteps=pipe.num_train_timesteps,
                device=pipe.device,
            )
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                for call_index, t in enumerate(timesteps):
                    t_scalar = float(t.item())
                    active = high if t_scalar >= boundary else low
                    should_collect = (
                        args.aggregation != "single_sample_single_timestep"
                        or (
                            (index - 1) == reference_sample_index
                            and int(t_scalar) == reference_timestep
                            and active is model
                        )
                    )
                    runtime_context.update(
                        sample_id=str(sample.videoid), sample_index=index - 1,
                        timestep=int(t_scalar), forward_call=call_index,
                        collect=bool(should_collect),
                    )
                    if active is model:
                        active_timesteps.append(int(t_scalar))
                        active_forward_calls += 1
                    timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                    noise_pred = active([latent], t=timestep, **model_args)[0]
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)
            completed += 1
            del raw, item, latent, y, context, model_args, scheduler, timesteps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.aggregation == "single_sample_single_timestep":
                break
    finally:
        for h in handles:
            h.remove()

    domain = "smooth_a4_qdq" if args.activation_domain == "a4" else "smooth_bf16_noqdq"
    main_input = "smooth_then_hif4_activation_qdq" if args.activation_domain == "a4" else "bf16_teacher_then_smooth_no_activation_qdq"
    dtype = save_dtype(args.save_dtype)
    payload: dict[str, Any] = {
        "__meta__": {
            "artifact": "wan_hif4_curvature_shard_v48",
            "schema_version": 48,
            "expert": args.expert,
            "block": args.block,
            "module_regex": args.module_regex,
            "num_calib_samples": completed,
            "start": args.start,
            "sample_tokens_per_call": args.sample_tokens,
            "max_tokens_per_call": args.max_tokens,
            "token_sampling": args.token_sampling,
            "aggregation": args.aggregation,
            "reference_sample_index": reference_sample_index,
            "reference_sample_id": reference_sample_id,
            "reference_timestep": reference_timestep,
            "reference_timestep_policy": args.reference_timestep_policy if reference_timestep is not None else None,
            "reference_timestep_candidates": reference_candidates,
            "normalization": "mean_xtx",
            "domain": domain,
            "activation_domain": args.activation_domain,
            "main_input": main_input,
            "teacher_trajectory": "bf16_unquantized_wan_expert",
            "act_path": str(act_path),
            "act_path_sha256": sha256_file(act_path),
            "smooth_alpha": args.smooth_alpha,
            "smooth_eps": args.smooth_eps,
            "sampling_steps": args.sampling_steps,
            "shift": args.shift,
            "target_frames": args.target_frames,
            "height": args.height,
            "width": args.width,
            "save_dtype": args.save_dtype,
            "estimated_fp32_hessian_gb": estimate_gb,
            "manifest_path": manifest["manifest_paths"],
            "manifest_sha256": manifest["manifest_sha256"],
            "ordered_sample_ids": manifest["ordered_sample_ids"],
            "ordered_sample_ids_sha256": manifest["ordered_sample_ids_sha256"],
            "resolved_image_paths_sha256": manifest["resolved_image_paths_sha256"],
            "expert_forward_calls": active_forward_calls,
            "expert_timesteps": active_timesteps,
            "seed": args.seed,
        },
        "low_noise_model": {}, "high_noise_model": {},
    }
    for name, collector in collectors.items():
        if collector.nsamples <= 0:
            raise RuntimeError(f"no curvature samples collected for {args.expert}.{name}")
        rec = collector.export(normalize=True)
        rec["hessian"] = rec["hessian"].to(dtype=dtype)
        rec.update({
            "normalization": "mean_xtx",
            "domain": domain,
            "main_input": main_input,
            "aggregation": args.aggregation,
            "hook_calls": hook_calls[name],
            "sampled_tokens_total": sampled_tokens[name],
            "retained_tokens": int(collector.nsamples),
        })
        payload[args.expert][name] = rec

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {
        "output": str(output), "expert": args.expert, "block": args.block,
        "module_count": len(payload[args.expert]), "num_samples": completed,
        "domain": domain, "aggregation": args.aggregation,
        "manifest_sha256": manifest["manifest_sha256"],
        "file_size_bytes": output.stat().st_size,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WAN_HIF4_CURVATURE_SHARD_V48_OK")


if __name__ == "__main__":
    main()
