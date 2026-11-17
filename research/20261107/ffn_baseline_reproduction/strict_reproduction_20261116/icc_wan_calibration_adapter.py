"""Native Wan D4 operations for the fixed ICC author calibration driver."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image
import torch

from icc_calibration_driver import NativeAdapter
from icc_four_boundary_collector import FourBoundaryCollector, FourBoundaryTap, write_json_exclusive


LATENT = (16, 21, 58, 104)
FRAMES, HEIGHT, WIDTH = 81, 464, 832


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def tensor_identity(value):
    if not torch.isfinite(value).all().item():
        raise ValueError("nonfinite Wan tensor")
    cpu = value.detach().cpu().contiguous()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy()).hexdigest()}


def decode_source(record, ffmpeg):
    file = Path(record["source_video"])
    if file_sha256(file) != record["source_sha256"]:
        raise RuntimeError("source video bytes changed")
    if file_sha256(record["image_condition"]) != record["image_condition_sha256"]:
        raise RuntimeError("first-frame condition changed")
    if (record["frames"], record["height"], record["width"], record["transformed_fps"]) != (FRAMES, HEIGHT, WIDTH, 16):
        raise ValueError("unexpected frozen geometry")
    args = [str(ffmpeg), "-nostdin", "-v", "error", "-threads", "1", "-i", str(file), "-an",
            "-vf", record["ffmpeg_filter"], "-frames:v", str(FRAMES), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    decoded = subprocess.run(args, capture_output=True, check=True, timeout=60).stdout
    if len(decoded) != FRAMES * HEIGHT * WIDTH * 3 or hashlib.sha256(decoded).hexdigest() != record["transformed_rgb24_sha256"]:
        raise RuntimeError("frozen transformed RGB stream mismatch")
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(FRAMES, HEIGHT, WIDTH, 3)
    with Image.open(record['image_condition']) as image:
        if not np.array_equal(frames[0], np.asarray(image.convert('RGB'))):
            raise RuntimeError('first-frame condition differs from the transformed video')
    return frames


def seeded_noise(reference, seed, dtype):
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    return torch.randn(reference.shape, generator=generator, device=reference.device, dtype=dtype)


def encode_posterior(vae, frames, eta_seed):
    """Read raw parameters from the same native encode that returns normalized mean."""
    if vae.parallel or vae.use_tiling or vae.cpu_offload or vae.model.training:
        raise RuntimeError("expected reviewed eval, resident, single-device native VAE")
    dtype = vae.scale[0].dtype
    device = vae.scale[0].device
    video = torch.from_numpy(frames.copy()).to(device, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.div_(255).sub_(0.5).div_(0.5).to(dtype=dtype)
    captured = []

    def capture(_module, _args, output):
        if captured:
            raise RuntimeError("unexpected extra native VAE posterior operation")
        captured.append(output.detach())

    hook = vae.model.conv1.register_forward_hook(capture)
    try:
        normalized = vae.encode(video)
    finally:
        hook.remove()
    if len(captured) != 1 or tuple(normalized.shape) != LATENT:
        raise RuntimeError("native VAE did not expose the expected posterior/shape")
    raw_mu, raw_logvar = (part.squeeze(0) for part in captured[0].chunk(2, dim=1))
    if tuple(raw_mu.shape) != LATENT or raw_mu.dtype != normalized.dtype:
        raise RuntimeError("native raw VAE posterior differs from main precision or geometry")
    shift = vae.scale[0].view(-1, 1, 1, 1)
    inv_std = vae.scale[1].view(-1, 1, 1, 1)
    if not torch.equal((raw_mu - shift) * inv_std, normalized):
        raise RuntimeError("raw posterior and native normalized mean do not agree")
    metadata = {"native_normalized_mean": tensor_identity(normalized), "raw_mu": tensor_identity(raw_mu),
                "raw_logvar": tensor_identity(raw_logvar), "scale_shift": tensor_identity(shift),
                "scale_inverse_std": tensor_identity(inv_std)}
    posterior = None
    if eta_seed is not None:
        eta = seeded_noise(raw_mu, eta_seed, torch.float32)
        raw_sample = raw_mu.float() + torch.exp(0.5 * raw_logvar.float().clamp(-30, 20)) * eta
        posterior = ((raw_sample - shift.float()) * inv_std.float()).to(dtype=normalized.dtype)
        metadata.update(eta_seed=eta_seed, eta=tensor_identity(eta), posterior=tensor_identity(posterior))
    return normalized, posterior, metadata


@contextmanager
def observe_dispatch(model_class, tap, step):
    original = model_class.get_current_model_index
    calls = []

    def dispatch(model):
        result = original(model)
        expert = {0: "high", 1: "low"}[int(model.cur_model_index)]
        calls.append((int(model.scheduler.step_index), expert))
        tap.dispatch = expert
        return result

    model_class.get_current_model_index = dispatch
    try:
        yield
        if calls != [(step, tap.collector.expert)]:
            raise RuntimeError(f"unexpected model dispatch {calls!r}")
    finally:
        model_class.get_current_model_index = original


@contextmanager
def verify_native_blocks(owner, weight_type):
    original_block = owner.infer_block
    original_ffn = owner.infer_ffn

    def block(model, weights, x, pre_infer):
        if tuple(pre_infer.grid_sizes.tuple) != (21, 29, 52):
            raise RuntimeError('unexpected native ICC raster grid')
        return original_block(model, weights, x, pre_infer)

    def ffn(model, phase, x, attn, shift, scale, gate=None):
        if model._use_mxfp8_quant_fuse() or hasattr(phase, 'smooth_norm2_weight'):
            raise RuntimeError('unexpected fused or folded Dense pathway')
        for op, shape in ((phase.ffn_0, (5120, 13824)), (phase.ffn_2, (13824, 5120))):
            if type(op) is not weight_type or op.has_lora_branch or hasattr(op, 'weight_diff') or hasattr(op, 'bias_diff'):
                raise RuntimeError('not the reviewed unfused dense MMWeight')
            w, b = op._get_actual_weight(), op._get_actual_bias()
            if (w is None or b is None or tuple(w.shape) != shape or b.numel() != shape[1]
                    or w.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
                    or w.data_ptr() != op.weight_cuda_buffer.data_ptr() or b.data_ptr() != op.bias_cuda_buffer.data_ptr()):
                raise RuntimeError('active checkpoint operand binding mismatch')
        return original_ffn(model, phase, x, attn, shift, scale, gate)

    owner.infer_block, owner.infer_ffn = block, ffn
    try:
        yield
    finally:
        owner.infer_block, owner.infer_ffn = original_block, original_ffn


class WanCalibrationAdapter(NativeAdapter):
    """Called only after a compute guard and V2 source/checkpoint verification."""

    def __init__(self, runner, output, manifest, schedule, checkpoint_identities, wan_types):
        self.runner = runner
        self.scheduler = runner.scheduler
        self.output = Path(output)
        self.output.mkdir()
        self.manifest = manifest
        self.schedule = schedule
        self.checkpoint_identities = checkpoint_identities
        self.info_type, self.model_class, self.infer_class, self.weight_type = wan_types
        if type(runner.model) is not self.model_class or runner.model.scheduler is not self.scheduler:
            raise RuntimeError("unexpected dual-expert model or scheduler ownership")
        if runner.config["enable_cfg"] or runner.config.get("feature_caching") != "NoCaching":
            raise RuntimeError("CFG/cache configuration mismatch")
        if file_sha256(manifest["ffmpeg_path"]) != manifest["ffmpeg_sha256"]:
            raise RuntimeError("changed video decoder")

    def encode_source(self, record):
        source = record["source_cluster_id"]
        if not re.fullmatch(r"pe-video:\d+", source):
            raise ValueError("unexpected source cluster")
        source_dir = self.output / source.replace(":", "_")
        source_dir.mkdir()
        info = self.info_type(seed=42, prompt=record["prompt"], negative_prompt="", image_path=record["image_condition"])
        self.runner.input_info = info
        native = self.runner.run_input_encoder()
        image = native["image_encoder_output"]
        if tuple(info.latent_shape) != LATENT or image.get("clip_encoder_out") is not None:
            raise RuntimeError("wrong native I2V visual condition")
        visual = image["vae_encoder_out"]
        if tuple(visual.shape) != (20, *LATENT[1:]):
            raise RuntimeError("expected native 4 mask channels +16 I2V conditioning channels")
        empty_info = self.info_type(seed=42, prompt="", negative_prompt="", image_path=info.image_path)
        empty = self.runner.run_text_encoder(empty_info)
        text = {False: native["text_encoder_output"], True: empty}
        contexts = {flag: tensor_identity(output["context"]) for flag, output in text.items()}
        if contexts[False] == contexts[True]:
            raise RuntimeError("literal empty text and actual prompt encoded identically")
        frames = decode_source(record, self.manifest["ffmpeg_path"])
        eta_seed = record["posterior_noise_seed"] if source in self.manifest["clean_cal3_source_clusters"] else None
        mean, posterior, vae_receipt = encode_posterior(self.runner.vae_encoder, frames, eta_seed)
        self.scheduler.generator = None
        self.scheduler.prepare(seed=42, latent_shape=list(LATENT), image_encoder_output=image)
        # prepare initializes FP32; the installed step_pre converts to native BF16.
        if self.scheduler.latents.dtype != torch.float32 or self.scheduler.keep_latents_dtype_in_scheduler:
            raise RuntimeError('native scheduler prepare/step_pre precision contract changed')
        actual = [{"timestep": float(t), "sigma": float(s)} for t, s in zip(self.scheduler.timesteps, self.scheduler.sigmas)]
        if actual != self.schedule:
            raise RuntimeError("native calibration schedule changed")
        identity = {"source_video_sha256": record["source_sha256"], "image_condition_sha256": record["image_condition_sha256"],
                    "transformed_rgb24_sha256": record["transformed_rgb24_sha256"], "condition": tensor_identity(visual),
                    "prompt_context": contexts[False], "empty_context": contexts[True], "vae": vae_receipt}
        write_json_exclusive(source_dir / "ENCODING.json", identity)
        return {"directory": source_dir, "mean": mean, "posterior": posterior,
                "text": text, "image": image, "contexts": contexts, "identity": identity}

    def prepare_transaction(self, encoded, record, txn):
        clean = encoded["posterior"] if txn.branch == "posterior" else encoded["mean"]
        if clean is None:
            raise RuntimeError("posterior was required but not encoded")
        first_step = txn.indices[0]
        sig = self.schedule[first_step]["sigma"]
        epsilon = seeded_noise(clean, txn.forward_noise_seed, torch.float32)
        # Native flow rule in FP32, one BF16 cast at the calibrated scheduler injection.
        noised = self.scheduler.add_noise(clean.float(), epsilon, sig).to(torch.bfloat16)
        empty = not txn.use_empty_text if txn.branch == "text_flip" else txn.use_empty_text
        identity = {"noised": tensor_identity(noised), "epsilon": tensor_identity(epsilon),
                    "image": encoded["identity"]["condition"], "context": encoded["contexts"][empty],
                    "steps": [self.schedule[step] for step in txn.indices], "expert": txn.expert,
                    "checkpoint": self.checkpoint_identities[txn.expert], "empty_context": empty,
                    "noising_precision": "native add_noise with FP32 target and epsilon, then BF16 latent injection"}
        label = f"{txn.family}_{txn.expert}_{txn.branch}"
        return {"latents": noised, "identity": identity, "clean_identity": tensor_identity(clean),
                "inputs": {"text_encoder_output": encoded["text"][empty], "image_encoder_output": encoded["image"]},
                "pending": encoded["directory"] / (label + "_pending"), "final": encoded["directory"] / label}

    def same_native_input_and_conditions(self, left, right):
        return left["identity"] == right["identity"]

    def begin_transaction(self, prepared, txn):
        self.scheduler.latents = prepared["latents"]
        prepared["collector"] = FourBoundaryCollector(
            torch, prepared["pending"], txn.source, txn.expert, txn.family,
            identity={"transaction": txn.identifier, **prepared["identity"]},
        )

    def forward(self, prepared, txn, step, phase):
        self.scheduler.step_pre(step_index=step)
        model_input = tensor_identity(self.scheduler.latents)
        expected = prepared['cd_second_input'] if phase == 'hit' else prepared['identity']['noised']
        if model_input != expected:
            raise RuntimeError('step_pre changed the registered first input or native adjacent CD state')
        tap = FourBoundaryTap(prepared["collector"], phase, step)
        with verify_native_blocks(self.infer_class, self.weight_type):
            try:
                tap.install(self.infer_class)
                with observe_dispatch(self.model_class, tap, step):
                    self.runner.model.infer(prepared["inputs"])
            finally:
                tap.uninstall()
        flow = self.scheduler.noise_pred.detach().to("cpu", copy=True)
        if tuple(flow.shape) != LATENT or flow.dtype != torch.float32:
            raise RuntimeError('native Wan flow shape or output dtype changed')
        return flow, {"step": step, "phase": phase, "expert": txn.expert, "model_input": model_input,
                      "flow": tensor_identity(flow), **self.schedule[step]}

    def native_step_post(self, prepared, txn, first):
        if int(self.scheduler.step_index) != txn.indices[0]:
            raise RuntimeError("invalid CD adjacency")
        self.scheduler.step_post()
        if self.scheduler.latents.dtype != torch.bfloat16:
            raise RuntimeError('native CD scheduler did not preserve BF16 latent state')
        prepared["cd_second_input"] = tensor_identity(self.scheduler.latents)

    def finish_transaction(self, prepared, txn, responses):
        if len(responses) != len(txn.indices) or (txn.family == "CD" and "cd_second_input" not in prepared):
            raise RuntimeError("incomplete transaction")
        collector = prepared["collector"]
        collector.finish()
        if prepared["final"].exists():
            raise FileExistsError(prepared["final"])
        info = {"status": "COMMITTED", "transaction": txn.identifier, "directory": str(prepared["final"]),
                "initial": prepared["identity"], "clean": prepared["clean_identity"],
                "cd_second_input": prepared.get("cd_second_input"), "forward_flows": [tensor_identity(x) for x in responses]}
        write_json_exclusive(prepared["pending"] / "TRANSACTION.json", info)
        os.rename(prepared["pending"], prepared["final"])
        return info

    def compare_diagnostic(self, main, alternate, txn):
        a, b = main["committed"], alternate["committed"]
        stats = []
        for i in range(40):
            ordinary = torch.load(Path(a["directory"]) / f"stats_{i:02d}.pt", map_location="cpu", weights_only=True)
            variant = torch.load(Path(b["directory"]) / f"stats_{i:02d}.pt", map_location="cpu", weights_only=True)
            stats.append({"layer": i, **{
                boundary: {"normalized_rms": float((variant[boundary]["normalized"] - values["normalized"]).square().mean().sqrt()),
                           "magnitude_rms": float((variant[boundary]["magnitude"] - values["magnitude"]).square().mean().sqrt())}
                for boundary, values in ordinary.items()
            }})
        differences = [float((one.float() - two.float()).square().mean().sqrt())
                       for one, two in zip(main["responses"], alternate["responses"], strict=True)]
        receipt = {"transaction": txn.identifier, "main": a, "alternate": b,
                   "flow_rms": differences, "per_layer_statistics_differences": stats}
        write_json_exclusive(Path(b["directory"]) / "PAIRED_DIAGNOSTIC.json", receipt)
        return receipt

    def abort_transaction(self, prepared, txn):
        pending = prepared["pending"]
        if pending.is_dir():
            for file in pending.glob("anchor_*.pt"):
                file.unlink()
            write_json_exclusive(pending / "ABORTED.json", {"transaction": txn.identifier})
