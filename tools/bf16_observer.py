"""Observe native D-BF16 FFN boundaries without replacing model arithmetic.

This creates a new capture series with uniform row sampling. It is not the old
D-Q R1 dataset. Structural checks do not prove observer no-op: a separate dense
trajectory comparison is required after the capture run completes.
"""
import contextvars
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


class BF16Observer:
    arm = "D-BF16"
    series = "WORKER_NEW_DBF16_UNIFORM_CAPTURE_20261020"
    required_tensors = {"P", "C", "R", "U", "Z", "H", "Y", "b2", "gate_broadcast", "Delta_writeback_operand", "X_next"}
    tensor_aliases = {"V2": "H"}
    alias_note = "BF16 analysis alias only: H is the native FFN2 input; no Q2 is executed"

    def __init__(self, root, *, source_id, seed, blocks=(0, 10, 20, 38), rows=512):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        if not blocks or any(block not in range(40) for block in blocks) or len(set(blocks)) != len(blocks):
            raise ValueError("Capture blocks must be unique native blocks 0..39")
        if not 1 <= rows <= 512:
            raise ValueError("Capture at most 512 rows per scope")
        self.blocks, self.rows = set(blocks), rows
        self.source_id, self.seed = source_id, seed
        self.active = contextvars.ContextVar("ffnds_bf16_capture", default=None)
        self.pending = None
        self.completed = []
        self.methods = []
        self.seen = set()
        self.status = "CAPTURE_IN_PROGRESS"

    def _manifest(self, error=None):
        data = {"status": self.status, "series": self.series,
                "arm": self.arm, "source_id": self.source_id, "seed": self.seed,
                "blocks": sorted(self.blocks), "steps": [0, 1, 2, 3],
                "expected_scopes": 4 * len(self.blocks), "completed_scopes": len(self.completed),
                "observer_noop": "PENDING_SEPARATE_SAME_INPUT_DENSE_TRAJECTORY_COMPARISON",
                "original_DQ_R1_replay": False,
                "capture_points": {"P": "native infer_ffn entry x, after self-attention residual",
                    "C": "native infer_ffn entry cross-attention output, before add_",
                    "R": "native x after original infer_ffn has added C",
                    "U": "native FFN0.apply input, after norm2/modulation",
                    "Z": "native FFN0.apply output including its bias once",
                    "H": "native FFN2.apply input, after native GELU(tanh)",
                    "Y": "native infer_ffn return, including FFN2 bias once",
                    "gate_broadcast": "native post_process gate, required to be row-constant",
                    "Delta_writeback_operand": "sampled Y times gate in native post_process dtype branch",
                    "X_next": "native post_process returned x"}, "scopes": self.completed}
        if error is not None:
            data["error"] = str(error)
        temp = self.root / "CAPTURE_MANIFEST.json.tmp"
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temp.replace(self.root / "CAPTURE_MANIFEST.json")

    def _new_scope(self, tag, x):
        step, block = tag["step"], tag["block"]
        if (step, block) in self.seen:
            raise RuntimeError(f"Duplicate native scope: {(step, block)}")
        grid = tag["grid"]
        flat_rows = x.reshape(-1, x.shape[-1]).shape[0]
        if len(grid) != 3 or math.prod(grid) != flat_rows:
            raise RuntimeError(f"Observed row/grid mismatch: {flat_rows}, {grid}")
        size = min(self.rows, flat_rows)
        key = f"FFNDS_RESTART_UNIFORM_V1|{self.source_id}|{self.seed}|{grid}"
        sampling_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        selected = np.sort(np.random.default_rng(sampling_seed).choice(flat_rows, size=size, replace=False)).tolist()
        indices = torch.tensor(selected, dtype=torch.long, device=x.device)
        return {"step": step, "block": block, "expert": tag["expert"], "grid": grid,
                "denoising_step": tag["denoising_step"], "sigma": tag["sigma"],
                "scheduler_timestep": tag["scheduler_timestep"],
                "row_count": size, "selected_rows": selected,
                "latent_t_h_w": [[i // (grid[1] * grid[2]), (i % (grid[1] * grid[2])) // grid[2], i % grid[2]] for i in selected],
                "sampling": "uniform_without_replacement_from_observed_flat_rows",
                "sampling_seed": sampling_seed, "inclusion_probabilities": [size / flat_rows] * size,
                "design_weights": [flat_rows / size] * size,
                "coordinate_provenance": "OBSERVED_NEW_WORKER_RUNTIME_GRID_AND_SELECTED_ROWS",
                "checks": {}, "_indices": indices, "_samples": {}}

    @staticmethod
    def _sample(scope, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        if flat.shape[0] != math.prod(scope["grid"]):
            raise RuntimeError("Tensor rows differ from the observed native grid")
        return flat.index_select(0, scope["_indices"])

    def _save_scope(self, scope):
        if set(scope["_samples"]) != self.required_tensors:
            raise RuntimeError(f"Incomplete native boundary fields: {set(scope['_samples'])}")
        if not all(scope["checks"].values()):
            raise RuntimeError(f"Native capture boundary check failed: {scope['checks']}")
        samples = scope["_samples"]
        if not all(bool(torch.isfinite(tensor).all()) for tensor in samples.values()):
            raise RuntimeError("Non-finite native capture value")
        path = self.root / f"step{scope['step']}_block{scope['block']:02d}.safetensors"
        save_file(samples, str(path))
        public = {key: value for key, value in scope.items() if not key.startswith("_")}
        public.update(tensor_file=str(path), tensor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      tensor_schema={key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in samples.items()},
                      tensor_keys=self.tensor_aliases,
                      V2_alias_note=self.alias_note,
                      Delta_status="derived from observed Y and gate in the native dtype branch; not a separately observed fused intermediate")
        self.completed.append(public)
        self.seen.add((scope["step"], scope["block"]))
        self._manifest()

    def __enter__(self):
        from HiFloat4.ptq.adapters.lightx2v_wan_distill import normalize_module_name
        from lightx2v.common.ops.mm.mm_weight import MMWeight
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
        old_mm, old_multi = MMWeight.apply, MultiDistillModelStruct.infer
        old_block, old_ffn, old_post = WanTransformerInfer.infer_block, WanTransformerInfer.infer_ffn, WanTransformerInfer.post_process
        if getattr(old_mm, "_ffnds_observer", False):
            raise RuntimeError("Another BF16 observer is already installed")
        observer = self

        def multi_infer(model, inputs):
            step = int(model.scheduler.step_index)
            if model.boundary_step_index != 2 or step not in range(4):
                raise RuntimeError("Capture requires the frozen four-step/two-expert contract")
            denoising_step = int(model.scheduler.denoising_step_list[step])
            if denoising_step != [1000, 750, 500, 250][step]:
                raise RuntimeError("Observed native timestep differs from frozen contract")
            token = observer.active.set({"step": step, "denoising_step": denoising_step,
                "sigma": float(model.scheduler.sigmas[step]),
                "scheduler_timestep": float(model.scheduler.timesteps[step]),
                "expert": "high_noise_model" if step < 2 else "low_noise_model"})
            try:
                return old_multi(model, inputs)
            finally:
                observer.active.reset(token)

        def infer_block(infer, block, x, pre):
            tag = observer.active.get()
            if tag is None:
                raise RuntimeError("Native block ran outside its observed expert/step")
            grid = [int(size) for size in pre.grid_sizes.tuple]
            token = observer.active.set({**tag, "block": int(infer.block_idx), "grid": grid})
            try:
                return old_block(infer, block, x, pre)
            finally:
                observer.active.reset(token)

        def infer_ffn(infer, phase, x, attn_out, shift, scale, gate=None):
            tag = observer.active.get()
            if tag["block"] not in observer.blocks:
                return old_ffn(infer, phase, x, attn_out, shift, scale, gate)
            scope = observer._new_scope(tag, x)
            if observer.pending is not None:
                raise RuntimeError("An earlier native FFN scope was not finalized")
            observer.pending = scope
            p, c = observer._sample(scope, x), observer._sample(scope, attn_out)
            scope["_expected_R"] = p.clone().add_(c)
            scope["_samples"].update(P=p.detach().cpu().contiguous(), C=c.detach().cpu().contiguous())
            scope["smooth_norm2_weight_branch"] = hasattr(phase, "smooth_norm2_weight")
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            result = old_ffn(infer, phase, x, attn_out, shift, scale, gate)
            actual_r = observer._sample(scope, x)
            scope["checks"].update(sampled_native_P_plus_C_matches_R=torch.equal(actual_r, scope.pop("_expected_R")),
                cpu_rng_unchanged=torch.equal(cpu_rng, torch.get_rng_state()),
                cuda_rng_unchanged=torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            scope["_samples"].update(R=actual_r.detach().cpu().contiguous(), Y=observer._sample(scope, result).detach().cpu().contiguous())
            return result

        def mm_apply(mm, input_tensor):
            scope = observer.pending
            tag = observer.active.get()
            name = normalize_module_name(mm.weight_name, tag.get("block")) if tag else ""
            targeted = scope is not None and name in [f"blocks.{scope['block']}.ffn.0", f"blocks.{scope['block']}.ffn.2"]
            if not targeted:
                return old_mm(mm, input_tensor)
            if mm.has_lora_branch or mm.has_diff:
                raise RuntimeError("This D-BF16 capture expects unmodified native weights")
            sample = observer._sample(scope, input_tensor)
            if name.endswith(".ffn.0"):
                scope["_samples"]["U"] = sample.detach().cpu().contiguous()
            else:
                scope["_samples"]["H"] = sample.detach().cpu().contiguous()
                scope["checks"]["sampled_gelu_tanh_matches_H"] = torch.equal(F.gelu(scope.pop("_z_gpu"), approximate="tanh"), sample)
                weight, bias = mm._get_actual_weight(), mm._get_actual_bias()
                scope["_samples"]["b2"] = torch.empty(0, dtype=weight.dtype) if bias is None else bias.detach().cpu().contiguous()
                if weight.dtype != torch.bfloat16 or list(weight.shape) != [13824, 5120]:
                    raise RuntimeError("Unexpected native BF16 W2 layout/dtype")
                scope["effective_W2"] = {"runtime_name": mm.weight_name, "runtime_shape": list(weight.shape),
                    "runtime_stride": list(weight.stride()), "dtype": str(weight.dtype),
                    "checkpoint_tensor_key": f"blocks.{scope['block']}.ffn.2.weight",
                    "checkpoint_to_runtime": "transpose [out,in] to [in,out] by the unmodified native loader",
                    "has_lora_or_diff": False, "bias_status": "EXPLICIT_NONE" if bias is None else "OBSERVED_NATIVE_BIAS",
                    "bias_dtype": None if bias is None else str(bias.dtype)}
            scope["_current_linear"] = name
            try:
                result = old_mm(mm, input_tensor)
            finally:
                scope.pop("_current_linear", None)
            if name.endswith(".ffn.0"):
                z = observer._sample(scope, result)
                scope["_samples"]["Z"] = z.detach().cpu().contiguous()
                scope["_z_gpu"] = z
            return result

        def post_process(infer, x, y, gate, pre_infer_out=None):
            scope = observer.pending
            if scope is None:
                return old_post(infer, x, y, gate, pre_infer_out)
            tag = observer.active.get()
            if (tag["step"], tag["block"]) != (scope["step"], scope["block"]):
                raise RuntimeError("Post-process capture escaped its native scope")
            r, yy = observer._sample(scope, x), observer._sample(scope, y)
            if gate.numel() != yy.shape[-1] or gate.squeeze().ndim != 1:
                raise RuntimeError(f"Expected a row-constant native gate; got {list(gate.shape)}")
            if infer.sensitive_layer_dtype != infer.infer_dtype:
                delta = yy.to(infer.sensitive_layer_dtype) * gate.squeeze()
                expected = r.to(infer.sensitive_layer_dtype) + delta
                scope["post_process_branch"] = "sensitive_add"
            else:
                delta = yy * gate.squeeze()
                expected = r.clone().add_(delta)
                scope["post_process_branch"] = "infer_add_"
            result = old_post(infer, x, y, gate, pre_infer_out)
            actual = observer._sample(scope, result)
            scope["checks"]["sampled_native_post_matches_derived_reference"] = torch.equal(actual, expected)
            scope["_samples"].update(gate_broadcast=gate.detach().cpu().contiguous(),
                Delta_writeback_operand=delta.detach().cpu().contiguous(), X_next=actual.detach().cpu().contiguous())
            observer._save_scope(scope)
            observer.pending = None
            return result

        mm_apply._ffnds_observer = True
        replacements = [(MMWeight, "apply", mm_apply), (MultiDistillModelStruct, "infer", multi_infer),
                        (WanTransformerInfer, "infer_block", infer_block), (WanTransformerInfer, "infer_ffn", infer_ffn),
                        (WanTransformerInfer, "post_process", post_process)]
        try:
            for cls, name, replacement in replacements:
                self.methods.append((cls, name, getattr(cls, name)))
                setattr(cls, name, replacement)
            self._manifest()
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for cls, name, original in reversed(self.methods):
            setattr(cls, name, original)
        self.methods.clear()

    def __exit__(self, exc_type, exc, tb):
        self._restore()
        if exc is not None:
            self.status = "FAILED_CAPTURE"
            self._manifest(exc)
            return False
        expected = {(step, block) for step in range(4) for block in self.blocks}
        if self.seen != expected or self.pending is not None:
            self.status = "INCOMPLETE_CAPTURE"
            self._manifest("Missing native scopes")
            raise RuntimeError(f"Capture incomplete: {self.seen ^ expected}")
        self.status = "PASS_CAPTURE_STRUCTURE_ONLY"
        self._manifest()
        return False
