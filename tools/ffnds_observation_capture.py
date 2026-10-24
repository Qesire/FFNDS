"""Passive dense D-Q discovery capture. Install after activate_lightx2v_hooks.

Copies selected rows; calls each original operation once; no extra FFN/QDQ.
This observer has NOT been accepted until plain/observed latent controls pass.
"""
import contextvars
from pathlib import Path

from ffnds_observation_common import FIELDS, row_plan, sha256, write_json


class DiscoveryCapture:
    def __init__(self, runtime, out, source_id, protocol, cache_identity):
        self.runtime, self.out = runtime, Path(out)
        self.source_id, self.protocol = source_id, protocol
        self.cache_identity = cache_identity
        self.context = contextvars.ContextVar("ffnds_discovery", default=None)
        self.scopes, self.weights = [], {}
        self.out.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        import torch
        from safetensors.torch import save_file
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.torch, self.save_file = torch, save_file
        self.adapter, self.infer = adapter, WanTransformerInfer
        if type(self.runtime) is not adapter.HIF4CacheRuntime:
            raise ValueError("Discovery requires the original dense HIF4CacheRuntime")
        if self.runtime.activation_qdq_override is not None or self.runtime.keep_fp is not None:
            raise ValueError("QDQ overrides are not the dense D-Q discovery reference")
        self.old_apply, self.old_qdq = self.runtime.apply, adapter.native_hif4_activation_qdq
        self.old_block, self.old_ffn = self.infer.infer_block, self.infer.infer_ffn
        self.old_post = self.infer.post_process
        owner = self

        def block(infer, weights, x, pre):
            token = owner.context.set({"grid": [int(v) for v in pre.grid_sizes.tuple]})
            try:
                result = owner.old_block(infer, weights, x, pre)
                state = owner.context.get()
                if state.get("active"):
                    raise RuntimeError("Selected FFN did not reach post_process")
                return result
            finally:
                owner.context.reset(token)

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            state = owner.context.get()
            step, bid = adapter._step_ctx.get(), int(infer.block_idx)
            selected = step in owner.protocol["steps"] and bid in owner.protocol["blocks"]
            if not selected:
                return owner.old_ffn(infer, phase, x, attn, shift, scale, gate)
            if state is None or state.get("active") or gate is None or gate.numel() != 5120:
                raise RuntimeError("Unexpected FFN context/gate layout")
            if getattr(infer, "use_compile", False):
                raise RuntimeError("Discovery observer requires non-compiled native inference")
            plan = row_plan(state["grid"], owner.source_id, owner.protocol["uniform_rows"])
            state.update(active=True, step=step, block=bid, plan=plan, samples={}, x_ref=x,
                         ids=torch.tensor(plan["rows"], device=x.device), seen=[], qseen=[])
            owner.put(state, "x_before_cross_add", x)
            owner.put(state, "cross_attention", attn)
            state["samples"]["gate"] = gate.detach().reshape(1, 5120).cpu().contiguous()
            result = owner.old_ffn(infer, phase, x, attn, shift, scale, gate)
            state["await_post"] = True
            state.pop("x_ref", None)
            return result

        def qdq(x, *, force_fp32=True):
            state = owner.context.get()
            slot = state.get("slot") if state is not None and state.get("active") else None
            if slot in (0, 2):
                if not force_fp32 or slot in state["qseen"]:
                    raise RuntimeError("Unexpected QDQ precision/count")
                owner.put(state, "q1_pre" if slot == 0 else "q2_pre", x)
            result = owner.old_qdq(x, force_fp32=force_fp32)
            if slot in (0, 2):
                owner.put(state, "q1_post" if slot == 0 else "q2_post", result.to(x.dtype))
                state["qseen"].append(slot)
            return result

        def apply(mm, x, *, expert, step_index, block_index, module_name):
            kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            state = owner.context.get()
            selected = state is not None and state.get("active") and module_name.endswith((".ffn.0", ".ffn.2"))
            if not selected:
                return owner.old_apply(mm, x, **kw)
            slot = 0 if module_name.endswith(".ffn.0") else 2
            if (step_index, block_index) != (state["step"], state["block"]):
                raise RuntimeError("Cross-position capture")
            expected_expert = "high_noise_model" if step_index < 2 else "low_noise_model"
            if expert != expected_expert or state["seen"] != ([] if slot == 0 else [0]):
                raise RuntimeError("Unexpected expert or FFN ordering")
            rec = owner.runtime.records[expert][module_name]
            expected = (13824, 5120) if slot == 0 else (5120, 13824)
            if tuple(rec["weight_qdq"].shape) != expected or not rec["activation_qdq"] or not rec["force_fp32"]:
                raise RuntimeError("Not full dense frozen D-Q")
            state["expert"], state["slot"] = expert, slot
            if slot == 0:
                owner.put(state, "residual_pre_ffn", state["x_ref"])
                owner.put(state, "modulated_input", x)
            else:
                owner.put(state, "gelu", x)
            bias = mm._get_actual_bias()
            if bias is None:
                raise RuntimeError("Frozen native FFN requires its original bias")
            state["samples"]["b1" if slot == 0 else "b2"] = bias.detach().cpu().contiguous()
            result = owner.old_apply(mm, x, **kw)
            owner.put(state, "z" if slot == 0 else "ffn_output", result)
            state["seen"].append(slot)
            state.pop("slot")
            return result

        def post(infer, x, y, gate, pre_infer_out=None):
            result = owner.old_post(infer, x, y, gate, pre_infer_out)
            state = owner.context.get()
            if state is not None and state.get("active") and state.get("await_post"):
                owner.put(state, "residual_post_ffn", result)
                owner.finish(state)
                state["active"] = False
            return result

        self.runtime.apply, adapter.native_hif4_activation_qdq = apply, qdq
        self.infer.infer_block, self.infer.infer_ffn, self.infer.post_process = block, ffn, post
        return self

    def put(self, state, name, value):
        if name in state["samples"]:
            raise RuntimeError("Duplicate observation: " + name)
        if value.shape[-1] != FIELDS[name] or value.numel() // value.shape[-1] != __import__("math").prod(state["grid"]):
            raise RuntimeError("Native row/width mismatch: " + name)
        # Copies before subsequent in-place residual operations.
        state["samples"][name] = value.detach().reshape(-1, value.shape[-1]).index_select(
            0, state["ids"]).cpu().contiguous()

    def weight_binding(self, expert, block):
        key = f"{expert}_block{block:02d}"
        if key in self.weights:
            return self.weights[key]
        groups = self.protocol["weight_panel_groups"]
        channels = self.torch.tensor([g * 64 + k for g in groups for k in range(64)])
        r0 = self.runtime.records[expert][f"blocks.{block}.ffn.0"]
        r2 = self.runtime.records[expert][f"blocks.{block}.ffn.2"]
        values = {
            "W1_panel": r0["weight_qdq"].detach().cpu().index_select(0, channels).contiguous(),
            "W2_panel": r2["weight_qdq"].detach().cpu().index_select(1, channels).contiguous(),
        }
        for name, rec in [("s1", r0), ("s2", r2)]:
            if rec.get("smooth_scale") is not None:
                values[name] = rec["smooth_scale"].detach().cpu().contiguous()
        path = self.out / f"weights_{key}.safetensors"
        if path.exists():
            raise FileExistsError(path)
        self.save_file(values, str(path))
        binding = {"path": path.name, "sha256": sha256(path), "groups": groups,
                   "cache_identity": self.cache_identity,
                   "scope": "fixed panels, not random hidden-channel population"}
        self.weights[key] = binding
        return binding

    def finish(self, state):
        if state["seen"] != [0, 2] or state["qseen"] != [0, 2]:
            raise RuntimeError("Missing/duplicate original FFN calls")
        if set(FIELDS) - set(state["samples"]):
            raise RuntimeError("Incomplete observation")
        for value in state["samples"].values():
            if not bool(self.torch.isfinite(value).all()):
                raise RuntimeError("Non-finite captured value")
        position = (state["step"], state["block"])
        if any((s["step"], s["block"]) == position for s in self.scopes):
            raise RuntimeError("Duplicate native scope")
        path = self.out / f"step{state['step']}_block{state['block']:02d}.safetensors"
        if path.exists():
            raise FileExistsError(path)
        self.save_file(state["samples"], str(path))
        record = {"source_id": self.source_id, "step": state["step"], "block": state["block"],
                  "expert": state["expert"], "row_plan": state["plan"], "path": path.name,
                  "sha256": sha256(path), "weight_binding": self.weight_binding(state["expert"], state["block"]),
                  "nextnorm_captured": False, "dynamic_quant_metadata_exported": False,
                  "acceptance": "PENDING_PLAIN_OBSERVED_NOOP"}
        self.scopes.append(record)
        write_json(self.out / "SCOPES.json", self.scopes)
        state["samples"].clear()

    def __exit__(self, *error):
        self.runtime.apply, self.adapter.native_hif4_activation_qdq = self.old_apply, self.old_qdq
        self.infer.infer_block, self.infer.infer_ffn, self.infer.post_process = self.old_block, self.old_ffn, self.old_post

