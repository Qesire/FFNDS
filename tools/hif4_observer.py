"""Observe the original HiF4 runtime's actual QDQ and effective W2 operands.

Shares the fresh uniform sampling plan with BF16Observer, while preserving the
original QDQ and F.linear callable. This is a new D-Q series, not old R1 tensors.
"""
import hashlib

import torch

from bf16_observer import BF16Observer


def tensor_digest(tensor):
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()


class HiF4Observer(BF16Observer):
    arm = "D-Q"
    series = "WORKER_NEW_HIF4_UNIFORM_CAPTURE_20261020"
    required_tensors = BF16Observer.required_tensors | {"Q1_input", "V1", "Q2_input", "V2"}
    tensor_aliases = {}
    alias_note = "V2 is the actual native HiF4 Q2 output subsequently passed to F.linear"

    def __init__(self, root, *, runtime, cache_manifest, **kwargs):
        super().__init__(root, **kwargs)
        self.runtime = runtime
        self.cache_manifest = cache_manifest
        self.weight_checks = {}

    def _new_scope(self, tag, x):
        scope = super()._new_scope(tag, x)
        scope["quantized_cache"] = self.cache_manifest
        scope["QDQ_order"] = "native input.float / smooth_scale.float -> BF16 -> native HiF4(force_fp32=True) -> BF16 -> F.linear/BF16 bias once"
        return scope

    def __enter__(self):
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        if adapter._ACTIVE_RUNTIME is not self.runtime:
            raise RuntimeError("Install the exact audited HiF4 runtime before D-Q observation")
        super().__enter__()
        original_qdq = adapter.native_hif4_activation_qdq
        original_weight = adapter.HIF4CacheRuntime._weight
        observer = self

        def observed_qdq(activation, *, force_fp32=True):
            scope = observer.pending
            name = scope.get("_current_linear", "") if scope is not None else ""
            slot = "1" if name.endswith(".ffn.0") else "2" if name.endswith(".ffn.2") else None
            if slot is None:
                return original_qdq(activation, force_fp32=force_fp32)
            if not force_fp32 or activation.dtype != torch.bfloat16:
                raise RuntimeError("Unexpected native activation QDQ dtype/force_fp32")
            input_key, output_key = f"Q{slot}_input", f"V{slot}"
            if input_key in scope["_samples"]:
                raise RuntimeError("Duplicate activation QDQ in a native FFN scope")
            scope["_samples"][input_key] = observer._sample(scope, activation).detach().cpu().contiguous()
            result = original_qdq(activation, force_fp32=force_fp32)
            scope["_samples"][output_key] = observer._sample(scope, result).detach().cpu().contiguous()
            scope.setdefault("QDQ_calls", []).append({"module": name, "input": input_key,
                "output": output_key, "input_dtype": str(activation.dtype), "output_dtype": str(result.dtype),
                "force_fp32": force_fp32, "dim": -1, "native_group_size": 64})
            return result

        def observed_weight(runtime, **kwargs):
            weight = original_weight(runtime, **kwargs)
            scope = observer.pending
            name = kwargs["module_name"]
            if runtime is not observer.runtime or scope is None or not name.endswith(".ffn.2"):
                return weight
            if kwargs["expert"] != scope["expert"] or kwargs["block_index"] != scope["block"]:
                raise RuntimeError("Effective quantized W2 escaped its native scope")
            expected = runtime.records[scope["expert"]][name]["weight_qdq"]
            if weight.dtype != torch.bfloat16 or list(weight.shape) != [5120, 13824] or expected.dtype != torch.bfloat16:
                raise RuntimeError("Unexpected effective QDQ W2 dtype/layout")
            key = (scope["expert"], scope["block"])
            if key not in observer.weight_checks:
                observer.weight_checks[key] = tensor_digest(expected)
            observed_hash = tensor_digest(weight)
            if observed_hash != observer.weight_checks[key]:
                raise RuntimeError("Runtime QDQ W2 bytes differ from verified cache record")
            native = scope["effective_W2"]
            scope["effective_W2"] = {"module_name": name, "runtime_shape": list(weight.shape),
                "runtime_stride": list(weight.stride()), "dtype": str(weight.dtype),
                "runtime_callable": "torch.nn.functional.linear(x, weight_qdq, original_bias)",
                "mathematical_W2_transform": "transpose cache [out,in] to [in,out] for analysis",
                "runtime_operand_sha256": observer.weight_checks[key], "cache_record_key": [scope["expert"], name, "weight_qdq"],
                "bias_status": native["bias_status"], "bias_dtype": native["bias_dtype"],
                "bias_checkpoint_tensor_key": f"blocks.{scope['block']}.ffn.2.bias",
                "has_lora_or_diff": False, "runtime_cache_bytes_match": True}
            scope["checks"]["actual_runtime_W2_matches_cache"] = True
            return weight

        try:
            for owner, name, function in [(adapter, "native_hif4_activation_qdq", observed_qdq),
                                          (adapter.HIF4CacheRuntime, "_weight", observed_weight)]:
                self.methods.append((owner, name, getattr(owner, name)))
                setattr(owner, name, function)
        except BaseException:
            self._restore()
            raise
        return self

    def _save_scope(self, scope):
        if len(scope.get("QDQ_calls", [])) != 2 or not scope["effective_W2"].get("runtime_cache_bytes_match"):
            raise RuntimeError("Missing actual QDQ/W2 capture provenance")
        super()._save_scope(scope)
