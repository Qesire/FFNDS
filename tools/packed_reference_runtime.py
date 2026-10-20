"""Closed-loop static packed reference, preserving the original HiF4 bridge.

Weights are packed offline in CPU memory before generation. This is an
operator/control baseline, not the proposed dynamic selector or acceptance.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ar2_checkpoint import digest


def create_runtime(cache_path, policy_path):
    from HiFloat4.ptq.adapters.lightx2v_wan_distill import HIF4CacheRuntime
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq

    class PackedReferenceRuntime(HIF4CacheRuntime):
        def __init__(self):
            super().__init__(cache_path)
            self.policy = json.loads(Path(policy_path).read_text())
            self.policy_sha = digest(policy_path)
            if self.policy["kind"] != "STATIC_PACKED_OPERATOR_REFERENCE":
                raise ValueError("Unsupported sparse reference policy")
            groups = torch.as_tensor(self.policy["group_ids"], dtype=torch.long)
            if groups.shape != (108,) or len(groups.unique()) != 108 or bool((groups < 0).any()) or bool((groups >= 216).any()):
                raise ValueError("Need 108 distinct original groups64")
            self.channels = (groups.sort().values[:, None] * 64 + torch.arange(64)).reshape(-1)
            if self.policy["blocks"] != list(range(40)) or self.policy["steps"] != list(range(4)):
                raise ValueError("Reference requires full 40x4 coverage")
            self.coverage = {}
            self.channel_devices = {}
            for expert, records in self.records.items():
                alpha = float(self.policy["gains_by_expert"][expert])
                if not 0 <= alpha < 10:
                    raise ValueError("Invalid fixed training gain")
                for block in range(40):
                    for slot, axis in [(0, 0), (2, 1)]:
                        name = f"blocks.{block}.ffn.{slot}"
                        old = records[name]
                        record = dict(old)
                        record["weight_qdq"] = old["weight_qdq"].index_select(axis, self.channels).contiguous()
                        if slot == 2:
                            record["smooth_scale"] = old["smooth_scale"].index_select(0, self.channels).contiguous()
                        records[name] = record

        def apply(self, mm_weight, input_tensor, *, expert, step_index, block_index, module_name):
            if not module_name.endswith((".ffn.0", ".ffn.2")):
                return super().apply(mm_weight, input_tensor, expert=expert, step_index=step_index,
                                     block_index=block_index, module_name=module_name)
            if step_index not in range(4) or expert != ("high_noise_model" if step_index < 2 else "low_noise_model"):
                raise RuntimeError("Sparse reference expert/step escaped contract")
            slot = 0 if module_name.endswith(".ffn.0") else 2
            expected_in, expected_out = (5120, 6912) if slot == 0 else (6912, 5120)
            if input_tensor.shape[-1] != expected_in:
                raise RuntimeError("Sparse FFN input shape mismatch")
            record = self._record(expert, module_name)
            x = (input_tensor.float() / record["smooth_scale"].to(input_tensor.device)).to(input_tensor.dtype)
            x = qdq(x, force_fp32=True)
            self.counters["activation_qdq_calls"] += 1
            if slot == 2:
                x = (x.float() * float(self.policy["gains_by_expert"][expert])).to(x.dtype)
            weight = self._weight(expert=expert, block_index=block_index, module_name=module_name,
                                  record=record, device=x.device, dtype=x.dtype)
            bias = mm_weight._get_actual_bias() if hasattr(mm_weight, "_get_actual_bias") else getattr(mm_weight, "bias", None)
            if bias is not None:
                bias = bias.to(device=x.device, dtype=x.dtype)
                if slot == 0:
                    if str(x.device) not in self.channel_devices:
                        self.channel_devices[str(x.device)] = self.channels.to(x.device)
                    bias = bias.index_select(0, self.channel_devices[str(x.device)])
            output = F.linear(x, weight, bias)
            if output.shape[-1] != expected_out:
                raise RuntimeError("Reduced FFN GEMM output shape mismatch")
            key = f"step{step_index}_block{block_index}_ffn{slot}"
            if key in self.coverage:
                raise RuntimeError("Sparse reference scope was executed twice")
            self.coverage[key] = {"rows": input_tensor.numel() // expected_in,
                                  "actual_weight_shape": list(weight.shape), "main_GEMM_density": .5}
            self.counters["linear_forward_calls"] += 1
            return output

        def report(self):
            return {**super().report(), "packed_reference": {"policy": self.policy,
                "policy_sha256": self.policy_sha, "source_sha256": digest(Path(__file__)),
                "actual_FFN_linear_calls": len(self.coverage), "expected_FFN_linear_calls": 320,
                "coverage": self.coverage, "weight_packing": "CPU before generation; packed weights transferred by original block cache",
                "full_FFN_speed_acceptance": False, "formal20_quality_acceptance": False}}

    return PackedReferenceRuntime()
