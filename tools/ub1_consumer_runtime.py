"""Independent UB1 hook with a fixed original R256 residual.

Neutral and fitted consumers use the same arithmetic and residency path.  This
module never calls the historical policy constructor: old recipe bytes and
their recorded historical protocol SHA retain their original identities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


GROUPS_BY_STEP = (54, 108, 108, 162)
RR_NAMES = {"first", "second", "bias_first", "bias_second"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_sha(value):
    import torch
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def expected_expert(step):
    if type(step) is not int or step not in range(4):
        raise ValueError("Expected a native step in 0..3")
    return "high_noise_model" if step < 2 else "low_noise_model"


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("Bound asset escaped its declared root")
    return path


def verify_binding(root, binding):
    path = confined(root, binding["path"])
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise ValueError("Bound asset path/SHA differs: " + binding["path"])
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise ValueError("Bound asset size differs: " + binding["path"])
    return path


def validate_pack_metadata(item, position, groups, alpha, rr):
    """Compare a new consumer to its exact original support/alpha/RR object."""
    if (item["step"], item["block"], item["expert"]) != (
        position["step"], position["block"], position["expert"]
    ):
        raise ValueError("Consumer position/expert differs")
    if (item["groups"] != groups or item["alpha"] != alpha or
            item["RR_parameter_sha256"] != {k: tensor_sha(v) for k, v in rr.items()}):
        raise ValueError("Consumer support/alpha/original RR differs")
    for key in ("original_w2_tensor_sha256", "original_b2_sha256"):
        if key in item and item[key] != position[key]:
            raise ValueError("Consumer original weight/bias source differs")


def load_original_biases(checkpoints):
    """Read only bias tensors from the two verified original checkpoints."""
    import torch
    from safetensors import safe_open
    biases = {}
    for expert in ("high_noise_model", "low_noise_model"):
        with safe_open(str(checkpoints[expert]), framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            for block in range(40):
                values = {}
                for slot, name, width in ((0, "b1", 13824), (2, "b2", 5120)):
                    suffix = f"blocks.{block}.ffn.{slot}.bias"
                    matched = [k for k in keys if k == suffix or k.endswith("." + suffix)]
                    if len(matched) != 1:
                        raise ValueError("Expected one original checkpoint bias: " + suffix)
                    value = checkpoint.get_tensor(matched[0]).contiguous()
                    if value.dtype != torch.bfloat16 or tuple(value.shape) != (width,) or not bool(torch.isfinite(value).all()):
                        raise ValueError("Original checkpoint bias shape/dtype/finite differs")
                    values[name] = value
                    values[name + "_sha256"] = tensor_sha(value)
                    values[name + "_checkpoint_key"] = matched[0]
                biases[expert, block] = values
    return biases


def prepare_positions(base, bank_protocol, private_root, native_biases, *, coefficient_loader=None):
    """Build CPU packs from the independently verified original coefficients.

    The caller verifies the complete report/identity/parent/cache chain before
    entering this function.  Coefficient tensor hashes are checked here again
    by the frozen loader, against the actual decoded native W2 tensor.
    """
    import torch
    if coefficient_loader is None:
        from e0_ub1_native_consistency import load_coeff as coefficient_loader
    specs = bank_protocol["positions"]
    expected = {(s, b) for s in range(4) for b in range(40)}
    if len(specs) != 160 or {(p["step"], p["block"]) for p in specs} != expected:
        raise ValueError("Exactly 160 original coefficient positions required")
    positions = {}
    original_w2_hashes = {}
    for spec in specs:
        step, block, expert = spec["step"], spec["block"], spec["expert"]
        if expert != expected_expert(step):
            raise ValueError("Original coefficient expert/step differs")
        up = base.records[expert][f"blocks.{block}.ffn.0"]
        down = base.records[expert][f"blocks.{block}.ffn.2"]
        w1, w2 = up["weight_qdq"], down["weight_qdq"]
        if (w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or
                tuple(w1.shape) != (13824, 5120) or tuple(w2.shape) != (5120, 13824)):
            raise ValueError("Original native W1/W2 geometry/dtype differs")
        original_key = (expert, block)
        if original_key not in original_w2_hashes:
            original_w2_hashes[original_key] = tensor_sha(w2)
        actual_w2_sha = original_w2_hashes[original_key]
        if actual_w2_sha != spec["original_w2_tensor_sha256"]:
            raise ValueError("Actual runtime W2 differs from original coefficient source")
        biases = native_biases[original_key]
        if tensor_sha(biases["b2"]) != spec["original_b2_sha256"]:
            raise ValueError("Original native checkpoint b2 differs from coefficient source")
        files = [confined(private_root, spec[k]) for k in ("main_npz", "main_json", "rr_npz", "rr_json")]
        channels, bias, alpha, residual = coefficient_loader(
            *files, step, block, "cpu", actual_w2_sha, spec["original_b2_sha256"])
        groups = (channels.reshape(-1, 64)[:, 0] // 64).tolist()
        if (len(groups) != GROUPS_BY_STEP[step] or groups != sorted(set(groups)) or
                channels.dtype != torch.int64 or channels.shape != (64 * len(groups),)):
            raise ValueError("Original sorted whole64 support differs")
        for record, width in ((up, 5120), (down, 13824)):
            scale = record["smooth_scale"]
            if (scale.numel() != width or scale.dtype != torch.float32 or
                    not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()) or
                    not record.get("activation_qdq", True) or not record.get("force_fp32", True)):
                raise ValueError("Original Smooth/QDQ contract differs")
        width = len(channels)
        neutral_w2 = w2.index_select(1, channels).contiguous()
        positions[step, block] = {
            "step": step, "block": block, "expert": expert, "groups": groups,
            "channels": channels.contiguous(), "alpha": alpha,
            "w1": w1.index_select(0, channels).contiguous(),
            "b1": biases["b1"].index_select(0, channels).contiguous(),
            "s1": up["smooth_scale"].reshape(-1).contiguous(),
            "s2": down["smooth_scale"].reshape(-1).index_select(0, channels).contiguous(),
            "w2": neutral_w2, "bias": bias.contiguous(), "rr": residual,
            "original_b2_sha256": spec["original_b2_sha256"],
            "original_b1_sha256": biases["b1_sha256"],
            "original_b1_checkpoint_key": biases["b1_checkpoint_key"],
            "original_b2_checkpoint_key": biases["b2_checkpoint_key"],
            "original_w2_tensor_sha256": actual_w2_sha,
            "neutral_weight_tensor_sha256": tensor_sha(neutral_w2),
            "neutral_bias_tensor_sha256": tensor_sha(bias),
            "groups_tensor_sha256": tensor_sha(torch.tensor(groups, dtype=torch.int64)),
            "RR_parameter_sha256": {k: tensor_sha(v) for k, v in residual.items()},
            "width": width,
        }
    return positions


def replace_consumers(positions, manifest, hf_root, *, loader=None):
    """Replace W2/bias only; called after every neutral W2 GPU ref is released."""
    import torch
    if loader is None:
        from safetensors.torch import load_file as loader
    packs = manifest["packs"]
    if len(packs) != 160 or {(p["step"], p["block"]) for p in packs} != set(positions):
        raise ValueError("Exactly 160 fitted consumers required")
    for item in packs:
        key = item["step"], item["block"]
        position = positions[key]
        validate_pack_metadata(item, position, position["groups"], position["alpha"], position["rr"])
        path = verify_binding(hf_root, item)
        tensors = loader(str(path), device="cpu")
        if (set(tensors) != {"weight", "bias"} or
                tuple(tensors["weight"].shape) != (5120, position["width"]) or
                tuple(tensors["bias"].shape) != (5120,) or
                any(v.dtype != torch.bfloat16 or not bool(torch.isfinite(v).all()) for v in tensors.values())):
            raise ValueError("Fitted BF16 consumer shape/dtype/finite contract differs")
        if (tensor_sha(tensors["weight"]) != item["weight_tensor_sha256"] or
                tensor_sha(tensors["bias"]) != item["bias_tensor_sha256"]):
            raise ValueError("Fitted consumer tensor SHA differs")
        position["w2"], position["bias"] = tensors["weight"], tensors["bias"]
        position["consumer_file_sha256"] = item["sha256"]
        position["consumer_weight_tensor_sha256"] = item["weight_tensor_sha256"]
        position["consumer_bias_tensor_sha256"] = item["bias_tensor_sha256"]


def drop_dense_ffn_records(base):
    """Release dense FFN CPU/cache references after the dense residency exits."""
    if base._resident_weights:
        raise RuntimeError("Dense block cache must be empty before packed preparation")
    removed = 0
    for records in base.records.values():
        for name in list(records):
            if name.endswith((".ffn.0", ".ffn.2")):
                removed += records.pop(name)["weight_qdq"].numel() * 2
    base._resident_key = None
    return removed


def bytes_of_tensors(value):
    import torch
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(bytes_of_tensors(v) for v in value.values())
    return 0


class ConsumerResidency:
    """Preload non-FFN weights and all 160 distinct native packed positions.

    FFN keys contain step, block and slot.  A single W1/RR copy is shared by
    neutral/candidate, and ``unload_consumers`` fences before candidate load.
    """
    def __init__(self, base, positions, device, max_bytes):
        self.base, self.positions = base, positions
        self.device, self.limit = device, max_bytes
        self.non_ffn, self.packed = {}, {}
        self.stats = {"retained_bytes": 0, "peak_retained_bytes": 0,
                      "setup_tensor_transfers": 0, "setup_H2D_bytes": 0,
                      "non_FFN_weight_requests": 0, "non_FFN_retained_hits": 0,
                      "packed_position_requests": 0, "consumer_switches": 0}

    def _copy(self, tensor):
        size = bytes_of_tensors(tensor)
        if self.stats["retained_bytes"] + size > self.limit:
            raise RuntimeError("Consumer residency budget exceeded before transfer")
        value = tensor.to(self.device, non_blocking=False).contiguous()
        self.stats["retained_bytes"] += size
        self.stats["peak_retained_bytes"] = max(self.stats["peak_retained_bytes"], self.stats["retained_bytes"])
        self.stats["setup_tensor_transfers"] += 1
        self.stats["setup_H2D_bytes"] += size if self.device.type == "cuda" else 0
        return value

    def __enter__(self):
        if self.base._resident_weights:
            raise RuntimeError("Dense resident weights remain before consumer residency")
        self.old_weight = self.base._weight
        owner = self
        def weight(*, expert, block_index, module_name, record, device, dtype):
            if module_name.endswith((".ffn.0", ".ffn.2")):
                raise RuntimeError("A packed FFN requested an original dense GPU weight")
            key = expert, module_name, str(device), str(dtype)
            owner.stats["non_FFN_weight_requests"] += 1
            if key not in owner.non_ffn:
                raise RuntimeError("Generation requested a non-preloaded original weight")
            owner.stats["non_FFN_retained_hits"] += 1
            return owner.non_ffn[key]
        self.base._weight = weight
        return self

    def preload(self):
        import torch
        started = time.perf_counter()
        for expert, records in self.base.records.items():
            for name, record in records.items():
                if name.endswith((".ffn.0", ".ffn.2")):
                    raise RuntimeError("Dense FFN source records still retained")
                if not torch.is_tensor(record.get("weight_qdq")):
                    continue
                key = expert, name, str(self.device), str(torch.bfloat16)
                self.non_ffn[key] = self._copy(record["weight_qdq"].to(dtype=torch.bfloat16))
                self.base.counters["weight_transfers"] += 1
        for key, position in self.positions.items():
            self.packed[key] = {name: self._copy(position[name]) for name in ("w1", "b1", "s1", "s2", "channels")}
            self.packed[key]["rr"] = {name: self._copy(value) for name, value in position["rr"].items()}
        self.load_consumers()
        self._fence()
        return time.perf_counter() - started

    def _fence(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def unload_consumers(self):
        self._fence()
        for packed in self.packed.values():
            for name in ("w2", "bias"):
                value = packed.pop(name, None)
                self.stats["retained_bytes"] -= bytes_of_tensors(value)
                del value
        self._fence()

    def load_consumers(self):
        if any("w2" in p or "bias" in p for p in self.packed.values()):
            raise RuntimeError("Neutral and candidate consumers cannot coexist on GPU")
        for key, position in self.positions.items():
            self.packed[key]["w2"] = self._copy(position["w2"])
            self.packed[key]["bias"] = self._copy(position["bias"])
        self.stats["consumer_switches"] += 1
        self._fence()

    def position(self, key):
        self.stats["packed_position_requests"] += 1
        return self.packed[key]

    def snapshot(self):
        return {**self.stats, "resident_original_FFN_weights": 0,
                "packed_positions": len(self.packed),
                "resident_consumer_positions": sum("w2" in p for p in self.packed.values()),
                "residency_key": "step/block/slot; one W1/RR shared by neutral and candidate"}

    def __exit__(self, *error):
        self._fence()
        self.base._weight = self.old_weight
        self.non_ffn.clear()
        self.packed.clear()
        self.stats["retained_bytes"] = 0


class UB1ConsumerRuntime:
    """Swap only the two FFN linears; native GELU and caller remain outside."""
    def __init__(self, base, positions, residency, qdq, *, mode="neutral",
                 dim=5120, hidden=13824, group_size=64, rank=256, expected_rows=31668):
        self.base, self.positions, self.residency, self.qdq = base, positions, residency, qdq
        self.mode, self.dim, self.hidden, self.group_size, self.rank = mode, dim, hidden, group_size, rank
        self.expected_rows = expected_rows
        self.counters = base.counters
        self.coverage, self.residual_coverage, self.bias_sources = {}, {}, {}
        self.pending = None
        self.counts = dict.fromkeys(("main_linear", "RR_linear", "Q1", "Q2", "up", "down"), 0)
        self.request_open = False

    def begin_request(self):
        if self.pending is not None or self.request_open:
            raise RuntimeError("Previous consumer request is incomplete")
        self.coverage.clear()
        self.residual_coverage.clear()
        self.bias_sources.clear()
        self.counts = dict.fromkeys(self.counts, 0)
        self.before = dict(self.base.counters)
        self.request_open = True

    def apply(self, mm, x, *, expert, step_index, block_index, module_name):
        import torch
        import torch.nn.functional as F
        kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
        if not self.request_open:
            raise RuntimeError("A native request must reset the consumer ledger first")
        if not module_name.endswith((".ffn.0", ".ffn.2")):
            return self.base.apply(mm, x, **kw)
        if (expert != expected_expert(step_index) or type(block_index) is not int or
                block_index not in range(40) or module_name != f"blocks.{block_index}.ffn.{module_name[-1]}"):
            raise ValueError("Native FFN module/expert/position differs")
        key, slot = (step_index, block_index), int(module_name[-1])
        position = self.positions[key]
        packed = self.residency.position(key)
        width = position["width"]
        expected_in, expected_out = (self.dim, width) if slot == 0 else (width, self.dim)
        if (x.dtype != torch.bfloat16 or x.shape[-1] != expected_in or
                x.numel() // expected_in != self.expected_rows or (step_index, block_index, slot) in self.coverage):
            raise ValueError("Native packed input shape/dtype or duplicate call differs")
        if slot == 0 and self.pending is not None:
            raise RuntimeError("Duplicate/interleaved original RR prediction")
        if slot == 2 and (self.pending is None or self.pending[:2] != (key, tuple(x.shape[:-1]))):
            raise RuntimeError("Consumer lacks its exact preceding post-Q1 residual")
        smooth = packed["s1"] if slot == 0 else packed["s2"]
        self.counters["activation_qdq_calls"] += 1
        self.counts["Q1" if slot == 0 else "Q2"] += 1
        q = self.qdq((x.float() / smooth).to(x.dtype), force_fp32=True).to(x.dtype)
        original_bias = mm._get_actual_bias()
        if original_bias is None or original_bias.dtype != torch.bfloat16:
            raise ValueError("Original BF16 FFN bias source missing")
        if slot == 0:
            if tuple(original_bias.shape) != (self.hidden,):
                raise ValueError("Original W1 bias geometry differs")
            # The first RR linear consumes the exact post-Q1 BF16 state.
            rr = packed["rr"]
            self.counts["RR_linear"] += 1
            hidden = F.linear(q, rr["first"], rr["bias_first"])
            self.counts["RR_linear"] += 1
            correction = F.linear(hidden, rr["second"], rr["bias_second"])
            self.pending = (key, tuple(x.shape[:-1]), correction)
            bias = packed["b1"]
            weight = packed["w1"]
            self.residual_coverage[key] = {"rows": self.expected_rows, "rank": self.rank,
                                         "linear_calls": 2, "RR_parameter_sha256": position["RR_parameter_sha256"]}
            # Hashing and bias selection were paid in setup, before timing.
            source_key = expert, block_index
            digest = position["original_b1_sha256"]
            if source_key in self.bias_sources and self.bias_sources[source_key]["W1_bias_sha256"] != digest:
                raise ValueError("Original W1 bias changed between expert steps")
            self.bias_sources[source_key] = {"W1_bias_sha256": digest,
                "bias_source": "verified original BF16 checkpoint; selected and resident before generation",
                "checkpoint_key": position.get("original_b1_checkpoint_key")}
        else:
            if tuple(original_bias.shape) != (self.dim,):
                raise ValueError("Original W2 bias source differs")
            q = (q.float() * position["alpha"]).to(q.dtype)
            weight, bias = packed["w2"], packed["bias"]
        self.counters["linear_forward_calls"] += 1
        self.counts["main_linear"] += 1
        self.counts["up" if slot == 0 else "down"] += 1
        result = F.linear(q, weight, bias)
        if slot == 2:
            correction = self.pending[2]
            if correction.shape != result.shape or correction.dtype != result.dtype:
                raise ValueError("Original RR output shape/dtype differs")
            result = result + correction
            self.pending = None
        if result.shape[-1] != expected_out:
            raise ValueError("Packed native output geometry differs")
        self.coverage[step_index, block_index, slot] = {
            "rows": self.expected_rows, "weight_shape": list(weight.shape),
            "main_MAC_density": width / self.hidden,
            "groups_tensor_sha256": position.get("groups_tensor_sha256"),
        }
        return result

    def end_request(self, *, full_scope=True):
        if not self.request_open or self.pending is not None:
            raise RuntimeError("Unmatched original RR remains at request end")
        expected = {(s, b, n) for s in range(4) for b in range(40) for n in (0, 2)}
        if full_scope and (set(self.coverage) != expected or len(self.residual_coverage) != 160 or
                           self.counts != {"main_linear": 320, "RR_linear": 320,
                                           "Q1": 160, "Q2": 160, "up": 160, "down": 160}):
            raise RuntimeError("Incomplete native 160-site consumer ledger")
        self.request_open = False
        return self.report()

    def report(self):
        return {"mode": self.mode, "source_sha256": sha256(Path(__file__)),
                "counts": dict(self.counts), "pending_RR": int(self.pending is not None),
                "counter_semantics": "linear/QDQ dispatch attempts; main/RR coverage records completed tensors",
                "main_coverage": {f"step{s}_block{b:02d}_ffn{n}": v for (s, b, n), v in self.coverage.items()},
                "RR_coverage": {f"step{s}_block{b:02d}": v for (s, b), v in self.residual_coverage.items()},
                "original_bias_sources": {f"{e}_block{b:02d}": v for (e, b), v in self.bias_sources.items()},
                "groups_by_step": list(GROUPS_BY_STEP), "fixed_original_RR_rank": self.rank,
                "nominal_main_MAC_density": .5, "RR_MAC_over_dense_FFN": self.rank / self.hidden,
                "consumer_precision": "BF16 folded W2 and bias; two original BF16 RR linears; BF16 sum before native gate",
                "caller_norm_GELU_gate_and_residual": "unchanged native caller",
                "residency": self.residency.snapshot(),
                "quality_or_speed_acceptance": False}
