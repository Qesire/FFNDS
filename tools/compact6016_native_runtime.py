"""Deploy folded ordinary-6016 FFNs through the original native MM hook.

Only the two FFN linears are intercepted. The existing transformer supplies
its original norm/AdaLN, GELU(tanh), e5 gate, residual and scheduler. Original
attention projections delegate to the unmodified native cache runtime.

No import here loads Torch, assets, a model, or CUDA. The future registered
runner must bind complete producer receipts, reserve its GPU, and supply the
original native HiF4 quantizer. Training factors are never read for deployment.
"""
from pathlib import Path
import math
import time

from ub1_consumer_runtime import (bytes_of_tensors, expected_expert, sha256,
                                  tensor_sha, verify_binding)


EXPERTS = ("high_noise_model", "low_noise_model")
INITIALIZATIONS = ("energy94", "random94")
DIM, HIDDEN, WIDTH, GROUP = 5120, 13824, 6016, 64
DEPLOY_KEYS = {"W1", "W2", "b1", "b2", "smooth1", "smooth2", "groups", "channels"}
TRAINING_KEYS = {"training_lora_A1", "training_lora_B1", "training_lora_A2", "training_lora_B2"}
PRODUCER_ID = "E0_COMPACT6016_FULL80_BANK_20261028"
CELL_STATUS = "COMPLETE_256_UPDATES_AND_INITIAL_FINAL_48_SOURCE_STEP_DIAGNOSTICS"


def steps_for(expert):
    if expert not in EXPERTS:
        raise ValueError("Unknown native expert")
    return (0, 1) if expert == EXPERTS[0] else (2, 3)


def function_key(expert, block):
    if expert not in EXPERTS or type(block) is not int or block not in range(40):
        raise ValueError("Native expert/block outside full80")
    return expert, block


def cell_id(expert, block, initialization):
    function_key(expert, block)
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown frozen ordinary6016 initialization")
    return f"compact6016bank_{'high' if expert == EXPERTS[0] else 'low'}_b{block:02d}_{initialization}"


def assemble_manifest(training_protocol, part_receipts, hf_root, *, repo_root):
    """Read metadata only; accept four completed parts and all160 final cells.

    This is an asset index, not a fit or run registration. The trained tensors
    are verified only by ``load_bank``; this function creates no model calls.
    """
    import json
    protocol_path = verify_binding(repo_root, training_protocol)
    plan = json.loads(protocol_path.read_text())
    if (plan.get("id") != PRODUCER_ID or plan.get("assets_finalized") is not True
            or plan.get("initializations") != list(INITIALIZATIONS)
            or [p["part"] for p in part_receipts] != list(range(4))):
        raise ValueError("Exactly four frozen full80 producer parts required")
    training = plan.get("training", {})
    if (any(training.get(k) != v for k, v in {"rank": 32, "epochs": 8, "steps_per_cell": 256,
            "rows_per_batch": 256, "optimizer": "Adam", "lr": .0005,
            "checkpoint_selection": False, "weight_requantization": False,
            "train_bias": False, "train_smooth": False,
            "shared_expert_block_weights_across_both_steps": True}.items())
            or plan.get("source_split", {}).get("training") != list(range(16))
            or plan.get("source_split", {}).get("diagnostic_heldout") != list(range(16, 24))):
        raise ValueError("Exact new matched ordinary6016 training provenance differs")
    result = {"schema": "ffnds.compact6016.deployment_bank.v1", "status": "METADATA_COMPLETE_TENSOR_LOAD_PENDING",
              "training_protocol": training_protocol, "producer_id": PRODUCER_ID,
              "asset_view": plan["asset_view"], "parts": part_receipts, "cells": [],
              "independent_training_sources": 16, "diagnostic_exposed_sources": 8,
              "deployment": "two folded BF16 matrices; no online LoRA or weight QDQ"}
    seen = set()
    for part in part_receipts:
        run = json.loads(verify_binding(hf_root, part["RUN"]).read_text())
        exit_record = json.loads(verify_binding(hf_root, part["launch_exit"]).read_text())
        expected_functions = {(e, b) for e in EXPERTS for b in range(part["part"] * 10, (part["part"] + 1) * 10)}
        if (run.get("id") != PRODUCER_ID or run.get("part") != part["part"]
                or run.get("protocol_sha256") != training_protocol["sha256"]
                or run.get("status") != "COMPLETE_FULL80_BANK_PART"
                or run.get("predebited") != plan["limits_per_part"] or run.get("observed") != plan["limits_per_part"]
                or exit_record.get("id") != PRODUCER_ID or exit_record.get("part") != part["part"]
                or exit_record.get("protocol_sha256") != training_protocol["sha256"]
                or exit_record.get("launcher_returncode") != 0 or exit_record.get("process_exited") is not True
                or len(run.get("cells", [])) != 40):
            raise ValueError("Producer completion/counts/protocol/actual exit0 differs")
        functions = {(r["expert"], r["block"]): r for r in run["functions"]}
        if len(run["functions"]) != 20 or set(functions) != expected_functions:
            raise ValueError("Fixed balanced twenty-function producer partition differs")
        for cell in run["cells"]:
            key = cell["expert"], cell["block"], cell["initialization"]
            if key in seen or key[:2] not in expected_functions:
                raise ValueError("Duplicate or cross-part folded cell")
            seen.add(key)
            row = json.loads(verify_binding(hf_root, cell["result"]).read_text())
            expected_groups = functions[key[:2]]["energy_groups" if key[2] == "energy94" else "random_groups"]
            if (cell.get("run_id") != cell_id(*key) or row.get("run_id") != cell["run_id"]
                    or cell.get("status") != CELL_STATUS or row.get("status") != CELL_STATUS
                    or row.get("expert") != key[0] or row.get("block") != key[1] or row.get("initialization") != key[2]
                    or row.get("steps") != list(steps_for(key[0])) or row.get("groups") != expected_groups
                    or row.get("export") != cell.get("export")
                    or cell["export"].get("reloaded_all_tensor_bytes_equal") is not True
                    or cell["export"].get("online_LoRA_GEMM") != 0 or cell["export"].get("online_weight_QDQ") != 0):
                raise ValueError("Completed folded cell/result/export provenance differs")
            result["cells"].append({**cell, "steps": row["steps"], "groups": row["groups"],
                                     "support_file": functions[key[:2]]["support_file"]})
    expected = {(e, b, init) for e in EXPERTS for b in range(40) for init in INITIALIZATIONS}
    if seen != expected:
        raise ValueError("All80 functions and both matched initializations required")
    result["cells"].sort(key=lambda c: (c["expert"], c["block"], c["initialization"]))
    return result


def validate_deployment_pack(pack, cell, original_up, original_down, native_b1, native_b2,
                             *, dim=DIM, hidden=HIDDEN, width=WIDTH, group_size=GROUP):
    """Check exact bias/scale/support bytes against the original dense assets.

    ``W1`` and ``W2`` are the final trained BF16 matrices. Their provenance is
    the completed cell export SHA; they are not expected to equal raw slices.
    """
    import torch
    if set(pack) != DEPLOY_KEYS:
        raise ValueError("Exactly eight deployment tensors; no online training factors")
    shapes = {"W1": (width, dim), "W2": (dim, width), "b1": (width,), "b2": (dim,),
              "smooth1": (dim,), "smooth2": (width,), "groups": (width // group_size,), "channels": (width,)}
    for name, shape in shapes.items():
        value = pack[name]
        dtype = torch.int64 if name in ("groups", "channels") else torch.float32 if name.startswith("smooth") else torch.bfloat16
        if value.dtype != dtype or tuple(value.shape) != shape or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("Deployment tensor dtype/shape/device/finite differs: " + name)
    groups = pack["groups"].tolist()
    if (width % group_size or hidden % group_size or groups != sorted(set(groups))
            or any(g < 0 or g >= hidden // group_size for g in groups) or groups != cell["groups"]):
        raise ValueError("Frozen sorted whole-group support differs")
    channels = (pack["groups"][:, None] * group_size + torch.arange(group_size)).reshape(-1)
    if not torch.equal(channels, pack["channels"]):
        raise ValueError("Whole-group channels are not original sorted columns")
    if (native_b1.dtype != torch.bfloat16 or tuple(native_b1.shape) != (hidden,)
            or native_b2.dtype != torch.bfloat16 or tuple(native_b2.shape) != (dim,)):
        raise ValueError("Original complete native bias geometry/dtype differs")
    for item, size in ((original_up, dim), (original_down, hidden)):
        smooth = item["smooth_scale"]
        if (smooth.dtype != torch.float32 or tuple(smooth.shape) != (size,) or not bool(torch.isfinite(smooth).all())
                or not bool((smooth > 0).all()) or item.get("activation_qdq") is not True or item.get("force_fp32") is not True):
            raise ValueError("Original positive Smooth/native QDQ contract differs")
    expected = {"smooth1": original_up["smooth_scale"],
                "smooth2": original_down["smooth_scale"].index_select(0, channels),
                "b1": native_b1.index_select(0, channels), "b2": native_b2}
    for name, value in expected.items():
        if value.dtype != pack[name].dtype or not torch.equal(value.contiguous().view(torch.uint8), pack[name].view(torch.uint8)):
            raise ValueError("Native selected bias/smooth byte identity differs: " + name)
    if any(not bool((pack[name] > 0).all()) for name in ("smooth1", "smooth2")):
        raise ValueError("Frozen smooth must be positive")
    return {"groups": groups, "width": width, "input_boundary": "native MM.apply before Smooth1/Q1",
            "original_b1_sha256": tensor_sha(native_b1), "original_b2_sha256": tensor_sha(native_b2),
            "deployment_tensor_sha256": {k: tensor_sha(v) for k, v in pack.items()},
            "W1_bias": "selected original b1; original 13824-vector is never used by the narrow GEMM"}


def load_bank(manifest, initialization, hf_root, base, native_biases, *, repo_root, loader=None):
    """Read only the eight deployment tensors for each of eighty functions.

    The caller binds the immutable manifest itself before invoking this loader.
    Original native biases come from the verified checkpoint CPU loader. The
    original cache remains intact for the matched Dense replay.
    """
    import json
    import torch
    if initialization not in INITIALIZATIONS or manifest.get("schema") != "ffnds.compact6016.deployment_bank.v1":
        raise ValueError("Frozen deployment bank/init required")
    if manifest.get("status") != "METADATA_COMPLETE_TENSOR_LOAD_PENDING" or len(manifest.get("cells", [])) != 160:
        raise ValueError("Four completed producer parts required before deployment")
    verified = assemble_manifest(manifest["training_protocol"], manifest["parts"], hf_root, repo_root=repo_root)
    if verified != manifest:
        raise ValueError("Frozen deployment manifest differs from actual complete producer receipts")
    import hashlib
    bank_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if loader is None:
        from safetensors import safe_open
        def loader(path):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != DEPLOY_KEYS | TRAINING_KEYS:
                    raise ValueError("Original folded export keys differ")
                return {k: handle.get_tensor(k).contiguous() for k in DEPLOY_KEYS}
    view = json.loads(verify_binding(hf_root, manifest["asset_view"]).read_text())
    specs = {(r["expert"], r["block"]): r for r in view["shards"]}
    expected = {(e, b) for e in EXPERTS for b in range(40)}
    if set(specs) != expected or len(view["shards"]) != 80:
        raise ValueError("Original full80 native asset identity required")
    positions = {}
    cells = [c for c in manifest["cells"] if c["initialization"] == initialization]
    if len(cells) != 80 or {(c["expert"], c["block"]) for c in cells} != expected:
        raise ValueError("Exactly eighty complete same-init folded FFNs required")
    for cell in cells:
        key = function_key(cell["expert"], cell["block"])
        if cell["run_id"] != cell_id(*key, initialization) or cell["steps"] != list(steps_for(key[0])):
            raise ValueError("Folded expert shares exactly its two original steps")
        path = verify_binding(hf_root, cell["export"])
        pack = loader(path)
        if set(pack) != DEPLOY_KEYS:
            raise ValueError("Deploy loader read unexpected tensors")
        for name, value in pack.items():
            recorded = cell["export"]["tensors"][name]
            if (list(value.shape) != recorded["shape"] or str(value.dtype) != recorded["dtype"]
                    or tensor_sha(value) != recorded["sha256"]):
                raise ValueError("Actual folded tensor bytes differ from producer export")
        up = base.records[key[0]][f"blocks.{key[1]}.ffn.0"]
        down = base.records[key[0]][f"blocks.{key[1]}.ffn.2"]
        for name, value in (("W1", up["weight_qdq"]), ("W2", down["weight_qdq"]),
                            ("smooth1", up["smooth_scale"]), ("smooth2", down["smooth_scale"])):
            spec = specs[key]["tensors"][name]
            if list(value.shape) != spec["shape"] or tensor_sha(value) != spec["sha256"]:
                raise ValueError("Actual original native cache differs from bank parent")
        biases = native_biases[key]
        metadata = validate_deployment_pack(pack, cell, up, down, biases["b1"], biases["b2"])
        positions[key] = {**pack, **metadata, "expert": key[0], "block": key[1], "initialization": initialization,
                          "run_id": cell["run_id"], "export_binding": cell["export"],
                          "producer_result_binding": cell["result"], "support_binding": cell["support_file"],
                          "bank_semantic_sha256": bank_sha}
    return positions


class Compact6016Residency:
    """Retain original non-FFN weights plus one80-function folded bank.

    Uses the already tested AllWeightResidency for every original attention
    matrix. The added FFN cache key has expert/block/init, never step: each
    expert's two steps use identical tensor objects. Dense and compact GPU
    contexts must be disjoint. CPU original cache records remain untouched.
    """
    def __init__(self, base, positions, device, max_bytes):
        self.base, self.positions, self.device, self.limit = base, positions, device, max_bytes
        self.packed = {}
        self.stats = {"narrow_bytes": 0, "setup_tensor_transfers": 0, "setup_matrix_transfers": 0,
                      "setup_H2D_bytes": 0, "position_requests": 0}
        self.attempted = False

    def _fence(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        from ffnds_all_weight_residency import AllWeightResidency
        if (self.base._resident_weights or getattr(self.base, "_compact6016_residency_active", False)
                or getattr(self.base._weight, "__self__", None) is not self.base):
            raise RuntimeError("Dense/other resident weight context still active")
        extra = sum(bytes_of_tensors(p[k]) for p in self.positions.values()
                    for k in ("W1", "W2", "b1", "b2", "smooth1", "smooth2"))
        if extra >= self.limit:
            raise RuntimeError("Narrow bank exceeds total residency budget")
        self.policy = AllWeightResidency(self.base, "all_resident_no_flush", self.limit - extra)
        self.policy.__enter__()
        self.original_weight = self.base._weight
        def weight(**kw):
            if kw["module_name"].endswith((".ffn.0", ".ffn.2")):
                raise RuntimeError("Compact FFN requested an original dense GPU matrix")
            return self.original_weight(**kw)
        self.base._weight = weight
        self.base._compact6016_residency_active = True
        self.reserved_narrow_bytes = extra
        return self

    def preload(self):
        import torch
        if self.attempted:
            raise RuntimeError("Single preload attempt; no retry/overlapping banks")
        self.attempted = True
        begin = time.perf_counter()
        for expert, records in self.base.records.items():
            for name, record in records.items():
                if name.endswith((".ffn.0", ".ffn.2")):
                    continue
                self.base._weight(expert=expert, block_index=int(name.split(".")[1]), module_name=name,
                    record=record, device=self.device, dtype=torch.bfloat16)
        for key, position in self.positions.items():
            entry = {}
            for name in ("W1", "W2", "b1", "b2", "smooth1", "smooth2"):
                value = position[name]
                size = bytes_of_tensors(value)
                if self.stats["narrow_bytes"] + size > self.reserved_narrow_bytes:
                    raise RuntimeError("Narrow residency changed after reservation")
                entry[name] = value.to(device=self.device, non_blocking=False).contiguous()
                self.stats["narrow_bytes"] += size
                self.stats["setup_tensor_transfers"] += 1
                if self.device.type == "cuda":
                    self.stats["setup_H2D_bytes"] += size
                if name in ("W1", "W2"):
                    self.base.counters["weight_transfers"] += 1
                    self.stats["setup_matrix_transfers"] += 1
            self.packed[key] = entry
        self._fence()
        self.stats["preload_seconds"] = time.perf_counter() - begin
        return self.stats["preload_seconds"]

    def position(self, key):
        if key not in self.packed:
            raise RuntimeError("Folded bank was not completely preloaded")
        self.stats["position_requests"] += 1
        return self.packed[key]

    def snapshot(self):
        original = self.policy.snapshot()
        return {**self.stats, "original_attention_residency": original,
                "total_tensor_bytes": original["retained_bytes"] + self.stats["narrow_bytes"],
                "FFN_function_copies": len(self.packed), "original_dense_FFN_decoded_cache_matrices": 0,
                "native_checkpoint_offload_buffers": "unchanged original full buffers; separate physical memory/H2D cost",
                "cache_key": "expert/block/initialization/tensor; both steps share exact objects"}

    def __exit__(self, *error):
        try:
            self._fence()
        finally:
            self.packed.clear()
            self.stats["narrow_bytes"] = 0
            self.base._weight = self.original_weight
            try:
                self.policy.__exit__(*error)
            finally:
                del self.base._compact6016_residency_active


class Compact6016Runtime:
    """Hook actual MM inputs before Smooth1/Q1 and before student Smooth2/Q2.

    No fitted factors, teacher V2, full13824 FFN, online observer/hash, or gate
    approximation is used. Timings come from the existing native FFN timer.
    """
    def __init__(self, base, positions, residency, quantizer, *, initialization,
                 dim=DIM, hidden=HIDDEN, width=WIDTH, expected_rows=31668, blocks=40):
        if initialization not in INITIALIZATIONS or any(p["initialization"] != initialization for p in positions.values()):
            raise ValueError("One consistent final folded initialization required")
        banks = {p["bank_semantic_sha256"] for p in positions.values()}
        if len(banks) != 1:
            raise ValueError("One immutable complete folded bank identity required")
        self.bank_semantic_sha256 = next(iter(banks))
        self.base, self.positions, self.residency, self.quantizer = base, positions, residency, quantizer
        self.initialization, self.dim, self.hidden, self.width = initialization, dim, hidden, width
        self.expected_rows, self.blocks = expected_rows, blocks
        self.counters = base.counters
        self.open, self.pending = False, None
        self.coverage, self.sites, self.counts = {}, {}, {}
        self.non_ffn_coverage = set()

    def begin_request(self):
        if self.open or self.pending is not None:
            raise RuntimeError("Previous full-depth native request incomplete")
        self.open, self.coverage, self.sites = True, {}, {}
        self.non_ffn_coverage = set()
        self.counts = dict.fromkeys(("W1", "W2", "Q1", "Q2", "narrow_BF16_GEMM", "online_LoRA_GEMM", "weight_QDQ"), 0)
        self.before = dict(self.counters)

    def apply(self, mm, x, *, expert, step_index, block_index, module_name):
        import torch
        import torch.nn.functional as F
        if not self.open:
            raise RuntimeError("Native request must explicitly begin before any target linear")
        kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
        if not module_name.endswith((".ffn.0", ".ffn.2")):
            target = (step_index, block_index, module_name)
            if (type(step_index) is not int or step_index not in range(4) or expert != expected_expert(step_index)
                    or type(block_index) is not int or block_index not in range(self.blocks)
                    or module_name not in self.base.records[expert] or target in self.non_ffn_coverage):
                raise ValueError("Unknown or duplicate original attention projection")
            result = self.base.apply(mm, x, **kw)
            if result is None:
                raise RuntimeError("Original attention cache returned fallback")
            self.non_ffn_coverage.add(target)
            return result
        if (type(step_index) is not int or step_index not in range(4) or expert != expected_expert(step_index)
                or type(block_index) is not int or block_index not in range(self.blocks)
                or module_name != f"blocks.{block_index}.ffn.{module_name[-1]}"):
            raise ValueError("Native FFN expert/step/block/module identity differs")
        key, site, slot = (expert, block_index), (step_index, block_index), int(module_name[-1])
        if key not in self.positions or (step_index, block_index, slot) in self.coverage:
            raise ValueError("Unknown folded bank entry or duplicate native dispatch")
        expected_in, expected_out = (self.dim, self.width) if slot == 0 else (self.width, self.dim)
        if (x.dtype != torch.bfloat16 or x.ndim != 2 or tuple(x.shape) != (self.expected_rows, expected_in)):
            raise ValueError("Actual compact native BF16 input geometry differs")
        if slot == 0 and self.pending is not None or slot == 2 and self.pending != site:
            raise RuntimeError("Original W1/GELU/W2 pair is missing or interleaved")
        original_bias = mm._get_actual_bias()
        if (original_bias is None or original_bias.dtype != torch.bfloat16
                or tuple(original_bias.shape) != ((self.hidden,) if slot == 0 else (self.dim,))):
            raise ValueError("Original full-width native MM bias source differs")
        packed, position = self.residency.position(key), self.positions[key]
        smooth = packed["smooth1" if slot == 0 else "smooth2"]
        self.counters["activation_qdq_calls"] += 1
        self.counts["Q1" if slot == 0 else "Q2"] += 1
        q = self.quantizer((x.float() / smooth.view(1, -1)).to(x.dtype), force_fp32=True).to(x.dtype)
        if q.shape != x.shape:
            raise ValueError("Native QDQ changed the actual narrow operand geometry")
        weight, bias = (packed["W1"], packed["b1"]) if slot == 0 else (packed["W2"], packed["b2"])
        self.counters["linear_forward_calls"] += 1
        self.counts["narrow_BF16_GEMM"] += 1
        self.counts["W1" if slot == 0 else "W2"] += 1
        result = F.linear(q, weight, bias)
        if tuple(result.shape) != (self.expected_rows, expected_out) or result.dtype != torch.bfloat16:
            raise ValueError("Actual narrow native output geometry differs")
        self.pending = site if slot == 0 else None
        self.coverage[step_index, block_index, slot] = {"input_shape": list(x.shape), "weight_shape": list(weight.shape),
            "output_shape": list(result.shape), "QDQ_shape": list(q.shape), "bias_shape": list(bias.shape),
            "export_sha256": position["export_binding"]["sha256"]}
        self.sites.setdefault(site, {"step": step_index, "block": block_index, "expert": expert,
            "initialization": self.initialization, "run_id": position["run_id"], "export_binding": position["export_binding"],
            "groups": position["groups"], "producer_result_binding": position["producer_result_binding"],
            "native_bias_parent_SHA": {"b1": position["original_b1_sha256"], "b2": position["original_b2_sha256"]}})
        return result

    def end_request(self, *, FFN_boundary_timings=None, full_scope=True):
        if not self.open or self.pending is not None:
            raise RuntimeError("Incomplete native W1/GELU/W2 pair")
        if full_scope:
            expected = {(s, b, n) for s in range(4) for b in range(self.blocks) for n in (0, 2)}
            count = 4 * self.blocks
            if (set(self.coverage) != expected or len(self.sites) != count
                    or self.counts != {"W1": count, "W2": count, "Q1": count, "Q2": count,
                                       "narrow_BF16_GEMM": 2 * count, "online_LoRA_GEMM": 0, "weight_QDQ": 0}):
                raise RuntimeError("Full4step forty-block narrow coverage incomplete")
            expected_other = {(s, b, name) for s in range(4) for b in range(self.blocks)
                for name in self.base.records[expected_expert(s)]
                if name.startswith(f"blocks.{b}.") and not name.endswith((".ffn.0", ".ffn.2"))}
            delta = {k: self.counters[k] - self.before[k] for k in self.before}
            if (len(expected_other) != 8 * count or self.non_ffn_coverage != expected_other
                    or delta != {"linear_forward_calls": 10 * count, "activation_qdq_calls": 10 * count,
                                 "weight_transfers": 0, "cache_misses": 0}):
                raise RuntimeError("Original all-depth attention/native counter ledger differs")
            timings = FFN_boundary_timings or []
            if (len(timings) != count or {(r["step"], r["block"]) for r in timings} != set(self.sites)
                    or any(not math.isfinite(r["elapsed_ms"]) or r["elapsed_ms"] < 0 for r in timings)):
                raise ValueError("Actual complete native FFN timer evidence required")
            for item in timings:
                self.sites[item["step"], item["block"]]["native_FFN_elapsed_ms"] = item["elapsed_ms"]
        self.open = False
        return self.report()

    def report(self):
        return {"initialization": self.initialization, "source_sha256": sha256(Path(__file__)),
                "bank_semantic_sha256": self.bank_semantic_sha256,
                "counts": dict(self.counts), "pending_pair": self.pending,
                "original_attention_projection_calls": len(self.non_ffn_coverage),
                "completed_native_sites": [self.sites[k] for k in sorted(self.sites)],
                "main_coverage": {f"step{s}_block{b:02d}_ffn{n}": value for (s, b, n), value in self.coverage.items()},
                "native_counter_delta": {k: self.counters[k] - self.before[k] for k in self.before},
                "input_boundary": "original norm2/AdaLN before original Smooth1/native Q1, once",
                "caller_GELU_gate_residual": "original native GELU(tanh), e5 and BF16 writeback unchanged",
                "Q2": "actual student6016 GELU/Smooth2/native HiF4; no teacher activation",
                "physical_scope": "two ordinary BF16 GEMMs per FFN; no online LoRA/weight QDQ",
                "nominal_two_GEMM_MAC_ratio": self.width / self.hidden,
                "residency": self.residency.snapshot(), "quality_or_speed_acceptance": False}
