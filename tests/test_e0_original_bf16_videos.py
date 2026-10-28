"""Tiny software controls; no Torch/NumPy/model/checkpoint tensor/GPU import."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_original_bf16_videos as q


class Tensor:
    def __init__(self, shape, dtype="BF16", device="cuda"):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = types.SimpleNamespace(type=device, index=0 if device == "cuda" else None)
        self.ndim = len(shape)

    def stride(self):
        return (1, self.shape[0]) if self.ndim == 2 else (1,)


class Torch:
    bfloat16 = "BF16"
    def __init__(self): self.mm_calls = []
    def is_tensor(self, value): return isinstance(value, Tensor)
    def empty(self, shape, *, dtype, device, requires_grad):
        assert requires_grad is False
        return Tensor(shape, dtype, device.type)
    def mm(self, value, weight, *, out): self.mm_calls.append(("mm", value, weight, out)); return out
    def addmm(self, bias, value, weight, *, out): self.mm_calls.append(("addmm", value, weight, out)); return out


def raw_mm(torch, *, k=5120, n=5120, bias=True):
    source = ROOT / ".runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py"
    cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name == "MMWeight")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "apply")
    env = {"torch": torch}; exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), env)
    MM = type("MMWeight", (), {"__module__": "lightx2v.common.ops.mm.mm_weight", "apply": env["apply"],
        "_get_actual_weight": lambda self: self.weight, "_get_actual_bias": lambda self: self.bias})
    mm = MM(); mm.weight, mm.bias = Tensor((k, n)), Tensor((n,)) if bias else None
    mm.bias_name = "bias" if bias else None; mm.has_lora_branch = mm.has_diff = False
    return mm


def base_plan():
    originals = q.read(ROOT / ".runtime/model_view.json")["files"]
    names = [("high", "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"),
             ("low", "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"),
             ("T5", "models_t5_umt5-xxl-enc-bf16.pth"), ("VAE", "Wan2.1_VAE.pth")]
    models = []
    for kind, name in names:
        row = next(x for x in originals if x["filename"] == name)
        models.append({"kind": kind, "path": "/fixture/model/" + name,
                       **{k: row[k] for k in ("filename", "sha256", "bytes", "repo", "revision")}})
    return {"id": q.ID, "numeric_path": "BF16_ORIGINAL", "limits": q.LIMITS, "resources": q.RESOURCE,
        "CPU_preflight_resources": q.CPU_RESOURCE, "runtime_overlay": q.recipe(), "model_view_root": "/fixture/model", "models": models,
        "cases": [{"index": s, "source_id": f"source{s}", "image_sha256": f"image{s}", "prompt_sha256": f"prompt{s}",
                   "prompt": f"p{s}", "image": {"path": f"images/{s}.png"}, "conditioning_id": f"condition{s}"} for s in (0, 12)],
        "repo_sources": [{"path": p} for p in q.TOOLS], "output_HF_relative": "ffnds-derived/tiny-original-bf16",
        "assets_finalized": True, "binding_status": "BOUND_READY_FOR_REGISTRATION"}


class PayloadControls(unittest.TestCase):
    def test_header_only_BF16_payload_offsets_and_no_tensor_imports(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "raw.safetensors"; h = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}).encode()
            path.write_bytes(struct.pack("<Q", len(h)) + h + b"\x00" * 12)
            record = q.header(path); self.assertEqual(record["tensor_count"], 1)
            h2 = json.dumps({"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
            path.write_bytes(struct.pack("<Q", len(h2)) + h2 + b"\x00" * 24)
            with self.assertRaisesRegex(ValueError, "Original BF16 checkpoint"):
                q.header(path)

    def test_header_overlap_gap_and_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "raw.safetensors"
            for offset, body in [([1, 13], 13), ([0, 12], 11)]:
                h = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": offset}}).encode()
                path.write_bytes(struct.pack("<Q", len(h)) + h + b"\x00" * body)
                with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "offsets|geometry"):
                    q.header(path)

    def test_original_schema_rejects_QDQ_and_narrow_payload(self):
        with self.assertRaisesRegex(ValueError, "unquantized DiT schema"):
            q.raw_header_check({"rows": {"smooth1": {"shape": [5120]}}})
        rows = {f"padding{i}": {} for i in range(1015)}
        for b in range(40):
            rows[f"blocks.{b}.ffn.0.weight"] = {"shape": [13824, 5120]}
            rows[f"blocks.{b}.ffn.2.weight"] = {"shape": [5120, 13824]}
        q.raw_header_check({"rows": rows})
        rows["blocks.17.ffn.0.weight"]["shape"] = [6016, 5120]
        with self.assertRaisesRegex(ValueError, "full-width FFN"):
            q.raw_header_check({"rows": rows})

    def test_actual_tiny_full_SHA_and_failure_preserve_read_debit(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "raw"; p.write_bytes(b"raw BF16 bytes")
            good = {"path": str(p), "bytes": p.stat().st_size, "sha256": q.sha(p)}
            result = q.full_hash(good); self.assertEqual(result["status"], "PASS_FRESH_FULL_SHA")
            self.assertEqual(result["actual_read_bytes"], good["bytes"])
            bad = {**good, "sha256": "f" * 64}; stop = threading.Event()
            failed = q.full_hash(bad, stop); self.assertTrue(stop.is_set())
            self.assertEqual(failed["actual_read_bytes"], failed["predebited_bytes"])
            stopped = q.full_hash(good, stop); self.assertEqual(stopped["actual_read_bytes"], 0)
            self.assertEqual(stopped["status"], "FAILED_OR_STOPPED_READ_PRESERVED_NO_RETRY")

    def test_actual_loader_result_keys_dtype_device_and_shape(self):
        torch = Torch(); rows = {"W1": {"shape": [3, 2]}, "b1": {"shape": [3]}}
        good = {k: Tensor(v["shape"], device="cpu") for k, v in rows.items()}
        q.loaded_raw_check(torch, good, rows)
        bads = [{**good, "smooth": Tensor((2,), device="cpu")}, {"W1": good["W1"]},
                {**good, "W1": Tensor((2, 3), device="cpu")}, {**good, "W1": Tensor((3, 2), dtype="F32", device="cpu")},
                {**good, "W1": Tensor((3, 2))}]
        for bad in bads:
            with self.subTest(keys=list(bad)), self.assertRaisesRegex(ValueError, "raw BF16 header"):
                q.loaded_raw_check(torch, bad, rows)


class DefaultExecutionControls(unittest.TestCase):
    def test_real_Default_apply_AST_delegates_original_mm_and_addmm_out(self):
        torch = Torch()
        for bias in (False, True):
            mm = raw_mm(torch, k=3, n=4, bias=bias); value = Tensor((2, 3))
            meta = q.operand_check(torch, mm, value); out = mm.apply(value)
            self.assertEqual(out.shape, (2, 4)); self.assertEqual(meta["dtype"], "BF16")
        self.assertEqual([x[0] for x in torch.mm_calls], ["mm", "addmm"])

    def test_wrong_default_class_adapter_and_diff_fail_before_GEMM(self):
        torch = Torch(); mm = raw_mm(torch, k=3, n=4); x = Tensor((2, 3))
        for flag in ("has_lora_branch", "has_diff"):
            setattr(mm, flag, True)
            with self.assertRaisesRegex(ValueError, "without adapters"):
                q.operand_check(torch, mm, x)
            setattr(mm, flag, False)
        mm.weight_diff = object()
        with self.assertRaisesRegex(ValueError, "without adapters"):
            q.operand_check(torch, mm, x)
        self.assertEqual(torch.mm_calls, [])

    def test_dtype_device_shape_and_bias_fail_before_GEMM(self):
        torch = Torch(); mm = raw_mm(torch, k=3, n=4)
        for value in (Tensor((2, 3), dtype="FP4"), Tensor((2, 3), device="cpu"), Tensor((2, 2))):
            with self.assertRaisesRegex(ValueError, "operand dtype/device/shape"):
                q.operand_check(torch, mm, value)
        mm.bias = Tensor((5,))
        with self.assertRaisesRegex(ValueError, "BF16 bias"):
            q.operand_check(torch, mm, Tensor((2, 3)))
        self.assertEqual(torch.mm_calls, [])

    def test_two_full_tiny_native_dispatch_traces_320FFN_3248Default_zero_QDQ(self):
        torch = Torch(); record, journal = {}, []; ledger = q.Ledger(record, journal.append)
        guard = types.SimpleNamespace(torch=torch, ledger=ledger)
        scheduler = types.SimpleNamespace(step_index=0); pipe = types.SimpleNamespace(runner=types.SimpleNamespace(scheduler=scheduler))
        for rid in q.RUN_IDS:
            tap = q.NativeTap(pipe, guard, rid, None, lambda: None); tap.raw_draws = 1
            tap.t5_model = types.SimpleNamespace(block_idx=0); tap.t5_encoding = True
            t5 = {}
            for suffix in q.T5_NAMES:
                mm = raw_mm(torch, k=4096, n=4096, bias=False); t5[suffix] = mm; tap.t5_mapping[id(mm)] = suffix
            for block in range(24):
                tap.t5_model.block_idx = block; tap.t5_blocks.append(block)
                for suffix in q.T5_NAMES:
                    mm = t5[suffix]; x = Tensor((5, 4096))
                    tap.mm_call(mm, x, q.operand_check(torch, mm, x), type(mm).apply)
            tap.t5_encoding = False
            matrices = {}
            for expert in (0, 1):
                for role, names in (("pre", q.PRE_NAMES), ("head", ("head",)), ("block", q.LINEAR_SUFFIXES)):
                    for name in names:
                        k, n = (5120, 13824) if name == "ffn.0" else (13824, 5120) if name == "ffn.2" else (5120, 5120)
                        mm = raw_mm(torch, k=k, n=n); matrices[expert, role, name] = mm; tap.mapping[id(mm)] = (role, expert, name)
            for step in range(4):
                scheduler.step_index = step; expert = 0 if step < 2 else 1
                for name in q.PRE_NAMES:
                    mm = matrices[expert, "pre", name]; x = Tensor((1, 5120))
                    tap.mm_call(mm, x, q.operand_check(torch, mm, x), type(mm).apply)
                for block in range(40):
                    tap.block = (expert, block)
                    for name in q.LINEAR_SUFFIXES:
                        mm = matrices[expert, "block", name]; x = Tensor((31668, mm.weight.shape[0]))
                        tap.mm_call(mm, x, q.operand_check(torch, mm, x), type(mm).apply)
                tap.block = None; mm = matrices[expert, "head", "head"]; x = Tensor((31668, 5120))
                tap.mm_call(mm, x, q.operand_check(torch, mm, x), type(mm).apply)
            tap.finish(); self.assertEqual(len(tap.lines), 1600)
        self.assertEqual(len(torch.mm_calls), 3248 + 336)
        self.assertEqual(record["observed"]["T5_Default_MM_calls"], 336)
        self.assertEqual(record["observed"]["native_FFN_linears"], 640)
        self.assertEqual(record["observed"]["native_non_FFN_linears"], 2560)
        self.assertEqual(record["observed"]["pre_Default_MM_calls"], 40)
        self.assertEqual(record["observed"]["head_Default_MM_calls"], 8)
        self.assertEqual(record["observed"].get("activation_QDQ", 0), 0)

    def test_T5_and_DiT_wrong_phase_expert_stride_and_order_stop_before_dispatch(self):
        torch = Torch(); record = {}; ledger = q.Ledger(record, lambda _: None)
        guard = types.SimpleNamespace(torch=torch, ledger=ledger)
        scheduler = types.SimpleNamespace(step_index=0); pipe = types.SimpleNamespace(runner=types.SimpleNamespace(scheduler=scheduler))
        tap = q.NativeTap(pipe, guard, q.RUN_IDS[0], None, lambda: None)
        mm = raw_mm(torch, k=3, n=4); x = Tensor((2, 3)); meta = q.operand_check(torch, mm, x)
        tap.t5_mapping[id(mm)] = q.T5_NAMES[0]; tap.t5_model = types.SimpleNamespace(block_idx=0)
        with self.assertRaisesRegex(ValueError, "Wrong phase/duplicate/out-of-order original T5"):
            tap.mm_call(mm, x, meta, type(mm).apply)
        tap.t5_mapping.clear(); tap.mapping[id(mm)] = ("block", 1, q.LINEAR_SUFFIXES[0]); tap.block = (1, 0)
        with self.assertRaisesRegex(ValueError, "step/expert"):
            tap.mm_call(mm, x, meta, type(mm).apply)
        tap.mapping[id(mm)] = ("block", 0, q.LINEAR_SUFFIXES[1]); tap.block = (0, 0)
        with self.assertRaisesRegex(ValueError, "out-of-order original full-depth MM"):
            tap.mm_call(mm, x, meta, type(mm).apply)
        mm.weight.stride = lambda: (4, 1)
        with self.assertRaisesRegex(ValueError, "weight stride"):
            q.operand_check(torch, mm, x)
        self.assertEqual(torch.mm_calls, []); self.assertEqual(record["predebited"], {})

    def test_failed_dispatch_keeps_predebit_and_never_observes_success(self):
        record = {}; ledger = q.Ledger(record, lambda _: None)
        def fail(): raise RuntimeError("real original failure")
        with self.assertRaisesRegex(RuntimeError, "real original failure"):
            ledger.call(q.RUN_IDS[0], {"video_generations": 1}, fail)
        self.assertEqual(record["predebited"]["video_generations"], 1); self.assertEqual(record["observed"], {})

    def test_QDQ_and_extra_teacher_budget_reject_before_dispatch(self):
        calls, record = [], {}; ledger = q.Ledger(record, lambda _: None)
        for key in ("activation_QDQ", "weight_quantizations", "extra_teacher_flows", "official_quality_calls"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "budget exceeded"):
                ledger.call(q.RUN_IDS[0], {key: 1}, lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_HiF4_imports_and_quant_constructor_are_explicit_failures(self):
        finder = q.NoHiF4()
        for name in ("HiFloat4", "HiFloat4.ptq.adapters.lightx2v_wan_distill", "quant.hif4_backend"):
            with self.assertRaisesRegex(ImportError, "outside BF16_ORIGINAL"):
                finder.find_spec(name)
        self.assertIsNone(finder.find_spec("lightx2v.common.ops.mm.mm_weight"))
        record = {}; ledger = q.Ledger(record, lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "quantized_MM_constructions"):
            ledger.forbidden("quantized_MM_constructions")()
        self.assertEqual(record["forbidden_attempts"], ["quantized_MM_constructions"])


class ProvenanceAndPolicyControls(unittest.TestCase):
    def test_fixed_recipe_reject_NVFP4_schedule_and_quantized_payload(self):
        plan = base_plan(); q.validate(plan, True)
        for key, value in (("dit_quantized", True), ("dit_quant_scheme", "nvfp4"), ("sample_shift", 1.), ("self_attn_1_type", "sage_attn2")):
            bad = copy.deepcopy(plan); bad["runtime_overlay"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "scope/recipe"):
                q.validate(bad, True)
        bad = copy.deepcopy(plan); bad["models"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "public original model"):
            q.validate(bad, True)

    def test_two_source_single_generation_cohort_has_correct_exposure_and_no_new_N(self):
        value = q.cohort(base_plan(), Path("/fixture/hf"))
        self.assertEqual(len(value["samples"]), 2); self.assertEqual(value["required_variants"], ["dense"])
        self.assertEqual([x["seed"] for x in value["samples"]], [42, 42])
        self.assertTrue(all(x["role"] == "DEVELOPMENT" and x["exposure_history"]["prior_evaluation"] for x in value["sources"]))

    @mock.patch.object(q, "FULL_READ_BYTES", 16)
    def test_CPU_proof_exited_exact_RUN_SHA_and_current_payload_stat(self):
        with tempfile.TemporaryDirectory() as t:
            hf = Path(t); models = []
            for i in range(4):
                p = hf / f"model{i}"; p.write_bytes(bytes([i + 1]) * 4)
                models.append({"path": str(p), "bytes": 4, "sha256": q.sha(p)})
            plan = {"repo_sources": [], "models": models, "model_view_metadata": []}; proof = q.CPU_expected(plan, "a" * 64)
            proof.update(wall_seconds=1., model_payloads=[q.full_hash(x) for x in models])
            rp, ep = hf / "CPU.json", hf / "EXIT.json"
            def resign(report, **overrides):
                rp.write_text(json.dumps(report)); rb = q.binding(rp, hf)
                exit = {"id": q.ID, "protocol_sha256": "a" * 64, "process_exited": True, "child_returncode": 0, "timeout": False, "automatic_retry": False, "run_sha256": rb["sha256"], **overrides}
                ep.write_text(json.dumps(exit)); return {"cpu_preflight": {"RUN": rb, "launch_exit": q.binding(ep, hf)}}
            q.require_CPU(resign(proof), plan, "a" * 64, hf)
            with self.assertRaisesRegex(ValueError, "CPU proof NOT_BOUND"): q.require_CPU({}, plan, "a" * 64, hf)
            for field, value in (("GPU_calls", 1), ("actual_full_payload_read_bytes", q.FULL_READ_BYTES - 1), ("protocol_sha256", "stale"), ("NumPy_or_Torch_imported", True)):
                bad = copy.deepcopy(proof); bad[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity/count"):
                    q.require_CPU(resign(bad), plan, "a" * 64, hf)
            for override in ({"child_returncode": 1}, {"child_returncode": False}, {"run_sha256": "f" * 64}, {"timeout": True}):
                with self.subTest(override=override), self.assertRaisesRegex(ValueError, "actual exit0/RUN SHA"):
                    q.require_CPU(resign(proof, **override), plan, "a" * 64, hf)
            Path(models[0]["path"]).write_bytes(b"xxxx")
            with self.assertRaisesRegex(ValueError, "changed after independent hash"):
                q.require_CPU(resign(proof), plan, "a" * 64, hf)

    def test_registered_job1_session2_dynamic_repository_and_unique_run_ids(self):
        plan = base_plan()
        active = {"user_continuation_authorization_20261027": {"status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION", "new_scope_registration_allowed": True},
            "resource_scheduling_policy_20261028": {"mode": "DYNAMIC_MAX_ACTUALLY_CALLABLE", "global_gpu_card_cap": None},
            "current_session_resource_constraint_20261028": {"max_simultaneously_assigned_idle_GPUs": 2},
            "experiment_authority": {"current_user_authorized_scopes": [{"id": q.ID, "protocol_sha256": "a" * 64, "execution_status": "AUTHORIZED_NOT_STARTED",
                "limits": q.LIMITS, "resources": q.RESOURCE, "max_gpu_cards": 1, "run_ids": list(q.RUN_IDS)}]}}
        original = q.read
        with mock.patch.object(q, "read", side_effect=lambda p: active if str(p).endswith("ACTIVE_RESEARCH.json") else original(p)):
            q.authorize(plan, "a" * 64, q.RUN_IDS[0])
            with self.assertRaisesRegex(ValueError, "registration"): q.authorize(plan, "b" * 64)
            with self.assertRaisesRegex(ValueError, "registration"): q.authorize(plan, "a" * 64, "other")
            active["resource_scheduling_policy_20261028"]["global_gpu_card_cap"] = 2
            with self.assertRaisesRegex(ValueError, "Repository/session"): q.authorize(plan, "a" * 64)

    def test_no_real_model_tensor_or_GPU_modules_imported(self):
        self.assertNotIn("torch", sys.modules); self.assertNotIn("numpy", sys.modules)
        self.assertFalse(any(x == "HiFloat4" or x.startswith("HiFloat4.") for x in sys.modules))


if __name__ == "__main__": unittest.main()
