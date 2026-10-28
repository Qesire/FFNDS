"""Small synthetic CPU controls; no real model/asset forward or CUDA use."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compact6016_native_runtime as runtime_module
from compact6016_adapter import compact_forward
from ffnds_all_weight_residency import AllWeightResidency


def fixture(initialization="energy94", block=0, expert="high_noise_model"):
    generator = torch.Generator().manual_seed(200 + block + (expert == "low_noise_model"))
    def sample(*shape):
        return torch.randn(shape, generator=generator).bfloat16()
    groups = torch.tensor([0, 2], dtype=torch.int64)
    channels = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    up = {"weight_qdq": sample(8, 4), "smooth_scale": torch.tensor([1., 2., 4., 8.]),
          "activation_qdq": True, "force_fp32": True}
    down = {"weight_qdq": sample(4, 8), "smooth_scale": torch.arange(8).float() + 1,
            "activation_qdq": True, "force_fp32": True}
    b1, b2 = sample(8), sample(4)
    pack = {"W1": sample(4, 4), "W2": sample(4, 4), "b1": b1[channels].contiguous(), "b2": b2,
            "smooth1": up["smooth_scale"].clone(), "smooth2": down["smooth_scale"][channels].contiguous(),
            "groups": groups, "channels": channels}
    cell = {"groups": [0, 2]}
    meta = runtime_module.validate_deployment_pack(pack, cell, up, down, b1, b2,
                                                   dim=4, hidden=8, width=4, group_size=2)
    position = {**pack, **meta, "expert": expert, "block": block, "initialization": initialization,
                "run_id": runtime_module.cell_id(expert, block, initialization),
                "export_binding": {"sha256": "a" * 64}, "producer_result_binding": {"sha256": "b" * 64},
                "bank_semantic_sha256": "c" * 64}
    return position, pack, up, down, b1, b2, sample(2, 4)


class Bias:
    def __init__(self, value):
        self.value = value
    def _get_actual_bias(self):
        return self.value


class FakeBase:
    def __init__(self, blocks=1):
        self.records = {expert: {} for expert in runtime_module.EXPERTS}
        for expert in runtime_module.EXPERTS:
            for block in range(blocks):
                for n in range(8):
                    self.records[expert][f"blocks.{block}.attn.{n}"] = {"weight_qdq": torch.eye(4).bfloat16()}
                self.records[expert][f"blocks.{block}.ffn.0"] = {"weight_qdq": torch.ones(8, 4).bfloat16()}
                self.records[expert][f"blocks.{block}.ffn.2"] = {"weight_qdq": torch.ones(4, 8).bfloat16()}
        self.counters = dict.fromkeys(("linear_forward_calls", "activation_qdq_calls", "weight_transfers", "cache_misses"), 0)
        self._resident_weights, self._resident_key = {}, None
        self.delegations = []
    def _weight(self, **kw):
        return kw["record"]["weight_qdq"]
    def apply(self, mm, value, **kw):
        self.delegations.append((mm, value, kw))
        self.counters["linear_forward_calls"] += 1
        self.counters["activation_qdq_calls"] += 1
        return value


def quantizer(value, *, force_fp32=True):
    if force_fp32 is not True:
        raise AssertionError("Original quantizer force_fp32 required")
    return ((value.float() * 4).round() / 4).to(value.dtype)


def cpu_residency(positions):
    return SimpleNamespace(position=lambda key: positions[key], snapshot=lambda: {"software_test_only": True})


class CompactNativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_runtime(self, init="energy94"):
        p, pack, up, down, b1, b2, x = fixture(init)
        base = FakeBase()
        positions = {("high_noise_model", 0): p}
        calls = []
        def observed(value, **kw):
            calls.append(value.clone())
            return quantizer(value, **kw)
        runtime = runtime_module.Compact6016Runtime(base, positions, cpu_residency(positions), observed,
            initialization=init, dim=4, hidden=8, width=4, expected_rows=2, blocks=1)
        return runtime, base, p, pack, b1, b2, x, calls

    def test_pack_preserves_exact_native_selected_bias_scale_support(self):
        p, pack, up, down, b1, b2, _ = fixture()
        self.assertEqual(p["groups"], [0, 2])
        self.assertEqual(p["width"], 4)
        self.assertFalse(torch.equal(pack["W1"], up["weight_qdq"][pack["channels"]]))
        for key in ("b1", "b2", "smooth1", "smooth2"):
            wrong = deepcopy(pack)
            wrong[key].view(-1)[0] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "byte identity"):
                runtime_module.validate_deployment_pack(wrong, {"groups": [0, 2]}, up, down, b1, b2,
                                                        dim=4, hidden=8, width=4, group_size=2)

    def test_pack_rejects_reordered_or_partial_groups_and_training_factor(self):
        _, pack, up, down, b1, b2, _ = fixture()
        for modify in (lambda p: p["channels"].__setitem__(0, 2),
                       lambda p: p["groups"].__setitem__(0, 2),
                       lambda p: p.__setitem__("training_lora_A1", torch.zeros(2, 4))):
            wrong = deepcopy(pack)
            modify(wrong)
            with self.assertRaises(ValueError):
                runtime_module.validate_deployment_pack(wrong, {"groups": [0, 2]}, up, down, b1, b2,
                                                        dim=4, hidden=8, width=4, group_size=2)

    def test_actual_hook_matches_frozen_postQ1_student_and_smooth_is_once(self):
        runtime, base, p, pack, b1, b2, x, calls = self.make_runtime()
        q1 = quantizer((x.float() / pack["smooth1"]).bfloat16())
        expected, trace = compact_forward(q1, pack["W1"], pack["b1"], pack["W2"], pack["b2"],
                                          pack["smooth2"], quantizer, return_trace=True)
        runtime.begin_request()
        kw = dict(expert="high_noise_model", step_index=0, block_index=0)
        original = F.linear
        with mock.patch("torch.nn.functional.linear", wraps=original) as linear:
            hidden = runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **kw)
            result = runtime.apply(Bias(b2), F.gelu(hidden, approximate="tanh"), module_name="blocks.0.ffn.2", **kw)
        self.assertEqual(linear.call_count, 2)
        self.assertIs(linear.call_args_list[0].args[2], p["b1"])
        self.assertEqual(tuple(linear.call_args_list[0].args[2].shape), (4,))
        self.assertTrue(torch.equal(result.view(torch.uint8), expected.view(torch.uint8)))
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0], (x.float() / pack["smooth1"]).bfloat16()))
        self.assertTrue(torch.equal(calls[1], trace["pre_Q2"]))
        report = runtime.end_request(full_scope=False)
        self.assertEqual(report["counts"]["online_LoRA_GEMM"], 0)
        self.assertEqual(report["counts"]["weight_QDQ"], 0)

    def test_attention_delegates_original_objects_and_binding(self):
        runtime, base, p, pack, b1, b2, x, _ = self.make_runtime()
        runtime.begin_request()
        mm = object()
        kw = dict(expert="high_noise_model", step_index=0, block_index=0, module_name="blocks.0.attn.0")
        result = runtime.apply(mm, x, **kw)
        self.assertIs(result, x)
        self.assertIs(base.delegations[0][0], mm)
        self.assertIs(base.delegations[0][1], x)
        self.assertEqual(base.delegations[0][2], kw)
        with self.assertRaises(ValueError):
            runtime.apply(mm, x, **kw)

    def test_pair_expert_shape_and_original_bias_guards(self):
        runtime, _, p, pack, b1, b2, x, _ = self.make_runtime()
        kw = dict(expert="high_noise_model", step_index=0, block_index=0)
        with self.assertRaises(RuntimeError):
            runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **kw)
        runtime.begin_request()
        with self.assertRaises(RuntimeError):
            runtime.apply(Bias(b2), x, module_name="blocks.0.ffn.2", **kw)
        for change in ({"expert": "low_noise_model"}, {"step_index": 2}, {"block_index": 1}):
            with self.assertRaises(ValueError):
                runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **{**kw, **change})
        for value in (x.float(), x[:1], x[:, :3]):
            with self.assertRaises(ValueError):
                runtime.apply(Bias(b1), value, module_name="blocks.0.ffn.0", **kw)
        with self.assertRaisesRegex(ValueError, "full-width"):
            runtime.apply(Bias(pack["b1"]), x, module_name="blocks.0.ffn.0", **kw)
        runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **kw)
        with self.assertRaises(RuntimeError):
            runtime.begin_request()

    def test_complete_coverage_requires_all_attention_and_timings(self):
        base, positions, originals = FakeBase(), {}, {}
        for expert in runtime_module.EXPERTS:
            p, _, _, _, b1, b2, x = fixture(expert=expert)
            positions[expert, 0], originals[expert] = p, (b1, b2, x)
        runtime = runtime_module.Compact6016Runtime(base, positions, cpu_residency(positions), quantizer,
            initialization="energy94", dim=4, hidden=8, width=4, expected_rows=2, blocks=1)
        runtime.begin_request()
        for step in range(4):
            expert = "high_noise_model" if step < 2 else "low_noise_model"
            b1, b2, x = originals[expert]
            kw = dict(expert=expert, step_index=step, block_index=0)
            for i in range(8):
                runtime.apply(None, x, module_name=f"blocks.0.attn.{i}", **kw)
            h = runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **kw)
            runtime.apply(Bias(b2), F.gelu(h, approximate="tanh"), module_name="blocks.0.ffn.2", **kw)
        with self.assertRaisesRegex(ValueError, "timer"):
            runtime.end_request()
        report = runtime.end_request(FFN_boundary_timings=[{"step": s, "block": 0, "elapsed_ms": .1} for s in range(4)])
        self.assertEqual(len(report["main_coverage"]), 8)
        self.assertEqual(len(report["completed_native_sites"]), 4)
        self.assertEqual(report["original_attention_projection_calls"], 32)
        self.assertEqual(report["native_counter_delta"]["linear_forward_calls"], 40)

    def test_no_online_tensor_hash_or_full_width_ffn(self):
        runtime, _, p, pack, b1, b2, x, _ = self.make_runtime()
        runtime.begin_request()
        kw = dict(expert="high_noise_model", step_index=0, block_index=0)
        with mock.patch.object(runtime_module, "tensor_sha", side_effect=AssertionError("online tensor observer")):
            h = runtime.apply(Bias(b1), x, module_name="blocks.0.ffn.0", **kw)
            runtime.apply(Bias(b2), F.gelu(h, approximate="tanh"), module_name="blocks.0.ffn.2", **kw)
        self.assertEqual(runtime.coverage[0, 0, 0]["weight_shape"], [4, 4])

    def test_residency_shares_expert_steps_and_prohibits_dense_cache(self):
        base = FakeBase()
        positions = {("high_noise_model", 0): fixture()[0]}
        original = base._weight
        with runtime_module.Compact6016Residency(base, positions, torch.device("cpu"), 1 << 20) as policy:
            policy.preload()
            first = policy.position(("high_noise_model", 0))
            second = policy.position(("high_noise_model", 0))
            self.assertIs(first["W1"], second["W1"])
            self.assertEqual(policy.snapshot()["FFN_function_copies"], 1)
            self.assertEqual(policy.snapshot()["setup_matrix_transfers"], 2)
            self.assertEqual(base.counters["weight_transfers"], 18)  # 16 original attention +2 new matrices.
            with self.assertRaisesRegex(RuntimeError, "dense GPU"):
                base._weight(expert="high_noise_model", block_index=0, module_name="blocks.0.ffn.0",
                    record=base.records["high_noise_model"]["blocks.0.ffn.0"], device=torch.device("cpu"), dtype=torch.bfloat16)
            with self.assertRaises(RuntimeError):
                policy.preload()
        self.assertEqual(base._weight, original)
        self.assertEqual(policy.stats["narrow_bytes"], 0)
        self.assertFalse(hasattr(base, "_compact6016_residency_active"))

    def test_residency_rejects_overlap_and_insufficient_budget(self):
        base = FakeBase()
        positions = {("high_noise_model", 0): fixture()[0]}
        with AllWeightResidency(base, "all_resident_no_flush", 1 << 20):
            with self.assertRaisesRegex(RuntimeError, "resident"):
                with runtime_module.Compact6016Residency(base, positions, torch.device("cpu"), 1 << 20):
                    pass
        with self.assertRaisesRegex(RuntimeError, "budget"):
            with runtime_module.Compact6016Residency(base, positions, torch.device("cpu"), 1):
                pass

    def test_manifest_rejects_incomplete_producers_before_any_tensor_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "protocol.json"
            path.write_text(json.dumps({"id": runtime_module.PRODUCER_ID, "assets_finalized": True,
                                       "initializations": list(runtime_module.INITIALIZATIONS)}))
            binding = {"path": path.name, "sha256": runtime_module.sha256(path)}
            with self.assertRaisesRegex(ValueError, "four"):
                runtime_module.assemble_manifest(binding, [], root, repo_root=root)


if __name__ == "__main__":
    unittest.main()
