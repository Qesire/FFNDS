import copy
import hashlib
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
from ub1_consumer_runtime import (ConsumerResidency, UB1ConsumerRuntime, bytes_of_tensors,
    confined, drop_dense_ffn_records, expected_expert, sha256, tensor_sha,
    validate_pack_metadata, verify_binding)
from run_ub1_consumer_videos import (ExecutionLedger, MODES, PER_CASE_LIMITS,
    compare_historical, object_sha, validate_authorization, validate_native_counts,
    validate_plan, verify_coefficient_chains, generate_phase)


class Bias:
    def __init__(self, value):
        self.value = value

    def _get_actual_bias(self):
        return self.value


class FakeBase:
    def __init__(self):
        self.records = {e: {} for e in ("high_noise_model", "low_noise_model")}
        self.counters = dict.fromkeys(("linear_forward_calls", "activation_qdq_calls", "weight_transfers", "cache_misses"), 0)
        self._resident_weights, self._resident_key = {}, None
        self._weight = lambda **kw: "old weight"

    def apply(self, *args, **kwargs):
        return "unchanged original non-FFN"


def fixture(step=0, block=0, width=4):
    generator = torch.Generator().manual_seed(10 + step * 40 + block)
    def sample(*shape):
        return torch.randn(shape, generator=generator).bfloat16()
    dim, hidden, rank = 4, 8, 2
    channels = torch.arange(width)
    bias_generator = torch.Generator().manual_seed(1000 + (step // 2) * 40 + block)
    b1 = torch.randn((hidden,), generator=bias_generator).bfloat16()
    original_b2 = torch.randn((dim,), generator=bias_generator).bfloat16()
    rr = {"first": sample(rank, dim), "second": sample(dim, rank),
          "bias_first": sample(rank), "bias_second": sample(dim)}
    p = {"step": step, "block": block, "expert": expected_expert(step),
         "width": width, "channels": channels, "groups": list(range(width // 2)),
         "alpha": 1.37, "w1": sample(width, dim), "b1": b1[:width].contiguous(),
         "w2": sample(dim, width), "bias": sample(dim), "rr": rr,
         "s1": torch.arange(dim).float() + 1, "s2": torch.arange(width).float() + 2,
         "original_b1_sha256": tensor_sha(b1), "original_b2_sha256": tensor_sha(original_b2),
         "original_w2_tensor_sha256": "original_w2",
         "RR_parameter_sha256": {k: tensor_sha(v) for k, v in rr.items()}}
    return p, Bias(b1), Bias(original_b2), sample(2, dim)


def fake_residency(positions):
    return SimpleNamespace(position=lambda key: positions[key], snapshot=lambda: {})


class ConsumerRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make(self, *, step=0, block=0):
        p, mm1, mm2, x = fixture(step, block)
        positions = {(step, block): p}
        calls = []
        def qdq(value, *, force_fp32):
            self.assertTrue(force_fp32)
            calls.append(value.clone())
            return value
        base = FakeBase()
        runtime = UB1ConsumerRuntime(base, positions, fake_residency(positions), qdq,
            dim=4, hidden=8, group_size=2, rank=2, expected_rows=2)
        kw = dict(expert=expected_expert(step), step_index=step, block_index=block)
        return runtime, base, p, mm1, mm2, x, calls, kw

    def test_exact_native_order_and_original_rr_once(self):
        runtime, base, p, mm1, mm2, x, calls, kw = self.make()
        q1 = (x.float() / p["s1"]).bfloat16()
        expected_up = F.linear(q1, p["w1"], p["b1"])
        h = F.gelu(expected_up, approximate="tanh")
        q2 = (h.float() / p["s2"]).bfloat16()
        expected_main = F.linear((q2.float() * p["alpha"]).bfloat16(), p["w2"], p["bias"])
        expected_rr = F.linear(F.linear(q1, p["rr"]["first"], p["rr"]["bias_first"]),
                               p["rr"]["second"], p["rr"]["bias_second"])
        runtime.begin_request()
        self.assertEqual(runtime.apply(None, x, module_name="blocks.0.self_attn.q", **kw), "unchanged original non-FFN")
        original_linear = F.linear
        with mock.patch("torch.nn.functional.linear", wraps=original_linear) as linear:
            z = runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **kw)
            result = runtime.apply(mm2, F.gelu(z, approximate="tanh"), module_name="blocks.0.ffn.2", **kw)
            self.assertEqual(linear.call_count, 4)
            torch.testing.assert_close(linear.call_args_list[0].args[0], q1, rtol=0, atol=0)
            self.assertIs(linear.call_args_list[0].args[1], p["rr"]["first"])
            self.assertIs(linear.call_args_list[1].args[1], p["rr"]["second"])
            self.assertIs(linear.call_args_list[2].args[1], p["w1"])
            self.assertIs(linear.call_args_list[3].args[1], p["w2"])
        torch.testing.assert_close(z, expected_up, rtol=0, atol=0)
        torch.testing.assert_close(result, expected_main + expected_rr, rtol=0, atol=0)
        torch.testing.assert_close(calls[0], q1, rtol=0, atol=0)
        torch.testing.assert_close(calls[1], q2, rtol=0, atol=0)
        report = runtime.end_request(full_scope=False)
        self.assertEqual(report["counts"], {"main_linear": 2, "RR_linear": 2, "Q1": 1, "Q2": 1, "up": 1, "down": 1})
        self.assertEqual(base.counters["activation_qdq_calls"], 2)
        self.assertEqual(report["pending_RR"], 0)
        runtime.begin_request()
        self.assertFalse(runtime.coverage)
        self.assertEqual(runtime.counts["RR_linear"], 0)

    def test_pair_order_reset_expert_shape_and_dtype_guards(self):
        runtime, _, _, mm1, mm2, x, _, kw = self.make()
        with self.assertRaisesRegex(RuntimeError, "reset"):
            runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **kw)
        runtime.begin_request()
        with self.assertRaisesRegex(RuntimeError, "preceding"):
            runtime.apply(mm2, x, module_name="blocks.0.ffn.2", **kw)
        for changed in ({**kw, "expert": "low_noise_model"}, {**kw, "block_index": 1}):
            with self.assertRaises(ValueError):
                runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **changed)
        for value in (x.float(), x[:1], x[:, :3]):
            with self.assertRaises(ValueError):
                runtime.apply(mm1, value, module_name="blocks.0.ffn.0", **kw)
        runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **kw)
        with self.assertRaises(RuntimeError):
            runtime.begin_request()
        with self.assertRaises(RuntimeError):
            runtime.end_request(full_scope=False)
        with self.assertRaises(ValueError):
            runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **kw)

    def test_native_bias_dtype_and_shape_guard_and_setup_sha(self):
        runtime, _, p, _, _, x, _, kw = self.make()
        self.assertEqual(p["original_b1_sha256"], tensor_sha(torch.cat((p["b1"], fixture()[1].value[4:]))))
        runtime.begin_request()
        with self.assertRaisesRegex(ValueError, "bias source"):
            runtime.apply(Bias(torch.zeros(8)), x, module_name="blocks.0.ffn.0", **kw)
        with self.assertRaisesRegex(ValueError, "bias geometry"):
            runtime.apply(Bias(torch.zeros(7).bfloat16()), x, module_name="blocks.0.ffn.0", **kw)

    def test_full160_actual_call_coverage_and_fresh_request(self):
        positions, biases = {}, {}
        for step in range(4):
            for block in range(40):
                p, mm1, mm2, x = fixture(step, block)
                positions[step, block], biases[step, block] = p, (mm1, mm2, x)
        runtime = UB1ConsumerRuntime(FakeBase(), positions, fake_residency(positions), lambda x, **kw: x,
            dim=4, hidden=8, group_size=2, rank=2, expected_rows=2)
        runtime.begin_request()
        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            for (step, block), (mm1, mm2, x) in biases.items():
                kw = dict(expert=expected_expert(step), step_index=step, block_index=block)
                z = runtime.apply(mm1, x, module_name=f"blocks.{block}.ffn.0", **kw)
                runtime.apply(mm2, F.gelu(z, approximate="tanh"), module_name=f"blocks.{block}.ffn.2", **kw)
            self.assertEqual(linear.call_count, 640)
        report = runtime.end_request()
        self.assertEqual(len(report["main_coverage"]), 320)
        self.assertEqual(len(report["RR_coverage"]), 160)
        self.assertEqual(report["counts"]["RR_linear"], 320)
        runtime.begin_request()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            runtime.end_request()

    def test_no_online_tensor_hash(self):
        runtime, _, _, mm1, mm2, x, _, kw = self.make()
        runtime.begin_request()
        with mock.patch("ub1_consumer_runtime.tensor_sha", side_effect=AssertionError("online CPU observer")):
            z = runtime.apply(mm1, x, module_name="blocks.0.ffn.0", **kw)
            runtime.apply(mm2, F.gelu(z, approximate="tanh"), module_name="blocks.0.ffn.2", **kw)
        runtime.end_request(full_scope=False)

    def test_residency_has_distinct_step_variants_and_releases_only_consumers(self):
        base = FakeBase()
        base.records["high_noise_model"]["blocks.0.self_attn.q"] = {"weight_qdq": torch.ones(4, 4).bfloat16()}
        positions = {key: fixture(*key, width=2 if key[0] == 0 else 4)[0] for key in ((0, 0), (1, 0))}
        old = base._weight
        with ConsumerResidency(base, positions, torch.device("cpu"), 1 << 20) as resident:
            resident.preload()
            first, second = resident.position((0, 0)), resident.position((1, 0))
            self.assertEqual(first["w1"].shape, (2, 4))
            self.assertEqual(second["w1"].shape, (4, 4))
            w1, rr = first["w1"], first["rr"]["first"]
            with self.assertRaisesRegex(RuntimeError, "coexist"):
                resident.load_consumers()
            resident.unload_consumers()
            self.assertEqual(resident.snapshot()["resident_consumer_positions"], 0)
            self.assertIs(first["w1"], w1)
            self.assertIs(first["rr"]["first"], rr)
            positions[0, 0]["w2"] = torch.full_like(positions[0, 0]["w2"], 5)
            resident.load_consumers()
            self.assertEqual(float(first["w2"][0, 0]), 5)
            value = base._weight(expert="high_noise_model", block_index=0, module_name="blocks.0.self_attn.q",
                record=None, device=torch.device("cpu"), dtype=torch.bfloat16)
            self.assertEqual(tuple(value.shape), (4, 4))
            with self.assertRaisesRegex(RuntimeError, "dense GPU"):
                base._weight(expert="high_noise_model", block_index=0, module_name="blocks.0.ffn.0",
                    record=None, device=torch.device("cpu"), dtype=torch.bfloat16)
            self.assertEqual(resident.stats["retained_bytes"], bytes_of_tensors(resident.non_ffn) + bytes_of_tensors(resident.packed))
        self.assertIs(base._weight, old)
        self.assertEqual(resident.stats["retained_bytes"], 0)

    def test_residency_limit_before_transfer_and_dense_reference_release(self):
        base = FakeBase()
        positions = {(0, 0): fixture()[0]}
        with ConsumerResidency(base, positions, torch.device("cpu"), 1) as resident:
            with self.assertRaisesRegex(RuntimeError, "before transfer"):
                resident.preload()
            self.assertEqual(resident.stats["setup_tensor_transfers"], 0)
        base.records["high_noise_model"]["blocks.0.ffn.0"] = {"weight_qdq": torch.ones(2, 4).bfloat16()}
        base._resident_weights["leftover"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "must be empty"):
            drop_dense_ffn_records(base)
        base._resident_weights.clear()
        self.assertEqual(drop_dense_ffn_records(base), 16)
        self.assertFalse(base.records["high_noise_model"])

    def test_manifest_support_alpha_rr_and_original_source_sha(self):
        p = fixture()[0]
        item = {k: p[k] for k in ("step", "block", "expert", "groups", "alpha", "RR_parameter_sha256",
                                   "original_w2_tensor_sha256", "original_b2_sha256")}
        validate_pack_metadata(item, p, p["groups"], p["alpha"], p["rr"])
        for key, wrong in (("groups", [1, 2]), ("alpha", 1.4), ("RR_parameter_sha256", {}),
                           ("original_b2_sha256", "different"), ("original_w2_tensor_sha256", "different")):
            changed = {**item, key: wrong}
            with self.assertRaises(ValueError):
                validate_pack_metadata(changed, p, p["groups"], p["alpha"], p["rr"])

    def test_asset_path_and_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "weight"
            path.write_bytes(b"immutable")
            self.assertEqual(verify_binding(root, {"path": "weight", "sha256": sha256(path)}), path)
            with self.assertRaises(ValueError):
                verify_binding(root, {"path": "weight", "sha256": "bad"})
            with self.assertRaises(ValueError):
                confined(root, "../escaped")
            with self.assertRaises(ValueError):
                confined(root, str(path))


class ConsumerDriverTest(unittest.TestCase):
    def test_neutral_video_identity_is_required_even_if_all_latents_match(self):
        hashes = [str(i) for i in range(4)]
        trajectory = [{"raw_tensor_sha256": h} for h in hashes]
        reference = {"latent_sha256": hashes, "video_sha256": "video"}
        self.assertTrue(compare_historical(trajectory, "video", reference)["passed"])
        result = compare_historical(trajectory, "different_codec_bytes", reference)
        self.assertTrue(result["four_raw_FP32_latents_match"])
        self.assertFalse(result["passed"])

    def test_single_attempt_and_neutral_gate_cannot_be_reset_by_new_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            case = {"id": "input00_seed42"}
            ledger = ExecutionLedger(Path(temp), case, "frozen")
            ledger.start()
            with self.assertRaises(RuntimeError):
                ledger.reserve_generation("candidate")
            ledger.reserve_generation("dense")
            ledger.complete_phase({"mode": "dense"})
            ledger.reserve_generation("neutral")
            ledger.complete_phase({"mode": "neutral", "historical_comparison": {"passed": False}})
            with self.assertRaisesRegex(RuntimeError, "Neutral"):
                ledger.reserve_generation("candidate")
            self.assertEqual(ledger.record["counts"]["video_generations"], 2)
            with self.assertRaises(FileExistsError):
                ExecutionLedger(Path(temp), case, "another protocol").start()

    def test_ledger_charges_before_calls_and_refuses_extra_or_zero_budget_work(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = ExecutionLedger(Path(temp), {"id": "case"}, "frozen")
            ledger.start()
            for mode in MODES:
                ledger.reserve_generation(mode)
                ledger.complete_phase({"mode": mode, "historical_comparison": {"passed": True}})
            with self.assertRaises(RuntimeError):
                ledger.reserve_generation("dense")
            with self.assertRaises(RuntimeError):
                ledger.charge("official_quality_calls")
            with self.assertRaises(RuntimeError):
                ledger.charge("new_fits")

    def test_failed_generation_keeps_full_prepaid_native_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = ExecutionLedger(Path(temp), {"id": "case"}, "frozen")
            ledger.start()
            ledger.reserve_generation("dense")
            counts = json.loads((Path(temp) / "RUN.json").read_text())["counts"]
            self.assertEqual(counts["video_generations"], 1)
            self.assertEqual(counts["native_full_model_flows"], 4)
            self.assertEqual(counts["native_main_FFN_linears"], 320)
            self.assertEqual(counts["native_FFN_activation_QDQ"], 320)
            self.assertEqual(counts["native_non_FFN_linears"], 1280)
            self.assertEqual(counts["native_non_FFN_activation_QDQ"], 1280)
            self.assertEqual(counts["native_RR_linears"], 0)
            # Simulate failure before any native operation: there is no refund
            # and the next generation cannot skip the incomplete dense phase.
            with self.assertRaises(RuntimeError):
                ledger.reserve_generation("neutral")
            self.assertEqual(ledger.record["counts"], counts)

    def test_mocked_native_failure_preserves_actual_partial_counts_and_prepaid_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            base = FakeBase()
            original_post = lambda: None
            scheduler = SimpleNamespace(step_post=original_post, generator="old")
            def failed_generate(**kwargs):
                self.assertEqual(kwargs["seed"], 42)
                base.counters["linear_forward_calls"] += 1
                base.counters["activation_qdq_calls"] += 1
                raise RuntimeError("injected native failure")
            pipe = SimpleNamespace(runner=SimpleNamespace(scheduler=scheduler), generate=failed_generate)
            case = {"id": "input00_seed42", "source_id": "image", "prompt_sha256": "prompt"}
            ledger = ExecutionLedger(Path(temp), case, "frozen")
            ledger.start()
            # A CPU mock must never initialize a real GPU or dispatch CUDA work.
            with mock.patch.object(torch.cuda, "init", side_effect=AssertionError("unexpected CUDA")), \
                    mock.patch.object(torch.cuda, "reset_peak_memory_stats"):
                with self.assertRaisesRegex(RuntimeError, "injected native"):
                    generate_phase(pipe, base, base, "dense", case, {"prompt": "test"}, Path(temp) / "image",
                        ledger, SimpleNamespace(events=[]), {"video_contract": {}},
                        SimpleNamespace(check=lambda: None), lambda **kw: None, .25)
            failed = json.loads((Path(temp) / "dense/PHASE.json").read_text())
            self.assertEqual(failed["status"], "FAILED_PRESERVED_NO_RETRY")
            self.assertEqual(failed["completed_native_flow_steps"], 0)
            self.assertEqual(failed["observed_native_counter_delta"]["linear_forward_calls"], 1)
            self.assertEqual(ledger.record["counts"]["native_main_FFN_linears"], 320)
            self.assertEqual(ledger.record["counts"]["native_full_model_flows"], 4)
            self.assertIs(scheduler.step_post, original_post)

    def test_exact_authorization_and_closed_scope(self):
        from run_ub1_consumer_videos import ID
        active = {"experiment_authority": {"current_user_authorized_scopes": [
            {"id": ID, "protocol_sha256": "exact", "execution_status": "RUNNING_FIXED_SCOPE"}]}}
        self.assertTrue(validate_authorization(active, "exact"))
        self.assertFalse(validate_authorization(active, "modified"))
        active["experiment_authority"]["current_user_authorized_scopes"][0]["execution_status"] = "COMPLETE"
        self.assertFalse(validate_authorization(active, "exact"))
        with self.assertRaisesRegex(ValueError, "Unfrozen"):
            validate_plan({"id": ID, "assets_finalized": False})

    def test_all160_timing_and_actual_counts_are_required(self):
        counts = {"linear_forward_calls": 1600, "activation_qdq_calls": 1600, "cache_misses": 0, "weight_transfers": 0}
        timings = [{"step": s, "block": b} for s in range(4) for b in range(40)]
        consumer = {"counts": {"main_linear": 320, "RR_linear": 320, "Q1": 160, "Q2": 160, "up": 160, "down": 160},
                    "pending_RR": 0, "main_coverage": dict.fromkeys(range(320)), "RR_coverage": dict.fromkeys(range(160))}
        validate_native_counts(counts, timings, consumer)
        with self.assertRaises(RuntimeError):
            validate_native_counts({**counts, "weight_transfers": 1}, timings, consumer)
        with self.assertRaises(RuntimeError):
            validate_native_counts(counts, timings[:-1], consumer)
        broken = copy.deepcopy(consumer)
        broken["counts"]["RR_linear"] = 318
        with self.assertRaises(RuntimeError):
            validate_native_counts(counts, timings, broken)

    def test_coefficient_chain_preserves_historical_protocol_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = []
            def store_file(name, content):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(content))
                assets.append({"path": name, "sha256": sha256(path)})
                return name
            def fit(name, identity, status, **extra):
                fp = object_sha(identity)
                ip = store_file(name + "/identity.json", {"fingerprint": fp, "identity": identity})
                rp = store_file(name + "/RUN.json", {"fingerprint": fp, "status": status, **extra})
                return ip, rp, fp
            cache = {"path": "original/cache.json", "sha256": "native_cache"}
            high = fit("high", {"cache_manifest_sha256": "native_cache", "expert": "high"},
                       "COMPLETED_CALIBRATION_PARAMETER_FIT", expert="high_noise_model")
            low = fit("low", {"cache_manifest_sha256": "native_cache", "expert": "low"},
                      "COMPLETED_CALIBRATION_PARAMETER_FIT", expert="low_noise_model")
            rr = fit("rr", {"cache_manifest_sha256": "native_cache", "base_fits": [{"fingerprint": high[2]}, {"fingerprint": low[2]}]},
                     "COMPLETED_RR1_FULL160_CALIBRATION_FIT", positions=list(range(160)))
            def binding(value):
                return {"path": value[1], "sha256": sha256(root / value[1]), "fingerprint": value[2]}
            parent = {"support": "C", "mode": "centered_gain_bias", "cache_manifest": cache,
                      "fit_reports": {"high_noise_model": binding(high), "low_noise_model": binding(low)}, "residual_fit": binding(rr)}
            parent_path = store_file("parent.json", parent)
            recipe = {"cache_manifest": cache, "parent_policy": {"sha256": sha256(root / parent_path)},
                      "protocol": {"sha256": "immutable_original_protocol"}, "changed_fits": {}}
            changed = {}
            for step, groups in ((0, 54), (3, 162)):
                changed[step] = fit(f"step{step}", {"cache_manifest_sha256": "native_cache",
                    "parent_policy_sha256": recipe["parent_policy"]["sha256"], "protocol_sha256": "immutable_original_protocol"},
                    "COMPLETED_UB1_STEP40_FIT", step=step, groups=groups, positions=list(range(40)))
                recipe["changed_fits"][str(step)] = binding(changed[step])
            positions = []
            for step, main, residual in ((0, changed[0], changed[0]), (1, high, rr), (2, low, rr), (3, changed[3], changed[3])):
                position = {"step": step, "expert": expected_expert(step)}
                for prefix, value in (("main", main), ("rr", residual)):
                    npz = store_file(f"{step}/{prefix}.npz", {"frozen": step})
                    sidecar = store_file(f"{step}/{prefix}.json", {"status": "COMPLETE", "fingerprint": value[2], "payload_sha256": sha256(root / npz)})
                    position.update({prefix + "_npz": npz, prefix + "_json": sidecar,
                                     prefix + "_identity": value[0], prefix + "_report": value[1], prefix + "_fingerprint": value[2]})
                positions.append(position)
            plan = {"positions": positions, "private_assets": assets, "private_common": [parent_path]}
            result = verify_coefficient_chains(plan, root, recipe)
            self.assertEqual(result["recorded_historical_protocol_sha256"], "immutable_original_protocol")
            changed_recipe = copy.deepcopy(recipe)
            changed_recipe["protocol"]["sha256"] = "new_public_sanitized_protocol"
            with self.assertRaisesRegex(ValueError, "parent/protocol"):
                verify_coefficient_chains(plan, root, changed_recipe)


if __name__ == "__main__":
    unittest.main()
