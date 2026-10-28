"""Meaningful tiny CPU controls; no real asset tensor, GPU, model, or QR."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_b_head_row_geometry_neutral as m
import e0_b_terminal_head_probe as original

TINY = {"M": 12, "D": 8, "head_outputs": 3, "selected_rows": 2}


def fixture():
    gen = torch.Generator().manual_seed(173)
    draw = lambda *shape: torch.randn(shape, generator=gen).bfloat16()
    u, weight, bias = draw(12, 8), draw(8, 3), draw(3)
    scale, shift = draw(1, 1, 8), draw(1, 1, 8)
    rows = torch.tensor([1, 9], dtype=torch.int64)
    head, hidden = original.native_head(torch, u, weight, bias, scale, shift)
    values = {"rows": rows, "u_rows": u.index_select(0, rows), "head_output": head,
              "head_input_rows": hidden.index_select(0, rows), "head_scale": scale, "head_shift": shift}
    def spec(tensor, stride=None):
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()) if stride is None else stride,
                "raw_tensor_sha256": m.tensor_sha(torch, tensor)}
    metadata = {"schema": "ffnds.b_terminal.capture_case.v1", "run_id": "b_final_capture_source01_seed42_step0",
        "source_index": 1, "seed": 42, "step": 0, "row_draw": {"rows": rows.tolist()},
        "actual_pre_head_u": spec(u, [8, 1]),
        "actual_head_operator": {"head_class": "lightx2v.common.ops.mm.mm_weight.MMWeight",
            "norm_class": "lightx2v.common.ops.norm.layer_norm_weight.LNWeight", "head_has_lora_branch": False,
            "norm_weight": None, "norm_bias": None, "norm_eps": 1e-6,
            "infer_dtype": "torch.bfloat16", "sensitive_dtype": "torch.bfloat16",
            "weight": spec(weight, [1, 8]), "bias": spec(bias, [1])},
        "actual_combined_head_modulation": {"scale": spec(scale, [16, 8, 1]), "shift": spec(shift, [16, 8, 1])},
        "capture_file": {"tensors": {name: spec(t) for name, t in values.items()}}}
    pack = {"head_weight_runtime_KN": weight, "head_bias": bias}
    weight = original.restore_layout(torch, weight, [1, 8], "cpu")
    return values, pack, metadata, weight, bias, scale, shift


def tiny_limits():
    keys = ("channel_pattern_builds", "padded_states", "selected_row_writes", "padded_state_hashes",
            "layer_norm_calls", "head_calls", "selected_input_row_checks", "selected_head_input_checks",
            "selected_head_row_checks", "same_fill_full_head_repeat_checks", "full_head_hashes")
    return {key: m.LIMITS[key] // 4 for key in keys}


def plan_and_authority():
    plan = {"id": m.ID, "geometry": m.GEOMETRY, "fills": list(m.FILLS), "fill_description": m.FILL_DESCRIPTION,
        "repeats_per_fill": 2, "epsilon": 1e-6, "limits": m.LIMITS, "resources": m.RESOURCES,
        "runtime_versions": m.RUNTIME, "automatic_retry": False, "assets_finalized": True,
        "historical_probe_protocol": {"sha256": m.HISTORICAL_PROBE_SHA},
        "repo_sources": [{"path": s} for s in sorted(m.REQUIRED_SOURCES)],
        "cases": [{"source_index": i, "step": s, "capture": {}, "case": {}, "head": {}} for i, s in m.CASES]}
    scope = {"id": m.ID, "protocol_sha256": "version", "execution_status": "AUTHORIZED_NOT_STARTED",
        "limits": m.LIMITS, "resources": m.RESOURCES, "max_gpu_cards": 1,
        "max_simultaneously_assigned_idle_GPUs": 2}
    active = {"user_continuation_authorization_20261027": {"id": "USER_RESUME_OFFLINE_IFC_20261027",
        "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION", "new_scope_registration_allowed": True,
        "max_simultaneously_assigned_idle_GPUs": 2},
        "experiment_authority": {"current_user_authorized_scopes": [scope]}}
    return plan, active


class NeutralControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_import_performs_no_numeric_or_device_work(self):
        cmd = [sys.executable, "-c", "import sys;sys.path.insert(0,'tools');import e0_b_head_row_geometry_neutral;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)

    def test_two_fills_preserve_genuine_rows_and_exact_call_counts(self):
        tensors, pack, metadata, weight, bias, scale, shift = fixture()
        report, limits = {}, tiny_limits()
        ledger = m.Ledger(report, limits=limits)
        before = torch.get_rng_state().clone()
        actual = m.neutral_case(torch, tensors, weight, bias, scale, shift, metadata, ledger,
                                device="cpu", geometry=TINY)
        ledger.finish()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(actual), 4)
        self.assertEqual(report["predebited"], limits)
        self.assertEqual(report["invoked"], limits)
        self.assertEqual(report["observed"], limits)
        self.assertEqual([r["fill"] for r in actual], ["zero", "zero", "fixed_channel_pattern", "fixed_channel_pattern"])
        self.assertEqual(actual[0]["full_head_sha256"], actual[1]["full_head_sha256"])
        self.assertEqual(actual[2]["full_head_sha256"], actual[3]["full_head_sha256"])
        self.assertNotEqual(actual[0]["full_head_sha256"], actual[2]["full_head_sha256"])

    def test_selected_head_byte_mismatch_stops_first_call(self):
        tensors, _, meta, weight, bias, scale, shift = fixture()
        tensors["head_output"][1, 0] += 1
        report = {}
        ledger = m.Ledger(report, limits=tiny_limits())
        with self.assertRaisesRegex(ValueError, "Selected head rows differ"):
            m.neutral_case(torch, tensors, weight, bias, scale, shift, meta, ledger, device="cpu", geometry=TINY)
        self.assertEqual(report["invoked"]["head_calls"], 1)
        self.assertEqual(report["observed"]["selected_head_row_checks"], 0)
        self.assertEqual(report["predebited"]["selected_head_row_checks"], 1)

    def test_mutating_an_unselected_input_row_is_detected(self):
        tensors, _, meta, weight, bias, scale, shift = fixture()
        report = {}
        ledger = m.Ledger(report, limits=tiny_limits())
        real = original.native_head
        def mutating(torch_api, u, *args, **kwargs):
            output = real(torch_api, u, *args, **kwargs)
            u[0, 0] += .5
            return output
        with mock.patch.object(original, "native_head", side_effect=mutating):
            with self.assertRaisesRegex(ValueError, "mutated padded input"):
                m.neutral_case(torch, tensors, weight, bias, scale, shift, meta, ledger, device="cpu", geometry=TINY)
        self.assertEqual(report["observed"]["head_calls"], 1)
        self.assertEqual(report["observed"]["selected_head_row_checks"], 0)

    def test_full_head_repeat_gate_catches_unselected_output_instability(self):
        tensors, _, meta, weight, bias, scale, shift = fixture()
        report = {}
        ledger = m.Ledger(report, limits=tiny_limits())
        real, calls = original.native_head, []
        def unstable(*args, **kwargs):
            output, hidden = real(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                output[0, 0] += 1
            return output, hidden
        with mock.patch.object(original, "native_head", side_effect=unstable):
            with self.assertRaisesRegex(ValueError, "Same-fill full head repeat differs"):
                m.neutral_case(torch, tensors, weight, bias, scale, shift, meta, ledger, device="cpu", geometry=TINY)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["observed"]["selected_head_row_checks"], 2)
        self.assertEqual(report["observed"]["same_fill_full_head_repeat_checks"], 0)

    def test_native_failure_keeps_invoked_without_completed_head(self):
        tensors, _, meta, weight, bias, scale, shift = fixture()
        report, records = {}, []
        ledger = m.Ledger(report, limits=tiny_limits())
        with mock.patch.object(torch, "addmm", side_effect=RuntimeError("tiny failed MM")):
            with self.assertRaisesRegex(RuntimeError, "tiny failed MM"):
                m.neutral_case(torch, tensors, weight, bias, scale, shift, meta, ledger, device="cpu", geometry=TINY, records=records)
        self.assertEqual(report["predebited"]["head_calls"], 1)
        self.assertEqual(report["invoked"]["head_calls"], 1)
        self.assertEqual(report["observed"]["head_calls"], 0)
        self.assertEqual(report["observed"]["layer_norm_calls"], 1)
        self.assertEqual(report["last_operation"], {"key": "head_calls", "phase": "invoked"})
        self.assertEqual(records, [{"fill": "zero", "repeat": 0, "status": "STARTED"}])
        with self.assertRaises(ValueError):
            ledger.finish()

    def test_atomic_predebit_rejects_overrun_and_unpaid_observation(self):
        report = {}
        ledger = m.Ledger(report, limits={"head_calls": 1, "layer_norm_calls": 1})
        ledger.reserve({"head_calls": 1})
        before = copy.deepcopy(report)
        with self.assertRaises(ValueError):
            ledger.reserve({"layer_norm_calls": 1, "head_calls": 1})
        self.assertEqual(report, before)
        with self.assertRaises(ValueError):
            ledger.observe("layer_norm_calls", "invoked")

    def test_pattern_is_nonconstant_exact_deterministic_and_has_no_rng(self):
        before = torch.get_rng_state().clone()
        first, second = m.channel_pattern(torch, 34), m.channel_pattern(torch, 34)
        self.assertTrue(original.same_bits(torch, first, second))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(len(torch.unique(first)), 17)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_saved_tensor_identity_and_row_metadata_are_strict(self):
        tensors, pack, meta, *_ = fixture()
        m.validate_case_metadata(meta, {"source_index": 1, "step": 0}, geometry=TINY)
        m.validate_loaded(torch, tensors, pack, meta, geometry=TINY)
        corrupt = copy.deepcopy(tensors)
        corrupt["u_rows"][0, 0] += .5
        with self.assertRaisesRegex(ValueError, "Saved tensor bytes differ"):
            m.validate_loaded(torch, corrupt, pack, meta, geometry=TINY)
        for change in ({"head_has_lora_branch": True}, {"norm_eps": 1e-5}):
            bad = copy.deepcopy(meta)
            bad["actual_head_operator"].update(change)
            with self.assertRaisesRegex(ValueError, "row-local"):
                m.validate_case_metadata(bad, {"source_index": 1, "step": 0}, geometry=TINY)

    def test_capture_header_validates_only_the_six_requested_tiny_tensors(self):
        from safetensors.torch import save_file
        tensors, _, meta, *_ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.safetensors"
            save_file(tensors, str(path))
            self.assertEqual(m.check_capture_header(path, meta, geometry=TINY), 6)
            tensors["u_rows"] = tensors["u_rows"].float()
            save_file(tensors, str(path))
            with self.assertRaisesRegex(ValueError, "tensor header differs"):
                m.check_capture_header(path, meta, geometry=TINY)

    def test_authority_cap_two_and_scope_are_exact_before_torch_entry(self):
        plan, active = plan_and_authority()
        m.authorize(plan, "version", active)
        for change in (lambda p, a: p.update(assets_finalized=False),
                       lambda p, a: a["user_continuation_authorization_20261027"].update(max_simultaneously_assigned_idle_GPUs=4),
                       lambda p, a: a["experiment_authority"]["current_user_authorized_scopes"][0].update(protocol_sha256="old")):
            p, a = copy.deepcopy(plan), copy.deepcopy(active)
            change(p, a)
            with self.assertRaises(ValueError):
                m.authorize(p, "version", a)
        bad = copy.deepcopy(plan)
        bad["cases"][0]["u"] = {}
        with self.assertRaisesRegex(ValueError, "never bind full-u"):
            m.validate_plan(bad)

    def test_actual_preflight_exit_must_bind_exact_report_and_version(self):
        plan, active = plan_and_authority()
        scope = active["experiment_authority"]["current_user_authorized_scopes"][0]
        with self.assertRaisesRegex(ValueError, "NOT_BOUND"):
            m.prior_preflight(plan, "version", scope, ROOT)
        with tempfile.TemporaryDirectory() as folder:
            hf = Path(folder)
            report = {"id": m.ID, "protocol_sha256": "version", "status": "PASS_B_HEAD_ROW_GEOMETRY_CPU_PREFLIGHT_NO_TORCH_NO_GPU",
                "repo_sources_sha256": m.semantic_sha(plan["repo_sources"]), "cases_sha256": m.semantic_sha(plan["cases"]),
                "cases": 4, "Torch_or_CUDA_imported": False, "all_CUDA_devices_hidden": True, "native_calls": 0}
            def put(name, value):
                path = hf / name
                path.write_text(json.dumps(value))
                return {"path": name, "bytes": path.stat().st_size, "sha256": m.sha(path)}
            run = put("PREFLIGHT.json", report)
            receipt = {"id": m.ID, "protocol_sha256": "version", "launcher_returncode": 0,
                       "process_exited": True, "automatic_retry": False, "RUN_sha256": run["sha256"]}
            scope["cpu_preflight"] = {"RUN": run, "launch_exit": put("EXIT.json", receipt)}
            m.prior_preflight(plan, "version", scope, hf)
            for change in ({"RUN_sha256": "0" * 64}, {"process_exited": False}, {"launcher_returncode": 1}):
                scope["cpu_preflight"]["launch_exit"] = put("EXIT.json", {**receipt, **change})
                with self.assertRaisesRegex(ValueError, "actual exit/RUN SHA"):
                    m.prior_preflight(plan, "version", scope, hf)
            scope["cpu_preflight"]["RUN"] = put("PREFLIGHT.json", {**report, "protocol_sha256": "stale"})
            receipt["RUN_sha256"] = scope["cpu_preflight"]["RUN"]["sha256"]
            scope["cpu_preflight"]["launch_exit"] = put("EXIT.json", receipt)
            with self.assertRaisesRegex(ValueError, "version or result"):
                m.prior_preflight(plan, "version", scope, hf)


if __name__ == "__main__":
    unittest.main()
