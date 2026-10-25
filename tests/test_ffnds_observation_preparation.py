import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_observation_common import FIELDS, audit_headers, read_header, row_plan
from analyze_ffnds_observations import analyze_run, load_arrays, paired, panel_geometry
from ffnds_observation_capture import DiscoveryCapture
from prepare_ffnds_observations import preflight, validate_protocol
from analyze_legacy_ffn_structure import profile
from analyze_e0_native_mechanism import panel_decomposition
from analyze_e0_residual_geometry import geometry


def fixture(path, fields):
    header, raw, offset = {}, b"", 0
    for name, (dtype, shape, data) in fields.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        raw += data
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + raw)


class Contracts(unittest.TestCase):
    def test_ideal_normalization_shift_radial_and_orthogonal_directions(self):
        x = np.array([[1., -1., 0., 0.]])
        shift = geometry(x, np.ones_like(x), eps=0)
        self.assertAlmostEqual(shift["channel_mean_removed_fraction"], 1)
        self.assertAlmostEqual(shift["finite_ideal_LN_response_over_rstd_scaled_direction_energy"], 0)
        radial = geometry(x, .1 * x, eps=0)
        self.assertAlmostEqual(radial["radial_removed_fraction"], 1)
        perpendicular = geometry(x, np.array([[0., 0., .1, -.1]]), eps=0)
        self.assertAlmostEqual(perpendicular["total_directional_energy_removed_fraction"], 0)
        self.assertAlmostEqual(perpendicular["isotropic_expected_removed_fraction"], .5)

    def test_panel_common_mean_does_not_imply_common_row_variation(self):
        q2 = np.zeros((2, 128))
        q2[0, 0], q2[1, 64] = 2, 2
        weights = np.zeros((5120, 128))
        weights[0, 0] = weights[0, 64] = 1
        r = panel_decomposition(q2, weights, np.ones((1, 5120)), [0, 1])
        self.assertEqual(r["mean_sum_energy_over_sum_individual_mean_energies"], 2)
        self.assertEqual(r["centered_sum_energy_over_sum_individual_energies"], 0)
        self.assertEqual(r["constant_mean_fraction_of_summed_panel_output_energy"], 1)
        self.assertEqual(r["mean_variation_identity_relative_error"], 0)

    def test_fixed_profile_amplitude_scaling_and_equal_RMS_counterexample(self):
        fixed = profile(np.stack([np.ones(128), np.ones(128) * 2]))
        self.assertAlmostEqual(fixed["row_amplitude_times_static_profile_relative_SSE"], 0)
        self.assertAlmostEqual(fixed["RMS_adjacent_pair_profile_cosine_quantiles"][2], 1)
        a = np.r_[np.ones(64), np.zeros(64)]
        shifted = profile(np.stack([a, a[::-1]]))
        self.assertEqual(shifted["RMS_adjacent_pair_relative_RMS_gap_quantiles"][2], 0)
        self.assertEqual(shifted["RMS_adjacent_pair_profile_cosine_quantiles"][2], 0)
        self.assertEqual(shifted["all_zero_group64_fraction"], .5)
        self.assertEqual(shifted["row_amplitude_times_static_profile_relative_SSE"], .5)

    def test_protocol_refuses_duplicate_axes(self):
        p = json.loads((Path(__file__).resolve().parents[1] / "config/observation_20261025_01a16d3d.json").read_text())
        validate_protocol(p)
        p["blocks"] = [0, 0]
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_preflight_does_not_claim_missing_environment_ready(self):
        p = json.loads((Path(__file__).resolve().parents[1] / "config/observation_20261025_01a16d3d.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            r = preflight(Path(d), p)
        self.assertEqual(r["status"], "ENVIRONMENT_INCOMPLETE")
        self.assertFalse(r["gpu_initialized"])
        self.assertFalse(r["native_observer_validated"])

    def test_rows_pair_and_rng(self):
        state = random.getstate()
        a = row_plan([21, 29, 52], "source-A")
        self.assertEqual(state, random.getstate())
        self.assertEqual(a, row_plan([21, 29, 52], "source-A"))
        self.assertNotEqual(a["uniform_rows"], row_plan([21, 29, 52], "source-B")["uniform_rows"])
        for p in a["neighbor_pairs"]:
            self.assertIn(p["a"], a["rows"])
            self.assertIn(p["b"], a["rows"])
        self.assertLessEqual(len(a["rows"]), 48)

    def test_tiny_grid_and_invalid_grid(self):
        self.assertEqual(row_plan([1, 1, 1], "a")["rows"], [0])
        with self.assertRaises(ValueError):
            row_plan([0, 3, 2], "a")

    def test_bf16_decode_and_legacy_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.safetensors"
            fixture(path, {"V2": ("BF16", [1, 2], struct.pack("<HH", 0x3F80, 0xC000))})
            np.testing.assert_array_equal(load_arrays(path)["V2"], [[1, -2]])
            r = audit_headers([path])["files"][0]
            self.assertTrue(r["legacy_post_q2_only"])
            self.assertFalse(r["paired_q2"])

    def test_header_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad"
            path.write_bytes(struct.pack("<Q", 99999) + b"{}")
            with self.assertRaises(ValueError):
                read_header(path)
            self.assertEqual(audit_headers([path])["files"][0]["status"], "HEADER_FAILED")

    def test_header_rejects_overlapping_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad"
            h = json.dumps({k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                            for k in ("a", "b")}).encode()
            path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 4)
            with self.assertRaises(ValueError):
                read_header(path)

    def test_paired_zeros_are_not_infinite(self):
        self.assertIsNone(paired(np.zeros((2, 3)), np.ones((2, 3)))["relative_SSE"])
        self.assertEqual(paired(np.ones((2, 3)), np.ones((2, 3)))["relative_SSE"], 0)

    def test_signed_panel_energy_is_not_clipped_to_one(self):
        a = np.ones((1, 128), dtype=np.float32)
        b = np.zeros((5120, 128), dtype=np.float32)
        b[0, 0] = b[0, 64] = 1
        r = panel_geometry({"q2_post": a, "gate": np.ones((1, 5120))}, {"W2_panel": b}, [0, 1])
        self.assertEqual(r["signed_sum_energy_over_sum_individual_energies"], 2)
        b[0, 64] = -1
        r = panel_geometry({"q2_post": a, "gate": np.ones((1, 5120))}, {"W2_panel": b}, [0, 1])
        self.assertEqual(r["signed_sum_energy_over_sum_individual_energies"], 0)

    def test_unaccepted_run_is_not_scientific_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "RUN.json"
            p.write_text(json.dumps({"status": "FAILED_NOT_ACCEPTED"}))
            with self.assertRaises(ValueError):
                analyze_run(p)


class Tensor:
    """Minimal transport fake, NOT a quantizer or a Wan model."""
    def __init__(self, a):
        self.a = np.asarray(a)
        self.device = "fake"
        self.dtype = "fake"
    @property
    def shape(self): return self.a.shape
    def numel(self): return self.a.size
    def detach(self): return self
    def reshape(self, *s): return Tensor(self.a.reshape(*s))
    def cpu(self): return self
    def contiguous(self): return self
    def to(self, *args, **kwargs): return self
    def index_select(self, dim, ids): return Tensor(np.take(self.a, ids.a.astype(int), axis=dim))


class HookContracts(unittest.TestCase):
    def test_capture_calls_original_once_copies_inplace_and_restores_on_error(self):
        adapter = types.ModuleType("HiFloat4.ptq.adapters.lightx2v_wan_distill")
        adapter._step_ctx = types.SimpleNamespace(get=lambda: 0)
        calls = {"apply": 0, "qdq": 0, "ffn": 0, "post": 0}
        def qdq(x, **kw):
            calls["qdq"] += 1
            return x
        adapter.native_hif4_activation_qdq = qdq
        class Runtime:
            activation_qdq_override = None
            keep_fp = None
            records = {"high_noise_model": {f"blocks.0.ffn.{s}": {
                "weight_qdq": types.SimpleNamespace(shape=shape), "activation_qdq": True, "force_fp32": True}
                for s, shape in [(0, (13824, 5120)), (2, (5120, 13824))]}}
            def apply(self, mm, x, **kw):
                calls["apply"] += 1
                adapter.native_hif4_activation_qdq(x, force_fp32=True)
                width = 13824 if kw["module_name"].endswith(".0") else 5120
                return Tensor(np.full((4, width), 4.0))
        adapter.HIF4CacheRuntime = Runtime
        runtime = Runtime()
        mm = types.SimpleNamespace(_get_actual_bias=lambda: Tensor(np.zeros(5120)))
        class Infer:
            block_idx = 0
            use_compile = False
            def infer_block(self, block, x, pre):
                y = self.infer_ffn(None, x, Tensor(np.full((4, 5120), 2.0)), None, None,
                                   Tensor(np.ones(5120)))
                return self.post_process(x, y, Tensor(np.ones(5120)), pre)
            def infer_ffn(self, phase, x, attn, shift, scale, gate=None):
                calls["ffn"] += 1
                x.a += attn.a
                kw = dict(expert="high_noise_model", step_index=0, block_index=0)
                z = runtime.apply(mm, x, module_name="blocks.0.ffn.0", **kw)
                return runtime.apply(mm, z, module_name="blocks.0.ffn.2", **kw)
            def post_process(self, x, y, gate, pre=None):
                calls["post"] += 1
                return Tensor(x.a + y.a)
        infer_module = types.ModuleType("lightx2v.models.networks.wan.infer.transformer_infer")
        infer_module.WanTransformerInfer = Infer
        torch = types.ModuleType("torch")
        torch.tensor = lambda x, **kw: Tensor(x)
        torch.isfinite = lambda x: np.isfinite(x.a)
        safe = types.ModuleType("safetensors.torch")
        safe.save_file = lambda *a, **k: None
        modules = {}
        for name in ["HiFloat4", "HiFloat4.ptq", "HiFloat4.ptq.adapters", "lightx2v",
                     "lightx2v.models", "lightx2v.models.networks", "lightx2v.models.networks.wan",
                     "lightx2v.models.networks.wan.infer", "safetensors"]:
            modules[name] = types.ModuleType(name)
        modules["HiFloat4.ptq.adapters"].lightx2v_wan_distill = adapter
        modules.update({"torch": torch, "safetensors.torch": safe,
                        "HiFloat4.ptq.adapters.lightx2v_wan_distill": adapter,
                        "lightx2v.models.networks.wan.infer.transformer_infer": infer_module})
        originals = (Infer.infer_block, Infer.infer_ffn, Infer.post_process, runtime.apply)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, modules):
            observer = DiscoveryCapture(runtime, d, "a", {"steps": [0], "blocks": [0], "uniform_rows": 4}, {})
            captured = {}
            def finish(s): captured.update(s["samples"])
            observer.finish = finish
            with observer:
                result = Infer().infer_block(None, Tensor(np.ones((4, 5120))),
                                             types.SimpleNamespace(grid_sizes=types.SimpleNamespace(tuple=(1, 1, 4))))
            self.assertEqual(calls, {"apply": 2, "qdq": 2, "ffn": 1, "post": 1})
            self.assertEqual(set(FIELDS) - captured.keys(), set())
            np.testing.assert_array_equal(captured["x_before_cross_add"].a, np.ones((4, 5120)))
            np.testing.assert_array_equal(captured["residual_pre_ffn"].a, np.full((4, 5120), 3))
            np.testing.assert_array_equal(result.a, np.full((4, 5120), 7))
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with DiscoveryCapture(runtime, d, "a", {}, {}):
                    raise RuntimeError("sentinel")
        self.assertIs(Infer.infer_block, originals[0])
        self.assertIs(Infer.infer_ffn, originals[1])
        self.assertIs(Infer.post_process, originals[2])
        self.assertEqual(runtime.apply, originals[3])
        self.assertIs(adapter.native_hif4_activation_qdq, qdq)


if __name__ == "__main__":
    unittest.main()
