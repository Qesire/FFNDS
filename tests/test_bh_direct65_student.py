"""Tiny synthetic B-H controls; no real labels, decompositions or training."""
from collections import Counter
from contextlib import contextmanager
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bh_direct65_student as bh
import e0_bh_direct65_pilot as driver


def parameters(D=8, outputs=2, hidden=4, rows=2):
    return {**bh.PARAMETERS, "D": D, "head_outputs": outputs, "hidden": hidden, "rows_per_case": rows}


def geometry(p=None, rank_deficient=False):
    p = parameters() if p is None else p
    rng = np.random.default_rng(91)
    w = rng.normal(size=(p["D"], p["head_outputs"]))
    if rank_deficient:
        w[:, -1] = 2 * w[:, 0]
    return bh.compile_geometry(w, np.ones(p["D"]), np.zeros(p["D"]), np.zeros(p["head_outputs"]), p)


def unit_normalizer(outputs=2):
    return {"x_rms": torch.tensor(1.), "eta_mean": torch.zeros(outputs + 1), "eta_std": torch.ones(outputs + 1), "target_rms": torch.ones(outputs + 1)}


def tiny_training_cases():
    generator = torch.Generator().manual_seed(41)
    cases = {}
    for i in range(16):
        for step in (0, 1):
            x = torch.randn((2, 8), generator=generator).bfloat16()
            eta = torch.randn((2, 3), generator=generator)
            target = torch.stack((x[:, 0].float() / 5, x[:, 1].float() / 5, eta[:, 0] / 10), dim=1)
            cases[(i, step)] = {"source_index": i, "step": step, "role": "TRAIN", "V1": x, "anchor": {"eta": eta}, "target65": target}
    return cases


class Direct65Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_import_has_no_numerical_or_GPU_side_effect(self):
        command = [sys.executable, "-c", "import sys;sys.path.insert(0,'tools');import bh_direct65_student,e0_bh_direct65_pilot;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)

    def test_exact_parameter_count_and_declared_cost(self):
        self.assertEqual(bh.parameter_count(), 672386)
        self.assertEqual(712704 - bh.parameter_count(), 40318)
        cost = bh.logical_cost()
        self.assertEqual(cost["logical_MAC_per_row"], 1016064)
        self.assertEqual(cost["online_GEMM_calls"], 8)
        model = bh.Direct65Student(D=8, outputs=2, hidden=4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), bh.parameter_count(8, 2, 4))

    def test_geometry_projector_inverse_and_rank_deficiency(self):
        g = geometry(rank_deficient=True)
        self.assertEqual(g["rank"], 1)
        np.testing.assert_allclose(g["K"].sum(axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(g["projector"] @ g["projector"], g["projector"], atol=1e-12)
        G = g["K"] @ g["K"].T
        np.testing.assert_allclose(G @ g["Gplus"] @ G, G, rtol=1e-12, atol=1e-12)
        outside = np.asarray([2., -1.])
        np.testing.assert_allclose(outside @ g["projector"], 0, atol=1e-12)

    def test_geometry_counts_include_actual_projector_validation_GEMM(self):
        p, counts = parameters(), Counter()
        @contextmanager
        def work(key, count=1):
            counts[key] += count
            yield
        w = np.random.default_rng(8).normal(size=(8, 2))
        bh.compile_geometry(w, np.ones(8), np.zeros(8), np.zeros(2), p, work)
        self.assertEqual(dict(counts), {"geometry_FP64_Gram_GEMM": 1, "geometry_symmetric_eigh": 1,
            "geometry_FP64_inverse_and_projector_GEMM": 2, "geometry_FP64_shift_GEMM": 1,
            "geometry_FP64_projector_validation_GEMM": 1})

    def test_zero_head_and_centered_kernel_degeneracy_fail(self):
        p = parameters(D=3, outputs=2)
        with self.assertRaisesRegex(ValueError, "zero"):
            bh.compile_geometry(np.zeros((3, 2)), np.ones(3), np.zeros(3), np.zeros(2), p)
        w = np.asarray([[1., 0], [0, 1], [-1, -1]])
        with self.assertRaisesRegex(ValueError, "centered kernel"):
            bh.compile_geometry(w, np.ones(3), np.zeros(3), np.zeros(2), p)

    def test_actual_gain_rounding_is_the_compiler_input(self):
        p = parameters()
        scale = torch.full((8,), .01, dtype=torch.bfloat16)
        gain = (1 + scale).double().numpy()
        weight = np.arange(16, dtype=float).reshape(8, 2) / 16
        # Add independent nonconstant columns to avoid artificial rank collapse.
        weight[:, 1] = np.asarray([1., -1, 2, -2, 3, -3, 4, -4])
        g = bh.compile_geometry(weight, gain, np.zeros(8), np.zeros(2), p)
        T = weight.T * gain[None]
        np.testing.assert_allclose(g["K"], T - T.mean(axis=1, keepdims=True), atol=0)
        self.assertGreater(float(np.max(np.abs(gain - (1 + scale.double().numpy())))), 1e-3)

    def test_zero_initialization_is_residual_anchor_with_explicit_floor(self):
        p = parameters()
        g = bh.device_geometry(geometry(), torch, "cpu")
        r = torch.tensor([[1., 2., -1., .5, 3., -2., 1., 4.], [0.] * 8], dtype=torch.bfloat16)
        a = bh.anchor_features(torch, r, g, p)
        model = bh.Direct65Student(D=8, outputs=2, hidden=4)
        raw = model.predict65(r, a["eta"], 0, unit_normalizer())
        self.assertEqual(raw.dtype, torch.bfloat16)
        self.assertTrue(bool((raw == 0).all()))
        decoded = bh.decode_statistics(torch, raw, a, unit_normalizer(), g, p)
        np.testing.assert_allclose(decoded["q"].detach().numpy(), a["qr"].numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(a["floor_rows"], 1)
        self.assertAlmostEqual(float(decoded["variance"][1].detach()), p["slack_floor"], places=17)
        self.assertGreater(float(decoded["variance"][1].detach()), float(a["variance"][1]))

    def test_rank_deficient_outputs_stay_reachable(self):
        p = parameters()
        g = bh.device_geometry(geometry(rank_deficient=True), torch, "cpu")
        r = torch.ones((2, 8), dtype=torch.bfloat16)
        anchor = bh.anchor_features(torch, r, g, p)
        raw = torch.tensor([[2., -1., .2], [-4., 3., -.1]], dtype=torch.bfloat16)
        out = bh.decode_statistics(torch, raw, anchor, unit_normalizer(), g, p)
        np.testing.assert_allclose(out["q"].numpy() @ g["projector"].numpy(), out["q"].numpy(), atol=1e-6)
        vmin = ((out["q"] @ g["Gplus"]) * out["q"]).sum(dim=1) / 8
        self.assertTrue(bool((out["variance"] >= vmin).all()))
        # Reachability ellipsoid around the fixed bias c0.
        h = out["head_FP32"] - g["c0"]
        self.assertTrue(bool((((h @ g["Gplus"]) * h).sum(dim=1) <= 8 + 1e-5).all()))

    def test_declared_negative_tolerance_repairs_and_material_failure(self):
        fixed, report = bh.checked_nonnegative(torch, torch.tensor([-1e-7, 2.]), torch.ones(2), 2e-5, 1e-6, "toy slack")
        self.assertEqual(float(fixed[0]), 0)
        self.assertEqual(report["negative_rows_repaired"], 1)
        with self.assertRaisesRegex(ValueError, "materially negative"):
            bh.checked_nonnegative(torch, torch.tensor([-.1]), torch.ones(1), 2e-5, 1e-6, "toy slack")

    def test_log_slack_overflow_fails_instead_of_target_clipping(self):
        p = parameters()
        g = bh.device_geometry(geometry(), torch, "cpu")
        r = torch.ones((2, 8), dtype=torch.bfloat16)
        anchor = bh.anchor_features(torch, r, g, p)
        raw = torch.tensor([[0., 0., 31.], [0., 0., 0.]], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "finite domain"):
            bh.decode_statistics(torch, raw, anchor, unit_normalizer(), g, p)

    def test_offline_truth_closes_continuous_head_and_reports_slack_floor(self):
        p = parameters()
        cpu = geometry()
        gpu = bh.device_geometry(cpu, torch, "cpu")
        r = torch.tensor([[.3, 1., -2., 3., 1., -1., 2., .5], [.5, 1., 2., 4., 3., 0., -1., 2.]], dtype=torch.bfloat16)
        u = r + torch.tensor([[.2, -.1, 0., .3, .1, -.2, .4, .2]], dtype=torch.bfloat16)
        anchor = bh.anchor_features(torch, r, gpu, p)
        labels = bh.truth_labels(u.double().numpy(), {k: anchor[k].numpy() for k in ("qr", "slack")}, cpu, p)
        raw = torch.from_numpy(labels["target65"].astype(np.float32))
        decoded = bh.decode_statistics(torch, raw, anchor, unit_normalizer(), gpu, p)
        np.testing.assert_allclose(decoded["head_FP32"].numpy(), labels["real_head"], rtol=3e-6, atol=3e-6)
        self.assertEqual(labels["diagnostic"]["target_clipping"].split(";")[0], "NONE")
        # Constant u has true variance/slack zero: tau explicitly changes it.
        zero = torch.zeros_like(r)
        zero_anchor = bh.anchor_features(torch, zero, gpu, p)
        labels = bh.truth_labels(zero.double().numpy(), {k: zero_anchor[k].numpy() for k in ("qr", "slack")}, cpu, p)
        self.assertEqual(labels["diagnostic"]["teacher_slack_floor_rows"], 2)
        self.assertEqual(labels["diagnostic"]["max_teacher_floor_variance_added"], p["slack_floor"])

    def test_predictor_has_no_teacher_input_and_truth_changes_only_labels(self):
        signature = set(inspect.signature(bh.Direct65Student.predict65).parameters)
        self.assertEqual(signature, {"self", "postQ1_V1", "residual_summary", "local_step", "normalizer"})
        p = parameters()
        cpu, normalizer = geometry(), unit_normalizer()
        g = bh.device_geometry(cpu, torch, "cpu")
        r = torch.arange(16).reshape(2, 8).bfloat16()
        a = bh.anchor_features(torch, r, g, p)
        model = bh.Direct65Student(D=8, outputs=2, hidden=4)
        before = model.predict65(r, a["eta"], 1, normalizer).detach().clone()
        l1 = bh.truth_labels(r.double().numpy(), {k: a[k].numpy() for k in ("qr", "slack")}, cpu, p)
        l2 = bh.truth_labels((r * 2).double().numpy(), {k: a[k].numpy() for k in ("qr", "slack")}, cpu, p)
        self.assertFalse(np.array_equal(l1["target65"], l2["target65"]))
        self.assertTrue(torch.equal(before, model.predict65(r, a["eta"], 1, normalizer)))

    def test_fixed_schedule_matches_A0_batch_source_and_step_budget(self):
        for expert in bh.EXPERTS:
            schedule = bh.schedule(expert)
            self.assertEqual(len(schedule), 256)
            self.assertEqual(set(source for _, source, _ in schedule), set(range(16)))
            self.assertEqual(Counter((i, s) for _, i, s in schedule), Counter({(i, s): 8 for i in range(16) for s in bh.steps_for(expert)}))

    def test_normalizers_reject_heldout_or_ragged_sources(self):
        p, cases = parameters(), list(tiny_training_cases().values())
        n = bh.normalizers(torch, cases, p)
        self.assertTrue(bool((n["target_rms"] > 0).all()))
        bad = copy.deepcopy(cases)
        bad[-1]["role"] = "HELDOUT_EXPOSED"
        with self.assertRaisesRegex(ValueError, "Only 16"):
            bh.normalizers(torch, bad, p)
        bad = copy.deepcopy(cases)
        bad[0]["V1"] = bad[0]["V1"][:1]
        with self.assertRaisesRegex(ValueError, "row geometry"):
            bh.normalizers(torch, bad, p)

    def test_tiny_backward_dispatch_is_four_native_BF16_mm_and_zero_start_is_trainable(self):
        m = bh.Direct65Student(D=8, outputs=2, hidden=4)
        n = unit_normalizer()
        x = torch.arange(16).reshape(2, 8).bfloat16() / 10
        eta, target = torch.ones((2, 3)), torch.ones((2, 3))
        raw = m.predict65(x, eta, 0, n)
        value, _, _ = bh.supervised_loss(torch, raw, target, n)
        calls = bh.backward_with_accounting(torch, value)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c["dtype"] == "torch.bfloat16" for c in calls))
        self.assertGreater(float(m.weights["Wout"].grad.norm()), 0)
        self.assertEqual(float(m.weights["Wx"].grad.norm()), 0)
        optimizer = torch.optim.Adam(m.parameters(), lr=.0005, foreach=False, fused=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        raw = m.predict65(x, eta, 0, n)
        value, _, _ = bh.supervised_loss(torch, raw, target, n)
        bh.backward_with_accounting(torch, value)
        self.assertGreater(float(m.weights["Wx"].grad.norm()), 0)

    def test_tiny_256_update_training_schedule_and_all_native_work_counts(self):
        p, cases, counts = parameters(), tiny_training_cases(), Counter()
        @contextmanager
        def work(key, count=1):
            counts[key] += count
            yield
        normalizer = bh.normalizers(torch, list(cases.values()), p)
        model = bh.Direct65Student(D=8, outputs=2, hidden=4, work=work)
        epochs = bh.training_loop(torch, model, cases, normalizer, bh.EXPERTS[0], p, work)
        self.assertEqual(len(epochs), 8)
        expected = {"student_random_weight_draws": 2, "student_predict65_calls": 256, "student_input_normalizations": 256,
            "student_BF16_forward_GEMM": 768, "student_BF16_tanh_GELU": 256, "training_source_batches": 256,
            "training_loss_evaluations": 256, "backward_calls": 256, "student_BF16_backward_GEMM": 1024,
            "finite_gradient_tensor_checks": 1280, "Adam_steps": 256, "Adam_parameter_tensor_updates": 1280,
            "finite_updated_parameter_tensor_checks": 1280}
        self.assertEqual(dict(counts), expected)

    def test_driver_ledger_retains_failed_dispatch_without_completion(self):
        report = {}
        ledger = driver.Ledger(report)
        with self.assertRaises(RuntimeError):
            with ledger("geometry_symmetric_eigh"):
                raise RuntimeError("intentional tiny failure")
        self.assertEqual(report["predebited"]["geometry_symmetric_eigh"], 1)
        self.assertEqual(report["invoked"]["geometry_symmetric_eigh"], 1)
        self.assertEqual(report["completed"]["geometry_symmetric_eigh"], 0)
        with self.assertRaises(ValueError):
            ledger.finish()

    def test_job_one_GPU_and_session_two_are_distinct_from_dynamic_repository_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research").mkdir()
            active = {"user_continuation_authorization_20261027": {"status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION", "new_scope_registration_allowed": True, "max_simultaneously_assigned_idle_GPUs": 99},
                "current_session_resource_constraint_20261028": {"max_simultaneously_assigned_idle_GPUs": 2},
                "resource_scheduling_policy_20261028": {"mode": "DYNAMIC_MAX_ACTUALLY_CALLABLE", "global_gpu_card_cap": None},
                "experiment_authority": {"current_user_authorized_scopes": [{"id": driver.ID, "protocol_sha256": "test", "execution_status": "AUTHORIZED_NOT_STARTED", "max_gpu_cards": 1, "session_max_simultaneously_assigned_idle_GPUs": 2, "limits": driver.exact_budget()}]}}
            path = root / "research/ACTIVE_RESEARCH.json"
            path.write_text(json.dumps(active))
            with patch.object(driver, "ROOT", root):
                driver.authorize({}, "test")
                active["current_session_resource_constraint_20261028"]["max_simultaneously_assigned_idle_GPUs"] = 1
                path.write_text(json.dumps(active))
                with self.assertRaisesRegex(ValueError, "session constraint"):
                    driver.authorize({}, "test")

    def test_declared_budget_accounts_all_main_tensor_H2D_payloads(self):
        budget = driver.exact_budget()
        self.assertEqual(budget["CUDA_H2D_tensor_copies"], 698)
        self.assertEqual(budget["CUDA_H2D_payload_bytes"], 536287248)
        self.assertEqual(budget["student_BF16_forward_GEMM"], 3 * budget["student_predict65_calls"])
        self.assertEqual(budget["student_BF16_backward_GEMM"], 4 * budget["backward_calls"])
        self.assertEqual(budget["student_direct_readout"], 384)
        self.assertEqual(budget["online_FP32_range_projection_GEMM"], 288 + 384)

    def test_environment_cannot_silently_override_FP32_contract(self):
        with patch.dict("os.environ", {"NVIDIA_TF32_OVERRIDE": "1"}):
            with self.assertRaisesRegex(ValueError, "forces TF32"):
                driver.reject_precision_overrides()
        with patch.dict("os.environ", {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}):
            with self.assertRaisesRegex(ValueError, "forces TF32"):
                driver.reject_precision_overrides()

    def test_scalar_normalizer_and_all_exported_master_deploy_geometry_bytes_reload(self):
        p = parameters()
        model = bh.Direct65Student(D=8, outputs=2, hidden=4)
        g = bh.device_geometry(geometry(), torch, "cpu")
        n = unit_normalizer()
        with tempfile.TemporaryDirectory() as directory:
            result = driver.save_export(torch, Path(directory) / "student.safetensors", model, n, {0: g, 1: g}, bh.EXPERTS[0], lambda *_: __import__("contextlib").nullcontext())
            self.assertGreater(result["bytes"], 0)
            self.assertEqual(result["tensors"]["normalizer.x_rms"]["shape"], [])
            self.assertEqual(len(result["tensors"]["normalizer.x_rms"]["raw_sha256"]), 64)
            self.assertEqual(result["tensors"]["BF16.Wout"]["dtype"], "torch.bfloat16")

    def test_actual_preparation_interface_reads_only_V1_and_checks_saved_row_payload(self):
        p, rng = parameters(), np.random.default_rng(107)
        g = geometry()
        gpu = bh.device_geometry(g, torch, "cpu")
        tables, device_tables = {0: g, 1: g}, {0: gpu, 1: gpu}
        packs, rows, conditioning, requests = {}, [], {}, []
        for step in (0, 1):
            condition = {"gate": torch.ones((1, 8), dtype=torch.bfloat16),
                         "head_scale": torch.zeros((1, 1, 8), dtype=torch.bfloat16),
                         "head_shift": torch.zeros((1, 1, 8), dtype=torch.bfloat16)}
            conditioning[step] = {k: driver.tensor_sha(torch, v) for k, v in condition.items()}
            for i in range(24):
                label_path, operand_path = f"label_{i}_{step}", f"operand_{i}_{step}"
                case = {"source_index": i, "step": step, "expert": bh.EXPERTS[0], "case_id": f"tiny_{i}_{step}",
                        "role": "TRAIN" if i < 16 else "HELDOUT_EXPOSED", "rows": [2 * i, 2 * i + 1],
                        "operand_path": operand_path, "operand_bytes": 1, "operand_sha256": "synthetic"}
                r = torch.from_numpy(rng.normal(size=(2, 8))).bfloat16()
                u = torch.from_numpy(rng.normal(size=(2, 8))).bfloat16()
                packs[label_path] = {**condition, "r_rows": r, "u_rows": u, "head_rows": torch.zeros((2, 2), dtype=torch.bfloat16), "rows": torch.tensor(case["rows"])}
                packs[operand_path] = {"V1": torch.from_numpy(rng.normal(size=(2, 8))).bfloat16()}
                rows.append({"case": case, "label": {"path": label_path}, "tensor_sha256": {k: driver.tensor_sha(torch, packs[label_path][k]) for k in ("r_rows", "u_rows", "head_rows")}})
        def load(path, names):
            requests.append((str(path), names))
            return {k: packs[str(path)][k] for k in names}
        @contextmanager
        def work(*_):
            yield
        with patch.object(driver.prior, "bound", side_effect=lambda _, item: Path(item["path"])), patch.object(driver, "tensors_cpu", side_effect=load), patch.object(bh, "copy_to_device", side_effect=lambda _, value, device, callback: value):
            cached, provenance = driver.prepare_cases(torch, {"cases": rows}, Path("/synthetic"), bh.EXPERTS[0], tables, device_tables, conditioning, p, work)
            self.assertEqual(len(cached), 48)
            self.assertEqual(len(provenance), 48)
            self.assertTrue(all(names == ("V1",) for path, names in requests if path.startswith("operand")))
            self.assertTrue(all(not row["teacher_Y_or_V2_numerically_loaded"] for row in provenance))
            packs["label_0_0"]["rows"] = torch.tensor([1, 0])
            with self.assertRaisesRegex(ValueError, "row payload"):
                driver.prepare_cases(torch, {"cases": rows}, Path("/synthetic"), bh.EXPERTS[0], tables, device_tables, conditioning, p, work)


if __name__ == "__main__":
    unittest.main()
