"""Tiny synthetic CPU controls; no CAL24 fits, real assets, or model calls."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import b_terminal_paired_discrimination as b


def tiny_data():
    rng = np.random.default_rng(104)
    heads = {expert: {"weight_KN": rng.normal(size=(8, 2))} for expert in ("high_noise_model", "low_noise_model")}
    gates = [rng.normal(size=8) for _ in range(4)]
    values, cases = {}, []
    for i in range(24):
        for step in range(4):
            name = f"cal24_{i:02d}_step{step}"
            cases.append({"case_id": name, "source_index": i, "step": step,
                          "expert": "high_noise_model" if step < 2 else "low_noise_model",
                          "rows": [100 * i + 3 * step + j for j in range(3)],
                          "role": "TRAIN" if i < 16 else "HELDOUT_EXPOSED"})
            values[name] = {"Y": rng.normal(size=(3, 8)) + i / 8,
                            "u": rng.normal(size=(3, 8)) * (1 + i / 24),
                            "gate": gates[step].copy(), "gain": np.ones(8) * (1 + step / 8),
                            "scale": np.ones(8) * (step / 8), "shift": np.ones(8) * (step / 11)}
    params = {**b.PARAMETERS, "rows_per_case": 3, "D": 8, "head_outputs": 2, "span_dimension": 4, "rank": 1}
    return cases, values, heads, params


class PairedDiscriminationControls(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        command = [sys.executable, "-c", "import sys;sys.path.insert(0,'tools');import b_terminal_paired_discrimination;assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)

    def test_implicit_actions_and_adjoint_match_explicit_jacobian(self):
        rng = np.random.default_rng(15)
        u, gate, T = rng.normal(size=(3, 7)), rng.normal(size=7), rng.normal(size=(2, 7))
        jac = b.HeadJacobian(u, gate, T)
        explicit = []
        for z, sigma in zip(jac.z, jac.sigma):
            explicit.append(T @ (np.eye(7) - np.ones((7, 7)) / 7 - np.outer(z, z) / (7 * sigma ** 2)) @ np.diag(gate) / sigma)
        explicit = np.stack(explicit)
        delta, W = rng.normal(size=(3, 7)), rng.normal(size=(7, 4))
        actual, direct, radial = jac.apply(delta)
        np.testing.assert_allclose(actual, np.einsum("nhd,nd->nh", explicit, delta), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual, direct + radial)
        H, _, _ = jac.basis(W)
        np.testing.assert_allclose(H, np.einsum("nhd,dq->nhq", explicit, W), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jac.adjoint_basis_sum(H), np.einsum("nhd,nhq->dq", explicit, H), rtol=1e-12, atol=1e-12)

    def test_marginal_change_matches_full_cartesian_state_row_product(self):
        rng = np.random.default_rng(23)
        D, n = 6, 5
        y = rng.normal(size=(n, D)) + np.arange(D)
        d = y - y.mean(axis=0)
        J = rng.normal(size=(n, 2, D))
        W, _ = np.linalg.qr(rng.normal(size=(D, 3)))
        candidates = [{"B": np.eye(3)[:, [j]]} for j in range(3)]
        C = d.T @ d / n
        G = np.einsum("nhd,nhe->de", J, J) / n
        Cq, Gq, M = W.T @ C @ W, W.T @ G @ W, W.T @ C @ G @ W
        predicted = b.marginal_changes(Cq, Gq, M, candidates)
        baseline = np.square(np.einsum("ihd,jd->ijh", J, d)).sum(axis=2).mean()
        brute = []
        for c in candidates:
            P = W @ c["B"] @ c["B"].T @ W.T
            residual = d @ (np.eye(D) - P)
            risk = np.square(np.einsum("ihd,jd->ijh", J, residual)).sum(axis=2).mean()
            brute.append(risk - baseline)
        np.testing.assert_allclose(predicted, brute, rtol=1e-12, atol=1e-12)
        # Pairing identical token ordinals is a different estimand.
        paired = np.square(np.einsum("ihd,id->ih", J, d)).sum(axis=1).mean()
        self.assertGreater(abs(paired - baseline), 1e-2)

    def test_public_counterexample_reverses_joint_and_marginal_rankings(self):
        # Four +/- rows make the same common mu=0; source-specific centering
        # would destroy precisely the between-source information being tested.
        d = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        G = np.asarray([np.diag([.1, 1]), np.diag([.1, 1]), np.diag([100, 10]), np.diag([100, 10])])
        paired, marginal = [], []
        Cbar = d.T @ d / 4
        for kept in (0, 1):
            R = np.eye(2)
            R[kept, kept] = 0
            paired.append(np.einsum("ni,nij,nj->n", d @ R, G, d @ R).mean())
            marginal.append(np.trace(G.mean(axis=0) @ R @ Cbar @ R.T))
        self.assertEqual(int(np.argmin(paired)), 1)
        self.assertEqual(int(np.argmin(marginal)), 0)
        self.assertAlmostEqual(paired[0] / paired[1], 100)

    def test_conditioning_on_step_removes_an_expert_time_only_ranking_confound(self):
        # No source dependence at all within a step. Unconditional averaging
        # alone reverses the ordering; step-conditioned marginal equals joint.
        G_steps = [np.diag([.1, 1]), np.diag([100, 10])] * 2
        C_steps = [np.diag([1., 0]), np.diag([0., 1.])] * 2
        conditioned, unconditional = [], []
        for kept in (0, 1):
            R = np.eye(2)
            R[kept, kept] = 0
            conditioned.append(np.mean([np.trace(G @ R @ C @ R) for G, C in zip(G_steps, C_steps)]))
            unconditional.append(np.trace(np.mean(G_steps, axis=0) @ R @ np.mean(C_steps, axis=0) @ R))
        self.assertEqual(int(np.argmin(conditioned)), 1)
        self.assertEqual(int(np.argmin(unconditional)), 0)

    def test_analysis_primary_marginal_matches_step_cartesian_brute_force(self):
        cases, values, heads, parameters = tiny_data()
        report, state = b.analyze(cases, lambda c: values[c["case_id"]], heads, parameters)
        W, mu = state["W"], state["mu"]
        changes = []
        for step in range(4):
            selected = [c for c in cases if c["source_index"] < 16 and c["step"] == step]
            d = np.concatenate([values[c["case_id"]]["Y"] for c in selected]) - mu
            u = np.concatenate([values[c["case_id"]]["u"] for c in selected])
            jac = b.HeadJacobian(u, state["refs"][step]["gate"], state["refs"][step]["T"])
            J, _, _ = jac.basis(np.eye(8))
            baseline = np.square(np.einsum("ihd,jd->ijh", J, d)).sum(axis=2).mean()
            risk_changes = []
            for candidate in state["candidates"]:
                basis = W @ candidate["B"]
                residual = d - (d @ basis) @ basis.T
                risk = np.square(np.einsum("ihd,jd->ijh", J, residual)).sum(axis=2).mean()
                risk_changes.append(risk - baseline)
            changes.append(risk_changes)
        np.testing.assert_allclose(report["training_scores"]["marginal_per_step_risk_changes"], changes, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(report["training_scores"]["marginal_risk_change_from_P0"], np.mean(changes, axis=0), rtol=1e-11, atol=1e-11)

    def test_finite_identity_and_linear_small_delta_limit(self):
        rng = np.random.default_rng(9)
        u, delta, T = rng.normal(size=(3, 7)), rng.normal(size=(3, 7)) * 1e-5, rng.normal(size=(2, 7))
        finite, linear = b.finite_formula(u, delta, T)
        def F(x):
            z = x - x.mean(axis=1, keepdims=True)
            return (z @ T.T) / np.sqrt(np.square(z).mean(axis=1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(finite, F(u + delta) - F(u), rtol=1e-9, atol=2e-15)
        self.assertLess(np.linalg.norm(finite - linear), 1e-8)

    def test_constant_shift_null_direction(self):
        rng = np.random.default_rng(33)
        jac = b.HeadJacobian(rng.normal(size=(2, 8)), np.ones(8), rng.normal(size=(2, 8)))
        actual, _, _ = jac.apply(np.ones((2, 8)))
        np.testing.assert_allclose(actual, 0, atol=1e-14)

    def test_candidate_pool_has_common_rank_and_idempotence(self):
        rng = np.random.default_rng(7)
        W, _ = np.linalg.qr(rng.normal(size=(9, 5)))
        X = rng.normal(size=(5, 5))
        candidates = b.candidate_family(W, X.T @ X, 2, 42)
        self.assertEqual(len(candidates), 7)
        for c in candidates:
            basis = b.projection_basis(W, c["B"])
            P = basis @ basis.T
            self.assertEqual(np.linalg.matrix_rank(P), 2)
            np.testing.assert_allclose(P @ P, P, atol=1e-12)
        self.assertEqual(candidates[0]["id"], "head_static_top_rank")

    def test_one_source_cycle_has_no_fixed_points_and_is_reproducible(self):
        mapping = b.source_derangement(range(16), 73)
        self.assertEqual(mapping, b.source_derangement(range(16), 73))
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(i != donor for i, donor in mapping.items()))
        with self.assertRaises(ValueError):
            b.source_derangement([0], 73)

    def test_source_aggregation_checks_all_four_steps(self):
        cases = [{"case_id": f"{i}:{s}", "source_index": i, "step": s} for i in (0, 1) for s in range(4)]
        scores = {c["case_id"]: np.asarray([100 * c["source_index"] + c["step"]]) for c in cases}
        mean, per_source = b.equal_source_scores(cases, scores, [0, 1])
        self.assertEqual(set(per_source), {0, 1})
        self.assertAlmostEqual(float(mean[0]), 51.5)
        with self.assertRaises(ValueError):
            b.equal_source_scores(cases[:-1], scores, [0, 1])

    def test_heldout_cannot_change_common_mu_basis_or_any_training_selection(self):
        cases, values, heads, parameters = tiny_data()
        result, state = b.analyze(cases, lambda c: values[c["case_id"]], heads, parameters)
        actual_mu = np.concatenate([values[c["case_id"]]["Y"] for c in cases if c["source_index"] < 16]).mean(axis=0)
        np.testing.assert_allclose(state["mu"], actual_mu, atol=1e-15)
        changed = copy.deepcopy(values)
        for c in cases:
            if c["source_index"] >= 16:
                changed[c["case_id"]]["Y"] += 10000
                changed[c["case_id"]]["u"] *= 30
        other, other_state = b.analyze(cases, lambda c: changed[c["case_id"]], heads, parameters)
        np.testing.assert_array_equal(state["mu"], other_state["mu"])
        np.testing.assert_array_equal(state["W"], other_state["W"])
        self.assertEqual(result["selection"], other["selection"])
        self.assertEqual(result["training_scores"], other["training_scores"])
        self.assertNotEqual(result["paired_evaluation"]["selected_heldout"], other["paired_evaluation"]["selected_heldout"])
        self.assertEqual(result["paired_evaluation"]["train_sources"], 16)
        self.assertEqual(result["paired_evaluation"]["heldout_exposed_sources"], 8)
        components = result["projected_head_vs_radial"]["heldout_exposed"]["selected"]
        for arm in b.ARMS:
            values = components[arm]
            self.assertAlmostEqual(values["total"], values["direct_head"] + values["radial"] + values["twice_signed_cross"], places=12)
        mapping = result["source_shuffle"]["permutation"]
        for row in result["source_shuffle"]["row_mapping"]:
            host = next(c for c in cases if c["case_id"] == row["host"])
            donor = next(c for c in cases if c["case_id"] == row["donor"])
            self.assertEqual(donor["source_index"], mapping[host["source_index"]])
            self.assertEqual(donor["step"], host["step"])
            self.assertNotEqual(row["host_rows"], row["donor_rows"])

    def test_ragged_rows_or_varying_conditioning_fail_without_fallback(self):
        cases, values, heads, parameters = tiny_data()
        bad = copy.deepcopy(values)
        bad[cases[0]["case_id"]]["Y"] = bad[cases[0]["case_id"]]["Y"][:2]
        with self.assertRaisesRegex(ValueError, "geometry"):
            b.analyze(cases, lambda c: bad[c["case_id"]], heads, parameters)
        bad = copy.deepcopy(values)
        bad["cal24_01_step0"]["gate"][0] += .01
        with self.assertRaisesRegex(ValueError, "conditioning"):
            b.analyze(cases, lambda c: bad[c["case_id"]], heads, parameters)

    def test_condition_byte_check_distinguishes_signed_zero(self):
        cases, values, heads, parameters = tiny_data()
        for c in cases:
            if c["step"] == 0:
                values[c["case_id"]]["shift"][0] = 0.0
        values["cal24_01_step0"]["shift"][0] = -0.0
        with self.assertRaisesRegex(ValueError, "conditioning"):
            b.analyze(cases, lambda c: values[c["case_id"]], heads, parameters)

    def test_rankings_use_exact_declared_non_io_budgets(self):
        cases, values, heads, parameters = tiny_data()
        parameters = {**parameters, "span_dimension": 4}
        actual = {}
        reads = []
        @contextmanager
        def work(key, count=1):
            actual[key] = actual.get(key, 0) + count
            yield
        def read(c):
            reads.append(c["case_id"])
            return values[c["case_id"]]
        b.analyze(cases, read, heads, parameters, work)
        self.assertEqual(len(reads), 384)  # before 32 heldout finite reads
        expected = {"shared_head_SVD": 1, "projected_covariance_eigh": 1, "random_basis_QR": 1,
            "paired_J_action": 96, "paired_J_basis": 96, "train_J_adjoint_basis": 64,
            "marginal_cross_moment": 64, "source_shuffled_J_action": 64,
            "step_conditioned_marginal_cross_moment": 64,
            "candidate_loss_evaluations": (96 * 2 + 64) * (4 + 2), "marginal_candidate_scores": 30,
            "radial_component_candidate_evaluations": 96 * 6}
        self.assertEqual(actual, expected)

    def test_failed_operation_keeps_debit_and_invocation_without_completion(self):
        record = {}
        ledger = b.WorkLedger(record)
        with self.assertRaises(RuntimeError):
            with ledger("shared_head_SVD"):
                raise RuntimeError("intentional tiny failure")
        self.assertEqual(record["predebited"]["shared_head_SVD"], 1)
        self.assertEqual(record["invoked"]["shared_head_SVD"], 1)
        self.assertEqual(record["completed"]["shared_head_SVD"], 0)
        with self.assertRaises(ValueError):
            with ledger("shared_head_SVD"):
                pass
        with self.assertRaises(ValueError):
            ledger.finish()

    def test_bf16_write_obeys_y_update_then_gate_rounding_then_add(self):
        rng = np.random.default_rng(12)
        y = torch.from_numpy(rng.normal(size=(3, 8))).bfloat16()
        r = torch.from_numpy(rng.normal(size=(3, 8))).bfloat16()
        gate = torch.from_numpy(rng.normal(size=(1, 8))).bfloat16()
        basis = np.eye(8)[:, :2]
        mu = np.full(8, .13)
        actual = b.bf16_projected_write(torch, r, y, gate, mu, basis, 1 / 4)
        d = y.double().numpy() - mu
        delta = np.asarray(-.25 * (d - (d @ basis) @ basis.T), dtype=np.float32)
        y2 = (y.float() + torch.from_numpy(delta)).bfloat16()
        expected = r.clone()
        expected.add_(y2 * gate)
        self.assertTrue(b.same_bits(torch, actual, expected))
        fused_real = (r.float() + y2.float() * gate.float()).bfloat16()
        self.assertFalse(b.same_bits(torch, actual, fused_real))

    def test_actual_gain_rounding_differs_from_real_one_plus_scale(self):
        scale = torch.tensor([.01, -.01], dtype=torch.bfloat16)
        native_gain = (1 + scale).double().numpy()
        real_gain = 1 + scale.double().numpy()
        self.assertGreater(np.max(np.abs(native_gain - real_gain)), 1e-3)

    def test_tiny_explicit_m256_helper_has_the_declared_head_operation_order(self):
        u = torch.tensor([[1., 2., 3., 5.], [4., -2., .5, 1.]], dtype=torch.bfloat16)
        weight = torch.arange(8).reshape(4, 2).bfloat16() / 7
        bias = torch.tensor([.1, -.2], dtype=torch.bfloat16)
        scale = torch.full((1, 1, 4), .01, dtype=torch.bfloat16)
        shift = torch.full((1, 1, 4), -.2, dtype=torch.bfloat16)
        actual = b.native_M256_head(torch, u, weight, bias, scale, shift, 1e-6)
        x = torch.nn.functional.layer_norm(u, (4,), None, None, 1e-6)
        x.mul_(1 + scale.reshape(-1)).add_(shift.reshape(-1))
        expected = torch.addmm(bias, x, weight)
        self.assertTrue(b.same_bits(torch, actual, expected))

    def test_finite_schedule_keeps_all_six_arms_and_exact_448_tiny_head_calls(self):
        cases, values, heads, parameters = tiny_data()
        work_counts = {}
        @contextmanager
        def work(key, count=1):
            work_counts[key] = work_counts.get(key, 0) + count
            yield
        class TinyBank:
            def __init__(self):
                self.torch = torch
                self.entries = {c["case_id"]: {"evidence": {"head_weight_stride": [1, 8]}} for c in cases}
                self.heads = {}
                for expert in heads:
                    weight = torch.from_numpy(heads[expert]["weight_KN"]).bfloat16()
                    self.heads[expert] = {"weight_KN": weight.double().numpy(), "BF16_weight_KN": weight, "BF16_bias": torch.zeros(2, dtype=torch.bfloat16)}
            @staticmethod
            def array(t):
                return t.double().numpy()
            def __call__(self, case, finite=False):
                data = values[case["case_id"]]
                pack = {"Y": torch.from_numpy(data["Y"]).bfloat16(), "r_rows": torch.zeros((3, 8), dtype=torch.bfloat16),
                        "gate": torch.from_numpy(data["gate"]).reshape(1, 8).bfloat16(),
                        "head_scale": torch.from_numpy(data["scale"]).reshape(1, 1, 8).bfloat16(),
                        "head_shift": torch.from_numpy(data["shift"]).reshape(1, 1, 8).bfloat16(),
                        "head_rows": torch.zeros((3, 2), dtype=torch.bfloat16)}
                pack["u_rows"] = pack["r_rows"].clone()
                pack["u_rows"].add_(pack["Y"] * pack["gate"])
                result = {"Y": self.array(pack["Y"]), "u": self.array(pack["u_rows"]),
                    "gate": self.array(pack["gate"]).reshape(-1), "scale": self.array(pack["head_scale"]).reshape(-1),
                    "gain": self.array(1 + pack["head_scale"]).reshape(-1), "shift": self.array(pack["head_shift"]).reshape(-1)}
                if finite:
                    result["BF16"] = pack
                return result
        bank = TinyBank()
        _, state = b.analyze(cases, bank, bank.heads, parameters)
        result = b.finite_replay(cases, bank, state, parameters, work)
        self.assertEqual(len(result["records"]), 32 + 384)
        self.assertEqual(result["full_M_native_noop"], "NOT_RUN_SEPARATE_SCOPE")
        expected = {"BF16_gate_mul_then_add": 416, "M256_layer_norm_calls": 448, "M256_head_calls": 448,
            "BF16_head_mod_gain_adds": 448, "BF16_head_mod_mul_then_add": 448,
            "M256_baseline_repeat_checks": 32, "BF16_Y_updates": 384, "finite_real_and_linear": 384}
        self.assertEqual(work_counts, expected)
        self.assertEqual(len(result["heldout_summary"]), 12)
        for value in result["heldout_summary"].values():
            self.assertEqual(value["independent_exposed_sources"], 8)

    def test_draft_cannot_be_promoted_by_manifest_path_alone(self):
        draft = {"id": b.ID, "parameters": b.PARAMETERS, "resources": b.RESOURCES, "limits": b.exact_budget(),
            "source_split": {"training": list(range(16)), "diagnostic_heldout": list(range(16, 24))},
            "parent_protocol": {"sha256": b.PARENT_SHA}, "assets_finalized": False,
            "binding_status": "NOT_BOUND", "label_manifest": "NOT_BOUND"}
        b.validate_plan(draft)
        with self.assertRaisesRegex(ValueError, "NOT_BOUND"):
            b.validate_plan(draft, frozen=True)


if __name__ == "__main__":
    unittest.main()
