#!/usr/bin/env python3
"""Registered CAL24 terminal-head discrimination in one shared projection family.

The default entry point only inspects a draft. Numerical work is CPU-only and
requires frozen assets plus an ACTIVE registration. Importing this file performs
no NumPy/Torch import, fitting, decomposition, model call, or device discovery.

All P are orthogonal rank-r projections in one finite, common head-derived span.
The paired and marginal selectors therefore optimize a *dictionary*, not the
unrestricted joint/oblique-projection problem. The marginal score is its exact
risk CHANGE from P=0; its potentially expensive constant is not estimated.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_B_TERMINAL_PAIRED_DISCRIMINATION_20261028"
PARENT_SHA = "956fead39051a68b1c9b896e937ba5459765060a1f42674052bf2641d42da71f"
SCHEMA = "ffnds.b_cal24.terminal_label_asset_manifest.v1"
ARMS = ("paired", "source_shuffled", "marginal", "PCA_in_shared_span", "fixed_random", "shared_head_only")
PARAMETERS = {"span_dimension": 16, "rank": 4, "epsilon": 1e-6,
              "random_basis_seed": 61028, "source_permutation_seed": 20261028,
              "finite_amplitudes": [1 / 16, 1 / 4], "CPU_threads": 8,
              "rows_per_case": 256, "D": 5120, "head_outputs": 64}
RESOURCES = {"assigned_GPUs": 0, "global_max_simultaneously_assigned_idle_GPUs": 2,
             "numerical_device": "cpu", "max_output_bytes": 64 * (1 << 20),
             "automatic_retry": False, "models_loaded": 0, "videos_generated": 0}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def bound(root, item):
    check(isinstance(item, dict) and set(("path", "bytes", "sha256")) <= set(item), "Asset NOT_BOUND")
    root = Path(root).resolve()
    path = (root / item["path"]).resolve()
    check(path != root and path.is_relative_to(root), "Asset escapes bound root")
    check(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Asset byte binding differs: " + str(item["path"]))
    return path


def np64(value):
    import numpy as np
    result = np.asarray(value, dtype=np.float64)
    check(np.isfinite(result).all(), "Nonfinite mathematical input")
    return result


class HeadJacobian:
    """Real LN/head Jacobian at BF16 native u, with fixed actual BF16 gain.

    T = W_head diag(fl_BF16(1+scale)). This is a continuous, source-anchored
    surrogate, not the derivative of a discrete BF16 implementation. No explicit
    rows x outputs x D Jacobian is ever constructed. shift/bias cancel here.
    """
    def __init__(self, u, gate, T, epsilon=1e-6):
        import numpy as np
        self.u, self.gate, self.T = np64(u), np64(gate).reshape(-1), np64(T)
        check(self.u.ndim == self.T.ndim == 2 and self.u.shape[1] == self.T.shape[1] == self.gate.size and epsilon > 0, "Jacobian shapes differ")
        self.D = self.u.shape[1]
        self.z = self.u - self.u.mean(axis=1, keepdims=True)
        self.sigma = np.sqrt(np.square(self.z).mean(axis=1) + epsilon)
        self.K = self.T - self.T.mean(axis=1, keepdims=True)
        self.q = self.z @ self.T.T

    def apply(self, delta_y):
        d = np64(delta_y)
        check(d.shape == self.u.shape, "Jacobian action shape differs")
        v = d * self.gate
        direct = v @ self.K.T / self.sigma[:, None]
        radial = -self.q * ((self.z * v).sum(axis=1) / (self.D * self.sigma ** 3))[:, None]
        return direct + radial, direct, radial

    def basis(self, W):
        W = np64(W)
        check(W.ndim == 2 and W.shape[0] == self.D, "Shared basis shape differs")
        v = W * self.gate[:, None]
        direct = (self.K @ v)[None, :, :] / self.sigma[:, None, None]
        radial = -self.q[:, :, None] * (self.z @ v)[:, None, :] / (self.D * self.sigma[:, None, None] ** 3)
        return direct + radial, direct, radial

    def adjoint_basis_sum(self, H):
        """Sum_i J_y,i^T H_i in D x q memory, for exact marginal ranking."""
        import numpy as np
        H = np64(H)
        check(H.ndim == 3 and H.shape[:2] == (self.u.shape[0], self.T.shape[0]), "Adjoint shape differs")
        first = self.K.T @ (H / self.sigma[:, None, None]).sum(axis=0)
        coeff = np.einsum("nh,nhq->nq", self.q, H) / (self.D * self.sigma[:, None] ** 3)
        return (first - self.z.T @ coeff) * self.gate[:, None]


def projection_basis(W, B):
    import numpy as np
    W, B = np64(W), np64(B)
    check(W.ndim == B.ndim == 2 and W.shape[1] == B.shape[0], "Projection basis shape differs")
    check(np.allclose(W.T @ W, np.eye(W.shape[1]), rtol=0, atol=1e-9) and np.allclose(B.T @ B, np.eye(B.shape[1]), rtol=0, atol=1e-9), "Projection is not orthonormal")
    return W @ B


def candidate_family(W, covariance, rank, seed, work=lambda *_: nullcontext()):
    """q cyclic supports + train-span PCA + one fixed random; every arm shares it."""
    import numpy as np
    W, covariance = np64(W), np64(covariance)
    q = W.shape[1]
    check(0 < rank < q and covariance.shape == (q, q), "Invalid candidate family dimensions")
    candidates = []
    for start in range(q):
        columns = [(start + j) % q for j in range(rank)]
        candidates.append({"id": "head_static_top_rank" if start == 0 else f"cyclic_{start:02d}", "B": np.eye(q)[:, columns], "construction": "cyclic shared head-SVD coordinate support"})
    with work("projected_covariance_eigh", 1):
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    candidates.append({"id": "PCA_in_shared_span", "B": vectors[:, -rank:], "construction": "training projected covariance eigh; constrained to shared span"})
    random = np.random.default_rng(seed).standard_normal((q, rank))
    with work("random_basis_QR", 1):
        random, triangular = np.linalg.qr(random, mode="reduced")
    check(np.abs(np.diag(triangular)).min() > 1e-12, "Fixed random draw lost rank; no retry")
    candidates.append({"id": "fixed_random", "B": random, "construction": "one frozen Gaussian QR draw, no selection or retries"})
    for c in candidates:
        projection_basis(W, c["B"])
    return candidates


def losses(feature, candidates, field="paired"):
    import numpy as np
    check(field in ("paired", "direct", "radial"), "Unknown response component")
    if field == "radial":
        f, H = feature["f"] - feature["direct_f"], feature["H"] - feature["direct_H"]
    else:
        f = feature["f" if field == "paired" else "direct_f"]
        H = feature["H" if field == "paired" else "direct_H"]
    a = feature["a"]
    result = []
    for c in candidates:
        B = c["B"]
        projected = (a @ B) @ B.T
        residual = f - np.einsum("nhq,nq->nh", H, projected)
        result.append(float(np.square(residual).sum(axis=1).mean()))
    return np.asarray(result)


def marginal_changes(Cq, Gq, M, candidates):
    """Exact tr(Gbar R Cbar R') minus tr(Gbar Cbar), no dense D x D matrix.

    Cq=W' Cbar W, Gq=W' Gbar W, M=W' Cbar Gbar W. Only the common
    constant is omitted, so all candidate orderings equal the full marginal.
    """
    import numpy as np
    Cq, Gq, M = np64(Cq), np64(Gq), np64(M)
    out = []
    for c in candidates:
        B = c["B"]
        S = B @ B.T
        out.append(float(-2 * np.trace(S @ M) + np.trace(S @ Gq @ S @ Cq)))
    return np.asarray(out)


def source_derangement(indices, seed):
    """One fixed source-cluster cycle; donor carries all states, steps and rows.

    Row ordinals, not physical coordinates, are aligned between donor and host.
    This is one artificial source pairing; it is neither a token shuffle nor an
    estimate of the Cartesian-product marginal or a permutation-test p-value.
    """
    import numpy as np
    indices = list(indices)
    check(len(indices) > 1 and len(set(indices)) == len(indices), "Need distinct source clusters")
    cycle = np.random.default_rng(seed).permutation(indices).tolist()
    return {source: cycle[(j + 1) % len(cycle)] for j, source in enumerate(cycle)}


def equal_source_scores(cases, case_scores, indices):
    """Four step means -> equal source means; row count never becomes N."""
    import numpy as np
    output = {}
    for i in indices:
        items = [(c["step"], case_scores[c["case_id"]]) for c in cases if c["source_index"] == i]
        check(len(items) == 4 and {step for step, _ in items} == set(range(4)), "Incomplete source state nesting")
        output[i] = np.mean(np.stack([v for _, v in sorted(items)]), axis=0)
    return np.mean(np.stack(list(output.values())), axis=0), output


def finite_formula(u, delta, T, epsilon=1e-6):
    """Exact real finite head difference and first-order prediction, rowwise."""
    import numpy as np
    u, delta, T = np64(u), np64(delta), np64(T)
    check(u.shape == delta.shape and u.ndim == T.ndim == 2 and T.shape[1] == u.shape[1], "Finite response shapes differ")
    z, d0 = u - u.mean(axis=1, keepdims=True), delta - delta.mean(axis=1, keepdims=True)
    sigma2 = np.square(z).mean(axis=1) + epsilon
    q, direct = z @ T.T, d0 @ T.T
    new_sigma2 = np.square(z + d0).mean(axis=1) + epsilon
    finite = (q + direct) / np.sqrt(new_sigma2[:, None]) - q / np.sqrt(sigma2[:, None])
    linear = (direct - q * ((z * delta).sum(axis=1) / (u.shape[1] * sigma2))[:, None]) / np.sqrt(sigma2[:, None])
    return finite, linear


def analyze(cases, read_case, heads, parameters, work=lambda *_: nullcontext()):
    """Multi-pass streaming analysis; no heldout information trains any selector.

    read_case(case, finite=False) returns numeric Y,u,gate,gain and, for finite
    replay, original BF16 tensors. It must verify provenance before loading.
    Feature memory is O(cases * rows * 64 * q), never O(rows * 64 * D).
    """
    import numpy as np
    p = parameters
    train, held = list(range(16)), list(range(16, 24))
    check(len(cases) == 96 and {(c["source_index"], c["step"]) for c in cases} == {(i, s) for i in range(24) for s in range(4)}, "Need exact CAL24 source-state cases")
    check(all(c["role"] == ("TRAIN" if c["source_index"] < 16 else "HELDOUT_EXPOSED") for c in cases), "Source role differs")
    cases = sorted(cases, key=lambda c: (c["source_index"], c["step"]))
    refs, total_y, train_rows = {}, None, 0
    for c in cases:
        v = read_case(c)
        check(len(c["rows"]) == p["rows_per_case"] and len(set(c["rows"])) == p["rows_per_case"] and v["Y"].shape == v["u"].shape == (p["rows_per_case"], p["D"]), "Fixed equal per-source/step row geometry differs")
        check(heads[c["expert"]]["weight_KN"].shape == (p["D"], p["head_outputs"]), "Head output geometry differs")
        for name in ("gate", "gain", "scale", "shift"):
            if c["step"] in refs:
                check(np.array_equal(v[name], refs[c["step"]][name]) and v[name].tobytes() == refs[c["step"]][name].tobytes(), "Step conditioning differs across sources; shared-head assumption failed, no fallback")
        if c["step"] not in refs:
            refs[c["step"]] = {k: v[k].copy() for k in ("gate", "gain", "scale", "shift")}
            refs[c["step"]]["expert"] = c["expert"]
        if c["source_index"] in train:
            row_sum = v["Y"].sum(axis=0)
            total_y = row_sum if total_y is None else total_y + row_sum
            train_rows += v["Y"].shape[0]
    mu = total_y / train_rows
    direct_operators = []
    for step in range(4):
        ref = refs[step]
        T = heads[ref["expert"]]["weight_KN"].T * ref["gain"][None, :]
        ref["T"] = T
        direct_operators.append((T - T.mean(axis=1, keepdims=True)) * ref["gate"][None, :])
    stack = np.concatenate(direct_operators, axis=0)
    with work("shared_head_SVD", 1):
        _, singular, right = np.linalg.svd(stack, full_matrices=False)
    q, r = p["span_dimension"], p["rank"]
    check(0 < r < q <= min(stack.shape) and singular[q - 1] > 1e-10 * singular[0], "Shared head span lost prescribed rank; no retry")
    W = right[:q].T.copy()
    feature, Cq, Gq, A = {}, np.zeros((q, q)), np.zeros((q, q)), np.zeros((W.shape[0], q))
    Cq_step, Gq_step, A_step = np.zeros((4, q, q)), np.zeros((4, q, q)), np.zeros((4, W.shape[0], q))
    direct_diagnostics = {}
    for c in cases:
        v = read_case(c)
        d = v["Y"] - mu
        jac = HeadJacobian(v["u"], v["gate"], refs[c["step"]]["T"], p["epsilon"])
        with work("paired_J_action", 1):
            f, direct_f, radial_f = jac.apply(d)
        with work("paired_J_basis", 1):
            H, direct_H, _ = jac.basis(W)
        a = d @ W
        feature[c["case_id"]] = {"f": f, "direct_f": direct_f, "H": H, "direct_H": direct_H, "a": a,
                                    "Y_second_moment": float(np.square(d).sum(axis=1).mean())}
        direct_diagnostics[c["case_id"]] = {"direct_head_SSE": float(np.square(direct_f).sum(axis=1).mean()),
            "radial_SSE": float(np.square(radial_f).sum(axis=1).mean()),
            "twice_signed_cross": float(2 * (direct_f * radial_f).sum(axis=1).mean()),
            "total_SSE": float(np.square(f).sum(axis=1).mean()),
            "sigma_min": float(jac.sigma.min()), "sigma_max": float(jac.sigma.max())}
        if c["source_index"] in train:
            covariance = a.T @ a / train_rows
            sensitivity = np.einsum("nhq,nhr->qr", H, H) / train_rows
            Cq += covariance
            Gq += sensitivity
            Cq_step[c["step"]] += 4 * covariance
            Gq_step[c["step"]] += 4 * sensitivity
            with work("train_J_adjoint_basis", 1):
                adjoint = jac.adjoint_basis_sum(H) / train_rows
                A += adjoint
                A_step[c["step"]] += 4 * adjoint
    candidates = candidate_family(W, Cq, r, p["random_basis_seed"], work)
    M, M_step = np.zeros((q, q)), np.zeros((4, q, q))
    for c in cases:
        if c["source_index"] in train:
            v = read_case(c)
            d = v["Y"] - mu
            with work("marginal_cross_moment", 1):
                M += (d @ W).T @ (d @ A) / train_rows
            with work("step_conditioned_marginal_cross_moment", 1):
                M_step[c["step"]] += 4 * (d @ W).T @ (d @ A_step[c["step"]]) / train_rows
    pair_scores, direct_scores, radial_scores = {}, {}, {}
    for c in cases:
        with work("candidate_loss_evaluations", len(candidates)):
            pair_scores[c["case_id"]] = losses(feature[c["case_id"]], candidates)
        with work("candidate_loss_evaluations", len(candidates)):
            direct_scores[c["case_id"]] = losses(feature[c["case_id"]], candidates, "direct")
        with work("radial_component_candidate_evaluations", len(candidates)):
            radial_scores[c["case_id"]] = losses(feature[c["case_id"]], candidates, "radial")
    pair_train, _ = equal_source_scores(cases, pair_scores, train)
    head_train, _ = equal_source_scores(cases, direct_scores, train)
    permutation = source_derangement(train, p["source_permutation_seed"])
    by_coord = {(c["source_index"], c["step"]): c for c in cases}
    shuffled_scores, donor_rows = {}, []
    for c in cases:
        if c["source_index"] not in train:
            continue
        donor = by_coord[(permutation[c["source_index"]], c["step"])]
        host_v, donor_v = read_case(c), read_case(donor)
        jac = HeadJacobian(donor_v["u"], donor_v["gate"], refs[donor["step"]]["T"], p["epsilon"])
        with work("source_shuffled_J_action", 1):
            f, _, _ = jac.apply(host_v["Y"] - mu)
        donor_feature = feature[donor["case_id"]]
        with work("candidate_loss_evaluations", len(candidates)):
            shuffled_scores[c["case_id"]] = losses({"f": f, "H": donor_feature["H"], "a": feature[c["case_id"]]["a"]}, candidates)
        donor_rows.append({"host": c["case_id"], "donor": donor["case_id"], "host_rows": c["rows"], "donor_rows": donor["rows"], "alignment": "row ordinal; physical coordinates may differ"})
    shuffled_train, _ = equal_source_scores(cases, shuffled_scores, train)
    with work("marginal_candidate_scores", len(candidates)):
        unconditional_marginal = marginal_changes(Cq, Gq, M, candidates)
    conditional_marginals = []
    for step in range(4):
        with work("marginal_candidate_scores", len(candidates)):
            conditional_marginals.append(marginal_changes(Cq_step[step], Gq_step[step], M_step[step], candidates))
    marginal = np.mean(np.stack(conditional_marginals), axis=0)
    ids = [c["id"] for c in candidates]
    select = {"paired": int(np.argmin(pair_train)), "source_shuffled": int(np.argmin(shuffled_train)),
              "marginal": int(np.argmin(marginal)), "PCA_in_shared_span": ids.index("PCA_in_shared_span"),
              "fixed_random": ids.index("fixed_random"), "shared_head_only": int(np.argmin(head_train))}
    train_mean, train_by_source = equal_source_scores(cases, pair_scores, train)
    held_mean, held_by_source = equal_source_scores(cases, pair_scores, held)
    components = {}
    for label, source_indices in (("train", train), ("heldout_exposed", held)):
        head, per_source_head = equal_source_scores(cases, direct_scores, source_indices)
        radial, per_source_radial = equal_source_scores(cases, radial_scores, source_indices)
        total, per_source_total = equal_source_scores(cases, pair_scores, source_indices)
        components[label] = {"independent_source_count": len(source_indices), "direct_head_SSE_all_candidates": head.tolist(),
            "radial_SSE_all_candidates": radial.tolist(), "twice_signed_cross_all_candidates": (total - head - radial).tolist(),
            "per_source": {str(i): {"direct_head": per_source_head[i].tolist(), "radial": per_source_radial[i].tolist(),
                "twice_signed_cross": (per_source_total[i] - per_source_head[i] - per_source_radial[i]).tolist()} for i in source_indices},
            "selected": {arm: {"direct_head": float(head[index]), "radial": float(radial[index]),
                "twice_signed_cross": float(total[index] - head[index] - radial[index]), "total": float(total[index])} for arm, index in select.items()}}
    result = {"evidence": "OBS_CAL24_M256_TERMINAL_REAL_SURROGATE_FINITE_DICTIONARY",
        "common_mu": {"training_source_count": 16, "steps_per_source": 4, "rows_per_case": p["rows_per_case"],
            "definition": "one equal source/step/row mean across all train16; no source-specific centering", "FP64_raw_sha256": hashlib.sha256(mu.astype("<f8").tobytes()).hexdigest()},
        "projection_family": {"q": q, "rank": r, "candidate_count": len(candidates), "all_arms_share_pool": True,
            "span": "top-q right singular vectors of stacked four actual centered-head times gate operators; source0 conditioning; confirmed constant across all source at same step",
            "scope": "orthogonal fixed dictionary, no claim of global joint or oblique optimum", "singular_values": singular.tolist(),
            "shared_static_head_candidate": "head_static_top_rank", "candidates": [{"id": c["id"], "construction": c["construction"]} for c in candidates]},
        "training_scores": {"paired": pair_train.tolist(), "source_shuffled": shuffled_train.tolist(),
            "marginal_risk_change_from_P0": marginal.tolist(), "marginal_absolute_baseline": "NOT_MEASURED_COMMON_CONSTANT",
            "marginal_statistical_identity": "PRIMARY: equal mean of four step-conditioned independent training source/row state versus training source/row Y products; one global training mu, step/expert retained",
            "marginal_per_step_risk_changes": [x.tolist() for x in conditional_marginals],
            "unconditional_marginal_risk_change_DIAGNOSTIC_ONLY": unconditional_marginal.tolist(),
            "unconditional_marginal_identity": "independent all-training source/step/row state versus Y; breaks time/expert pairing; never selects a finite arm",
            "direct_head_only": head_train.tolist()},
        "selection": {arm: ids[index] for arm, index in select.items()},
        "source_shuffle": {"permutation": permutation, "statistics": "one fixed source-cluster derangement; no token shuffle, no marginal estimator, no p-value", "row_mapping": donor_rows},
        "paired_evaluation": {"train_sources": 16, "heldout_exposed_sources": 8, "historical_exposure": "all CAL24 are development/exposed, not fresh confirmation",
            "train_mean_all_candidates": train_mean.tolist(), "heldout_mean_all_candidates": held_mean.tolist(),
            "train_per_source": {str(i): x.tolist() for i, x in train_by_source.items()},
            "heldout_per_source": {str(i): x.tolist() for i, x in held_by_source.items()},
            "case_scores": {k: x.tolist() for k, x in pair_scores.items()},
            "selected_heldout": {arm: float(held_mean[index]) for arm, index in select.items()},
            "independent_unit": "source image cluster; rows/step nested, no token-level inference"},
        "head_vs_radial": direct_diagnostics,
        "projected_head_vs_radial": components,
        "gate_and_modulation_same_step_equal_across_all_sources": True,
        "full_M_native_noop": "NOT_RUN_IN_THIS_SCOPE; previous B0 covers only its four separate cases",
        "video_or_runtime_claim": "NOT_MEASURED"}
    return result, {"mu": mu, "W": W, "candidates": candidates, "selected_indices": select, "refs": refs}


def exact_budget():
    """High-level numerical invocations, not fictitious backend GEMM counts.

    96 first-pass reads + 96 paired-feature reads + 64 marginal reads +
    128 host/donor shuffle reads + 32 finite replay reads = 416. Every gain add,
    BF16 gate write, decomposition and M256 LN/head dispatch is explicit.
    Metadata/SHA IO and ordinary scalar/array bookkeeping stay in total wall.
    """
    return {"head_pack_loads": 2, "case_tensor_reads": 416, "BF16_gain_adds": 416,
            "saved_rows_payload_validations": 416, "shared_head_SVD": 1,
            "projected_covariance_eigh": 1, "random_basis_QR": 1,
            "paired_J_action": 96, "paired_J_basis": 96,
            "train_J_adjoint_basis": 64, "marginal_cross_moment": 64,
            "step_conditioned_marginal_cross_moment": 64,
            "source_shuffled_J_action": 64, "candidate_loss_evaluations": 4608,
            "radial_component_candidate_evaluations": 1728,
            "marginal_candidate_scores": 90, "BF16_gate_mul_then_add": 416,
            "BF16_Y_updates": 384, "M256_layer_norm_calls": 448,
            "BF16_head_mod_gain_adds": 448, "BF16_head_mod_mul_then_add": 448,
            "M256_head_calls": 448, "M256_baseline_repeat_checks": 32,
            "finite_real_and_linear": 384, "numeric_artifact_writes": 1,
            "full_M_head_calls": 0, "model_calls": 0, "GPU_calls": 0,
            "decoder_calls": 0, "student_training_updates": 0}


class WorkLedger:
    """Predebit before dispatch; failed invocations cannot become completions."""
    def __init__(self, record, persist=lambda: None):
        self.record, self.persist = record, persist
        self.limits = exact_budget()
        for key in ("predebited", "invoked", "completed"):
            record[key] = dict.fromkeys(self.limits, 0)

    def __call__(self, key, count=1):
        from contextlib import contextmanager
        @contextmanager
        def operation():
            check(key in self.limits and type(count) is int and count > 0, "Invalid work debit")
            check(self.record["predebited"][key] + count <= self.limits[key], "Fixed numerical budget exceeded: " + key)
            self.record["predebited"][key] += count
            self.persist()
            self.record["invoked"][key] += count
            self.record["last_invoked"] = key
            self.persist()
            yield
            self.record["completed"][key] += count
            self.persist()
        return operation()

    def finish(self):
        check(all(self.record[key] == self.limits for key in ("predebited", "invoked", "completed")), "Incomplete exact numerical budget")


def validate_plan(plan, *, frozen=False):
    check(plan.get("id") == ID and plan.get("parameters") == PARAMETERS and plan.get("resources") == RESOURCES and plan.get("limits") == exact_budget(), "Changed mathematical scope or budget")
    check(plan.get("source_split") == {"training": list(range(16)), "diagnostic_heldout": list(range(16, 24))}, "Changed source split")
    check(plan.get("parent_protocol", {}).get("sha256") == PARENT_SHA, "Only frozen CAL24 public _v1 is accepted")
    if frozen:
        check(plan.get("assets_finalized") is True and plan.get("binding_status") == "BOUND_READY_FOR_REGISTRATION", "Numerical assets NOT_BOUND")
        check(isinstance(plan.get("repo_sources"), list) and len(plan["repo_sources"]) >= 3, "Missing immutable code bindings")
        check({"tools/b_terminal_paired_discrimination.py", "tools/b_cal24_label_contract.py", "tests/test_b_terminal_paired_discrimination.py"} <= {x["path"] for x in plan["repo_sources"]}, "Missing required executable/test bindings")


def verify_manifest(manifest, plan, hf_root):
    """Consume the separate byte audit; preserve its exact 96-case provenance."""
    import b_cal24_label_contract as contract
    check(manifest.get("schema") == SCHEMA and manifest.get("status") == "AUDITED_COMPLETE_96_CASES" and manifest.get("parent_protocol") == plan["parent_protocol"], "Completed audited CAL24 manifest differs")
    parent_path = bound(ROOT, plan["parent_protocol"])
    parent = json.loads(parent_path.read_text())
    expected = {c["case_id"]: c for c in contract.build_cases(parent, hf_root)}
    check(manifest.get("source_split") == parent["source_split"] and manifest.get("source_exposure") == parent["source_exposure"], "Source exposure/split changed")
    parts = manifest.get("parts", [])
    check(len(parts) == 4 and [x.get("part") for x in parts] == list(range(4)), "Need four ordered completed parts")
    for part in parts:
        run_path = bound(hf_root, part["RUN"])
        receipt = json.loads(bound(hf_root, part["exit_receipt"]).read_text())
        run = json.loads(run_path.read_text())
        check(run.get("status") == "COMPLETE_CAL24_TERMINAL_LABEL_PART_POST_AND_OPERAND_BYTES_MATCH" and run.get("part") == part["part"] and run.get("protocol_sha256") == PARENT_SHA, "Producer part did not complete")
        check(receipt.get("child_returncode") == 0 and receipt.get("process_exited") is True and receipt.get("timeout") is False and receipt.get("automatic_retry") is False and receipt.get("error_type") is None and receipt.get("run_sha256") == part["RUN"]["sha256"] and receipt.get("protocol_sha256") == PARENT_SHA, "Producer actual exit/RUN differs")
    rows = manifest.get("cases", [])
    check(len(rows) == 96 and len({x["case"]["case_id"] for x in rows}) == 96, "Need 96 distinct audited cases")
    for item in rows:
        case = item["case"]
        check(expected.get(case["case_id"]) == case, "Case source/state/row identity changed")
        check(json.loads(bound(hf_root, item["case_record"]).read_text())["case"] == case, "CASE.json identity changed")
        label_path = bound(hf_root, item["label"])
        contract.validate_label_header(label_path, case)
        bound(hf_root, {"path": case["operand_path"], "bytes": case["operand_bytes"], "sha256": case["operand_sha256"]})
        check(item["evidence"]["actual_post_sha256"] == case["historical_post_sha256"], "Historical post SHA differs")
    heads = manifest.get("head_assets", [])
    check(len(heads) == 2 and {h.get("expert") for h in heads} == {"high_noise_model", "low_noise_model"}, "Two original expert heads required")
    for head in heads:
        bound(hf_root, head["binding"])
    return rows, heads


def preflight(protocol, hf_root):
    """Independent binding-only process: no numeric package imports or fitting."""
    import importlib.metadata
    check("torch" not in sys.modules and "numpy" not in sys.modules, "Preflight must be an independent byte-only process")
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide every CUDA device before CPU preflight")
    plan = json.loads(Path(protocol).read_text())
    validate_plan(plan, frozen=True)
    for binding in plan["repo_sources"]:
        bound(ROOT, binding)
    actual_versions = {k: importlib.metadata.version(k) for k in ("numpy", "torch", "safetensors")}
    check(actual_versions == plan["runtime_versions"], "Numerical package versions differ")
    manifest_path = bound(hf_root, plan["label_manifest"])
    manifest = json.loads(manifest_path.read_text())
    cases, heads = verify_manifest(manifest, plan, hf_root)
    result = {"id": ID, "status": "PASS_B_TERMINAL_PAIRED_CPU_BINDINGS_NO_NUMPY_NO_TORCH_NO_GPU",
              "protocol_sha256": sha(protocol), "repo_sources_sha256": state_sha(plan["repo_sources"]),
              "label_manifest_sha256": plan["label_manifest"]["sha256"], "cases": len(cases), "head_assets": len(heads),
              "source_split": plan["source_split"], "runtime_versions": actual_versions,
              "NumPy_imported": "numpy" in sys.modules, "Torch_or_CUDA_imported": "torch" in sys.modules or "torch.cuda" in sys.modules,
              "all_CUDA_devices_hidden": True, "decompositions": 0, "numeric_fits": 0, "model_calls": 0, "GPU_calls": 0}
    return plan, manifest, result


def authorize(plan, protocol_sha):
    active = json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())
    user = active.get("user_continuation_authorization_20261027", {})
    check(user.get("id") == "USER_RESUME_OFFLINE_IFC_20261027" and user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True and user.get("max_simultaneously_assigned_idle_GPUs") == 2, "Current offline session/resource authorization differs")
    entries = [x for x in active["experiment_authority"]["current_user_authorized_scopes"] if x.get("id") == ID and x.get("protocol_sha256") == protocol_sha and x.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")]
    check(len(entries) == 1, "Exactly one frozen registered numerical scope required")
    entry = entries[0]
    check(entry.get("max_gpu_cards") == 0 and entry.get("limits") == exact_budget(), "Registered CPU scope/budget differs")
    return entry


def prior_preflight(entry, plan, digest, hf_root):
    """ACTIVE binds prior report+actual exit, avoiding a protocol self-hash cycle."""
    check(isinstance(entry.get("cpu_preflight"), dict), "Independent exited CPU preflight NOT_BOUND in ACTIVE")
    previous = entry["cpu_preflight"]
    report = json.loads(bound(hf_root, previous["binding"]).read_text())
    receipt = json.loads(bound(hf_root, previous["launch_exit"]).read_text())
    check(report.get("id") == ID and report.get("status") == "PASS_B_TERMINAL_PAIRED_CPU_BINDINGS_NO_NUMPY_NO_TORCH_NO_GPU" and report.get("protocol_sha256") == digest and report.get("repo_sources_sha256") == state_sha(plan["repo_sources"]) and report.get("label_manifest_sha256") == plan["label_manifest"]["sha256"] and report.get("NumPy_imported") is False and report.get("Torch_or_CUDA_imported") is False and report.get("all_CUDA_devices_hidden") is True, "Prior independent CPU preflight differs")
    check(receipt.get("id") == ID and receipt.get("protocol_sha256") == digest and receipt.get("launcher_returncode") == 0 and receipt.get("process_exited") is True and receipt.get("RUN_sha256") == previous["binding"]["sha256"], "Prior CPU preflight actual exit/RUN binding differs")
    return previous


class AssetBank:
    """Audit-bound CPU operands, with no model import or GPU/device discovery."""
    def __init__(self, manifest, hf_root, work):
        import numpy as np
        import torch
        from safetensors.torch import load_file
        self.np, self.torch, self.hf, self.work = np, torch, Path(hf_root).resolve(), work
        self.entries = {v["case"]["case_id"]: v for v in manifest["cases"]}
        self.heads = {}
        for item in manifest["head_assets"]:
            path = bound(self.hf, item["binding"])
            with work("head_pack_loads", 1):
                pack = load_file(str(path), device="cpu")
                self.heads[item["expert"]] = {"weight_KN": self.array(pack["head_weight_runtime_KN"]),
                    "BF16_weight_KN": pack["head_weight_runtime_KN"], "BF16_bias": pack["head_bias"]}

    def array(self, tensor):
        return tensor.to(dtype=self.torch.float64).numpy()

    def __call__(self, case, finite=False):
        from safetensors import safe_open
        item, torch = self.entries[case["case_id"]], self.torch
        check(case == item["case"], "Numerical source/state/row identity differs")
        with self.work("case_tensor_reads", 1):
            label = bound(self.hf, item["label"])
            operand = bound(self.hf, {"path": case["operand_path"], "bytes": case["operand_bytes"], "sha256": case["operand_sha256"]})
            names = ["u_rows", "gate", "head_scale", "head_shift", "rows"] + (["r_rows", "head_rows"] if finite else [])
            with safe_open(str(label), framework="pt", device="cpu") as handle:
                tensors = {k: handle.get_tensor(k).contiguous() for k in names}
            with safe_open(str(operand), framework="pt", device="cpu") as handle:
                tensors["Y"] = handle.get_tensor("Y").contiguous()
            check(all(t.device.type == "cpu" for t in tensors.values()), "CPU-only tensors required")
            with self.work("saved_rows_payload_validations", 1):
                check(tensors["rows"].dtype == torch.int64 and tensors["rows"].tolist() == case["rows"], "Actual saved row payload differs")
            with self.work("BF16_gain_adds", 1):
                gain = 1 + tensors["head_scale"].reshape(-1)
            result = {"Y": self.array(tensors["Y"]), "u": self.array(tensors["u_rows"]),
                      "gate": self.array(tensors["gate"]).reshape(-1), "gain": self.array(gain),
                      "scale": self.array(tensors["head_scale"]).reshape(-1), "shift": self.array(tensors["head_shift"]).reshape(-1)}
            check(all(self.np.isfinite(v).all() for v in result.values()), "Nonfinite actual numerical operands")
            if finite:
                result["BF16"] = tensors
            return result


def same_bits(torch, a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def native_M256_head(torch, u, weight, bias, scale, shift, epsilon, work=lambda *_: nullcontext()):
    """Explicit CPU BF16 M256 LN/mod/addmm; no full-M or native-neutral claim."""
    check(u.device.type == weight.device.type == "cpu" and u.ndim == 2 and u.dtype == weight.dtype == torch.bfloat16, "CPU BF16 head operands required")
    with work("M256_layer_norm_calls", 1):
        normalized = torch.nn.functional.layer_norm(u, (u.shape[1],), None, None, epsilon)
    with work("BF16_head_mod_gain_adds", 1):
        gain = 1 + scale.reshape(-1)
    with work("BF16_head_mod_mul_then_add", 1):
        normalized.mul_(gain).add_(shift.reshape(-1))
    output = torch.empty((u.shape[0], weight.shape[1]), dtype=u.dtype, device="cpu")
    with work("M256_head_calls", 1):
        torch.addmm(bias, normalized, weight, out=output)
    return output


def bf16_projected_write(torch, r, y, gate, mu, basis, amplitude, work=lambda *_: nullcontext()):
    """fl_BF16(r + fl_BF16(g * fl_BF16(y + FP32 delta_y)))."""
    import numpy as np
    y64 = y.to(dtype=torch.float64).numpy()
    d = y64 - mu
    residual = d - (d @ basis) @ basis.T
    delta = torch.from_numpy(np.asarray(-amplitude * residual, dtype=np.float32))
    with work("BF16_Y_updates", 1):
        changed_y = (y.float() + delta).to(dtype=torch.bfloat16)
    with work("BF16_gate_mul_then_add", 1):
        updated = r.clone()
        updated.add_(changed_y * gate)
    return updated


def finite_replay(cases, bank, state, parameters, work=lambda *_: nullcontext()):
    import numpy as np
    torch = bank.torch
    records = []
    for case in sorted(cases, key=lambda c: (c["source_index"], c["step"])):
        if case["source_index"] < 16:
            continue
        v, head = bank(case, finite=True), bank.heads[case["expert"]]
        pack = v["BF16"]
        # Restoring runtime KN strides on CPU retains layout, but still does not
        # turn a 256-row addmm into an original M31668 native call.
        stride = bank.entries[case["case_id"]]["evidence"]["head_weight_stride"]
        weight = torch.empty_strided(head["BF16_weight_KN"].shape, stride, dtype=torch.bfloat16, device="cpu")
        weight.copy_(head["BF16_weight_KN"])
        with work("BF16_gate_mul_then_add", 1):
            replayed_u = pack["r_rows"].clone()
            replayed_u.add_(pack["Y"] * pack["gate"])
        check(same_bits(torch, replayed_u, pack["u_rows"]), "Actual CPU gate multiply/add no longer recovers native u rows")
        call = lambda u: native_M256_head(torch, u, weight, head["BF16_bias"], pack["head_scale"], pack["head_shift"], parameters["epsilon"], work)
        baseline, repeat = call(pack["u_rows"]), call(pack["u_rows"])
        with work("M256_baseline_repeat_checks", 1):
            check(same_bits(torch, baseline, repeat), "M256 repeated neutral call unstable; no retry")
        full_native = pack["head_rows"]
        baseline_shape_difference = bank.array(baseline) - bank.array(full_native)
        records.append({"case_id": case["case_id"], "source_index": case["source_index"], "step": case["step"],
            "kind": "M256_CPU_BASELINE", "M": case.get("probe_rows", 256),
            "repeated_same_shape_bitwise_equal": True,
            "captured_full_M_selected_rows_bitwise_equal": same_bits(torch, baseline, full_native),
            "captured_full_M_selected_rows_difference_SSE": float(np.square(baseline_shape_difference).sum()),
            "scope": "CPU M256; full-M native-neutral gate NOT_RUN and not inferred from equality"})
        T = state["refs"][case["step"]]["T"]
        for arm in ARMS:
            candidate = state["candidates"][state["selected_indices"][arm]]
            basis = projection_basis(state["W"], candidate["B"])
            for amplitude in parameters["finite_amplitudes"]:
                changed_u = bf16_projected_write(torch, pack["r_rows"], pack["Y"], pack["gate"], state["mu"], basis, amplitude, work)
                output = call(changed_u)
                actual = bank.array(output) - bank.array(baseline)
                effective = bank.array(changed_u) - v["u"]
                with work("finite_real_and_linear", 1):
                    finite, linear = finite_formula(v["u"], effective, T, parameters["epsilon"])
                records.append({"case_id": case["case_id"], "source_index": case["source_index"], "step": case["step"], "kind": "M256_CPU_BF16_FINITE",
                    "arm": arm, "candidate": candidate["id"], "amplitude": amplitude,
                    "amplitude_definition": "fraction of pre-gate projection residual removed, not fixed L2-relative delta-u",
                    "effective_delta_u_SSE": float(np.square(effective).sum()),
                    "effective_delta_u_zero_fraction": float((effective == 0).mean()),
                    "head_response_mean_row_SSE": float(np.square(actual).sum(axis=1).mean()),
                    "real_finite_mean_row_SSE": float(np.square(finite).sum(axis=1).mean()),
                    "real_linear_mean_row_SSE": float(np.square(linear).sum(axis=1).mean()),
                    "actual_vs_finite_mean_row_SSE": float(np.square(actual - finite).sum(axis=1).mean()),
                    "actual_vs_linear_mean_row_SSE": float(np.square(actual - linear).sum(axis=1).mean())})
    summary = {}
    for amplitude in parameters["finite_amplitudes"]:
        for arm in ARMS:
            selected = [r for r in records if r.get("arm") == arm and r.get("amplitude") == amplitude]
            by_source = {i: [r for r in selected if r["source_index"] == i] for i in range(16, 24)}
            check(all(len(values) == 4 for values in by_source.values()), "Finite replay missing nested source/step")
            source_means = {str(i): sum(r["head_response_mean_row_SSE"] for r in values) / 4 for i, values in by_source.items()}
            summary[f"{arm}@{amplitude}"] = {"source_means": source_means, "equal_source_mean": sum(source_means.values()) / 8,
                "independent_exposed_sources": 8, "tokens_are_not_independent_N": True}
    return {"records": records, "heldout_summary": summary, "full_M_native_noop": "NOT_RUN_SEPARATE_SCOPE", "video_quality": "NOT_MEASURED"}


def run(args):
    protocol, hf = args.protocol.resolve(), args.hf_root.resolve()
    plan, manifest, verification = preflight(protocol, hf)
    digest = sha(protocol)
    entry = authorize(plan, digest)
    cpu_receipt = prior_preflight(entry, plan, digest, hf)
    out = (hf / plan["output_HF_relative"]).resolve()
    check(out != hf and out.is_relative_to(hf) and not out.exists(), "Fresh HF result directory required; no retries")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"id": ID, "protocol_sha256": digest, "status": "RUNNING_FIXED_CPU_M256_SCOPE",
              "binding_preflight": verification, "prior_independently_exited_preflight": cpu_receipt,
              "resources": RESOURCES, "declared_budget": exact_budget(), "native_full_M_calls": 0}
    persist = lambda: write(out / "RUN.json", record)
    ledger = WorkLedger(record, persist)
    persist()
    try:
        # Environment is set before importing numeric libraries in this worker.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(PARAMETERS["CPU_threads"])
        import numpy as np
        import torch
        torch.set_num_threads(PARAMETERS["CPU_threads"])
        torch.set_num_interop_threads(1)
        record["numerical_versions"] = {"numpy": np.__version__, "torch": torch.__version__}
        with torch.inference_mode():
            bank = AssetBank(manifest, hf, ledger)
            cases = [item["case"] for item in manifest["cases"]]
            analysis, state = analyze(cases, bank, bank.heads, PARAMETERS, ledger)
            analysis_path = out / "ANALYSIS.json"
            write(analysis_path, analysis)
            record["analysis"] = {"root": "hf", "path": str(analysis_path.relative_to(hf)), "bytes": analysis_path.stat().st_size,
                "sha256": sha(analysis_path), "selection": analysis["selection"], "selected_heldout": analysis["paired_evaluation"]["selected_heldout"]}
            persist()
            record["finite_M256"] = finite_replay(cases, bank, state, PARAMETERS, ledger)
            arrays = {"mu": state["mu"], "W": state["W"],
                      "candidate_B": np.stack([c["B"] for c in state["candidates"]]),
                      "selected_indices": np.asarray([state["selected_indices"][arm] for arm in ARMS], dtype=np.int64)}
            path = out / "projection_dictionary.npz"
            with ledger("numeric_artifact_writes", 1):
                with path.open("xb") as stream:
                    np.savez(stream, **arrays)
            record["projection_asset"] = {"root": "hf", "path": str(path.relative_to(hf)), "bytes": path.stat().st_size, "sha256": sha(path),
                "tensor_raw_sha256": {k: hashlib.sha256(v.tobytes(order="C")).hexdigest() for k, v in arrays.items()},
                "candidate_order": [c["id"] for c in state["candidates"]], "arm_order": list(ARMS),
                "future_replay_identity": "pre-gate y' = BF16(y + FP32(-amplitude*(I-WBB'W')*(y-mu))); r/g from same case, separate full-M scope required"}
            write(out / "source_state_row_map.json", {"parent_manifest": plan["label_manifest"], "cases": cases})
            ledger.finish()
            record["status"] = "COMPLETE_B_TERMINAL_PAIRED_FINITE_DICTIONARY_M256_CPU"
    except BaseException as error:
        record.update(status="FAILED_FIXED_CPU_SCOPE_NO_AUTOMATIC_RETRY", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        persist()
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if size > RESOURCES["max_output_bytes"]:
            record["status"] = "FAILED_OUTPUT_BYTES_EXCEEDED_NO_AUTOMATIC_RETRY"
            record["actual_output_bytes_before_final_status"] = size
            persist()
            raise RuntimeError("Registered output size exceeded")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "preflight", "execute"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        plan = json.loads(args.protocol.read_text())
        validate_plan(plan)
        print(json.dumps({"id": ID, "binding_status": plan.get("binding_status"), "assets_finalized": plan.get("assets_finalized"), "limits": exact_budget(), "GPU_calls": 0, "numerical_calls": 0}, indent=2))
    elif args.command == "preflight":
        check(args.hf_root is not None and args.out is not None, "Preflight requires --hf-root and fresh --out")
        check(not args.out.exists(), "Fresh independent CPU report required")
        _, _, report = preflight(args.protocol, args.hf_root)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        check(args.hf_root is not None, "Execute requires --hf-root")
        run(args)


if __name__ == "__main__":
    main()
