#!/usr/bin/env python3
"""B-H residual-anchored direct65 student; no teacher Y in the predictor.

Import is stdlib-only. The preparation-stage tests use tiny synthetic arrays.
Real geometry compilation, saved-data training and GPU work require a separately
frozen/registered pilot. The student replaces a terminal FFN+write+LN/head graph;
it does not produce a D-wide y/u tensor or establish video/runtime quality.
"""
from contextlib import nullcontext
import hashlib
import json
import math

PARAMETERS = {"D": 5120, "head_outputs": 64, "hidden": 128, "rows_per_case": 256,
              "epsilon": 1e-6, "slack_floor": 1e-6 * 2 ** -16,
              "geometry_rank_rtol": 1e-10, "max_geometry_condition": 1e6,
              "FP64_slack_rtol": 1e-9, "FP32_slack_rtol": 2e-5,
              "normalization_floor": 1e-6, "log_slack_scale_floor": 1e-3,
              "max_abs_log_slack": 30.0, "epochs": 8, "seed": 42,
              "Adam_lr": .0005, "Adam_betas": [.9, .999], "Adam_epsilon": 1e-8}
TRAIN = tuple(range(16))
HELD = tuple(range(16, 24))
EXPERTS = ("high_noise_model", "low_noise_model")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def steps_for(expert):
    check(expert in EXPERTS, "Unknown expert")
    return (0, 1) if expert == EXPERTS[0] else (2, 3)


def schedule(expert, epochs=8):
    """Exactly A0's fixed source-then-step order, no validation-driven choice."""
    return [(epoch, source, step) for epoch in range(epochs) for source in TRAIN for step in steps_for(expert)]


def parameter_count(D=5120, outputs=64, hidden=128):
    # Wx, Weta, Wout, two hidden step biases, two output step biases.
    return hidden * D + hidden * (outputs + 1) + (outputs + 1) * hidden + 2 * hidden + 2 * (outputs + 1)


def logical_cost(D=5120, outputs=64, hidden=128, M=31668):
    """Fixed implementation always applies both range projections, even rank-full.

    Four small GEMMs: q_r range projection, q_r G+, q_hat range projection,
    q_hat G+. This explicit cost avoids assuming actual head full-row rank.
    Reductions, FP32 casts, normalizations, exp/rsqrt, BF16 rounding and buffers
    are recorded separately and are not included in these logical MAC counts.
    """
    matrices = {"residual_head_projection": D * outputs,
                "student_Wx": D * hidden, "student_Weta": (outputs + 1) * hidden,
                "student_Wout": hidden * (outputs + 1),
                "two_range_and_two_quadratic_GEMM": 4 * outputs * outputs}
    total = sum(matrices.values())
    return {"parameters_per_expert": parameter_count(D, outputs, hidden), "logical_MAC_per_row": total,
            "logical_MAC_full_M": total * M, "logical_matrix_MAC_per_row": matrices,
            "online_GEMM_calls": 8, "BF16_student_GEMM_calls": 3,
            "FP32_residual_GEMM_calls": 1, "FP32_small_GEMM_calls": 4,
            "original_Smooth1_Q1": "retained upstream cost; input here is actual post-Q1 V1",
            "student_hidden_Q2": "NONE_IN_THIS_BF16_PREDICTABILITY_PILOT",
            "D_wide_y_or_u_materialization": False,
            "online_extra_buffers_per_row": {"FP32_residual_cast": 4 * D, "BF16_hidden": 2 * hidden,
                                              "BF16_normalized_V1": 2 * D, "FP32_normalized_V1_temporary": 4 * D},
            "runtime_speed": "NOT_MEASURED"}


def compile_geometry(weight_KN, actual_BF16_gain, shift, bias, p=PARAMETERS, work=lambda *_: nullcontext()):
    """Paid FP64 head compilation, one Gram and symmetric eigendecomposition.

    The gain must already be fl_BF16(1+actual scale), not real 1+scale.
    A rank-deficient K gets its actual range projector; no invented inverse.
    """
    import numpy as np
    w, gain, shift, bias = [np.asarray(x, dtype=np.float64) for x in (weight_KN, actual_BF16_gain, shift, bias)]
    D, h = p["D"], p["head_outputs"]
    check(w.shape == (D, h) and gain.shape == shift.shape == (D,) and bias.shape == (h,), "Head geometry shape differs")
    check(all(np.isfinite(x).all() for x in (w, gain, shift, bias)), "Nonfinite head geometry")
    T = w.T * gain[None, :]
    K = T - T.mean(axis=1, keepdims=True)
    with work("geometry_FP64_Gram_GEMM", 1):
        G = K @ K.T
    with work("geometry_symmetric_eigh", 1):
        eigenvalues, vectors = np.linalg.eigh((G + G.T) / 2)
    maximum = float(eigenvalues[-1])
    check(maximum > 0 and float(eigenvalues[0]) >= -p["geometry_rank_rtol"] * maximum, "Head Gram is zero or materially indefinite")
    selected = eigenvalues > p["geometry_rank_rtol"] * maximum
    rank = int(selected.sum())
    check(D > rank + 1, "No centered kernel: positive arbitrary slack is not attainable")
    condition = maximum / float(eigenvalues[selected][0])
    check(condition <= p["max_geometry_condition"], "Head Gram condition exceeds fixed FP32 pilot gate; no retry")
    U = vectors[:, selected]
    with work("geometry_FP64_inverse_and_projector_GEMM", 2):
        inverse = (U / eigenvalues[selected][None, :]) @ U.T
        projector = U @ U.T
    with work("geometry_FP64_shift_GEMM", 1):
        c0 = shift @ w + bias
    with work("geometry_FP64_projector_validation_GEMM", 1):
        idempotence_error = float(np.abs(projector @ projector - projector).max())
    record = {"rank": rank, "condition_on_support": condition, "eigenvalues": eigenvalues.tolist(),
              "rank_cutoff": p["geometry_rank_rtol"] * maximum,
              "K_centering_max_abs_sum": float(np.abs(K.sum(axis=1)).max()),
              "projector_idempotence_max_abs": idempotence_error,
              "gain_identity": "actual fl_BF16(1+scale) promoted to FP64",
              "evidence": "continuous real surrogate anchored to original BF16 assets; not native bit equivalence"}
    return {"K": K, "Gplus": inverse, "projector": projector, "c0": c0, "D": D, "rank": rank, "record": record}


def copy_to_device(torch, tensor, device, work=lambda *_: nullcontext()):
    if tensor.device.type == "cpu" and str(device).startswith("cuda"):
        with work("CUDA_H2D_tensor_copies", 1), work("CUDA_H2D_payload_bytes", tensor.numel() * tensor.element_size()):
            return tensor.to(device)
    return tensor.to(device)


def device_geometry(geometry, torch, device, work=lambda *_: nullcontext()):
    # Cast on CPU first so the accounted H2D payload has an unambiguous dtype.
    return {**{k: copy_to_device(torch, torch.as_tensor(geometry[k], dtype=torch.float32, device="cpu"), device, work) for k in ("K", "Gplus", "projector", "c0")},
            "D": geometry["D"], "rank": geometry["rank"]}


def checked_nonnegative(torch, raw, reference, relative_tolerance, epsilon, label):
    """Explicit tolerance repair with diagnostics; substantial negatives fail."""
    check(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(reference).all()), "Nonfinite " + label)
    tolerance = relative_tolerance * torch.maximum(reference.abs(), torch.full_like(reference, epsilon))
    check(bool((raw >= -tolerance).all()), label + " materially negative; no silent clipping or retry")
    diagnostic = {"negative_rows_repaired": int((raw < 0).sum().item()),
                  "minimum_raw": float(raw.min().item()), "max_allowed_absolute_repair": float(tolerance.max().item())}
    return raw.clamp_min(0), diagnostic


def anchor_features(torch, residual, geometry, p=PARAMETERS, work=lambda *_: nullcontext()):
    """Legal, paid r -> q_r, v_r, slack and 65-element online summary."""
    check(residual.ndim == 2 and residual.shape[1] == p["D"] and residual.dtype == torch.bfloat16, "Original BF16 residual required")
    with work("online_FP32_residual_projection_GEMM", 1):
        r32 = residual.float()
        qr = r32 @ geometry["K"].T
    with work("online_FP32_range_projection_GEMM", 1):
        qr = qr @ geometry["projector"].T
    with work("online_residual_variance_reductions", 1):
        centered = r32 - r32.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1)
    with work("online_FP32_quadratic_GEMM", 1):
        qmin_raw = ((qr @ geometry["Gplus"]) * qr).sum(dim=1) / p["D"]
    qmin, qmin_diagnostic = checked_nonnegative(torch, qmin_raw, variance, p["FP32_slack_rtol"], p["epsilon"], "residual minimum variance")
    slack, slack_diagnostic = checked_nonnegative(torch, variance - qmin, variance, p["FP32_slack_rtol"], p["epsilon"], "residual slack")
    with work("online_anchor_summary", 1):
        eta = torch.cat((qr * torch.rsqrt(variance[:, None] + p["epsilon"]), torch.log(variance[:, None] + p["epsilon"])), dim=1)
    check(bool(torch.isfinite(eta).all()), "Nonfinite legal residual summary")
    return {"qr": qr, "variance": variance, "slack": slack, "eta": eta,
            "qmin_diagnostic": qmin_diagnostic, "slack_diagnostic": slack_diagnostic,
            "floor_rows": int((slack < p["slack_floor"]).sum().item())}


def truth_labels(u, online_anchor, geometry, p=PARAMETERS, work=lambda *_: nullcontext()):
    """Teacher u only in offline labels, relative to actual paid FP32 anchor.

    Floor/truncation affects only documented slack tolerance and fixed tau.
    No target magnitude clipping or heldout-fitted normalizer is used.
    """
    import numpy as np
    u = np.asarray(u, dtype=np.float64)
    check(u.ndim == 2 and u.shape[1] == p["D"] and np.isfinite(u).all(), "Offline native u label differs")
    with work("offline_FP64_teacher_head_projection_GEMM", 1):
        qu = u @ geometry["K"].T
    with work("offline_FP64_teacher_quadratic_GEMM", 1):
        qmin = ((qu @ geometry["Gplus"]) * qu).sum(axis=1) / p["D"]
    z = u - u.mean(axis=1, keepdims=True)
    variance = np.square(z).mean(axis=1)
    slack_raw = variance - qmin
    tolerance = p["FP64_slack_rtol"] * np.maximum(np.abs(variance), p["epsilon"])
    check(np.isfinite(qmin).all() and np.all(qmin >= -tolerance) and np.all(slack_raw >= -tolerance), "FP64 label feasibility failed; no hidden clipping")
    slack = np.maximum(slack_raw, 0)
    qr = np.asarray(online_anchor["qr"], dtype=np.float64)
    sr = np.asarray(online_anchor["slack"], dtype=np.float64)
    check(qr.shape == qu.shape and sr.shape == variance.shape and np.all(sr >= 0), "Operational anchor label shapes differ")
    log_ratio = np.log(np.maximum(slack, p["slack_floor"]) / np.maximum(sr, p["slack_floor"]))
    check(np.isfinite(log_ratio).all() and np.max(np.abs(log_ratio)) <= p["max_abs_log_slack"], "True log-slack label exceeds fixed numerical domain; no clipping")
    with work("offline_FP64_teacher_readout", 1):
        real_head = qu / np.sqrt(variance[:, None] + p["epsilon"]) + geometry["c0"]
    return {"target65": np.column_stack((qu - qr, log_ratio)), "q": qu, "variance": variance,
            "slack": slack, "real_head": real_head,
            "diagnostic": {"teacher_slack_floor_rows": int((slack < p["slack_floor"]).sum()),
                           "FP64_negative_slack_repaired_rows": int((slack_raw < 0).sum()),
                           "max_teacher_floor_variance_added": float(np.maximum(p["slack_floor"] - slack, 0).max()),
                           "label_anchor": "actual paid FP32 r-summary, not an unavailable ideal anchor",
                           "target_clipping": "NONE; fixed physical slack floor and bounded tolerance repair reported"}}


class Direct65Student:
    """FP32 Adam masters, actual BF16 linear/GELU forward, cast-gradient training.

    The class intentionally exposes only (postQ1 V1, legal residual summary,
    original local step) to predict65. It contains no teacher Y/u/head argument.
    """
    def __init__(self, D=5120, outputs=64, hidden=128, seed=42, device="cpu", work=lambda *_: nullcontext()):
        import torch
        self.torch, self.D, self.outputs, self.hidden, self.work = torch, D, outputs, hidden, work
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with work("student_random_weight_draws", 2):
            wx = torch.randn((hidden, D), generator=generator, dtype=torch.float32) / math.sqrt(D)
            we = torch.randn((hidden, outputs + 1), generator=generator, dtype=torch.float32) / math.sqrt(outputs + 1)
        values = {"Wx": wx, "Weta": we, "Wout": torch.zeros((outputs + 1, hidden), dtype=torch.float32),
                  "b_step": torch.zeros((2, hidden), dtype=torch.float32), "a_step": torch.zeros((2, outputs + 1), dtype=torch.float32)}
        self.weights = torch.nn.ParameterDict({k: torch.nn.Parameter(copy_to_device(torch, v, device, work)) for k, v in values.items()})

    def parameters(self):
        return self.weights.parameters()

    def state_dict(self):
        return self.weights.state_dict()

    def predict65(self, postQ1_V1, residual_summary, local_step, normalizer):
        torch, w = self.torch, self.weights
        check(type(local_step) is int and local_step in (0, 1), "Original expert-local step must be 0 or 1")
        check(postQ1_V1.ndim == residual_summary.ndim == 2 and postQ1_V1.shape[1] == self.D and residual_summary.shape == (postQ1_V1.shape[0], self.outputs + 1) and postQ1_V1.dtype == torch.bfloat16, "Legal V1/residual-summary shapes differ")
        with self.work("student_predict65_calls", 1):
            with self.work("student_input_normalizations", 1):
                x = (postQ1_V1.float() / normalizer["x_rms"]).to(torch.bfloat16)
                eta = ((residual_summary - normalizer["eta_mean"]) / normalizer["eta_std"]).to(torch.bfloat16)
            with self.work("student_BF16_forward_GEMM", 1):
                h = torch.nn.functional.linear(x, w["Wx"].to(torch.bfloat16))
            with self.work("student_BF16_forward_GEMM", 1):
                h = h + torch.nn.functional.linear(eta, w["Weta"].to(torch.bfloat16))
            h = h + w["b_step"][local_step].to(torch.bfloat16)
            with self.work("student_BF16_tanh_GELU", 1):
                h = torch.nn.functional.gelu(h, approximate="tanh")
            with self.work("student_BF16_forward_GEMM", 1):
                raw = torch.nn.functional.linear(h, w["Wout"].to(torch.bfloat16))
            raw = raw + w["a_step"][local_step].to(torch.bfloat16)
            check(raw.dtype == torch.bfloat16, "Student forward lost original BF16 arithmetic")
            return raw


def decode_statistics(torch, raw65, anchor, normalizer, geometry, p=PARAMETERS, work=lambda *_: nullcontext()):
    """Constrained direct65 -> FP32 q/v/head, with an explicit slack floor."""
    with work("student_direct_readout", 1):
        target = raw65.float() * normalizer["target_rms"]
        delta_q, ell = target[:, :p["head_outputs"]], target[:, p["head_outputs"]]
        check(bool(torch.isfinite(target).all()) and bool((ell.abs() <= p["max_abs_log_slack"]).all()), "Student log-slack outside fixed finite domain; no clamp/retry")
        with work("online_FP32_range_projection_GEMM", 1):
            q = (anchor["qr"] + delta_q) @ geometry["projector"].T
        with work("online_FP32_quadratic_GEMM", 1):
            qmin_raw = ((q @ geometry["Gplus"]) * q).sum(dim=1) / p["D"]
        qmin, diagnostic = checked_nonnegative(torch, qmin_raw, anchor["variance"], p["FP32_slack_rtol"], p["epsilon"], "predicted minimum variance")
        slack = anchor["slack"].clamp_min(p["slack_floor"]) * torch.exp(ell)
        variance = qmin + slack
        output = q * torch.rsqrt(variance[:, None] + p["epsilon"]) + geometry["c0"]
        check(bool(torch.isfinite(output).all()) and bool((variance >= qmin).all()), "Nonfinite or unattainable student head")
        return {"q": q, "variance": variance, "slack": slack, "head_FP32": output,
                "head_BF16": output.to(torch.bfloat16), "log_slack": ell, "qmin_diagnostic": diagnostic}


def normalizers(torch, training_cases, p=PARAMETERS, work=lambda *_: nullcontext()):
    """Train16 only; source/step/row weights equal and no validation refitting."""
    check(len(training_cases) == 32 and {c["source_index"] for c in training_cases} == set(TRAIN) and all(c["role"] == "TRAIN" for c in training_cases), "Only 16 train sources x2 states may normalize")
    check(len({(c["source_index"], c["step"]) for c in training_cases}) == 32 and len({c["step"] for c in training_cases}) == 2, "Training states repeat or mix experts")
    for case in training_cases:
        check(case["V1"].shape == (p["rows_per_case"], p["D"]) and case["target65"].shape == (p["rows_per_case"], p["head_outputs"] + 1), "Unequal calibration row geometry")
    eta_sum = target_square_sum = x_square_sum = None
    for case in training_cases:
        with work("train_normalization_mean_moment_cases", 1):
            e = case["anchor"]["eta"].double().mean(dim=0)
            t = case["target65"].double().square().mean(dim=0)
            x = case["V1"].double().square().mean()
            eta_sum = e if eta_sum is None else eta_sum + e
            target_square_sum = t if target_square_sum is None else target_square_sum + t
            x_square_sum = x if x_square_sum is None else x_square_sum + x
    eta_mean = eta_sum / 32
    eta_square = torch.zeros_like(eta_mean)
    for case in training_cases:
        with work("train_normalization_centered_moment_cases", 1):
            eta_square += (case["anchor"]["eta"].double() - eta_mean).square().mean(dim=0) / 32
    with work("normalizer_compiles", 1):
        target_rms = torch.sqrt(target_square_sum / 32).clamp_min(p["normalization_floor"])
        target_rms[-1] = target_rms[-1].clamp_min(p["log_slack_scale_floor"])
        return {"x_rms": torch.sqrt(x_square_sum / 32).float().clamp_min(p["normalization_floor"]),
                "eta_mean": eta_mean.float(), "eta_std": torch.sqrt(eta_square).float().clamp_min(p["normalization_floor"]),
                "target_rms": target_rms.float()}


def supervised_loss(torch, raw65, target65, normalizer):
    """Equal two terms: mean standardized delta-q MSE + log-slack MSE."""
    expected = target65 / normalizer["target_rms"]
    error = raw65.float() - expected
    q_loss, slack_loss = error[:, :-1].square().mean(), error[:, -1].square().mean()
    return q_loss + slack_loss, q_loss, slack_loss


def backward_with_accounting(torch, value, work=lambda *_: nullcontext()):
    """Observe actual BF16 mm dispatches, not a guessed backward count.

    Cached V1/eta are constants: Wx and Weta each need one weight-gradient mm;
    Wout needs weight-gradient and hidden-gradient mm. Biases use reductions.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    class BackwardMatmuls(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default):
                matrices = [t for t in args if isinstance(t, torch.Tensor) and t.ndim >= 2]
                check(func == torch.ops.aten.mm.default and len(matrices) == 2 and all(t.dtype == torch.bfloat16 for t in matrices), "Unexpected backward GEMM/dtype")
                with work("student_BF16_backward_GEMM", 1):
                    output = func(*args, **kwargs)
                self.calls.append({"operator": str(func), "input_shapes": [list(t.shape) for t in matrices], "dtype": "torch.bfloat16"})
                return output
            return func(*args, **kwargs)
    mode = BackwardMatmuls()
    with mode:
        value.backward()
    check(len(mode.calls) == 4, "Backward differed from fixed four BF16 matrix multiplications")
    return mode.calls


def training_loop(torch, model, cases_by_coord, normalizer, expert, p=PARAMETERS, work=lambda *_: nullcontext(), on_epoch=lambda *_: None):
    """256 fixed Adam updates/expert; only saved TRAIN cases can dispatch."""
    check(p["epochs"] == 8, "Fixed pilot has exactly eight epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=p["Adam_lr"], betas=tuple(p["Adam_betas"]),
                                 eps=p["Adam_epsilon"], weight_decay=0, foreach=False, fused=False)
    metrics, epoch_losses = [], []
    for epoch, source, step in schedule(expert, p["epochs"]):
        case = cases_by_coord[(source, step)]
        check(case["source_index"] in TRAIN and case["role"] == "TRAIN" and case["step"] == step, "Heldout or wrong state reached training")
        optimizer.zero_grad(set_to_none=True)
        with work("training_source_batches", 1):
            raw = model.predict65(case["V1"], case["anchor"]["eta"], step - steps_for(expert)[0], normalizer)
            with work("training_loss_evaluations", 1):
                value, q_loss, slack_loss = supervised_loss(torch, raw, case["target65"], normalizer)
                check(bool(torch.isfinite(value)), "Nonfinite training loss")
            with work("backward_calls", 1):
                backward_with_accounting(torch, value, work)
            for parameter in model.parameters():
                with work("finite_gradient_tensor_checks", 1):
                    check(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing/nonfinite parameter gradient")
            with work("Adam_steps", 1):
                with work("Adam_parameter_tensor_updates", 5):
                    optimizer.step()
            for parameter in model.parameters():
                with work("finite_updated_parameter_tensor_checks", 1):
                    check(bool(torch.isfinite(parameter).all()), "Nonfinite Adam parameter")
        epoch_losses.append(float(value.detach()))
        if len(epoch_losses) == 32:
            row = {"epoch": epoch + 1, "mean_source_step_standardized_loss": sum(epoch_losses) / 32,
                   "last_q_loss": float(q_loss.detach()), "last_log_slack_loss": float(slack_loss.detach())}
            metrics.append(row)
            on_epoch(row)
            epoch_losses = []
    check(len(metrics) == 8 and not epoch_losses, "Incomplete fixed optimizer schedule")
    return metrics
