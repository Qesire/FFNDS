"""Fit a fixed residual mean and scalar gain, excluding original FFN bias."""
import numpy as np


def fit_signal_moments(weight, sum_signal, sum_dense, sum_gated_signal2,
                       sum_gated_signal_dense, gate):
    mu_s = np.asarray(sum_signal, np.float64) / weight
    mu_y = np.asarray(sum_dense, np.float64) / weight
    gate = np.asarray(gate, np.float64).reshape(-1)
    if weight <= 0 or mu_s.shape != mu_y.shape or mu_s.shape != gate.shape:
        raise ValueError("Invalid signal moment geometry")
    if any(not np.isfinite(x).all() for x in [mu_s, mu_y, gate, sum_gated_signal2, sum_gated_signal_dense]):
        raise ValueError("Nonfinite signal moments")
    denominator = float(sum_gated_signal2)
    numerator = float(sum_gated_signal_dense)
    centered_den = denominator - weight * float(np.sum((mu_s * gate) ** 2))
    centered_num = numerator - weight * float(np.sum(mu_s * mu_y * gate ** 2))
    floor = len(gate) * 1e-24
    uncentered = max(0., numerator / denominator) if denominator > floor else None
    centered = max(0., centered_num / centered_den) if centered_den > floor else None
    return {"uncentered_alpha": uncentered, "centered_alpha": centered,
            "uncentered_numerator": numerator, "uncentered_denominator": denominator,
            "centered_numerator": centered_num, "centered_denominator": centered_den,
            "mean_signal": mu_s, "mean_dense": mu_y}
