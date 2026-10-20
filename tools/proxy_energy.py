"""Donor-only scalar-energy correction for a matrix of group scores."""
import numpy as np


def fit_energy_residual(raw_scores, true_scores, energy, weights):
    raw, target = np.asarray(raw_scores, np.float64), np.asarray(true_scores, np.float64)
    energy, weights = np.asarray(energy, np.float64), np.asarray(weights, np.float64)
    if raw.ndim != 2 or target.shape != raw.shape or energy.shape != (len(raw),) or weights.shape != energy.shape:
        raise ValueError("Training score/energy/weight shapes differ")
    if any(not np.isfinite(x).all() for x in [raw, target, energy, weights]) or np.any(energy <= 0) or np.any(weights <= 0):
        raise ValueError("Finite training data and positive energy/weights required")
    mean = float(weights @ energy / weights.sum())
    e = energy / mean
    beta = (weights * e) @ (target - raw) / float(weights @ (e * e))
    return {"beta": beta, "mean_energy": mean}


def corrected_scores(raw_scores, energy, model):
    raw, energy = np.asarray(raw_scores, np.float64), np.asarray(energy, np.float64)
    if raw.ndim != 2 or energy.shape != (len(raw),) or model["beta"].shape != (raw.shape[1],):
        raise ValueError("Prediction geometry differs from donor model")
    result = np.maximum(raw + (energy / model["mean_energy"])[:, None] * model["beta"], 0)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predicted group scores")
    return result
