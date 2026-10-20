"""Training-only centered kernel ridge for a pre-FFN0 support proxy."""
import numpy as np


def raw_kernel(aa, bb, kind):
    dot = (aa @ bb.T) / aa.shape[1]
    return dot if kind == "linear" else (1 + dot) ** 2


def prepare_fit(features, labels, kind):
    xx, yy = np.asarray(features, np.float64), np.asarray(labels, np.float64)
    if xx.ndim != 2 or yy.ndim != 2 or len(xx) != len(yy) or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("Finite aligned training features/labels required")
    mean_x, mean_y = xx.mean(0), yy.mean(0)
    if kind == "static":
        return {"kind": kind, "mean_y": mean_y}
    if kind not in ["linear", "poly2"]:
        raise ValueError("Unknown frozen kernel family")
    scale_x = max(float(np.sqrt(np.mean((xx - mean_x) ** 2))), 1e-12)
    train = (xx - mean_x) / scale_x
    kernel = raw_kernel(train, train, kind)
    kernel_scale = max(float(np.diag(kernel).mean()), 1e-12)
    column_mean, grand_mean = kernel.mean(0), float(kernel.mean())
    centered = (kernel - column_mean[None, :] - column_mean[:, None] + grand_mean) / kernel_scale
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    return {"kind": kind, "mean_x": mean_x, "mean_y": mean_y, "scale_x": scale_x,
            "train": train, "kernel_scale": kernel_scale, "column_mean": column_mean,
            "grand_mean": grand_mean, "eigenvalues": np.maximum(eigenvalues, 0),
            "eigenvectors": eigenvectors, "rotated_y": eigenvectors.T @ (yy - mean_y)}


def fit_prepared(prepared, regularization):
    model = {key: value for key, value in prepared.items() if key not in ["eigenvalues", "eigenvectors", "rotated_y"]}
    if model["kind"] != "static":
        if regularization <= 0:
            raise ValueError("Ridge regularization must be positive")
        model["coefficients"] = prepared["eigenvectors"] @ (
            prepared["rotated_y"] / (prepared["eigenvalues"][:, None] + len(prepared["train"]) * regularization))
    model["regularization"] = regularization
    return model


def predict(model, features):
    xx = np.asarray(features, np.float64)
    if xx.ndim != 2 or not np.isfinite(xx).all():
        raise ValueError("Predictor receives finite known-input features only")
    if model["kind"] == "static":
        return np.broadcast_to(model["mean_y"], (len(xx), len(model["mean_y"]))).copy()
    kernel = raw_kernel((xx - model["mean_x"]) / model["scale_x"], model["train"], model["kind"])
    centered = (kernel - model["column_mean"][None, :] - kernel.mean(1)[:, None] + model["grand_mean"]) / model["kernel_scale"]
    return np.maximum(centered @ model["coefficients"] + model["mean_y"], 0)


def select_training_only(features, labels, latent_times, grid):
    times = np.asarray(latent_times)
    candidates = [("static", 0.0)] + [(kind, float(lam)) for kind in ["linear", "poly2"] for lam in grid]
    errors = {candidate: 0.0 for candidate in candidates}
    denominator = 0.0
    for fold in range(4):
        validation = times % 4 == fold
        training = ~validation
        if training.sum() < 2 or validation.sum() < 1:
            raise ValueError("Every registered time fold must contain rows")
        train_x, test_x = features[training], features[validation]
        train_y, test_y = labels[training], labels[validation]
        denominator += float(np.square(test_y).sum())
        for kind in ["static", "linear", "poly2"]:
            prepared = prepare_fit(train_x, train_y, kind)
            for candidate in [item for item in candidates if item[0] == kind]:
                model = fit_prepared(prepared, candidate[1])
                errors[candidate] += float(np.square(predict(model, test_x) - test_y).sum())
    if denominator <= 0:
        raise ValueError("Nonpositive training-only CV normalization")
    winner = min(candidates, key=lambda candidate: (errors[candidate], candidates.index(candidate)))
    model = fit_prepared(prepare_fit(features, labels, winner[0]), winner[1])
    return model, [{"kind": kind, "regularization": lam, "normalized_CV_MSE": errors[(kind, lam)] / denominator}
                   for kind, lam in candidates]
