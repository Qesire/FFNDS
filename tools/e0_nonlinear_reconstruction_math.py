"""Temporary ridge corrections to a retained nonlinear-feature operator."""


def dual_correction(features, errors, fraction, torch):
    """Return dual coefficients; errors may contain several matched controls.

    Caller centers features and errors using calibration data only. The implied
    new retained weight is features.T @ coefficients, not a free online oracle.
    """
    gram = features @ features.T
    penalty = (fraction * gram.trace() / len(features)).clamp_min(1e-24)
    regularized = gram + penalty * torch.eye(len(features), device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(regularized, errors), gram, float(penalty)
