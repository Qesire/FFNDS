"""Statistical GELU gains and fixed-input low-rank regression, no CUDA on import."""
import math


def gaussian_gain(mean, variance, nodes, weights):
    from e0_gelu_tangent import derivative
    samples = mean[None] + (2 * variance.clamp_min(0)).sqrt()[None] * nodes[:, None]
    return (derivative(samples) * weights[:, None]).sum(0) / math.sqrt(math.pi)


def ridge_factors(x, y, ridge_fraction, rank, svd):
    """Return output-PCA, predictive reduced-rank, and token-shuffled factors.

    The predictive rank constraint minimizes training SSE + lambda*||B||².
    This does not promise the smallest held-source or gated error.
    """
    mx, my = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - mx, y - my
    ux, sx, vx = svd(xc)
    _, sy, vy = svd(yc)
    lam = (ridge_fraction * sx.square().sum() / len(x)).clamp_min(1e-24)
    cross = ux.T @ yc
    inverse = sx / (sx.square() + lam)
    output_basis = vy[:rank]
    pca_left = vx.T @ (inverse[:, None] * (cross @ output_basis.T))
    transformed = (sx / (sx.square() + lam).sqrt())[:, None] * cross
    up, sp, vp = svd(transformed)
    pred_left = (vx.T @ (up[:, :rank] / (sx.square() + lam).sqrt()[:, None])) * sp[:rank]
    return {'mean_x': mx, 'mean_y': my, 'pca_left': pca_left, 'pca_right': output_basis,
            'predictive_left': pred_left, 'predictive_right': vp[:rank],
            'input_u': ux, 'input_v': vx, 'ridge_inverse': inverse,
            'singular_values': sy, 'all_output_right': vy, 'lambda': float(lam)}


def shuffled_factors(fit, y, permutation, rank):
    centered = y[permutation] - fit['mean_y']
    right = fit['pca_right'][:rank]
    left = fit['input_v'].T @ (fit['ridge_inverse'][:, None] *
                              (fit['input_u'].T @ centered @ right.T))
    return left, right
