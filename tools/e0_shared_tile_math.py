"""Shared-column diagnostic primitives. No model calls or policy installation."""


def block_means(value, rows):
    if value.shape[0] % rows:
        raise ValueError('Complete diagnostic tiles required')
    return value.reshape(-1, rows, value.shape[-1]).mean(1)


def shared_mask(score, rows, retained):
    from e0_intrinsic_math import top_support
    return top_support(score, retained).repeat_interleave(rows, dim=0)


def selector_scores(delta, mean_preact_current, mean_preact_previous,
                    smooth, value_norm, rows, micro_rows=16):
    """Mean predictors only use pre-W1 Q1 operands and original static weights.

    Actual delta is used exclusively by explicitly named posthoc descriptions.
    mean_preact arrays were projected at micro_rows granularity, before selection.
    """
    import torch.nn.functional as F
    ratio = rows // micro_rows
    if rows % micro_rows or min(rows, micro_rows) <= 0:
        raise ValueError('Invalid observer subdivision')
    n = mean_preact_current.shape[-1]
    ac = mean_preact_current.reshape(-1, ratio, n)
    ap = mean_preact_previous.reshape(-1, ratio, n)
    mc, mp = ac.mean(1), ap.mean(1)
    dg = (F.gelu(ac, approximate='tanh')-F.gelu(ap, approximate='tanh'))/smooth
    mean_dg = (F.gelu(mc, approximate='tanh')-F.gelu(mp, approximate='tanh'))/smooth
    return {
        'posthoc_shared_diagonal':block_means(delta.square(), rows)*value_norm,
        'posthoc_shared_mean':block_means(delta, rows).square()*value_norm,
        'preW1_mean_preact':(mc-mp).abs(),
        'preW1_mean_gelu_value':mean_dg.square()*value_norm,
        'preW1_micro16_preact':(ac-ap).abs().sum(1),
        'preW1_micro16_gelu_value':dg.square().mean(1)*value_norm,
    }

