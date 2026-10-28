"""A fixed conditioning-only normalization gain; lazy numerical interface."""
import math
import struct
from contextlib import nullcontext

EPS=1e-6


def need(value,message):
    if not value:raise ValueError(message)


def gain_scalar(kappa):
    need(type(kappa) in (float,int) and math.isfinite(kappa) and kappa>=1.,'Fixed sigma_D/sigma_S must be finite and >=1')
    # This deployed reciprocal is frozen before any Tensor calculation.
    return struct.unpack('<f',struct.pack('<f',1./kappa))[0]


def normalized_gain(torch,normalized,raw_u,kappa,work=lambda _:nullcontext()):
    """Modify only the fresh LN result, using explicit FP32 arithmetic.

    The original LN's BF16 result remains its return object. Its source u and
    all original modulation/shift/head-bias paths remain untouched. kappa=1
    traverses the same storage gate but performs no new Tensor arithmetic.
    """
    alpha=gain_scalar(kappa)
    with work('normalization_storage_identity_checks'):
        need(normalized.dtype==raw_u.dtype==torch.bfloat16 and normalized.shape==raw_u.shape
          and normalized.device==raw_u.device and normalized.numel()>0
          and normalized.untyped_storage().data_ptr()!=raw_u.untyped_storage().data_ptr(),
          'LN result aliases original u or has wrong BF16 geometry')
    if kappa==1.:
        with work('normalization_identity_noops'):return normalized
    with work('conditioning_normalization_gain_calls'):
        with work('norm_gain_BF16_to_FP32_casts'):temporary=normalized.float()
        with work('norm_gain_FP32_multiplies'):temporary.mul_(alpha)
        with work('norm_gain_FP32_to_BF16_copies'):normalized.copy_(temporary)
    return normalized


def effective_statistics(torch,stats,kappa,work=lambda _:nullcontext()):
    """Raw-u q unchanged; effective denominator is sigma_u / FP32(alpha)."""
    alpha=gain_scalar(kappa);effective_kappa=1./alpha
    if kappa==1.:effective=stats
    else:
        with work('effective_normalizer_cases'):
            effective={'q':stats['q'],'variance':(stats['variance']+EPS)*(effective_kappa**2)-EPS}
    with work('raw_effective_normalizer_diagnostic_cases'):
        raw=(stats['variance']+EPS).sqrt();eff=(effective['variance']+EPS).sqrt()
        need(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(eff).all()) and bool((eff>0).all()),'Invalid raw/effective normalizer')
        diagnostic={'fitted_kappa_sigmaD_over_sigmaS':float(kappa),'deployed_FP32_reciprocal_alpha':alpha,
          'effective_kappa_from_deployed_alpha':effective_kappa,
          'raw_u_sigma_mean':float(raw.mean()),'raw_u_sigma_min':float(raw.min()),'raw_u_sigma_max':float(raw.max()),
          'effective_readout_sigma_mean':float(eff.mean()),'effective_readout_sigma_min':float(eff.min()),'effective_readout_sigma_max':float(eff.max()),
          'identity':'Raw u was not modified; gain acts on native BF16 LN output; gain/LN/modulation/GEMM rounding stays in native remainder'}
    return effective,diagnostic
