"""S2 statistical geometry of a true [latent_tau, spatial_anchor, channels] packet.

Pure numerical functions: no saved data, model, torch, GPU, filesystem or RUN.
Caller must verify source/step/block/row, tensor raw SHAs, teacher/student role,
actual BF16 postwrite and independent source history before using the numbers.
NumPy is injected to keep the module inert when imported for protocol review.
"""
import math


BAND_RANGES = (("low_1_2", 1, 2), ("mid_3_6", 3, 6),
               ("high_7_plus", 7, None))
EPS = 1e-30


def need(ok, why):
    if not ok:
        raise ValueError(why)


def finite_or_none(x):
    return float(x) if math.isfinite(float(x)) else None


def checked(x, np):
    x = np.asarray(x, dtype=np.float64)
    need(x.ndim == 3 and 3 <= x.shape[0] <= 21 and x.shape[1] > 0
         and x.shape[2] > 0 and bool(np.isfinite(x).all()),
         "Expected finite [T>=3,anchors>=1,channels>=1] latent-time trajectory")
    return x


def centered(x, np):
    return x-x.mean(axis=0, keepdims=True)


def spectrum(x, np):
    """Parseval-normalized rFFT band energies; latent time != diffusion step."""
    t = x.shape[0]
    f = np.fft.rfft(centered(x, np), axis=0)
    weights = np.ones(f.shape[0], dtype=np.float64)*2
    weights[0] = 1
    if t % 2 == 0:
        weights[-1] = 1
    energy = weights*np.sum(abs(f)**2, axis=(1, 2))/t
    real_energy = float(np.sum(centered(x, np)**2))
    need(abs(float(energy.sum())-real_energy) < 1e-8*max(real_energy, 1),
         "Parseval failed for latent-time spectrum")
    total = float(energy[1:].sum())
    bands = {}
    for key, lo, hi in BAND_RANGES:
        upper = min(f.shape[0], hi+1) if hi is not None else f.shape[0]
        bands[key] = (float(energy[lo:upper].sum())/total
                      if total > EPS and lo < upper else None)
    power = energy[1:]/total if total > EPS else None
    spectral_erank = (math.exp(float(-np.sum(power[power > 0]*np.log(power[power > 0]))))
                      if power is not None else None)
    return {"AC_energy": total, "band_fractions": bands,
            "bin_fraction": [float(v/total) if total > EPS else None
                             for v in energy[1:]],
            "spectral_effective_rank": spectral_erank,
            "frequency_unit": "integer cycles per T latent-video slots",
            "T": t}


def temporal_gram(x, np):
    """T x T PSD Gram, not a low-rank theorem about W2 weights."""
    z = centered(x, np)
    f = z.reshape(z.shape[0], -1)
    gram = f @ f.T
    eig = np.linalg.eigvalsh(gram)[::-1]
    need(float(eig.min()) >= -1e-9*max(float(eig.max()), 1),
         "Temporal Gram is not PSD within rounding tolerance")
    eig = np.maximum(eig, 0)
    total = float(eig.sum())
    if total <= EPS:
        return {"effective_rank_participation": None,
                "effective_rank_entropy": None, "rank90": None, "rank95": None,
                "spectrum_fraction": [None]*len(eig)}
    prob = eig/total
    cumsum = np.cumsum(prob)
    rank90 = int(np.searchsorted(cumsum, .9, side="left")+1)
    rank95 = int(np.searchsorted(cumsum, .95, side="left")+1)
    participation = float(total*total/np.sum(eig*eig))
    nonzero = prob[prob > 0]
    erank = math.exp(float(-np.sum(nonzero*np.log(nonzero))))
    return {"effective_rank_participation": participation,
            "effective_rank_entropy": erank, "rank90": rank90,
            "rank95": rank95, "spectrum_fraction": [float(v) for v in prob]}


def lag_statistics(x, np, max_lag=10):
    z = centered(x, np)
    t = x.shape[0]
    result = []
    for k in range(1, min(max_lag, t-1)+1):
        a, b = z[k:], z[:-k]
        aa = float(np.sum(a*a)); bb = float(np.sum(b*b))
        corr = (float(np.sum(a*b))/math.sqrt(aa*bb)
                if aa > EPS and bb > EPS else None)
        # Raw, uncentered variogram ratio-of-sums; never a mean of row ratios.
        aa0, bb0 = x[k:], x[:-k]
        den = float(np.sum(aa0*aa0)+np.sum(bb0*bb0))
        variogram = (float(np.sum((aa0-bb0)**2))/den if den > EPS else None)
        result.append({"lag":k, "centered_correlation":corr,
                       "raw_normalized_structure_function":variogram,
                       "eligible_temporal_pairs_per_anchor":t-k})
    return result


def temporal_events(x, np):
    t = x.shape[0]
    dx = x[1:]-x[:-1]
    energy = np.sum(dx*dx, axis=(1, 2))
    mean = float(energy.mean())
    first = x[:t//2].mean(axis=0)
    last = x[t//2:].mean(axis=0)
    z = centered(x, np)
    ac = float(np.sum(z*z))
    shift = (float(np.sum((first-last)**2))*t*x.shape[1]/max(ac, EPS)
             if ac > EPS else None)
    dot = np.sum(dx[:-1]*dx[1:], axis=(1, 2))
    norm = np.sqrt(np.sum(dx[:-1]**2, axis=(1, 2))*
                   np.sum(dx[1:]**2, axis=(1, 2)))
    valid = norm > EPS
    cos = (float(np.mean(dot[valid]/norm[valid])) if valid.any() else None)
    return {"first_difference_mean_energy":mean,
            "first_difference_peak_to_mean":float(energy.max()/mean) if mean > EPS else None,
            "first_difference_p90_to_median":
                float(np.quantile(energy, .9)/np.median(energy))
                if float(np.median(energy)) > EPS else None,
            "second_difference_mean_energy":
                float(np.mean(np.sum((dx[1:]-dx[:-1])**2, axis=(1, 2)))),
            "successive_difference_direction_cosine":cos,
            "half_window_mean_shift_over_total_AC":shift,
            "nonzero_differences":int(np.count_nonzero(energy > EPS)),
            "total_temporal_differences":t-1}


def summarize(x, np):
    x = checked(x, np)
    t, anchors, channels = x.shape
    z = centered(x, np)
    total = float(np.sum(x*x))
    ac = float(np.sum(z*z))
    return {"schema":"ffnds.s2.temporal_packet_statistics.v1",
            "geometry":{"T":t,"anchors":anchors,"channels":channels},
            "AC_fraction_of_total_signal_energy":ac/total if total > EPS else None,
            "spectrum":spectrum(x,np),
            "lag":lag_statistics(x,np),
            "temporal_gram":temporal_gram(x,np),
            "events":temporal_events(x,np)}


def compare(reference, candidate, np):
    """Matched same-coordinate BF16 factor replay spectrum, not video motion."""
    r, c = checked(reference, np), checked(candidate, np)
    need(r.shape == c.shape, "Reference/candidate time-coordinate mismatch")
    t = r.shape[0]
    fr = np.fft.rfft(centered(r,np), axis=0)
    fc = np.fft.rfft(centered(c,np), axis=0)
    weights = np.full(fr.shape[0], 2., dtype=np.float64)
    weights[0] = 1
    if t % 2 == 0:
        weights[-1] = 1
    output={}
    for label,lo,hi in BAND_RANGES:
        end = min(fr.shape[0],hi+1) if hi is not None else fr.shape[0]
        if lo >= end:
            output[label] = {"relative_error":None,"signed_gain":None,
                             "complex_coherence_real":None,"complex_coherence_imag":None,
                             "phase_angle_radians":None,"energy_ratio":None}
            continue
        a,b=fr[lo:end],fc[lo:end]
        w=weights[lo:end,None,None]
        er=float(np.sum(w*abs(a)**2)); ec=float(np.sum(w*abs(b)**2))
        complex_cross=complex(np.sum(w*np.conj(a)*b))
        cross=float(complex_cross.real)
        e=float(np.sum(w*abs(a-b)**2))
        valid=er>EPS and ec>EPS
        output[label]={"relative_error":e/er if er>EPS else None,
                       "signed_gain":cross/er if er>EPS else None,
                       "complex_coherence_real":cross/math.sqrt(er*ec) if valid else None,
                       "complex_coherence_imag":float(complex_cross.imag)/math.sqrt(er*ec) if valid else None,
                       "phase_angle_radians":math.atan2(float(complex_cross.imag),cross)
                           if valid and abs(complex_cross)>EPS else None,
                       "energy_ratio":ec/er if er>EPS else None}
    return {"schema":"ffnds.s2.temporal_factor_frequency_comparison.v1",
            "T":t,"bands":output,
            "interpretation":"Same Dense h and native factors required; matched tau/xy/step/block; no online motion inference."}
