"""Pure local-versus-prefix geometry on matched time/space operands.

All four operands must come from the caller's real same-checkpoint numerical
replays. These are fixed-state decompositions, not downstream causal videos.
"""
import math

EPS = 1e-30
BAND_MIN_AC_FRACTION = 1e-12
BANDS = (('low_1_2', 1, 2), ('mid_3_6', 3, 6), ('high_7_plus', 7, None))


def need(ok, reason):
    if not ok: raise ValueError(reason)


def component_geometry(teacher, student_raw, student_candidate, teacher_candidate, np,
                       *, weight=None, minimum_reference_energy=EPS):
    arrays = [np.asarray(x) for x in (teacher, student_raw, student_candidate, teacher_candidate)]
    need(len({x.shape for x in arrays}) == 1 and all(bool(np.isfinite(x).all()) for x in arrays), 'Four matched finite operands required')
    t, s, cs, ct = arrays
    w = 1. if weight is None else np.asarray(weight)
    energy = lambda x: float(np.sum(w*np.abs(x)**2, dtype=np.float64))
    cross = lambda x, y: complex(np.sum(w*np.conj(x)*y))
    local, prefix, total, teacher_local = cs-s, s-t, cs-t, ct-t
    interaction = local-teacher_local
    denominator = energy(t); available = denominator > minimum_reference_energy
    ratio = lambda x: x/denominator if available else None
    e_local, e_prefix, e_total = energy(local), energy(prefix), energy(total)
    signed = 2*cross(local, prefix).real
    closure = float(np.max(np.abs(total-local-prefix)))
    scale = max(float(np.max(np.abs(x))) for x in arrays)
    need(closure <= 1e-10*max(scale, 1), 'Local-plus-prefix operand identity failed')
    ec = energy(cs); ct_energy = energy(ct)
    inner = cross(t, cs); coherence_ok = available and ec > minimum_reference_energy
    return {'reference_energy': denominator, 'reference_available': available,
        'teacher_local_error_fraction': ratio(energy(teacher_local)),
        'student_local_error_fraction': ratio(e_local), 'prefix_error_fraction': ratio(e_prefix),
        'interaction_error_fraction': ratio(energy(interaction)),
        'twice_teacher_local_prefix_cross_fraction': ratio(2*cross(teacher_local, prefix).real),
        'twice_teacher_local_interaction_cross_fraction': ratio(2*cross(teacher_local, interaction).real),
        'twice_prefix_interaction_cross_fraction': ratio(2*cross(prefix, interaction).real),
        'L_P_I_energy_signed_remainder_fraction': ratio(e_total-energy(teacher_local)-e_prefix-energy(interaction)
            -2*cross(teacher_local, prefix).real-2*cross(teacher_local, interaction).real-2*cross(prefix, interaction).real),
        'twice_local_prefix_cross_fraction': ratio(signed), 'total_error_fraction': ratio(e_total),
        'energy_identity_signed_remainder_fraction': ratio(e_total-e_local-e_prefix-signed),
        'operand_identity_max_absolute': closure,
        'candidate_energy_ratio': ratio(ec), 'teacher_candidate_energy_ratio': ratio(ct_energy),
        'candidate_signed_aligned_gain': ratio(inner.real),
        'candidate_complex_coherence_real': inner.real/math.sqrt(denominator*ec) if coherence_ok else None,
        'candidate_complex_coherence_imag': inner.imag/math.sqrt(denominator*ec) if coherence_ok else None,
        'candidate_phase_radians': math.atan2(inner.imag, inner.real)
            if coherence_ok and abs(inner) > max(minimum_reference_energy, 1e-12*math.sqrt(denominator*ec)) else None}


def temporal_views(teacher, student_raw, student_candidate, teacher_candidate, masks, np, *, spectrum=True):
    arrays = [np.asarray(x, dtype=np.float64) for x in (teacher, student_raw, student_candidate, teacher_candidate)]
    need(len({x.shape for x in arrays}) == 1 and arrays[0].ndim == 3 and arrays[0].shape[0] == 21,
         'Expected four same [21,spatial_anchor,channel] arrays')
    anchors = arrays[0].shape[1]
    for mask in masks.values():
        need(mask and len(set(mask)) == len(mask) and all(type(i) is int and 0 <= i < anchors for i in mask), 'Invalid spatial mask')
    result = {}
    for view, sl in (('all21', slice(None)), ('tau1_to20', slice(1, None)), ('tau0', slice(0, 1))):
        selected = [x[sl] for x in arrays]; length = selected[0].shape[0]
        means = [x.mean(axis=0, keepdims=True) for x in selected]
        centered = [x-mu for x, mu in zip(selected, means)]
        fourier = [np.fft.rfft(x, axis=0) for x in centered] if spectrum and length > 1 else None
        weights = None
        if fourier is not None:
            weights = np.full(fourier[0].shape[0], 2., dtype=np.float64); weights[0] = 1.
            if length % 2 == 0: weights[-1] = 1.
            weights = weights[:, None, None]/length
            need(math.isclose(float(np.sum(weights*abs(fourier[0])**2)), float(np.sum(centered[0]**2)),
                              rel_tol=1e-10, abs_tol=1e-10), 'Reference Parseval identity failed')
        rois = {}
        for name, mask in masks.items():
            spatial = lambda values: [x[:, mask, :] for x in values]
            item = {'total': component_geometry(*spatial(selected), np),
                    'DC': component_geometry(*spatial(means), np, weight=float(length)),
                    'AC': component_geometry(*spatial(centered), np)}
            if fourier is not None:
                ref_ac = item['AC']['reference_energy']
                bands = {}
                for label, lo, hi in BANDS:
                    end = min(fourier[0].shape[0], hi+1) if hi is not None else fourier[0].shape[0]
                    values = [x[lo:end, mask, :] for x in fourier]
                    bands[label] = component_geometry(*values, np, weight=weights[lo:end],
                        minimum_reference_energy=max(EPS, BAND_MIN_AC_FRACTION*ref_ac))
                item['frequency'] = bands
                item['frequency_bins'] = {str(k): component_geometry(
                    *(x[k:k+1, mask, :] for x in fourier), np, weight=weights[k:k+1],
                    minimum_reference_energy=max(EPS, BAND_MIN_AC_FRACTION*ref_ac))
                    for k in range(1, fourier[0].shape[0])}
            rois[name] = item
        result[view] = rois
    return result


def spatial_masks(geometry):
    points = [tuple(p) for p in geometry['anchors_yx']]
    lookup = {p: i for i, p in enumerate(points)}
    need(len(lookup) == 52, 'Exactly52 unique spatial anchors required')
    return {'all52': list(range(52)),
        'high_flow25': [lookup[tuple(p)] for p in geometry['high_flow_tile_yx']],
        'central25': [lookup[tuple(p)] for p in geometry['central_tile_yx']],
        'far2': [lookup[tuple(p)] for p in geometry['far_controls_yx']]}


def relative_sse(candidate, reference, np):
    c, r = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    need(c.shape == r.shape and bool(np.isfinite(c).all()) and bool(np.isfinite(r).all()), 'Matched finite SSE operands required')
    denominator = float(np.sum(r*r)); numerator = float(np.sum((c-r)**2))
    return numerator/denominator if denominator > EPS else None


def native_rounding_bridge(native_teacher, native_candidate, replay_teacher, replay_candidate, masks, np, *, spectrum=True):
    """Exact saved-fullM = replay-smallM + measured signed shape bridge.

    Targets may already include own-state gate, typed residual or rowLN. No
    smoothness or linearity of their numerical implementations is assumed.
    """
    values = [np.asarray(x, np.float64) for x in (native_teacher, native_candidate, replay_teacher, replay_candidate)]
    need(len({v.shape for v in values}) == 1 and values[0].shape[:2] == (21, 52), 'Matched native/replay spatial targets required')
    result = {}
    for view, sl in (('all21', slice(None)), ('tau1_to20', slice(1, None)), ('tau0', slice(0, 1))):
        selected = [x[sl] for x in values]; length = selected[0].shape[0]
        dc = [x.mean(axis=0, keepdims=True) for x in selected]
        ac = [x-m for x, m in zip(selected, dc)]
        fourier = [np.fft.rfft(x, axis=0) for x in ac] if spectrum and length > 1 else None
        weights = None
        if fourier is not None:
            weights = np.full(fourier[0].shape[0], 2.); weights[0] = 1.
            if length % 2 == 0: weights[-1] = 1.
            weights = weights[:, None, None]/length
        rois = {}
        for roi, mask in masks.items():
            parts = {}
            comparisons = [('total', selected, float(1.)), ('DC', dc, float(length)), ('AC', ac, float(1.))]
            if fourier is not None:
                for label, lo, hi in BANDS:
                    end = min(fourier[0].shape[0], hi+1) if hi is not None else fourier[0].shape[0]
                    comparisons.append((label, [x[lo:end] for x in fourier], weights[lo:end]))
            for part, arrays, multiplier in comparisons:
                nt, nc, rt, rc = [x[:, mask] for x in arrays]
                error, replay, teacher_bridge, candidate_bridge = nc-nt, rc-rt, nt-rt, nc-rc
                bridge = candidate_bridge-teacher_bridge
                energy = lambda x: float(np.sum(multiplier*np.abs(x)**2))
                denominator = energy(nt)
                threshold = max(EPS, BAND_MIN_AC_FRACTION*parts['AC']['native_reference_energy']) if part in {b[0] for b in BANDS} else EPS
                ratio = lambda x: x/denominator if denominator > threshold else None
                signed = float(2*np.sum(multiplier*np.conj(replay)*bridge).real)
                closure = float(np.max(abs(error-replay-bridge)))
                need(closure <= 1e-10*max(1., *(float(np.max(abs(x))) for x in arrays)), 'Native anchor bridge identity failed')
                parts[part] = {'native_reference_energy': denominator, 'native_total_error_fraction': ratio(energy(error)),
                    'replay_total_error_fraction': ratio(energy(replay)),
                    'teacher_shape_bridge_fraction': ratio(energy(teacher_bridge)),
                    'candidate_shape_bridge_fraction': ratio(energy(candidate_bridge)),
                    'combined_shape_bridge_fraction': ratio(energy(bridge)),
                    'twice_replay_bridge_cross_fraction': ratio(signed),
                    'energy_signed_remainder_fraction': ratio(energy(error)-energy(replay)-energy(bridge)-signed),
                    'operand_identity_max_absolute': closure}
            rois[roi] = parts
        result[view] = rois
    return result
