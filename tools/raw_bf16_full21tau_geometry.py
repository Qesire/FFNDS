"""Pure saved-row geometry for the registered full21tau CPU observation.

No model, Torch, NumPy, or tensor payload is imported/read at module import.
NumPy is supplied by the authorized caller (or synthetic CPU controls).
"""
import math
import warnings

GRID = (21, 29, 52)
ANCHORS = ((6, 12), (6, 13), (6, 38), (6, 39),
           (21, 12), (21, 13), (21, 38), (21, 39))
NEAR = ((0, 1), (2, 3), (4, 5), (6, 7))
FAR = ((0, 6), (1, 7), (2, 4), (3, 5))
VIEWS = ('all21', 'tau1_to20', 'tau0')
OPERANDS = ('V1', 'h', 'Y', 'gY', 'R', 'u', 'rowLN_u')
ACTIONS = ('T10_KEEP_T0', 'NEAR_COPY_KEEP_T0', 'FAR_COPY_KEEP_T0')
TARGETS = ('gY_FP64', 'native_rounded_update', 'u_BF16', 'rowLN_u')
GEOMETRY_METRICS = (
    'mean_row_energy', 'grand_fraction', 'common_tau_fraction',
    'static_xy_fraction', 'tau_xy_interaction_fraction',
    'temporal_constant_oracle_residual_fraction',
    'orthogonality_signed_remainder_fraction', 'reconstruction_max_abs',
    'all_temporal_pairs_energy_ratio_of_sums',
    'near_energy_mean', 'far_energy_mean', 'near_minus_far_energy_mean',
    'near_cosine_mean', 'far_cosine_mean', 'near_minus_far_cosine_mean',
    'near_cosine_eligible_pairs', 'far_cosine_eligible_pairs',
    'matched_cosine_eligible_pairs')
ACTION_METRICS = (
    'error_energy', 'reference_energy', 'error_fraction',
    'error_temporal_mean_energy', 'error_temporal_residual_energy',
    'reference_temporal_residual_energy', 'error_temporal_residual_fraction',
    'candidate_temporal_residual_energy_ratio',
    'signed_error_reference_temporal_residual_fraction',
    'candidate_total_energy_ratio', 'cosine',
    'temporal_error_decomposition_signed_remainder_fraction')
EPS = 1e-30
LN_EPS = 1e-6


def need(ok, message):
    if not ok:
        raise ValueError(message)


def rows():
    return [t*29*52+y*52+x for t in range(21) for y, x in ANCHORS]


def validate_rows(actual):
    need(actual == rows() and len(set(actual)) == 168,
         'Complete21tau fixed-eight-xy native row order differs')
    return True


def time_view(x, name):
    need(name in VIEWS and x.shape[0] == 21 and x.shape[1] == 8,
         'Expected complete [21tau,8anchor,C]')
    return x if name == 'all21' else (x[1:] if name == 'tau1_to20' else x[:1])


def round_bf16(x, np):
    """Explicit finite FP32 -> BF16 round-to-nearest-even, returned as FP32.

    This is a numerical emulator, not evidence that a GPU fused its operations.
    The producer must match every original saved u byte before reporting actions.
    """
    value = np.asarray(x, dtype=np.float32)
    need(np.isfinite(value).all(), 'Nonfinite BF16 rounding input')
    bits = value.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    result = (rounded.astype(np.uint32) << 16).view(np.float32)
    need(np.isfinite(result).all(), 'Finite FP32 overflowed BF16')
    return result


def bf16_bytes(x, np):
    value = np.asarray(x, dtype=np.float32)
    bits = value.view(np.uint32)
    need(np.isfinite(value).all() and ((bits & 0xffff) == 0).all(),
         'Value is not exactly finite decoded BF16')
    return (bits >> 16).astype('<u2').tobytes()


def postwrite(r, y, gate, np):
    """Original eager x.add_(y*gate): BF16 product, then BF16 in-place add.

    No in-place operation touches the supplied arrays. The gate is produced by
    the current step/block modulation before FFN and is kept at the target row.
    """
    need(r.shape == y.shape and r.shape[-1] == gate.shape[-1],
         'Residual/output/gate geometry differs')
    for value in (r, y, gate):
        bf16_bytes(value, np)
    update = round_bf16(np.multiply(y, gate, dtype=np.float32), np)
    u = round_bf16(np.add(r, update, dtype=np.float32), np)
    return update, u


def neutral_postwrite(packet, np):
    update, restored = postwrite(packet['r'], packet['Y'], packet['gate'], np)
    need(bf16_bytes(restored, np) == bf16_bytes(packet['u'], np),
         'Original BF16 product-then-add did not restore saved u bytes; stop without tolerance/retry')
    return update


def row_ln(u, np):
    value = np.asarray(u, dtype=np.float64)
    z = value-value.mean(axis=-1, keepdims=True)
    result = z/np.sqrt(np.mean(z*z, axis=-1, keepdims=True)+LN_EPS)
    need(np.isfinite(result).all(), 'Nonfinite FP64 rowLN geometry')
    return result


def operands(packet, np):
    return {'V1': packet['V1'], 'h': packet['h'], 'Y': packet['Y'],
            'gY': packet['Y'].astype(np.float64)*packet['gate'].astype(np.float64),
            'R': packet['r'], 'u': packet['u'], 'rowLN_u': row_ln(packet['u'], np)}


def two_way(x, np):
    """Orthogonal balanced time x xy ANOVA; no learned basis or decomposition.

    The all-unordered-temporal-pairs ratio is a ratio of sums, using
    sum_{i<j}||xi-xj||^2 = T sum_i||xi-mean_t(x)||^2. It is not the mean
    of normalized pair ratios. Temporal mean is a paid post-hoc oracle.
    """
    x = np.asarray(x, dtype=np.float64)
    need(x.ndim == 3 and x.shape[0] > 0 and x.shape[1] == 8
         and x.shape[2] > 0 and np.isfinite(x).all(), 'Invalid balanced trajectory')
    t, a, _ = x.shape
    grand = x.mean(axis=(0, 1), keepdims=True)
    common_t = x.mean(axis=1, keepdims=True)-grand
    static_a = x.mean(axis=0, keepdims=True)-grand
    interaction = x-grand-common_t-static_a
    total = float(np.sum(x*x, dtype=np.float64))
    energies = [float(t*a*np.sum(grand*grand)),
                float(a*np.sum(common_t*common_t)),
                float(t*np.sum(static_a*static_a)),
                float(np.sum(interaction*interaction))]
    temporal_residual = energies[1]+energies[3]
    reconstruction = grand+common_t+static_a+interaction
    result = [total/(t*a), *(e/(total+EPS) for e in energies),
              temporal_residual/(total+EPS), (total-sum(energies))/(total+EPS),
              float(np.max(np.abs(x-reconstruction))),
              t*temporal_residual/((t-1)*total+EPS) if t > 1 else None]
    need(abs(total-sum(energies)) <= 1e-11*(total+EPS),
         'Balanced time/space orthogonality failed')
    return result


def spatial_pairs(x, np):
    """Same time, four disjoint near/far pairs, with matched cosine eligibility."""
    x = np.asarray(x, dtype=np.float64)
    def measure(pairs):
        a, b = x[:, [p[0] for p in pairs]], x[:, [p[1] for p in pairs]]
        aa, bb = np.sum(a*a, axis=-1), np.sum(b*b, axis=-1)
        energy = np.sum((a-b)**2, axis=-1)/(aa+bb+EPS)
        denominator = np.sqrt(aa*bb)
        cosine = np.full(denominator.shape, np.nan)
        np.divide(np.sum(a*b, axis=-1), denominator, out=cosine, where=denominator > 0)
        return energy, cosine
    ne, nc = measure(NEAR)
    fe, fc = measure(FAR)
    nv, fv = np.isfinite(nc), np.isfinite(fc)
    joint = nv & fv
    def eligible_mean(x, mask):
        return float(x[mask].mean()) if mask.any() else None
    return [float(ne.mean()), float(fe.mean()), float((ne-fe).mean()),
            eligible_mean(nc, nv), eligible_mean(fc, fv),
            eligible_mean(nc-fc, joint), int(nv.sum()), int(fv.sum()), int(joint.sum())]


def geometry(x, np):
    x = np.asarray(x).reshape(21, 8, -1)
    return [[*two_way(time_view(x, view), np),
             *spatial_pairs(time_view(x, view), np)] for view in VIEWS]


def representative_map(action):
    need(action in ACTIONS, 'Unknown fixed pre-W1 representative action')
    mapping = [[t*8+a for a in range(8)] for t in range(21)]
    for t in range(1, 21):
        if action == 'T10_KEEP_T0':
            mapping[t] = [10*8+a for a in range(8)]
        else:
            for a, b in (NEAR if action == 'NEAR_COPY_KEEP_T0' else FAR):
                mapping[t][b] = t*8+a
    return [i for group in mapping for i in group]


def action_contract():
    result = {}
    for action in ACTIONS:
        mapping = representative_map(action)
        count = len(set(mapping))
        result[action] = {
            'row_source_indices': mapping, 'representative_rows_of_168': count,
            'theoretical_FFN_row_fraction': count/168,
            'theoretical_FFN_row_saving_fraction': 1-count/168,
            'destination_rows_requiring_copy': sum(a != b for a, b in enumerate(mapping)),
            'known_before_W1': True, 'protect_tau0': True,
            'time_identity': 'CURRENT_DENOISING_CALL_LATENT_TAU_NOT_DIFFUSION_HISTORY',
            'online_work': 'GATHER_CURRENT_V1_REPRESENTATIVES_W1_GELU_W2_SCATTER_Y_THEN_ORIGINAL_TARGET_GATE_RESIDUAL',
            'not_measured': 'PACKING_GATHER_SCATTER_SMALL_M_GEMM_ROUNDING_KERNEL_TIME_STUDENT_PREFIX_SUFFIX_AND_VIDEO'}
    return result


def error_geometry(candidate, reference, np):
    """Time-mean vs innovation error at fixed Dense prefix; no suffix replay."""
    a, b = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    need(a.shape == b.shape and a.ndim == 3 and a.shape[1] == 8,
         'Action comparison geometry differs')
    delta = a-b
    dmean, bmean, amean = (x.mean(axis=0, keepdims=True) for x in (delta, b, a))
    dac, bac, aac = delta-dmean, b-bmean, a-amean
    error = float(np.sum(delta*delta))
    energy = float(np.sum(b*b))
    ddc = float(a.shape[0]*np.sum(dmean*dmean))
    dacre = float(np.sum(dac*dac))
    bacre, aacre = float(np.sum(bac*bac)), float(np.sum(aac*aac))
    aenergy = float(np.sum(a*a))
    den = math.sqrt(aenergy*energy)
    result = [error, energy, error/(energy+EPS), ddc, dacre, bacre,
              dacre/bacre if bacre > 0 else None,
              aacre/bacre if bacre > 0 else None,
              float(np.sum(dac*bac))/bacre if bacre > 0 else None,
              aenergy/(energy+EPS), float(np.sum(a*b))/den if den > 0 else None,
              (error-ddc-dacre)/(error+EPS)]
    need(abs(error-ddc-dacre) <= 1e-11*(error+EPS), 'Temporal action error identity failed')
    return result


def action_effects(packet, reference_operands, native_update, np):
    reference = (reference_operands['gY'], native_update,
                 packet['u'], reference_operands['rowLN_u'])
    results = []
    for action in ACTIONS:
        y = packet['Y'][representative_map(action)]
        update, u = postwrite(packet['r'], y, packet['gate'], np)
        candidate = (y.astype(np.float64)*packet['gate'].astype(np.float64),
                     update, u, row_ln(u, np))
        result = []
        for view in VIEWS:
            result.append([error_geometry(time_view(a.reshape(21, 8, -1), view),
                                          time_view(b.reshape(21, 8, -1), view), np)
                           for a, b in zip(candidate, reference, strict=True)])
        results.append(result)
    return results


def metric_labels():
    return ([f'geometry/{operand}/{view}/{metric}' for operand in OPERANDS
             for view in VIEWS for metric in GEOMETRY_METRICS]
            +[f'action/{action}/{view}/{target}/{metric}' for action in ACTIONS
              for view in VIEWS for target in TARGETS for metric in ACTION_METRICS])


def flatten(geometry_values, action_values):
    values = [v for operand in geometry_values for view in operand for v in view]
    values += [v for action in action_values for view in action for target in view for v in target]
    need(len(values) == len(metric_labels())
         and all(v is None or (isinstance(v, (float, int)) and math.isfinite(v)) for v in values),
         'Nonfinite or malformed packet scalar vector')
    return values


class SourceEqualSummary:
    """Position means within image; equal independent images, never row N."""
    def __init__(self, source_ids, np):
        need(len(source_ids) == len(set(source_ids)) > 0, 'Unique independent source IDs required')
        self.source_ids, self.np = list(source_ids), np
        self.values = np.full((len(source_ids), 4, 40, len(metric_labels())), np.nan)
        self.seen = set()

    def add(self, source_id, step, block, values):
        key = (source_id, step, block)
        need(key not in self.seen and source_id in self.source_ids
             and type(step) is int and 0 <= step < 4 and type(block) is int and 0 <= block < 40
             and len(values) == len(metric_labels()), 'Duplicate or invalid source/position')
        self.values[self.source_ids.index(source_id), step, block] = [self.np.nan if v is None else v for v in values]
        self.seen.add(key)

    def result(self):
        np = self.np
        need(len(self.seen) == len(self.source_ids)*160, 'All sources must have all160 unique positions')
        def mean(values, axis):
            valid = np.isfinite(values)
            count = valid.sum(axis=axis)
            total = np.where(valid, values, 0).sum(axis=axis)
            result = np.full(total.shape, np.nan)
            np.divide(total, count, out=result, where=count > 0)
            return result, count
        site_equal, site_n = mean(self.values, 0)
        source_step, source_step_n = mean(self.values, 2)
        source_all, source_all_n = mean(self.values.reshape(len(self.source_ids), 160, -1), 1)
        equal_step, equal_step_n = mean(source_step, 0)
        equal_all, equal_all_n = mean(source_all, 0)
        # No pooling rows or positions before giving each source equal weight.
        tails = []
        for quantile in (0, .5, 1):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tails.append(np.nanquantile(self.values, quantile, axis=0))
        def clean(x):
            if isinstance(x, np.ndarray):
                return clean(x.tolist())
            if isinstance(x, list):
                return [clean(v) for v in x]
            return None if isinstance(x, float) and not math.isfinite(x) else x
        return {
            'metric_labels': metric_labels(), 'source_order': self.source_ids,
            'joint_position_axes': ['step4', 'block40', 'metric'],
            'source_equal_joint_position': clean(site_equal), 'joint_position_source_N': clean(site_n),
            'source_tails_min_median_max_joint_position': clean(np.stack(tails)),
            'source_all_position_equal': clean(source_all), 'source_all_valid_positions': clean(source_all_n),
            'source_step_position_equal': clean(source_step), 'source_step_valid_positions': clean(source_step_n),
            'source_equal_all_positions': clean(equal_all), 'source_equal_all_positions_N': clean(equal_all_n),
            'source_equal_by_step': clean(equal_step), 'source_equal_by_step_N': clean(equal_step_n),
            'weights': 'PAIR_OR_ROW_WITHIN_POSITION; EQUAL_POSITION_WITHIN_SOURCE; EQUAL_UNIQUE_IMAGE_SOURCE',
            'tails': 'TEN_EXPLICIT_SOURCE_VECTORS_AND_PER_SOURCE_POSITION_RECORDS; MIN_MEDIAN_MAX_OVER_SOURCES_PER_SITE',
            'source_N': len(self.source_ids), 'independent_token_N': False}
