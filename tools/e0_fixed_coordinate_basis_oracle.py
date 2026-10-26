"""Fixed16/pooled128 OFFLINE_FUNCTION_CLASS_ORACLE; default metadata only.

An exact-SHA ACTIVE registration is required for --execute-cpu. No model,
native quantizer, GPU, weight decomposition or persisted readout coefficients.
Only the small feature matrix is factorized. The standard-eps numerical span
is primary; all-positive extra modes are explicit expanded-span sensitivity.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import e0_vector_curve_cpu_feasibility as old

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = 'E0_FIXED_COORDINATE_BASIS_ORACLE_20261026'
OLD_PLAN = 'research/method_discussion/discoveries/E0/independent-vector-curve-cpu-feasibility-plan-20261026.json'
OLD_RESULT = 'research/20261026/E0_VECTOR_CURVE_CPU_FEASIBILITY.json'
OLD_RESULT_SHA = 'd8c94d4da6d8839887d2ae85fe038e91b69a6ba277303829301173e76ef85065'
IDS = [0, 505, 1593, 4367, 4607, 5848, 6958, 7555, 7802, 8145, 8860, 9215, 10080, 11914, 13651, 13823]
SEED = 20261026
MAX_MAC = 20_000_000
TARGETS = ['continuous', 'saved_preQ2', 'saved_postQ2']
FIELDS = ['q1_post', 'b1', 'q2_pre', 'q2_post', 'gate']


def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    expected = {'id': STUDY_ID, 'label': 'OFFLINE_FUNCTION_CLASS_ORACLE',
        'selected_ids': IDS, 'anchors': old.ANCHORS, 'sources': [0, 12], 'steps': [0, 1],
        'block': 10, 'expert': 'high_noise_model', 'baselines': ['constant_only', 'anchor_affine'],
        'rows_per_case': 32, 'pooled_rows': 128, 'max_cpu_threads': 1,
        'max_RSS_bytes': 2 << 30, 'max_wall_seconds': 30, 'max_output_bytes': 1 << 20,
        'max_CPU_matmul_MAC': MAX_MAC, 'random_seed': SEED, 'automatic_retry': False,
        'rank_policy': 'STANDARD_EPS_PRIMARY_ALL_POSITIVE_EXPANDED_SENSITIVITY'}
    if any(plan.get(k) != value for k, value in expected.items()):
        raise ValueError('Scope differs from frozen fixed16/pooled128 oracle')
    required = {'tools/e0_fixed_coordinate_basis_oracle.py', 'tests/test_e0_fixed_coordinate_basis_oracle.py',
                'tools/e0_vector_curve_cpu_feasibility.py', OLD_PLAN, OLD_RESULT,
                'research/20261025/E0_NATIVE_OBSERVATIONS.json', '.runtime/hif4_rebuilt_cache.json'}
    if not required <= set(plan['bound_sources']):
        raise ValueError('Missing prior result/code/operand provenance')
    for name, expected_sha in plan['bound_sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or old.digest(ROOT / name) != expected_sha:
            raise ValueError('Bound source identity/path mismatch: ' + name)
    if old.digest(ROOT / OLD_RESULT) != OLD_RESULT_SHA:
        raise ValueError('Original cluster result changed')
    prior = json.loads((ROOT / OLD_RESULT).read_text())
    if (prior['status'] != 'PASS_CPU_FEASIBILITY_DESCRIPTION_NOT_METHOD_OR_QUALITY_ACCEPTANCE'
            or sorted({i for c in prior['clusters'] for i in c['ids']}) != IDS):
        raise ValueError('Reuse the exact original16 coordinates without neighbor search')
    old_plan, cases, weight = old.preflight(ROOT / OLD_PLAN)
    if old_plan['weight_shard_sha256'] != plan['weight_shard_sha256']:
        raise ValueError('Original resolved private shard identity changed')
    signature = [{k: c[k] for k in ['source', 'step', 'seed', 'sha256', 'run_sha256', 'row_ids', 'positions']}
                 for c in cases]
    if signature != plan['cases']:
        raise ValueError('Original fixed rows/case ordering changed')
    return plan, cases, weight


class Work:
    """Every explicit matrix product is charged before execution."""
    def __init__(self, np, maximum=MAX_MAC, fence=lambda: None):
        self.np, self.maximum, self.fence = np, maximum, fence
        self.mac = 0
        self.products = 0
        self.feature_factorizations = []

    def dot(self, a, b):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('Only compatible2D products are budgeted')
        cost = int(a.shape[0] * a.shape[1] * b.shape[1])
        if self.mac + cost > self.maximum:
            raise RuntimeError('CPU matmul MAC budget would be exceeded')
        self.mac += cost
        self.products += 1
        self.fence()
        return a @ b

    def factor(self, kind, shape):
        # LAPACK work is disclosed separately; no weight matrix is factorized.
        self.feature_factorizations.append({'kind': kind, 'shape': list(shape),
                                           'scope': 'SMALL_FEATURE_MATRIX_ONLY'})
        self.fence()


def features(coordinates, family, np):
    """Shared affine input transform preserves either full polynomial span."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError('Exactly four fixed anchor projections required')
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    scale = np.sqrt((centered * centered).mean(axis=0))
    safe_scale = np.where(scale > 0, scale, 1.)
    t = centered / safe_scale
    columns = [np.ones(len(t))]
    names = ['constant']
    if family == 'constant_only':
        pass
    elif family == 'anchor_affine':
        for j in range(4):
            columns.append(t[:, j]); names.append(f'T{j}')
    elif family == 'joint_quadratic':
        for j in range(4):
            columns.append(t[:, j]); names.append(f'T{j}')
        for j in range(4):
            for k in range(j, 4):
                columns.append(t[:, j] * t[:, k]); names.append(f'T{j}*T{k}')
    elif family == 'separable15':
        for power in [1, 2, 3]:
            for j in range(4):
                columns.append(t[:, j] ** power); names.append(f'T{j}^{power}')
        for j in [0, 1]:
            columns.append(t[:, j] ** 4); names.append(f'T{j}^4')
    else:
        raise ValueError('Unknown fixed basis family')
    return np.column_stack(columns), {'column_names': names, 'nominal_columns': len(columns),
        'pooled_coordinate_center': center.tolist(), 'pooled_coordinate_scale': scale.tolist(),
        'transform_scope': 'ONE_POOLED_AFFINE_TRANSFORM_NO_CASE_INDICATORS'}


def span(matrix, work, np):
    """Construct positive modes, disclose the primary identifiable prefix."""
    n = len(matrix)
    centered = matrix[:, 1:] - matrix[:, 1:].mean(axis=0)
    centered -= centered.mean(axis=0)  # reduce constant contamination
    norms = np.sqrt((centered * centered).sum(axis=0))
    nonzero = norms > 0
    scaled = centered[:, nonzero] / norms[nonzero]
    if scaled.shape[1]:
        work.factor('SVD', scaled.shape)
        u, s, vh = np.linalg.svd(scaled, full_matrices=False)
    else:
        u, s, vh = np.empty((n, 0)), np.empty(0), np.empty((0, 0))
    keep = s > 0
    candidate = np.column_stack([np.ones(n) / math.sqrt(n), u[:, keep]])
    # A very small feature singular mode can acquire a constant component
    # through roundoff. Reorthogonalize the retained span with the explicit
    # constant; this retains its width rather than silently truncating it.
    if candidate.shape[1] > 1:
        work.factor('QR_REORTHOGONALIZE_RETAINED_FEATURE_MODES', candidate.shape)
        q, _ = np.linalg.qr(candidate, mode='reduced')
    else:
        q = candidate
    tol = float(s.max() * max(scaled.shape) * np.finfo(np.float64).eps) if len(s) else 0.
    reconstruction = work.dot(u * s, vh) - scaled
    orthogonal = work.dot(q.T, q) - np.eye(q.shape[1])
    orthogonal_defect = float(np.max(np.abs(orthogonal)))
    if orthogonal_defect > 1e-10 or not np.isfinite(q).all():
        raise ValueError('Numerical feature span failed orthogonality/finite guard')
    primary_rank = 1 + int((s > tol).sum())
    condition = float(s.max() / s[keep].min()) if keep.any() else None
    amplified_roundoff = condition * np.finfo(np.float64).eps * max(matrix.shape) if condition else 0.
    stable = primary_rank == q.shape[1]
    info = {'retained_rank': primary_rank, 'standard_eps_rank': primary_rank,
        'expanded_numerical_retained_width': int(q.shape[1]), 'rank_stable': stable,
        'primary_policy': 'STANDARD_EPS_IDENTIFIABLE_NUMERICAL_SPAN',
        'eligible_fixed_family_comparison': bool(stable and amplified_roundoff <= 1e-4),
        'condition_amplified_machine_roundoff': amplified_roundoff,
        'well_conditioned_comparison_threshold': 1e-4,
        'standard_eps_tolerance': tol, 'centered_singular_values': s.tolist(),
        'column_equilibration_norms': norms.tolist(), 'zero_centered_columns': int((~nonzero).sum()),
        'condition_number_positive_spectrum': condition,
        'orthogonality_max_abs_defect': orthogonal_defect,
        'SVD_reconstruction_SSE': float((reconstruction ** 2).sum()),
        'reorthogonalization': 'QR_OF_EXPLICIT_CONSTANT_PLUS_ALL_POSITIVE_SVD_MODES_NO_RANK_CUTOFF',
        'rank_warning': 'EXTRA_POSITIVE_ROUNDOFF_MODES_ARE_EXPANDED_SENSITIVITY_NOT_FIXED_FAMILY_OPTIMUM; NO_CERTIFIED_EXCLUSION'}
    return q, s[keep], info


def random_span(n, rank, work, np):
    """One seeded nested row-space family, independent of target/input values."""
    if not 1 <= rank <= 15 or rank > n:
        raise ValueError('Unsupported dimension-matched control rank')
    g = np.random.Generator(np.random.PCG64(SEED)).standard_normal((n, 14))
    g -= g.mean(axis=0); g -= g.mean(axis=0)
    work.factor('QR_RANDOM_CONSTANT_ORTHOGONAL_COMPLEMENT', g.shape)
    q, _ = np.linalg.qr(g, mode='reduced')
    q = np.column_stack([np.ones(n) / math.sqrt(n), q[:, :rank-1]])
    defect = float(np.max(np.abs(work.dot(q.T, q) - np.eye(rank))))
    if defect > 1e-10:
        raise ValueError('Random control orthogonality failed')
    return q, {'retained_rank': rank, 'seed': SEED, 'generator': 'NumPy_PCG64_standard_normal',
               'constant_included': True, 'orthogonality_max_abs_defect': defect,
               'same_rank_same_seed_control_is_identical': True,
               'availability': 'OFFLINE_ROW_SPACE_CAPACITY_CONTROL_NOT_ONLINE_FEATURES'}


def energy(coefficients, gram, work, np):
    value = float((work.dot(coefficients, gram) * coefficients).sum())
    if not math.isfinite(value) or value < 0:
        raise ValueError('Nonfinite/negative signed Gram energy')
    return value


def score_residual(residual, target, grams, references, work, np):
    raw_rows = (work.dot(residual, grams[0]) * residual).sum(axis=1)
    raw = float(raw_rows.sum())
    if not math.isfinite(raw) or raw < 0:
        raise ValueError('Nonfinite/negative raw residual energy')
    case_scores = []
    for case in range(4):
        sl = slice(case * 32, (case + 1) * 32)
        gated = energy(residual[sl], grams[case+1], work, np)
        diagonal = float((residual[sl] ** 2 * np.diag(grams[case+1])).sum())
        case_scores.append({'case_index': case, 'raw_signed_SSE': float(raw_rows[sl].sum()),
            'raw_reference_energy': references[case+5],
            'raw_relative_SSE': old.ratio(float(raw_rows[sl].sum()), references[case+5]),
            'raw_centered_innovation_reference_energy': references[case+14],
            'raw_SSE_over_pooled_centered_innovation_case_energy': old.ratio(float(raw_rows[sl].sum()), references[case+14]),
            'current_gate_SSE_from_common_RAW_projection': gated,
            'current_gate_reference_energy': references[case+1],
            'current_gate_relative_SSE': old.ratio(gated, references[case+1]),
            'current_gate_pooled_centered_innovation_reference_energy': references[case+10],
            'current_gate_SSE_over_pooled_centered_innovation_case_energy': old.ratio(gated, references[case+10]),
            'unsigned_diagonal_residual_energy': diagonal,
            'signed_over_diagonal_residual': old.ratio(gated, diagonal)})
    return {'pooled_raw_signed_SSE': raw, 'pooled_raw_reference_energy': references[0],
        'pooled_raw_relative_SSE': old.ratio(raw, references[0]), 'cases': case_scores,
        'pooled_raw_centered_innovation_reference_energy': references[9],
        'pooled_raw_SSE_over_centered_innovation_energy': old.ratio(raw, references[9]),
        'centering_scope': 'ONE_TARGET_MEAN_OVER_ALL128_ROWS; NO_PER_CASE_RECENTERING',
        'gated_objective': 'SCORE_COMMON_RAW_ORACLE_RESIDUAL; NOT_COMMON_GATED_OPTIMUM'}


def analyze(coordinates, targets, weights, gates, work, np):
    """Small-array numerical kernel; also used with synthetic software fixtures."""
    if (coordinates.shape != (128, 4) or weights.shape[1] != 16 or len(gates) != 4
            or any(c.shape != (128, 16) for c in targets.values()) or list(targets) != TARGETS):
        raise ValueError('Expected fixed pooled128/four-anchor/16-coefficient geometry')
    if (any(g.shape != (1, weights.shape[0]) for g in gates)
            or any(not np.isfinite(a).all() for a in [coordinates, weights, *gates, *targets.values()])):
        raise ValueError('Expected finite original coefficient/value/gate geometry')
    grams = [work.dot(weights.T, weights)]
    for gate in gates:
        w = weights * gate.reshape(-1, 1)
        grams.append(work.dot(w.T, w))
    references = {}
    for name, c in targets.items():
        references[name] = [energy(c, grams[0], work, np)] + [
            energy(c[i*32:(i+1)*32], grams[i+1], work, np) for i in range(4)] + [
            energy(c[i*32:(i+1)*32], grams[0], work, np) for i in range(4)]
        centered = c - c.mean(axis=0)
        centered_raw_rows = (work.dot(centered, grams[0]) * centered).sum(axis=1)
        references[name] += [float(centered_raw_rows.sum())] + [
            energy(centered[i*32:(i+1)*32], grams[i+1], work, np) for i in range(4)] + [
            float(centered_raw_rows[i*32:(i+1)*32].sum()) for i in range(4)]
    result = {'basis_spaces': {}, 'comparisons': {},
        'raw_W2_Gram': grams[0].tolist(), 'current_gate_W2_Grams': [g.tolist() for g in grams[1:]],
        'bias_scope': 'FIXED16_PARTIAL_OUTPUT; ORIGINAL_GLOBAL_FFN_BIAS_EXCLUDED',
        'coefficient_targets': TARGETS, 'shared_raw_projection': True}
    random_cache = {}
    for family in ['constant_only', 'anchor_affine', 'joint_quadratic', 'separable15']:
        matrix, definition = features(coordinates, family, np)
        q, spectrum, info = span(matrix, work, np)
        result['basis_spaces'][family] = {**definition, **info}
        result['comparisons'][family] = {}
        for name, c in targets.items():
            modes = work.dot(q.T, c)
            primary_rank = info['retained_rank']
            residual = c - work.dot(q[:, :primary_rank], modes[:primary_rank])
            metrics = score_residual(residual, c, grams, references[name], work, np)
            # Positive-direction capture quantifies exactly what a threshold
            # would discard; no target magnitude is inferred from sigma alone.
            mode_energy = (work.dot(modes, grams[0]) * modes).sum(axis=1)
            metrics['raw_capture_energy_per_expanded_mode_constant_first'] = mode_energy.tolist()
            extra_capture = float(mode_energy[primary_rank:].sum())
            expanded_unclamped = metrics['pooled_raw_signed_SSE'] - extra_capture
            expanded_sse = max(0., expanded_unclamped)
            metrics['primary_scope'] = 'STANDARD_EPS_NUMERICAL_SPAN_DIRECT_COEFFICIENT_RESIDUAL'
            metrics['expanded_numerical_span_sensitivity'] = {
                'label': 'EXPANDED_NUMERICAL_SPAN_SENSITIVITY_NOT_ACTUAL_FUNCTION_FAMILY_OPTIMUM',
                'numerical_retained_width': int(q.shape[1]),
                'raw_capture_energy_in_extra_positive_modes': extra_capture,
                'raw_SSE_primary_minus_extra_capture_unclamped': expanded_unclamped,
                'raw_SSE_estimate_nonnegative': expanded_sse,
                'roundoff_negative_clipped_to_zero': expanded_unclamped < 0,
                'computation': 'ORTHOGONAL_MODE_ENERGY_DIFFERENCE; PRIMARY_RESIDUAL_IS_DIRECT; NO_CERTIFIED_BOUND',
                'eligible_fixed_family_comparison': False}
            metrics['truncation_sensitivity'] = []
            for factor in [0., 1., 10., 100.]:
                mask = spectrum > factor * info['standard_eps_tolerance']
                omitted = float(mode_energy[1:][~mask].sum())
                metrics['truncation_sensitivity'].append({'standard_eps_factor': factor,
                    'numerical_retained_width': 1 + int(mask.sum()), 'raw_energy_in_discarded_positive_modes': omitted,
                    'raw_SSE_expanded_estimate_plus_omitted_mode_energy': expanded_sse + omitted,
                    'numerical_scope': 'ORTHOGONAL_MODE_IDENTITY_WITH_REPORTED_FINITE_DEFECTS; NOT_CERTIFIED'})
            result['comparisons'][family][name] = metrics
        if family in ['constant_only', 'anchor_affine']:
            continue
        rank = info['retained_rank']
        control_name = 'seeded_random_rank' + str(rank)
        if rank not in random_cache:
            rq, rinfo = random_span(128, rank, work, np)
            result['basis_spaces'][control_name] = rinfo
            result['comparisons'][control_name] = {}
            for name, c in targets.items():
                residual = c - work.dot(rq, work.dot(rq.T, c))
                result['comparisons'][control_name][name] = score_residual(
                    residual, c, grams, references[name], work, np)
            random_cache[rank] = control_name
        result['basis_spaces'][family]['dimension_matched_control'] = random_cache[rank]
    eligible = all(result['basis_spaces'][family]['eligible_fixed_family_comparison']
                   for family in ['joint_quadratic', 'separable15'])
    result['fixed_family_comparison_status'] = ('ELIGIBLE_EXPOSED_NUMERICAL_COMPARISON_NOT_CERTIFIED'
        if eligible else 'OPEN_RANK_OR_CONDITION_AMBIGUITY')
    return result


def load_selected_fields(path, positions, np):
    offset, descriptors = old.header(path)
    arrays = {}
    with Path(path).open('rb') as stream:
        for name in FIELDS:
            item = descriptors[name]
            start, end = item['data_offsets']; stream.seek(8 + offset + start)
            bits = np.frombuffer(stream.read(end-start), dtype='<u2').astype('<u4') << 16
            values = bits.view('<f4').reshape(item['shape'])
            if name in ['q2_pre', 'q2_post']:
                values = values[positions][:, IDS]
            elif name == 'q1_post':
                values = values[positions]
            elif name == 'b1':
                values = values[IDS]
            arrays[name] = values.astype(np.float64)
    return arrays


def execute(plan_path, plan, cases, weight_path):
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    if sum(s.get('id') == STUDY_ID and s.get('protocol_sha256') == old.digest(plan_path)
           and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'] for s in scopes) != 1:
        raise RuntimeError('Exact CPU plan SHA must be registered once before numerical work')
    out = Path(plan['canonical_output']); out.mkdir(parents=True, exist_ok=False)
    result = {'id': STUDY_ID, 'label': 'OFFLINE_FUNCTION_CLASS_ORACLE', 'status': 'STARTED_NOT_ACCEPTED',
        'protocol_sha256': old.digest(plan_path), 'model_calls': 0, 'GPU_calls': 0, 'native_QDQ_calls': 0,
        'weight_decompositions': 0, 'deployed_readout_fits': 0, 'new_banks': 0,
        'offline_projection_oracle': True, 'persisted_readout_coefficients': False, 'quality_accepted': False,
        'selected_ids': IDS, 'anchors': old.ANCHORS, 'cases': plan['cases'],
        'weight_shard_filename': old.SHARD_FILENAME, 'weight_shard_sha256': plan['weight_shard_sha256']}
    started = time.monotonic()
    def fence():
        result['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if result['peak_RSS_bytes'] > plan['max_RSS_bytes'] or time.monotonic()-started > 30:
            raise RuntimeError('Fixed CPU RSS/time budget reached')
    def save():
        data = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        if len(data) > plan['max_output_bytes']:
            raise RuntimeError('Fixed output cap exceeded')
        target = out / 'RUN.json'; temp = out / 'RUN.json.tmp'; temp.write_bytes(data); temp.replace(target)
    def timeout(signum, frame):
        raise TimeoutError('Frozen30second numerical phase elapsed')
    work = None
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        save()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            os.environ[key] = '1'
        import numpy as np
        import torch  # exact CPU mmap archive loader/cast only, never CUDA/model APIs
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        work = Work(np, fence=fence)
        params = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)['high_noise_model']
        up, down = params['blocks.10.ffn.0'], params['blocks.10.ffn.2']
        w1, w2 = up['weight_qdq'], down['weight_qdq']
        smooth = down['smooth_scale']
        if (tuple(w1.shape) != (13824, 5120) or tuple(w2.shape) != (5120, 13824)
                or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
                or smooth.dtype != torch.float32 or w1.device.type != 'cpu' or w2.device.type != 'cpu'):
            raise ValueError('Original CPU BF16 weights/FP32Smooth2 required')
        selected_up = w1[IDS].float().numpy().astype(np.float64)
        weights = w2[:, IDS].float().numpy().astype(np.float64)
        scales = smooth.reshape(-1)[IDS].double().numpy()
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Positive original Smooth2 required')
        coordinates, continuous, pre, post, gates = [], [], [], [], []
        anchor_positions = [IDS.index(i) for i in old.ANCHORS]
        for case in cases:
            arrays = load_selected_fields(case['path'], case['positions'], np)
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Nonfinite saved operands')
            projection = work.dot(arrays['q1_post'], selected_up.T)
            a = projection + arrays['b1']
            c = .5 * a * (1 + np.tanh(math.sqrt(2/math.pi) * (a + .044715 * a**3))) / scales
            coordinates.append(projection[:, anchor_positions]); continuous.append(c)
            pre.append(arrays['q2_pre']); post.append(arrays['q2_post']); gates.append(arrays['gate'])
            fence()
        targets = dict(zip(TARGETS, [np.concatenate(v) for v in [continuous, pre, post]]))
        result['analysis'] = analyze(np.concatenate(coordinates), targets, weights, gates, work, np)
        result['environment'] = {'numpy': np.__version__, 'torch': torch.__version__,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'BLAS_thread_environment': {k: os.environ[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}
        fence()
        result.update(status='PASS_OFFLINE_FUNCTION_CLASS_ORACLE_NOT_DEPLOYMENT_OR_QUALITY',
            interpretation='Pooled exposed rows; shared RAW output projection; case gates score that same residual; numerical ranks and small modes disclosed; no certified exclusions or heldout claim')
    except BaseException as error:
        result.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if work:
            result.update(CPU_matmul_MAC=work.mac, explicit_matmul_calls=work.products,
                          feature_factorizations=work.feature_factorizations)
        result['elapsed_seconds'] = time.monotonic()-started
        result['timing_scope'] = 'NUMERICAL_PHASE_INCLUDING_IMPORTS_AND_CPU_MMAP; PRIOR_HASH_PREFLIGHT_EXCLUDED'
        result['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--execute-cpu', action='store_true')
    args = parser.parse_args()
    plan, cases, weight_path = preflight(args.protocol)
    print(json.dumps({'status': 'HASH_HEADER_FIXED_SELECTION_PREFLIGHT_ONLY', 'numerical_work': False,
                      'cases': len(cases), 'GPU_initialized': False}), flush=True)
    if args.execute_cpu:
        execute(args.protocol, plan, cases, weight_path)


if __name__ == '__main__':
    main()
