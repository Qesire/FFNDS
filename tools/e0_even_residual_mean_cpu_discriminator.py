"""Four saved-panel conditional-mean discriminator; default SHA/byte preflight.

Numerical work needs this exact plan SHA registered by the serialized root.
Uses saved W1 outputs, CPU archive reads and mean-vector products only.
Panel means describe these rows, never full-M DC or downstream video repair.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import zipfile

import e0_vector_curve_cpu_feasibility as archive

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = 'E0_EVEN_RESIDUAL_MEAN_CPU_DISCRIMINATOR_20261026'
D, H, BLOCK, STEP = 5120, 13824, 10, 1
MAX_MAC = 34 * D * H  # 8DH input moments + 26DH signed mean readouts.
ALL_MANIFEST = 'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json'
PASSIVE_MANIFEST = 'research/20261025/E0_NATIVE_OBSERVATIONS.json'
CACHE_MANIFEST = '.runtime/hif4_rebuilt_cache.json'
MATRIX_WIDTHS = {'q1_post': D, 'z': H, 'gelu': H, 'q2_post': H, 'ffn_output': D}
STATIC_SHAPES = {'b1': [H], 'b2': [D], 'gate': [1, D]}
METHODS = ['postW1_erf_Gaussian', 'postW1_tanh_GH32_Gaussian',
           'preW1_diagonal_erf_Gaussian', 'empirical_continuous_tanh_over_Smooth2',
           'empirical_saved_gelu_over_Smooth2', 'empirical_saved_postQ2']
# Fixed normalized physicists' Gauss-Hermite32 rule, generated once on synthetic
# coordinates. No adaptive order, fit, feature or weight factorization at run time.
GH_NODES = [-7.125813909830728, -6.409498149269661, -5.812225949515914,
    -5.2755509865158805, -4.777164503502596, -4.305547953351199,
    -3.853755485471445, -3.417167492818571, -2.992490825002374,
    -2.5772495377323175, -2.169499183606112, -1.7676541094632015,
    -1.3703764109528718, -0.9765004635896828, -0.5849787654359324,
    -0.19484074156939934, 0.19484074156939934, 0.5849787654359324,
    0.9765004635896828, 1.3703764109528718, 1.7676541094632015,
    2.169499183606112, 2.5772495377323175, 2.992490825002374,
    3.417167492818571, 3.853755485471445, 4.305547953351199,
    4.777164503502596, 5.2755509865158805, 5.812225949515914,
    6.409498149269661, 7.125813909830728]
GH_WEIGHTS = [4.1246074890182323e-23, 5.208449591960842e-19,
    6.755290223670129e-16, 2.378064855777791e-13, 3.3475012398012285e-11,
    2.312518412074231e-09, 8.881290713105862e-08, 2.0596221039534304e-06,
    3.055980306089635e-05, 0.00030255702581706253, 0.002062051051307884,
    0.009903461702320596, 0.034109847726092095, 0.08534480827208067,
    0.1565389937575985, 0.21170556988047937, 0.21170556988047937,
    0.1565389937575985, 0.08534480827208067, 0.034109847726092095,
    0.009903461702320596, 0.002062051051307884, 0.00030255702581706253,
    3.055980306089635e-05, 2.0596221039534304e-06, 8.881290713105862e-08,
    2.312518412074231e-09, 3.3475012398012285e-11, 2.378064855777791e-13,
    6.755290223670129e-16, 5.208449591960842e-19, 4.1246074890182323e-23]


def relative_path(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Relative asset path required')
    result = (Path(root) / p).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Asset escapes fixed root')
    return result


def unique_scope(run):
    matches = [s for s in run['scopes'] if (s['step'], s['block']) == (STEP, BLOCK)]
    if len(matches) != 1 or matches[0]['expert'] != 'high_noise_model':
        raise ValueError('Unique high-noise block10 step1 scope required')
    return matches[0]


def checked_run(root, name, sha, status, source, seed):
    path = relative_path(root, name)
    if archive.digest(path) != sha:
        raise ValueError('Frozen observation RUN checksum drift')
    run = json.loads(path.read_text())
    if (run['status'] != status or run['seed'] != seed
            or run.get('source_index', source) != source):
        raise ValueError('Original observation status/source/seed drift')
    return path, run


def checked_payload(run_path, scope, old=False):
    path = relative_path(run_path.parent / 'observed', scope['path'])
    if archive.digest(path) != scope['sha256']:
        raise ValueError('Frozen operand checksum drift')
    _, fields = archive.header(path)
    count = len(scope['row_plan']['rows'])
    shapes = {k: [count, v] for k, v in MATRIX_WIDTHS.items()} | STATIC_SHAPES
    if old:
        shapes['q2_pre'] = [count, H]
    elif set(fields) != set(shapes):
        raise ValueError('Expected exact eight-field step1 all-block archive')
    if any(k not in fields or fields[k]['dtype'] != 'BF16' or fields[k]['shape'] != v
           for k, v in shapes.items()):
        raise ValueError('Original BF16 field geometry drift')
    return path


def field_bytes(path, name, positions=None):
    """Hash/identity access only: no dtype decoding or numerical tensor work."""
    size, fields = archive.header(path)
    item = fields[name]
    if item['dtype'] != 'BF16':
        raise ValueError('BF16 byte identity required')
    start, end = item['data_offsets']
    shape = item['shape']
    expected = 2 * math.prod(shape)
    if end - start != expected or start < 0 or end < start:
        raise ValueError('Malformed safetensors BF16 span')
    with Path(path).open('rb') as stream:
        if positions is None:
            stream.seek(8 + size + start)
            data = stream.read(end - start)
        else:
            if len(shape) != 2 or any(type(p) is not int or p < 0 or p >= shape[0] for p in positions):
                raise ValueError('Row positions outside original matrix')
            width = 2 * shape[1]
            pieces = []
            for p in positions:
                stream.seek(8 + size + start + p * width)
                chunk = stream.read(width)
                if len(chunk) != width:
                    raise ValueError('Truncated selected row')
                pieces.append(chunk)
            data = b''.join(pieces)
        if positions is None and len(data) != expected:
            raise ValueError('Truncated original field')
        return data


def discover_inputs(hf_root):
    """Strict metadata and selected-byte join, including exact old pre-Q2 source."""
    accepted = json.loads((ROOT / ALL_MANIFEST).read_text())
    original = json.loads((ROOT / PASSIVE_MANIFEST).read_text())
    inputs = []
    for source, seed in [(0, 42), (12, 54)]:
        bindings = [a for a in accepted['raw_bindings'] if a['source_index'] == source]
        records = [a for a in original['sources'] if a['index'] == source]
        sources = [a for a in accepted['sources'] if a['source_index'] == source]
        if len(bindings) != 1 or len(records) != 1 or len(sources) != 1:
            raise ValueError('Unique accepted source records required')
        binding, rec, src = bindings[0], records[0], sources[0]
        run_name = str(Path('ffnds-derived') / binding['HF_relative'])
        rp, run = checked_run(hf_root, run_name, binding['sha256'],
            'PASS_NATIVE_ALL_BLOCK_TANGENT_OPERANDS_NOOP', source, seed)
        old_name = str(Path(rec['HF_relative']) / 'RUN.json')
        op, old = checked_run(hf_root, old_name, rec['run_sha256'],
            'PASS_PASSIVE_CAPTURE_NOOP', source, seed)
        if run['source_id'] != old['source_id'] or run['source_id'] != src['source_id']:
            raise ValueError('Cross-archive source identity drift')
        scope, old_scope = unique_scope(run), unique_scope(old)
        payload = checked_payload(rp, scope)
        old_payload = checked_payload(op, old_scope, old=True)
        row, older = scope['row_plan'], old_scope['row_plan']
        uniform = row['uniform_rows']
        center = row['contiguous_tiles'][0]
        if (row['grid'] != [21, 29, 52] or older['grid'] != row['grid']
                or uniform != older['uniform_rows'] or len(uniform) != 32 or len(set(uniform)) != 32
                or center != list(range(15744, 15872)) or len(row['contiguous_tiles']) != 1
                or row['sampling_seed'] != older['sampling_seed']
                or len(set(row['rows'])) != len(row['rows'])
                or len(set(older['rows'])) != len(older['rows'])
                or row['rows'] != sorted(set(center) | set(uniform))):
            raise ValueError('Frozen original row plan drift')
        positions = [row['rows'].index(i) for i in uniform]
        old_positions = [older['rows'].index(i) for i in uniform]
        identity = {}
        for field in [*MATRIX_WIDTHS, *STATIC_SHAPES]:
            ids = positions if field in MATRIX_WIDTHS else None
            old_ids = old_positions if field in MATRIX_WIDTHS else None
            left = field_bytes(payload, field, ids)
            if left != field_bytes(old_payload, field, old_ids):
                raise ValueError('Pre-Q2 join requires exact eight-field byte identity: ' + field)
            identity[field] = hashlib.sha256(left).hexdigest()
        inputs.append({'source': source, 'seed': seed, 'source_id': run['source_id'],
            'sampling_seed': row['sampling_seed'], 'grid': row['grid'],
            'all_RUN_relative': run_name, 'all_RUN_sha256': binding['sha256'],
            'all_payload_relative': str(payload.relative_to(Path(hf_root).resolve())),
            'all_payload_sha256': scope['sha256'], 'all_payload_bytes': payload.stat().st_size,
            'passive_RUN_relative': old_name, 'passive_RUN_sha256': rec['run_sha256'],
            'passive_payload_relative': str(old_payload.relative_to(Path(hf_root).resolve())),
            'passive_payload_sha256': old_scope['sha256'],
            'passive_payload_bytes': old_payload.stat().st_size,
            'uniform32_common_eight_fields_byte_SHA256': identity,
            'uniform32_preQ2_selected_byte_SHA256': hashlib.sha256(
                field_bytes(old_payload, 'q2_pre', old_positions)).hexdigest(),
            'panels': [{'name': 'contiguous128', 'row_ids': center,
                        'positions': [row['rows'].index(i) for i in center], 'preQ2_saved': False},
                       {'name': 'uniform32', 'row_ids': uniform, 'positions': positions,
                        'passive_positions': old_positions, 'preQ2_saved': True}],
            'panel_row_overlap': len(set(center) & set(uniform))})
    return inputs


def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    expected = {'id': STUDY_ID, 'block': BLOCK, 'steps': [STEP], 'sources': [0, 12],
        'expert': 'high_noise_model', 'panels': ['contiguous128', 'uniform32'],
        'max_cpu_threads': 1, 'max_RSS_bytes': 4 << 30, 'max_wall_seconds': 180,
        'max_output_bytes': 1 << 19, 'max_CPU_matmul_MAC': MAX_MAC,
        'tanh_Gaussian_quadrature_order': 32, 'automatic_retry': False,
        'new_W1_row_replay_calls': 0, 'native_QDQ_calls': 0, 'GPU_calls': 0,
        'model_calls': 0, 'fits': 0, 'weight_decompositions': 0, 'new_banks': 0,
        'methods': METHODS, 'saved_preQ2_panels': ['source00/uniform32', 'source12/uniform32']}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Scope differs from fixed four-panel mean diagnostic')
    required = {'tools/e0_even_residual_mean_cpu_discriminator.py',
                'tests/test_e0_even_residual_mean_cpu_discriminator.py',
                'tools/e0_vector_curve_cpu_feasibility.py', ALL_MANIFEST, PASSIVE_MANIFEST, CACHE_MANIFEST,
                'tools/ffnds_observation_capture.py', 'tools/e0_all_block_tangent_capture.py',
                'tools/e0_contiguous_capture.py', 'tools/ffnds_observation_common.py',
                '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py'}
    if not required <= set(plan['bound_sources']):
        raise ValueError('Missing exact software/native capture/asset lineage bindings')
    for name, sha in plan['bound_sources'].items():
        if archive.digest(relative_path(ROOT, name)) != sha:
            raise ValueError('Frozen source checksum drift: ' + name)
    inputs = discover_inputs(plan['hf_root'])
    if inputs != plan['inputs']:
        raise ValueError('Frozen original panels or cross-archive join drift')
    cache = json.loads((ROOT / CACHE_MANIFEST).read_text())
    weight = archive.resolve_original_shard(plan, cache)
    if (weight.stat().st_size != plan['weight_shard_bytes']
            or archive.digest(weight) != plan['weight_shard_sha256'] or not zipfile.is_zipfile(weight)):
        raise ValueError('Original one-block CPU mmap shard size/SHA/ZIP drift')
    if not Path(plan['canonical_output']).resolve().is_relative_to(
            (Path(plan['hf_root']) / 'ffnds-derived').resolve()):
        raise ValueError('Fixed output must remain inside HF derived cache')
    return plan, inputs, weight


class Work:
    """Precharge every explicit matrix/vector product; reductions are separate."""
    def __init__(self, maximum=MAX_MAC, fence=lambda: None):
        self.maximum, self.fence, self.mac, self.products = maximum, fence, 0, 0
        self.completed_mac, self.completed_products = 0, 0

    def dot(self, a, b):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('Compatible two-dimensional product required')
        cost = int(a.shape[0] * a.shape[1] * b.shape[1])
        if self.mac + cost > self.maximum:
            raise RuntimeError('Fixed CPU MAC budget would be exceeded')
        self.mac += cost
        self.products += 1
        self.fence()
        result = a @ b
        self.completed_mac += cost
        self.completed_products += 1
        return result


def tanh_gelu(x, np):
    return .5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))


def validate_moments(mu, var, np):
    if mu.shape != var.shape or not np.isfinite(mu).all() or not np.isfinite(var).all() or (var < 0).any():
        raise ValueError('Finite equal-shape means and nonnegative population variances required')


def erf_gaussian_mean(mu, var, np):
    """E[X Phi(X)], X~N(mu,var); exact analytic identity, FP64 evaluation."""
    validate_moments(mu, var, np)
    s = np.sqrt(1 + var)
    k = mu / s
    cdf = np.fromiter((.5 * math.erfc(-float(v) / math.sqrt(2)) for v in k.flat),
                      dtype=np.float64, count=k.size).reshape(k.shape)
    density = np.exp(-.5 * k*k) / math.sqrt(2 * math.pi)
    return mu * cdf + var / s * density


def tanh_gaussian_mean(mu, var, np):
    """Fixed32-node estimate; quadrature error remains part of this comparator."""
    validate_moments(mu, var, np)
    std = np.sqrt(2 * var)
    value = np.zeros_like(mu)
    for node, weight in zip(GH_NODES, GH_WEIGHTS):
        value += weight * tanh_gelu(mu + std * node, np)
    return value


def moments(x, np):
    mean = x.mean(axis=0)
    return mean, ((x - mean)**2).mean(axis=0)


def input_moments(mean_q, variance_q, w1, b1, work, np):
    """Tiny synthetic helper and real chunk kernel; no rowwise W1 replay."""
    validate_moments(mean_q, variance_q, np)
    return work.dot(mean_q, w1.T) + b1, work.dot(variance_q, (w1*w1).T)


def coefficient_means(arrays, scales, predicted_mu, predicted_var, np):
    mu, var = moments(arrays['z'], np)
    values = [erf_gaussian_mean(mu, var, np) / scales,
              tanh_gaussian_mean(mu, var, np) / scales,
              erf_gaussian_mean(predicted_mu, predicted_var, np) / scales,
              (tanh_gelu(arrays['z'], np) / scales).mean(axis=0),
              (arrays['gelu'] / scales).mean(axis=0), arrays['q2_post'].mean(axis=0)]
    names = list(METHODS)
    if 'q2_pre' in arrays:
        names.append('empirical_saved_preQ2')
        values.append(arrays['q2_pre'].mean(axis=0))
    means = np.stack(values)
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite conditional coefficient means')
    return names, means, mu, var


def array_hash(x, np):
    return hashlib.sha256(np.asarray(x, dtype='<f8', order='C').tobytes()).hexdigest()


def scalar_description(x, np):
    return {'sum': float(x.sum()), 'L2_squared': float((x*x).sum()),
            'max_abs': float(np.max(np.abs(x))), 'FP64_array_sha256': array_hash(x, np)}


def mean_error(prediction, reference, gate, count, np):
    result = {}
    for label, g in [('raw', 1.), ('FP64_current_gate', gate)]:
        p, r = prediction * g, reference * g
        e = p - r
        energy, err = float((r*r).sum()), float((e*e).sum())
        result[label] = {'mean_vector_error_L2_squared': err,
            'mean_vector_reference_L2_squared': energy,
            'relative_mean_vector_SSE': err / energy if energy > 0 else None,
            'error_reference_dot': float((e*r).sum()),
            'panel_constant_component_error_energy': count * err,
            'scope': 'ONE_PANEL_MEAN_VECTOR; GATE_IS_REAL_FP64_CHANNEL_WEIGHT_NOT_NATIVE_RESIDUAL_WRITEBACK'}
    return result


def summarize_panel(names, means, readouts, arrays, predicted_mu, predicted_var, actual_mu, actual_var, np):
    count = len(arrays['z'])
    y = dict(zip(names, readouts))
    c = dict(zip(names, means))
    gate, b2 = arrays['gate'].reshape(-1), arrays['b2']
    native = arrays['ffn_output'].mean(axis=0)
    post = y['empirical_saved_postQ2']
    comparisons = {}
    pairs = [
        ('Gaussian_tanh_minus_empirical_continuous', METHODS[1], METHODS[3]),
        ('Gaussian_erf_minus_empirical_continuous', METHODS[0], METHODS[3]),
        ('Gaussian_erf_minus_Gaussian_tanh', METHODS[0], METHODS[1]),
        ('preW1_diagonal_minus_postW1_moment_erf', METHODS[2], METHODS[0]),
        ('saved_gelu_over_Smooth2_minus_continuous', METHODS[4], METHODS[3]),
        ('saved_postQ2_minus_gelu_over_Smooth2_COMBINED', METHODS[5], METHODS[4])]
    if 'empirical_saved_preQ2' in y:
        pairs += [('saved_preQ2_minus_gelu_over_Smooth2_FP32divide_BF16cast', 'empirical_saved_preQ2', METHODS[4]),
                  ('saved_postQ2_minus_saved_preQ2_Q2_ONLY', METHODS[5], 'empirical_saved_preQ2')]
    for label, a, b in pairs:
        comparisons[label] = mean_error(y[a], y[b], gate, count, np)
        comparisons[label]['coefficient_mean_difference_L2_squared'] = float(((c[a]-c[b])**2).sum())
    deployment_target = {}
    for name in names:
        deployment_target[name] = mean_error(y[name] + b2, native, gate, count, np)
    deployment_target['known_b2_only'] = mean_error(b2, native, gate, count, np)
    deployment_target['decoded_postQ2_plus_b2'] = mean_error(post + b2, native, gate, count, np)
    return {'rows': count,
        'coefficient_mean_statistics': {n: scalar_description(v, np) for n, v in c.items()},
        'signed_W2_bias_free_mean_statistics': {n: scalar_description(v, np) for n, v in y.items()},
        'known_b2': scalar_description(b2, np), 'b1_included_inside_GELU': scalar_description(arrays['b1'], np),
        'native_saved_FFN_mean': scalar_description(native, np),
        'stage_bridge_mean_differences': comparisons,
        'means_vs_actual_saved_FFN_mean_b2_added_once': deployment_target,
        'preW1_mean_prediction_minus_saved_z_mean': scalar_description(predicted_mu-actual_mu, np),
        'preW1_diagonal_variance_minus_saved_z_population_variance': scalar_description(predicted_var-actual_var, np),
        'saved_z_mean_statistics': scalar_description(actual_mu, np),
        'saved_z_population_variance_statistics': scalar_description(actual_var, np),
        'predicted_preW1_diagonal_variance_statistics': scalar_description(predicted_var, np),
        'preQ2_availability': 'EXACT_SAVED_OLD_ARCHIVE_BYTE_JOIN' if 'q2_pre' in arrays else
            'NOT_SAVED_FOR_CONTIGUOUS128; COMBINED_CAST_AND_Q2_REMAINDER_ONLY',
        'interpretation': 'Covariance/mean mismatch includes BF16 W1 rounding; Gaussian/tanh error includes fixed quadrature; signed mean agreement alone establishes neither tokenwise accuracy nor full-M DC, transport or video quality.'}


def load_panel(path, positions, np, fields=None):
    names = list(MATRIX_WIDTHS) + list(STATIC_SHAPES) if fields is None else fields
    result = {}
    _, descriptors = archive.header(path)
    for name in names:
        ids = positions if name in MATRIX_WIDTHS or name == 'q2_pre' else None
        raw = field_bytes(path, name, ids)
        shape = descriptors[name]['shape']
        shape = [len(positions), shape[1]] if ids is not None else shape
        bits = np.frombuffer(raw, dtype='<u2').astype('<u4') << 16
        result[name] = bits.view('<f4').reshape(shape).astype(np.float64)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite original saved panel')
    return result


def require_registration(plan_path):
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    allowed = [s for s in scopes if s.get('id') == STUDY_ID
               and s.get('protocol_sha256') == archive.digest(plan_path)
               and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE']]
    if len(allowed) != 1:
        raise RuntimeError('Root must register this exact CPU protocol SHA once before numerical execution')


def save_result(out, result, cap):
    data = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    existing = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if existing + len(data) > cap:
        raise RuntimeError('Fixed output byte cap reached')
    temp = out / 'RUN.json.tmp'
    temp.write_bytes(data)
    temp.replace(out / 'RUN.json')


def execute(plan_path, plan, inputs, weight_path):
    require_registration(plan_path)
    if 'numpy' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Fresh numerical process required for single-thread environment binding')
    out = Path(plan['canonical_output'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'id': STUDY_ID, 'status': 'STARTED_NOT_ACCEPTED',
        'protocol_sha256': archive.digest(plan_path), 'cases': [],
        'model_calls': 0, 'GPU_calls': 0, 'native_QDQ_calls': 0, 'new_W1_row_replay_calls': 0,
        'fits': 0, 'weight_decompositions': 0, 'new_banks': 0, 'automatic_retry': False,
        'quality_accepted': False, 'full_M_DC_measured': False, 'CPU_matmul_MAC': 0,
        'weight_shard_sha256': plan['weight_shard_sha256'], 'input_bindings': inputs}
    work = None
    def fence():
        result['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if result['peak_RSS_bytes'] > plan['max_RSS_bytes'] or time.monotonic()-started > plan['max_wall_seconds']:
            raise RuntimeError('Fixed CPU RSS/time fence reached')
    def timeout(signum, frame):
        raise TimeoutError('Frozen180second numerical phase elapsed')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, plan['max_wall_seconds'])
    try:
        save_result(out, result, plan['max_output_bytes'])
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        threads = ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                   'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
        for key in threads:
            os.environ[key] = '1'
        import numpy as np
        import torch  # weights_only CPU mmap/casting; no model or CUDA APIs.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        work = Work(fence=fence)
        result['environment'] = {'numpy': np.__version__, 'torch': torch.__version__,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {k: os.environ[k] for k in threads}, 'CUDA_VISIBLE_DEVICES': ''}
        params = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)['high_noise_model']
        up, down = params['blocks.10.ffn.0'], params['blocks.10.ffn.2']
        w1, w2, smooth = up['weight_qdq'], down['weight_qdq'], down['smooth_scale']
        if (tuple(w1.shape) != (H, D) or tuple(w2.shape) != (D, H)
                or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
                or smooth.dtype != torch.float32
                or any(t.device.type != 'cpu' for t in [w1, w2, smooth])):
            raise ValueError('Original CPU BF16 W1/W2 and FP32 Smooth2 required')
        scales = smooth.reshape(-1).double().numpy()
        if scales.shape != (H,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Positive finite original Smooth2 required')
        panels, identities, mean_q, var_q = [], [], [], []
        for record in inputs:
            for panel in record['panels']:
                arrays = load_panel(relative_path(plan['hf_root'], record['all_payload_relative']), panel['positions'], np)
                if panel['preQ2_saved']:
                    arrays.update(load_panel(relative_path(plan['hf_root'], record['passive_payload_relative']),
                                             panel['passive_positions'], np, ['q2_pre']))
                mean, variance = moments(arrays['q1_post'], np)
                panels.append(arrays)
                mean_q.append(mean)
                var_q.append(variance)
                identities.append({'source': record['source'], 'seed': record['seed'],
                    'source_id': record['source_id'], 'step': STEP, 'block': BLOCK, 'panel': panel['name'],
                    'row_ids': panel['row_ids'], 'preQ2_saved': panel['preQ2_saved']})
                fence()
        mean_q, var_q = np.stack(mean_q), np.stack(var_q)
        bias = np.stack([a['b1'] for a in panels])
        predicted_mu, predicted_var = np.empty((4, H)), np.empty((4, H))
        for start in range(0, H, 256):
            stop = min(start+256, H)
            block = w1[start:stop].double().numpy()
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite original W1 chunk')
            pm, pv = input_moments(mean_q, var_q, block, bias[:, start:stop], work, np)
            predicted_mu[:, start:stop], predicted_var[:, start:stop] = pm, pv
            fence()
        del block
        all_means, descriptions = [], []
        for j, arrays in enumerate(panels):
            names, means, mu, variance = coefficient_means(arrays, scales, predicted_mu[j], predicted_var[j], np)
            all_means.append(means)
            descriptions.append((names, mu, variance))
            fence()
        coefficient_stack = np.concatenate(all_means)
        if coefficient_stack.shape != (26, H):
            raise ValueError('Exactly26 frozen signed mean readouts required')
        readout = np.empty((26, D))
        for start in range(0, D, 256):
            stop = min(start+256, D)
            block = w2[start:stop].double().numpy()
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite original W2 chunk')
            readout[:, start:stop] = work.dot(coefficient_stack, block.T)
            fence()
        del block
        offset = 0
        for j, (arrays, identity, desc) in enumerate(zip(panels, identities, descriptions)):
            names, mu, variance = desc
            n = len(names)
            item = summarize_panel(names, all_means[j], readout[offset:offset+n], arrays,
                                   predicted_mu[j], predicted_var[j], mu, variance, np)
            result['cases'].append(dict(identity, analysis=item))
            offset += n
            fence()
            result.update(CPU_matmul_MAC=work.mac, explicit_matmul_calls=work.products)
            save_result(out, result, plan['max_output_bytes'])
        if work.mac != MAX_MAC or len(result['cases']) != 4 or offset != 26:
            raise RuntimeError('Frozen four-panel work closure failed')
        result['status'] = 'PASS_SAVED_PANEL_CONDITIONAL_MEAN_DISCRIMINATOR_NOT_VIDEO_OR_COMPUTE_ACCEPTANCE'
    except BaseException as error:
        result.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if work is not None:
            result.update(CPU_matmul_MAC=work.mac, explicit_matmul_calls=work.products,
                          completed_CPU_matmul_MAC=work.completed_mac,
                          completed_explicit_matmul_calls=work.completed_products)
        result.update(elapsed_seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            timing_scope='NUMERICAL_PHASE_WITH_IMPORTS_CPU_MMAP; HASH_PREFLIGHT_SEPARATE',
            MAC_accounting='CPU_matmul_MAC precharges attempted products; completed counts update after return; elementwise work, reductions and copies are additional')
        save_result(out, result, plan['max_output_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--execute-cpu', action='store_true')
    args = parser.parse_args()
    plan, inputs, weight = preflight(args.protocol)
    print(json.dumps({'status': 'HASH_HEADER_SELECTED_BYTES_PREFLIGHT_ONLY', 'sources': len(inputs),
                      'panels': 4, 'numerical_work': False, 'GPU_initialized': False}), flush=True)
    if args.execute_cpu:
        execute(args.protocol, plan, inputs, weight)


if __name__ == '__main__':
    main()
