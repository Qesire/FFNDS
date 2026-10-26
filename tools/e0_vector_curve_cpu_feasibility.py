"""Frozen four-cluster saved-operand diagnostic. Default: hash/header preflight only.

No model, CUDA, native quantizer, fitting, polynomial bank, or generation is used.
Numerical CPU execution additionally requires the exact plan SHA in ACTIVE.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = 'E0_VECTOR_CURVE_CPU_FEASIBILITY_20261026'
ANCHORS = [0, 4607, 9215, 13823]
N, D, H, BLOCK = 32, 5120, 13824, 10
REQUIRED = {'q1_post': [48, D], 'z': [48, H], 'gelu': [48, H],
            'q2_pre': [48, H], 'q2_post': [48, H], 'gate': [1, D],
            'b1': [H], 'ffn_output': [48, D]}
MAX_MAC = 346030080  # 4*H*D + 4*N*D*16 + 4*4*5*N*4*D
SHARD_FILENAME = 'high_noise_model_block_10.pt'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def header(path):
    with Path(path).open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size > 1 << 20:
            raise ValueError('Unexpected operand header size')
        return size, json.loads(f.read(size))


def resolve_original_shard(plan, cache):
    """Resolve the private original path from a checksum-bound runtime manifest."""
    if plan['weight_shard_filename'] != SHARD_FILENAME:
        raise ValueError('Fixed original shard filename required')
    shard = next(s for s in cache['shards'] if s['expert'] == 'high_noise_model' and s['block'] == BLOCK)
    path = Path(shard['path'])
    if path.name != SHARD_FILENAME or not path.is_absolute() or shard['sha256'] != plan['weight_shard_sha256']:
        raise ValueError('Original shard manifest identity differs from fixed block')
    return path


def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    if (plan['id'] != STUDY_ID or plan['anchors'] != ANCHORS or plan['neighbors'] != 3
            or plan['block'] != BLOCK or plan['sources'] != [0, 12] or plan['steps'] != [0, 1]
            or plan['rows_per_case'] != N or plan['max_cpu_threads'] != 1
            or plan['max_RSS_bytes'] != 2 << 30 or plan['max_output_bytes'] != 1 << 20
            or plan['max_wall_seconds'] != 120 or plan['max_CPU_matmul_MAC'] != MAX_MAC):
        raise ValueError('Scope differs from fixed cheap diagnostic')
    required_sources = {'tools/e0_vector_curve_cpu_feasibility.py',
                        'research/20261025/E0_NATIVE_OBSERVATIONS.json', '.runtime/hif4_rebuilt_cache.json'}
    if not required_sources <= set(plan['bound_sources']):
        raise ValueError('Missing code/operand/native-cache lineage bindings')
    for name, expected in plan['bound_sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Bound source changed: ' + name)
    manifest = json.loads((ROOT / 'research/20261025/E0_NATIVE_OBSERVATIONS.json').read_text())
    cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    weight_path = resolve_original_shard(plan, cache)
    if digest(weight_path) != plan['weight_shard_sha256']:
        raise ValueError('Original single-shard lineage/checksum mismatch')
    if not zipfile.is_zipfile(weight_path):
        raise ValueError('CPU mmap requires the original torch.save ZIP shard')
    cases = []
    for source in plan['sources']:
        accepted = next(s for s in manifest['sources'] if s['index'] == source)
        run_path = Path(plan['hf_root']) / accepted['HF_relative'] / 'RUN.json'
        if digest(run_path) != accepted['run_sha256']:
            raise ValueError('Original observation receipt changed')
        run = json.loads(run_path.read_text())
        if run['status'] != 'PASS_PASSIVE_CAPTURE_NOOP':
            raise ValueError('Only accepted original passive operands')
        for step in plan['steps']:
            scope = next(s for s in run['scopes'] if s['block'] == BLOCK and s['step'] == step)
            path = (run_path.parent / 'observed' / scope['path']).resolve()
            if not path.is_relative_to((run_path.parent / 'observed').resolve()) or digest(path) != scope['sha256']:
                raise ValueError('Saved payload path/checksum drift')
            _, fields = header(path)
            if any(fields[k]['dtype'] != 'BF16' or fields[k]['shape'] != shape for k, shape in REQUIRED.items()):
                raise ValueError('Actual BF16 operand shape drift')
            rows = scope['row_plan']['uniform_rows']
            if len(rows) != N or len(set(rows)) != N:
                raise ValueError('Original fixed uniform32 required')
            cases.append({'source': source, 'step': step, 'seed': run['seed'], 'path': str(path),
                          'sha256': scope['sha256'], 'run_sha256': accepted['run_sha256'],
                          'row_ids': rows, 'positions': [scope['row_plan']['rows'].index(r) for r in rows]})
    return plan, cases, weight_path


def rank_neighbors(dots, norm2, anchors=ANCHORS, count=3):
    """Pure scalar tie policy; anchor rows are excluded from every neighbor list."""
    clusters = []
    for j, anchor in enumerate(anchors):
        if norm2[anchor] <= 0:
            raise ValueError('Zero representative row')
        scores = {i: float(dots[j][i]) / math.sqrt(norm2[anchor] * norm2[i])
                  for i in range(len(norm2)) if i not in anchors and norm2[i] > 0}
        chosen = sorted(scores, key=lambda i: (-abs(scores[i]), i))[:count]
        if len(chosen) != count:
            raise ValueError('Insufficient finite nonzero neighbor rows')
        ids = [anchor] + chosen
        clusters.append({'anchor': anchor, 'ids': ids,
                         'cosine': [1.] + [scores[i] for i in chosen],
                         'alpha': [1.] + [float(dots[j][i]) / norm2[anchor] for i in chosen]})
    return clusters


def cancellation_example():
    # phi(eps)-phi(-eps)=eps for real-arithmetic tanh-GELU, despite v+ + v- = 0.
    eps = 1e-3
    phi = lambda x: .5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))
    return {'epsilon': eps, 'centroid_merge_output': 0., 'actual_signed_pair_output': phi(eps)-phi(-eps),
            'expected_output': eps, 'relative_output_SSE_lost_by_merge': 1.,
            'status': 'ANALYTIC_CONTINUOUS_COUNTEREXAMPLE_NOT_A_WAN_MEASUREMENT'}


def load_fields(path, np):
    offset, descriptors = header(path)
    arrays = {}
    with Path(path).open('rb') as f:
        for name in REQUIRED:
            item = descriptors[name]; a, b = item['data_offsets']; f.seek(8 + offset + a)
            bits = np.frombuffer(f.read(b-a), dtype='<u2').astype('<u4') << 16
            arrays[name] = bits.view('<f4').reshape(item['shape']).astype(np.float64)
    return arrays


def ratio(a, b):
    return float(a / b) if b > 0 else None


def metrics(reference, error, coefficient_error, weights, gate, full_ffn, np):
    ref, err, full = reference * gate, error * gate, full_ffn * gate
    norm2 = ((weights * gate.reshape(-1, 1))**2).sum(axis=0)
    diagonal = float((coefficient_error**2 * norm2).sum())
    energy, sse = float((ref**2).sum()), float((err**2).sum())
    return {'reference_energy': energy, 'signed_error_SSE': sse,
            'diagonal_unsigned_error_energy': diagonal,
            'signed_over_diagonal_error': ratio(sse, diagonal), 'relative_SSE': ratio(sse, energy),
            'error_reference_dot': float((err * ref).sum()),
            'error_full_native_FFN_dot': float((err * full).sum()),
            'error_over_full_native_FFN_energy': ratio(sse, float((full**2).sum())),
            'bias_scope': 'PARTIAL_CLUSTER_BIAS_EXCLUDED; full-native denominator includes original global bias'}


def execute(plan_path, plan, cases, weight_path):
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    allowed = [s for s in active['experiment_authority']['current_user_authorized_scopes']
               if s.get('id') == STUDY_ID and s.get('protocol_sha256') == digest(plan_path)
               and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE']]
    if len(allowed) != 1:
        raise RuntimeError('Exact CPU scope must be registered once by root before numerical work')
    out = Path(plan['canonical_output']); out.mkdir(parents=True, exist_ok=False)
    result = {'id': STUDY_ID, 'status': 'STARTED_NOT_ACCEPTED', 'protocol_sha256': digest(plan_path),
              'cases': [], 'model_calls': 0, 'GPU_calls': 0, 'native_QDQ_calls': 0,
              'fits': 0, 'new_banks': 0, 'quality_accepted': False, 'CPU_matmul_MAC': 0,
              'source_cases': cases, 'weight_shard_sha256': plan['weight_shard_sha256'],
              'counterexample': cancellation_example()}
    started = time.monotonic()

    def save():
        data = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        if len(data) > plan['max_output_bytes']:
            raise RuntimeError('Fixed output cap exceeded')
        target = out / 'RUN.json'; temp = out / 'RUN.json.tmp'; temp.write_bytes(data); temp.replace(target)

    def fence():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux executor.
        result['peak_RSS_bytes'] = rss
        if rss > plan['max_RSS_bytes'] or time.monotonic()-started > plan['max_wall_seconds']:
            raise RuntimeError('Fixed CPU RSS/time budget reached')

    try:
        save()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            os.environ[key] = '1'
        import numpy as np
        import torch  # Tensor archive loader/cast only; no model/native imports or CUDA APIs.
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        params = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)['high_noise_model']
        up, down = params['blocks.10.ffn.0'], params['blocks.10.ffn.2']
        w1, w2 = up['weight_qdq'], down['weight_qdq']
        if (tuple(w1.shape) != (H, D) or tuple(w2.shape) != (D, H)
                or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
                or w1.device.type != 'cpu' or w2.device.type != 'cpu'):
            raise ValueError('Original BF16 CPU weight geometry required')
        smooth = down['smooth_scale'].reshape(-1).double().numpy()
        if smooth.shape != (H,) or not np.isfinite(smooth).all() or (smooth <= 0).any():
            raise ValueError('Actual positive Smooth2 required')
        reps = w1[ANCHORS].float().numpy().astype(np.float64)
        dots = np.empty((4, H), np.float64); norm2 = np.empty(H, np.float64)
        for start in range(0, H, 256):
            block = w1[start:start+256].float().numpy().astype(np.float64)
            dots[:, start:start+len(block)] = reps @ block.T
            norm2[start:start+len(block)] = (block**2).sum(axis=1)
            result['CPU_matmul_MAC'] += 4 * D * len(block)
            fence()
        if not np.isfinite(dots).all() or not np.isfinite(norm2).all():
            raise ValueError('Nonfinite original weights')
        clusters = rank_neighbors(dots, norm2); result['clusters'] = clusters
        ids = sorted({i for c in clusters for i in c['ids']}); pos = {i: j for j, i in enumerate(ids)}
        selected_up = w1[ids].float().numpy().astype(np.float64)
        selected_down = w2[:, ids].float().numpy().astype(np.float64)
        if not np.isfinite(selected_down).all():
            raise ValueError('Nonfinite original W2')
        phi = lambda a: .5*a*(1+np.tanh(math.sqrt(2/math.pi)*(a+.044715*a**3)))
        for case in cases:
            arrays = load_fields(case['path'], np); rows = case['positions']
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Nonfinite original saved operands')
            q, b1 = arrays['q1_post'][rows], arrays['b1'][ids]
            a = q @ selected_up.T + b1; result['CPU_matmul_MAC'] += N*D*len(ids)
            continuous = phi(a) / smooth[ids]
            item = {'source': case['source'], 'step': case['step'], 'seed': case['seed'],
                    'row_ids': case['row_ids'], 'clusters': []}
            for cluster in clusters:
                cols = [pos[i] for i in cluster['ids']]; chosen = cluster['ids']
                t = a[:, pos[cluster['anchor']]] - arrays['b1'][cluster['anchor']]
                curve = phi(t[:, None]*np.asarray(cluster['alpha'])+arrays['b1'][chosen]) / smooth[chosen]
                exact = continuous[:, cols]; pre = arrays['q2_pre'][rows][:, chosen]
                post = arrays['q2_post'][rows][:, chosen]; weights = selected_down[:, cols]
                coefficients = [exact, curve, post, post-pre, pre-exact]
                outputs = np.concatenate(coefficients, axis=0) @ weights.T
                result['CPU_matmul_MAC'] += 5*N*len(chosen)*D
                y0, yc, yn, eq, ep = np.split(outputs, 5, axis=0)
                record = {'anchor': cluster['anchor'], 'saved_z_vs_FP64_replay_SSE':
                          float(((arrays['z'][rows][:, chosen]-a[:, cols])**2).sum()),
                          'saved_gelu_over_smooth_vs_continuous_coeff_SSE':
                          float(((arrays['gelu'][rows][:, chosen]/smooth[chosen]-exact)**2).sum()),
                          'native_remainder_recomposition_SSE': float(((yn-y0-eq-ep)**2).sum()),
                          'metrics': {}}
                for name, ref, error, coeff_error in [
                        ('curve_minus_continuous', y0, yc-y0, curve-exact),
                        ('native_postQ2_minus_continuous', y0, yn-y0, post-exact),
                        ('saved_Q2_only_remainder', y0+ep, eq, post-pre),
                        ('preQ2_minus_continuous_remainder', y0, ep, pre-exact)]:
                    record['metrics'][name] = {}
                    for label, gate in [('raw', np.ones((1,D))), ('current_gate', arrays['gate'])]:
                        record['metrics'][name][label] = metrics(ref, error, coeff_error, weights, gate,
                                                              arrays['ffn_output'][rows], np)
                item['clusters'].append(record)
            result['cases'].append(item); fence(); save()
        if result['CPU_matmul_MAC'] > MAX_MAC or len(result['cases']) != 4:
            raise RuntimeError('Unexpected CPU work or missing frozen cases')
        result.update(status='PASS_CPU_FEASIBILITY_DESCRIPTION_NOT_METHOD_OR_QUALITY_ACCEPTANCE',
                      elapsed_seconds=time.monotonic()-started,
                      interpretation='Same weight-selected clusters across four exposed cases; no fitted bank, native replay, full-M or downstream causal claim')
    except BaseException as exc:
        result.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--execute-cpu', action='store_true')
    args = parser.parse_args(); plan, cases, weight_path = preflight(args.protocol)
    print(json.dumps({'status': 'HASH_AND_HEADER_PREFLIGHT_ONLY', 'cases': len(cases),
                      'numerical_work': False, 'GPU_initialized': False}), flush=True)
    if args.execute_cpu:
        execute(args.protocol, plan, cases, weight_path)


if __name__ == '__main__':
    main()
