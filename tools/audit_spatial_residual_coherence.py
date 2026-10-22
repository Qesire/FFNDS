"""Read actual OP2 pairs to test same-step spatial residual sharing on CPU.

No FFN forward, quantizer, weight loading, training or new tensor output.
Pair evidence is deliberately separate from an unexecuted full-stripe policy.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/20261022/R13_spatial_residual_audit_protocol.json'


def coordinates(row, grid):
    t, h, w = grid
    if not 0 <= row < t * h * w:
        raise ValueError('Native row outside the recorded grid')
    return row // (h * w), row // w % h, row % w


def pairs(selected, grid, kind):
    if selected != sorted(set(selected)):
        raise ValueError('Saved native coordinates must be sorted and unique')
    result = []
    for a, left in enumerate(selected):
        t, y, x = coordinates(left, grid)
        for b in range(a + 1, len(selected)):
            u, z, v = coordinates(selected[b], grid)
            if t != u:
                continue
            match = (abs(y - z) + abs(x - v) == 1 if kind == 'near'
                     else y == z and x // 4 == v // 4)
            if match:
                result.append((a, b))
    return result


def far_donor(selected, grid, target, minimum=8):
    t, y, x = coordinates(selected[target], grid)
    for offset, native in enumerate(selected):
        u, z, v = coordinates(native, grid)
        if u == t and abs(y - z) + abs(x - v) >= minimum:
            return offset
    return None


def scalar_summary(rows):
    if not rows:
        return {'directed_pairs': 0}
    keys = ['target_energy', 'donor_energy', 'cross', 'copy_SSE', 'FP64_UB1_SSE', 'UB1_SSE', 'BF16_copy_SSE']
    total = {k: math.fsum(r[k] for r in rows) for k in keys}
    coefficient = total['cross'] / total['donor_energy'] if total['donor_energy'] else 0.
    ratio = lambda a, b: a / b if b else None
    return {'directed_pairs': len(rows), **total,
        'FP64_copy_over_UB1': ratio(total['copy_SSE'], total['FP64_UB1_SSE']),
        'BF16_copy_over_UB1': ratio(total['BF16_copy_SSE'], total['UB1_SSE']),
        'copy_over_same_base_uncorrected': ratio(total['copy_SSE'], total['target_energy']),
        'pooled_residual_cosine': ratio(total['cross'], math.sqrt(total['target_energy'] * total['donor_energy'])),
        'descriptive_unconstrained_projection_coefficient': coefficient,
        'descriptive_projection_error_over_UB1': ratio(total['target_energy'] - coefficient * total['cross'], total['FP64_UB1_SSE']),
        'copy_beats_UB1_directed_pairs': sum(r['copy_SSE'] < r['FP64_UB1_SSE'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scalar-out', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.scalar_out.exists():
        raise RuntimeError('Preserve prior spatial audit')
    import torch
    from safetensors.torch import load_file
    from ar2_checkpoint import atomic_json, digest
    from audit_r12_saved_operands import verified_safetensors
    from ub1_native_data import load_ub1_collections
    torch.set_num_threads(8)
    started = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text())
    frozen = digest(PROTOCOL)
    paths = [ROOT / f'reports/worker_20261020/OP2_UB1_calibration24_shard{s}_RUN.json' for s in (0, 1)]
    records, bindings, identity = load_ub1_collections(paths, ROOT)
    all_rows, scopes, totals = [], [], defaultdict(list)
    payload_bytes = 0
    for step, block in protocol['positions']:
        for index in sorted(records):
            scope = next(s for s in records[index]['scopes'] if (s['step'], s['block']) == (step, block))
            if digest(PROTOCOL) != frozen:
                raise RuntimeError('Saved-data audit scope changed during execution')
            audit, _, _ = verified_safetensors(scope['path'], scope['sha256'])
            data = load_file(scope['path'], device='cpu')
            if (set(data) != {'V1', 'BaseY', 'StudentY', 'TeacherY', 'b2', 'gate'}
                    or any(x.dtype != torch.bfloat16 or not bool(torch.isfinite(x).all()) for x in data.values())
                    or any(data[k].shape != (256, 5120) for k in ('V1', 'BaseY', 'StudentY', 'TeacherY'))
                    or scope['native_V1_shape'] != [scope['full_rows'], 5120]
                    or scope['native_V1_stride'] != [5120, 1]):
                raise RuntimeError('Original saved operand geometry/dtype differs')
            payload_bytes += audit['bytes']
            gate = data['gate'].reshape(1, 5120)
            x, teacher, base, student = [data[k].double() for k in ('V1', 'TeacherY', 'BaseY', 'StudentY')]
            residual = {'full_output': teacher, 'raw_missing': teacher - base, 'innovation': teacher - student}
            bases = {'full_output': torch.zeros_like(base), 'raw_missing': base, 'innovation': student}
            true_gated = (data['TeacherY'] * gate).double()
            old_gated = (data['StudentY'] * gate).double()
            true_UB1_error = (true_gated - old_gated).square().sum(1)
            ideal_UB1_error = ((teacher - student) * gate.double()).square().sum(1)
            selected, grid = scope['selected_rows'], scope['grid']
            selected_set = set(selected)
            local_rows = []
            counts = {}
            for kind in ('near', 'stripe'):
                edges = pairs(selected, grid, kind)
                counts[kind] = len(edges)
                for left, right in edges:
                    for donor, target in ((left, right), (right, left)):
                        choices = [(kind, donor)]
                        if kind == 'near':
                            far = far_donor(selected, grid, target)
                            if far is not None:
                                choices.append(('far', far))
                        for graph, source in choices:
                            dx = float((x[target] - x[source]).square().sum())
                            cosine = float(torch.dot(x[target], x[source]) / (x[target].norm() * x[source].norm()).clamp_min(1e-30))
                            for family, values in residual.items():
                                a, b = values[source] * gate[0].double(), values[target] * gate[0].double()
                                aa, bb, cross = float(a.square().sum()), float(b.square().sum()), float((a * b).sum())
                                loss = float((b - a).square().sum())
                                if not math.isclose(loss, aa + bb - 2 * cross, rel_tol=1e-10, abs_tol=1e-7):
                                    raise RuntimeError('Native-output residual difference identity failed')
                                # Explicit CPU BF16 add and gate-product proxy;
                                # full original residual-stream x is unavailable.
                                candidate = bases[family][target].bfloat16() + values[source].bfloat16()
                                bf16_loss = float(((candidate * gate[0]).double() - true_gated[target]).square().sum())
                                row = {'index': index, 'step': step, 'graph': graph, 'family': family,
                                    'donor_native_row': selected[source], 'target_native_row': selected[target],
                                    'input_L2_squared': dx, 'input_cosine': cosine,
                                    'donor_energy': aa, 'target_energy': bb, 'cross': cross,
                                    'copy_SSE': loss, 'FP64_UB1_SSE': float(ideal_UB1_error[target]),
                                    'UB1_SSE': float(true_UB1_error[target]), 'BF16_copy_SSE': bf16_loss}
                                local_rows.append(row)
                                totals[step, graph, family].append(row)
            scopes.append({'index': index, 'step': step, 'block': block, 'grid': grid,
                'declared_main_width': scope['declared_main_width'], 'payload_locator': audit['locator'],
                'payload_sha256': audit['sha256'], 'decoded_tensor_bindings': audit['tensors'],
                'pair_counts': counts, 'complete_observed_horizontal4_stripes': sum(
                    1 for t in range(grid[0]) for y in range(grid[1]) for z in range(grid[2] // 4)
                    if all(t * grid[1] * grid[2] + y * grid[2] + 4 * z + k in selected_set for k in range(4))),
                'per_family': {f'{g}/{f}': scalar_summary([r for r in local_rows if r['graph'] == g and r['family'] == f])
                               for g in ('near', 'stripe', 'far') for f in residual}})
            all_rows.extend(local_rows)
            print(f'CPU spatial saved-data step{step} source{index:02d}: near{counts["near"]}, stripe{counts["stripe"]}', flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('Spatial saved-data audit initialized CUDA')
    sources = ['tools/audit_spatial_residual_coherence.py', 'tools/ub1_native_data.py',
               'tools/native_teacher_capture.py', 'tools/collect_ub1_native_states.py',
               '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py']
    args.scalar_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.scalar_out, {'protocol_sha256': frozen, 'rows': all_rows})
    result = {'id': protocol['id'], 'status': 'COMPLETED_SAVED_NATIVE_OUTPUT_SPATIAL_AUDIT_CPU_ONLY',
        'evidence_level': 'SAMPLED_LOCAL_GATED_OUTPUT_GEOMETRY_NOT_EXECUTED_DYNAMIC_ACTION',
        'protocol_sha256': frozen, 'source_sha256': {p: digest(ROOT / p) for p in sources},
        'original_collection_bindings': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings],
        'original_policy_sha256': identity['original_UB1_policy_sha256'], 'source_count': len(records),
        'payloads_read': len(scopes), 'payload_bytes_checked': payload_bytes,
        'scopes': scopes, 'paired_scalar_evidence': {'locator': str(args.scalar_out.relative_to(ROOT)),
            'sha256': digest(args.scalar_out), 'bytes': args.scalar_out.stat().st_size, 'rows': len(all_rows)},
        'summaries': [{'step': s, 'graph': g, 'family': f, **scalar_summary(v)} for (s, g, f), v in sorted(totals.items())],
        'new_model_generations': 0, 'models_loaded': 0, 'fitted_parameters': 0,
        'CUDA_initialized': False, 'new_tensor_writes': 0,
        'free_video_or_FFN_speed_acceptance': False,
        'elapsed_seconds': time.monotonic() - started,
        'scope': protocol['limits']}
    atomic_json(args.output, result)
    print('Completed spatial residual audit;0 model generations.', flush=True)


if __name__ == '__main__':
    main()
