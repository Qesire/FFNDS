"""CPU-only posthoc finite-map and fixed-direction oracle analysis.

No parameter fitting, learned-model calls, GPU work or corrected trajectories.
Projection bounds describe this fixed state, not an executable repair policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection_bounds(tensors, h):
    import torch
    a, b = [tensors[k].double().reshape(tensors[k].shape[0], -1) for k in ['z_dense', 'z_student']]
    dense, student = [tensors[k].double().reshape(a.shape) for k in ['v_dense', 'v_student']]
    y = (b - a) / h - (student - dense)
    y_energy = float((y * y).sum())
    target_mean = y.mean(dim=1, keepdim=True)
    yc = y - target_mean
    center = lambda value: value - value.mean(dim=1, keepdim=True)
    bias_energy = float((target_mean ** 2).sum()) * y.shape[1]
    student_energy = float((student ** 2).sum())
    flow_projection = float((student * y).sum()) ** 2 / student_energy if student_energy else 0.
    bases = [center(student).reshape(-1), center(b).reshape(-1)]
    gram = torch.stack([torch.stack([(u * v).sum() for v in bases]) for u in bases])
    rhs = torch.stack([(v * yc.reshape(-1)).sum() for v in bases])
    # Normalize each direction before two-pass Gram-Schmidt. A scale-based
    # eigencut can discard a tiny but functionally decisive direction and
    # incorrectly exclude the FULL correction span.
    orthogonal, residual_norms = [], []
    for vector in bases:
        norm = float((vector * vector).sum()) ** .5
        if norm == 0:
            residual_norms.append(0.)
            continue
        value = vector / norm
        for _ in range(2):
            for previous in orthogonal:
                value = value - (value * previous).sum() * previous
        residual_norm = float((value * value).sum()) ** .5
        residual_norms.append(residual_norm)
        if residual_norm > 0:
            orthogonal.append(value / residual_norm)
    projected = sum(float((direction * yc.reshape(-1)).sum()) ** 2 for direction in orthogonal)
    flow_center_energy = float((bases[0] ** 2).sum())
    flow_center_projection = float(rhs[0]) ** 2 / flow_center_energy if flow_center_energy else 0.
    candidates = {'flow_scale_only': flow_projection,
                  'channel_bias_only': bias_energy,
                  'channel_bias_and_flow_scale': bias_energy + flow_center_projection,
                  'channel_bias_and_flow_and_latent_scale': bias_energy + projected}
    report = {}
    for key, explained in candidates.items():
        residual = y_energy - explained
        if residual < -1e-9 * max(y_energy, 1e-30):
            raise ValueError('Projection energy exceeds target')
        residual = max(residual, 0.)
        report[key] = {'ideal_next_SSE_oracle_residual_estimate': h * h * residual,
                       'fraction_of_uncorrected_ideal_next_SSE': residual / y_energy if y_energy else None,
                       'target_projection_fraction': explained / y_energy if y_energy else None}
    return {'target_energy': y_energy, 'centered_2x2_gram': gram.tolist(),
            'centered_basis_target_dot': rhs.tolist(), 'centered_basis_numerical_rank': len(orthogonal),
            'normalized_reorthogonalization_residual_norms': residual_norms,
            'channel_bias_rank': a.shape[0], 'bounds': report,
            'oracle_only': True, 'coefficients_fitted_or_deployed': False,
            'certified_lower_bound': False, 'strict_family_exclusion_allowed': False,
            'interpretation': 'FP64 numerical estimate of the exact-span oracle optimum, with normalized two-pass reorthogonalization and no scale-based rank cutoff. Ill-conditioned ranks are not certified and cannot support strict exclusion. Uses teacher target offline; no online reachability, native BF16 correction or held-out quality guarantee.'}


def analyze(study, protocol, *, allow_partial=False):
    import torch
    from safetensors.torch import load_file
    from e0_whole_flow_counterfactual import flow_split_geometry, native_map_split_geometry
    from e0_paired_wan_controller import tensor_identity
    if torch.cuda.is_initialized():
        raise RuntimeError('Analysis must run in a CPU-only process')
    torch.set_num_threads(8)
    plan = json.loads(protocol.read_text())
    ledger = json.loads((study / 'STUDY.json').read_text())
    if ledger['protocol_sha256'] != sha(protocol) or ledger['id'] != plan['id']:
        raise ValueError('Fixed study protocol identity differs')
    ordered = [p['id'] for p in plan['pairs']]
    attempts = [a['pair'] for a in ledger['attempts']]
    if attempts != ordered[:len(attempts)] or len(attempts) > len(ordered):
        raise ValueError('Unique ordered fixed pair identities required')
    consumer_paths = [Path(__file__).with_name(name) for name in
                      ['e0_whole_flow_counterfactual.py', 'e0_paired_wan_controller.py', 'e0_full_depth_trace_contract.py']]
    for path in consumer_paths:
        if sha(path) != plan['driver_sources_sha256']['tools/' + path.name]:
            raise ValueError('Frozen mathematical consumer changed: ' + path.name)
    complete = len(ledger['attempts']) == 2 and all(a['status'] == 'PASS_NATIVE_LATENT_IDENTITY_AND_FLOW_SPLIT_NOT_QUALITY' for a in ledger['attempts'])
    if not allow_partial and not complete:
        raise ValueError('Whole fixed two-source scope has not passed')
    sources = []
    for attempt in ledger['attempts']:
        if attempt['status'] != 'PASS_NATIVE_LATENT_IDENTITY_AND_FLOW_SPLIT_NOT_QUALITY':
            continue
        out = study / attempt['pair']
        if sha(out / 'RUN.json') != attempt['RUN_sha256']:
            raise ValueError('Final native receipt changed')
        run = json.loads((out / 'RUN.json').read_text())
        trace = json.loads((out / 'TRACE.json').read_text())
        pair = next(p for p in plan['pairs'] if p['id'] == attempt['pair'])
        if (run['status'] != attempt['status'] or run['id'] != plan['id'] or run['protocol_sha256'] != sha(protocol)
                or run['pair'] != pair or run['source_id'] != pair['source_id'] or run['seed'] != pair['seed']
                or trace['source_index'] != pair['source_index'] or trace['seed'] != pair['seed']):
            raise ValueError('Actual source/seed/run identity differs')
        if (sha(out / 'TRACE.json') != run['trace_sha256'] or len(run['counterfactual_records']) != 4
                or sorted(r['step'] for r in run['counterfactual_records']) != [0, 1, 2, 3]):
            raise ValueError('Native trace or whole-flow coverage differs')
        steps = []
        for record in run['counterfactual_records']:
            binding = record['saved_tensors']
            path = out / binding['path']
            if sha(path) != binding['sha256'] or record['status'] != 'PASS_FLOW_AND_NATIVE_MAP_IDENTITIES':
                raise ValueError('Whole-state artifact changed/incomplete')
            tensors = load_file(str(path), device='cpu')
            if set(tensors) != set(binding['tensors']):
                raise ValueError('Whole-state tensor inventory differs')
            for key, value in tensors.items():
                if binding['tensors'][key] != {'shape': list(value.shape), 'dtype': str(value.dtype)}:
                    raise ValueError('Whole-state shape/dtype differs')
            for key, label in [('v_dense', 'dense_teacher'), ('v_student', 'candidate_student'), ('v_dense_at_student', 'dense_student')]:
                if tensor_identity(tensors[key]) != record['flow_hashes'][label]:
                    raise ValueError('Actual full-flow identity differs')
            step = record['step']
            old = next(s for s in trace['steps'] if s['step'] == step)
            for branch, before, after in [('teacher', 'z_dense', 'z_dense_after'), ('student', 'z_student', 'z_student_after')]:
                identity = tensor_identity(tensors[before])
                if (identity != record['input_hashes'][branch]
                        or identity != old['boundary_hashes'][branch + '_after_step_pre']
                        or tensor_identity(tensors[after]) != old['boundary_hashes'][branch + '_after_step_post']):
                    raise ValueError('Actual native input/after-state identity differs')
            h = old['h']
            geometry = flow_split_geometry(*(tensors[k] for k in ['z_dense', 'z_student', 'v_dense', 'v_student', 'v_dense_at_student']), h=h)
            native = native_map_split_geometry(*(tensors[k] for k in ['z_dense_after', 'z_student_after', 'z_dense_at_student_after']))
            for key, value in geometry.items():
                if type(value) in [int, float] and not math.isclose(value, record['geometry'][key], rel_tol=1e-9, abs_tol=1e-14):
                    raise ValueError('CPU/GPU full-flow geometry disagrees: ' + key)
            for key, value in native.items():
                if type(value) in [int, float] and not math.isclose(value, record['native_map_geometry'][key], rel_tol=1e-9, abs_tol=1e-14):
                    raise ValueError('CPU/GPU native geometry disagrees: ' + key)
            for before, flow, after in [('z_dense', 'v_dense', 'z_dense_after'), ('z_student', 'v_student', 'z_student_after'),
                                        ('z_student', 'v_dense_at_student', 'z_dense_at_student_after')]:
                expected = tensors[before].float() - tensors[flow].float() * old['sigma']
                if step < 3:
                    expected = expected + tensors[flow].float() * old['sigma_next']
                if not torch.equal(expected.to(tensors[before].dtype), tensors[after]):
                    raise ValueError('Actual native step_post arithmetic cannot be reproduced')
            native['dense_discrete_map_energy_gain'] = native['dense_map_response_SSE'] / geometry['delta_SSE'] if geometry['delta_SSE'] else None
            native['candidate_net_effect_on_next_SSE'] = native['candidate_effect_SSE'] + 2 * native['dense_map_candidate_effect_dot']
            steps.append({'step': step, 'tensor_sha256': binding['sha256'], 'ideal': geometry, 'native': native,
                          'fixed_online_direction_oracle': projection_bounds(tensors, h)})
        sources.append({'source_index': run['pair']['source_index'], 'seed': run['seed'],
                        'RUN_sha256': sha(out / 'RUN.json'), 'TRACE_sha256': sha(out / 'TRACE.json'),
                        'native_runtime_counts': run['native_runtime_counts'], 'cost_scope': run['cost_scope'],
                        'observation_wall_seconds': run['observation_wall_seconds'], 'setup_seconds': run['setup_seconds'],
                        'resource_report': run['assigned_GPU_resources'], 'steps': steps})
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU analysis unexpectedly initialized CUDA')
    return {'id': 'E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS_20261026',
            'status': 'COMPLETE_FIXED_SCOPE_NOT_QUALITY' if complete else 'PARTIAL_COMPLETED_SOURCES_ONLY',
            'protocol_sha256': sha(protocol), 'STUDY_sha256': sha(study / 'STUDY.json'),
            'analysis_source_sha256': sha(Path(__file__)), 'runs': sources,
            'mathematical_consumer_source_sha256': {p.name: sha(p) for p in consumer_paths},
            'new_learned_model_calls_by_analysis': 0, 'GPU_initialized': False,
            'fitted_or_deployed_parameters': False, 'new_videos': 0, 'quality_accepted': False,
            'causal_scope': 'Finite whole-model responses on observed state pairs; no earliest-layer cause, global stability or tested repair policy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.study, args.protocol, allow_partial=args.allow_partial), indent=2) + '\n')


if __name__ == '__main__':
    main()
