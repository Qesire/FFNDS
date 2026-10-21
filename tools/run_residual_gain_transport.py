"""R7D2: fixed two-parameter local-vs-flow objective experiment."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from collect_ub1_native_states import policy_alias, verify_runtime
from historical_source_snapshot import HistoricalSources
from residual_gain_transport import NativeResidualGain, box_quadratic
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from run_cross_layer_transport import references, SOURCES as BASE_SOURCES
from run_supplement_teacher_flow import exact_native_latent, error_stats
from variable_teacher_capture import TeacherRecords

METHOD = 'R7D2_RESIDUAL_GAIN_OBJECTIVE'
PROTOCOL = ROOT / 'research/20261021/R7D2_residual_gain_protocol.json'
TRAIN = [0, 5, 10]
EPSILON = .125


def source_binding():
    names = sorted(set(BASE_SOURCES + ['run_residual_gain_transport.py', 'residual_gain_transport.py']))
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in names}


def capture(pipe, runtime, teacher, gains, label, index, refs, folder):
    if folder.exists():
        raise RuntimeError('Preserve every attempted gain experiment')
    folder.mkdir(parents=True)
    dq, ub = refs['DQ'][index], refs['UB1'][index]
    sample = dq['case']['input']
    if sample['seed'] != 42 or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
        raise RuntimeError('Original frozen image/seed changed')
    scheduler = pipe.runner.scheduler
    old_pre, old_post = scheduler.step_pre, scheduler.step_post
    scheduler.generator = None
    runtime.begin_request()
    before = dict(runtime.counters)
    inputs, steps = {}, []

    def pre(step_index):
        result = old_pre(step_index)
        value = scheduler.latents.detach().float().cpu().contiguous()
        if not torch.equal(value, dq['payloads'][step_index]['input']):
            raise RuntimeError('Gain experiment did not receive original DQ input')
        inputs[step_index] = tensor_sha(value)
        return result

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        result = old_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        control = ub['payloads'][step]
        match = label == 'identity' or step != 3
        if match and (not torch.equal(flow, control['flow']) or not torch.equal(latent, control['post'])):
            raise RuntimeError('Original UB1 identity/pre-intervention step changed')
        record = {'step': step, 'input_sha256': inputs[step], 'flow_sha256': tensor_sha(flow),
                  'post_sha256': tensor_sha(latent), 'original_UB1_bitwise_required': match,
                  'flow_error': error_stats(flow, dq['payloads'][step]['flow']),
                  'post_error': error_stats(latent, dq['payloads'][step]['post'])}
        if step == 3:
            path = folder / 'step3.safetensors'
            save_file({'flow': flow, 'post': latent}, str(path))
            record.update(path=str(path), sha256=digest(path))
        steps.append(record)
        scheduler.latents = exact_native_latent(dq['payloads'][step]['post'], scheduler.latents)
        return result

    observer = NativeResidualGain(runtime, teacher, gains)
    scheduler.step_pre, scheduler.step_post = pre, post
    report = {'status': 'RUNNING', 'index': index, 'label': label, 'gains': gains, 'steps': steps,
              'quality_or_speed_acceptance': False}
    atomic_json(folder / 'INPUT.json', report)
    started = time.monotonic()
    try:
        with torch.inference_mode(), observer:
            pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(folder / 'diagnostic_only.mp4'))
        torch.cuda.synchronize()
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']:
            raise RuntimeError('Full160 native FFN/all-linear coverage differs')
        verify_runtime(runtime, 'UB1')
        if [s['block'] for s in observer.scopes] != [30, 31] or [s['step'] for s in steps] != list(range(4)):
            raise RuntimeError('Fixed pair/native step coverage differs')
        report.update(status='PASS_NATIVE_GAIN_RUN', scopes=observer.scopes, counter_delta=delta,
                      elapsed_seconds=time.monotonic() - started, main_MAC_fraction=.5,
                      extra_runtime_work='Two BF16 residual scalar multiplications when gain differs from1',
                      diagnostic_video_is_DQ_by_construction=True)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error), scopes=observer.scopes)
        raise
    finally:
        scheduler.step_pre, scheduler.step_post = old_pre, old_post
        atomic_json(folder / 'INPUT.json', report)
    print(f'R7D2 {label} input{index:02d} PASS; gains={gains}; flowSSE={steps[3]["flow_error"]["squared_error"]:.6f}', flush=True)
    return {'index': index, 'label': label, 'path': str(folder / 'INPUT.json'), 'sha256': digest(folder / 'INPUT.json')}


def barrier(out, identity, stage, expected):
    """Two live worker jobs rendezvous; failures abort rather than bypass controls."""
    deadline = time.monotonic() + 1200
    while True:
        reports = []
        for shard in (0, 1):
            path = out / f'shard{shard}/RUN.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if report['scientific_identity'] != identity or report['status'].startswith('FAILED'):
                raise RuntimeError('Paired job failed or scientific identity differs')
            reports.append(report)
        rows = [row for r in reports for row in r['completed'] if row['label'] in stage]
        if {(r['index'], r['label']) for r in rows} == expected and len(rows) == len(expected):
            for row in rows:
                if digest(row['path']) != row['sha256']:
                    raise RuntimeError('Paired native input receipt changed')
                case = json.loads(Path(row['path']).read_text())
                if case['status'] != 'PASS_NATIVE_GAIN_RUN':
                    raise RuntimeError('Paired native input did not pass')
            return rows
        if time.monotonic() >= deadline:
            raise RuntimeError('Paired job rendezvous timed out; no stage bypass allowed')
        time.sleep(2)


def fit_objectives(rows, refs, identity):
    cases = {(r['index'], r['label']): json.loads(Path(r['path']).read_text()) for r in rows}
    local_g, local_b = np.zeros((2, 2)), np.zeros(2)
    flow_g, flow_b = np.zeros((2, 2)), np.zeros(2)
    for index in TRAIN:
        base = cases[index, 'identity']
        for j, s in enumerate(base['scopes']):
            if s['gain'] != 1.:
                raise RuntimeError('Local objective requires original identity operands')
            local_g[j, j] += s['local_original_residual_norm2']
            local_b[j] += s['local_error_dot_original_residual']
        flow = []
        for label in ['identity', 'direction_A', 'direction_B']:
            record = cases[index, label]['steps'][3]
            if digest(record['path']) != record['sha256']:
                raise RuntimeError('Direction payload changed')
            value = load_file(record['path'], device='cpu')['flow']
            if tensor_sha(value) != record['flow_sha256']:
                raise RuntimeError('Direction decoded flow changed')
            flow.append(value.double())
        error = flow[0] - refs['DQ'][index]['payloads'][3]['flow'].double()
        directions = [(value - flow[0]) / EPSILON for value in flow[1:]]
        for j in range(2):
            flow_b[j] += float((error * directions[j]).sum())
            for k in range(2):
                flow_g[j, k] += float((directions[j] * directions[k]).sum())
    return {'status': 'FROZEN_TWO_OBJECTIVE_GAIN_FITS', 'scientific_identity': identity,
            'training_indices': TRAIN, 'heldout_index_not_used_in_fit': 15, 'epsilon': EPSILON,
            'local_fit': box_quadratic(local_g, local_b), 'flow_fit': box_quadratic(flow_g, flow_b),
            'input_receipts': [{k: v for k, v in row.items() if k != 'path'} for row in rows],
            'fit_source_sha256': digest(ROOT / 'tools/residual_gain_transport.py'),
            'scope': 'Two gains; native BF16 quantized update validated by later runs, not assumed equal to quadratic surrogate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(ROOT)
    if validate_records(*records) or not method_eligibility(records[0], records[1], METHOD)[0]:
        raise RuntimeError('Two-objective diagnostic is not registered')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'REGISTERED_IDENTITY_DIRECTIONS_FITS_NATIVE_COMPARISON':
        raise RuntimeError('Unexpected fixed objective protocol')
    torch.set_num_threads(8)
    args.out = args.out.resolve(); folder = args.out / f'shard{args.shard}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve previous job; no implicit resume')
    history = HistoricalSources(); alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    refs, ref_bindings = references(history, assets)
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': policy_sha, 'original_TF2_reports': ref_bindings,
                'R7D1_result_sha256': digest(ROOT / 'research/20261021/R7D1_results.json')}
    raw = Path(cache['path']).parents[1] / 'r7d2_residual_gain_20261021' / args.out.name / f'shard{args.shard}'
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30:
        raise RuntimeError('Preserve80GiB node reserve plus2GiB run allowance')
    indices = [0, 10] if args.shard == 0 else [5, 15]
    report = {'status': 'RUNNING', 'stage': 'identity', 'scientific_identity': identity,
              'completed': [], 'quality_or_speed_acceptance': False, 'historical_source_receipts': history.receipts}
    atomic_json(folder / 'RUN.json', report)
    try:
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Incomplete strict raw placeholders')
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        from step_budget_runtime import create_runtime
        runtime = create_runtime(cache['path'], alias); report['residency'] = make_resident(runtime)
        teacher = TeacherRecords(cache); adapter.activate_lightx2v_hooks(runtime=runtime)

        def run(index, label, gains):
            if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                raise RuntimeError('Frozen objective experiment source changed')
            row = capture(pipe, runtime, teacher, gains, label, index, refs, raw / f'input{index:02d}_{label}')
            if audit['fallback_calls']:
                raise RuntimeError('Forbidden raw fallback')
            report['completed'].append(row); atomic_json(folder / 'RUN.json', report)

        try:
            for index in indices: run(index, 'identity', [1., 1.])
            identity_rows = barrier(args.out, identity, {'identity'}, {(i, 'identity') for i in [0, 5, 10, 15]})
            report['stage'] = 'directions'; atomic_json(folder / 'RUN.json', report)
            for index in indices:
                if index in TRAIN:
                    run(index, 'direction_A', [1 + EPSILON, 1.])
                    run(index, 'direction_B', [1., 1 + EPSILON])
            direction_rows = barrier(args.out, identity, {'direction_A', 'direction_B'}, {(i, a) for i in TRAIN for a in ['direction_A', 'direction_B']})
            fit = fit_objectives(identity_rows + direction_rows, refs, identity)
            fit_lock = (args.out / 'FIT.lock').open('w'); fcntl.flock(fit_lock.fileno(), fcntl.LOCK_EX)
            path = args.out / 'FIT.json'
            if path.exists() and json.loads(path.read_text()) != fit:
                raise RuntimeError('Independent deterministic objective fits differ')
            if not path.exists(): atomic_json(path, fit)
            fcntl.flock(fit_lock.fileno(), fcntl.LOCK_UN)
            report.update(stage='native_comparison', fit_sha256=digest(path)); atomic_json(folder / 'RUN.json', report)
            for index in indices:
                for label in ['local_fit', 'flow_fit']: run(index, label, fit[label]['gains'])
        finally:
            adapter.deactivate_lightx2v_hooks()
        report.update(status='PASS_FIXED_OBJECTIVE_SHARD', stage='complete')
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)


if __name__ == '__main__':
    main()
