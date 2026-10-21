"""R7D1: full native four-step controls, then the fixed four-arm diagnostic."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from collect_ub1_native_states import policy_alias, verify_runtime
from cross_layer_transport import NativePairRestoration
from historical_source_snapshot import HistoricalSources
from restored_assets import ROOT, verified_restored_assets
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from run_ar2_restart import tensor_sha
from run_supplement_teacher_flow import exact_native_latent, error_stats
from variable_teacher_capture import TeacherRecords

METHOD = 'R7D1_CROSS_LAYER_TRANSPORT'
PROTOCOL = ROOT / 'research/20261021/R7D1_cross_layer_transport_protocol.json'
RUNS = ROOT / 'runs/worker_restart_20261020'
INDICES = [0, 5, 10, 15]
SOURCES = ['run_cross_layer_transport.py', 'cross_layer_transport.py', 'variable_teacher_capture.py',
           'run_supplement_teacher_flow.py', 'collect_ub1_native_states.py', 'restored_assets.py',
           'historical_source_snapshot.py', 'resident_ffn_runtime.py', 'step_budget_runtime.py',
           'trained_packed_runtime.py', 'reduced_rank_residual.py']


def source_binding():
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in SOURCES}


def references(history, assets):
    result, bindings = {}, {}
    for arm in ['DQ', 'UB1']:
        p = RUNS / f'tf2_flow/{arm}/RUN.json'
        report = json.loads(p.read_text())
        expected = 'PASS_TF2_FOUR_SOURCE_DENSE_CONTROLS' if arm == 'DQ' else 'PASS_TF2_FOUR_SOURCE_CANDIDATE_SAME_DENSE_STATES'
        if report['status'] != expected or report['identity']['assets'] != assets:
            raise RuntimeError('Original TF2 controls or model asset identity differ')
        history.verify(ROOT / 'tools/run_supplement_teacher_flow.py', report['identity']['source_sha256'])
        for path, sha in report['identity']['source_identity']['native_sources'].items():
            if digest(path) != sha:
                raise RuntimeError('Original TF2 native implementation changed')
        rows = {}
        for row in report['completed']:
            if digest(row['path']) != row['sha256']:
                raise RuntimeError('Original TF2 input report changed')
            case = json.loads(Path(row['path']).read_text())
            payloads = []
            for step, record in enumerate(case['steps']):
                if record['step'] != step or digest(record['path']) != record['sha256']:
                    raise RuntimeError('Original TF2 payload binding changed')
                data = load_file(record['path'], device='cpu')
                if set(data) != {'input', 'flow', 'post'}:
                    raise RuntimeError('Unexpected TF2 payload fields')
                for key, value in data.items():
                    if value.dtype != torch.float32 or list(value.shape) != [16, 21, 58, 104] or tensor_sha(value) != record[key + '_sha256']:
                        raise RuntimeError('Original TF2 decoded tensor identity differs')
                payloads.append(data)
            if len(payloads) != 4:
                raise RuntimeError('Incomplete TF2 steps')
            rows[row['index']] = {'case': case, 'payloads': payloads}
        if set(rows) != set(INDICES):
            raise RuntimeError('Fixed TF2 source membership differs')
        result[arm] = rows
        bindings[arm] = digest(p)
    for index in INDICES:
        if result['DQ'][index]['case']['input'] != result['UB1'][index]['case']['input']:
            raise RuntimeError('DQ/UB1 conditioning differs')
        for step in range(4):
            if not torch.equal(result['DQ'][index]['payloads'][step]['input'], result['UB1'][index]['payloads'][step]['input']):
                raise RuntimeError('TF2 arms did not receive matched states')
    return result, bindings


def capture(pipe, runtime, teacher, arm, index, refs, folder):
    if folder.exists():
        raise RuntimeError('Refusing to overwrite an attempted native run')
    folder.mkdir(parents=True)
    scheduler = pipe.runner.scheduler
    old_pre, old_post = scheduler.step_pre, scheduler.step_post
    dq, baseline = refs['DQ'][index], refs['DQ' if arm == 'DQ' else 'UB1'][index]
    sample = dq['case']['input']
    if digest(sample['image_path']) != sample['input_provenance']['image_sha256'] or sample['seed'] != 42:
        raise RuntimeError('Original image or seed changed')
    steps = []
    current = {}
    if hasattr(runtime, 'begin_request'):
        runtime.begin_request()
    before = dict(runtime.counters)
    scheduler.generator = None

    def pre(step_index):
        value = old_pre(step_index)
        latent = scheduler.latents.detach().float().cpu().contiguous()
        if not torch.equal(latent, dq['payloads'][step_index]['input']):
            raise RuntimeError('Native input does not match original DQ state bitwise')
        current[int(step_index)] = tensor_sha(latent)
        return value

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        value = old_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        target = dq['payloads'][step]
        control = baseline['payloads'][step]
        must_match = arm in ('DQ', 'none', 'none_repeat') or step != 3
        if must_match and (not torch.equal(flow, control['flow']) or not torch.equal(latent, control['post'])):
            raise RuntimeError(f'Original TF2 flow/post bitwise control failed at step{step}')
        row = {'step': step, 'input_sha256': current[step], 'flow_sha256': tensor_sha(flow),
               'post_sha256': tensor_sha(latent), 'required_original_bitwise_match': must_match,
               'flow_error': error_stats(flow, target['flow']), 'post_error': error_stats(latent, target['post']),
               'sigma': float(scheduler.sigmas[step]), 'timestep': float(scheduler.timesteps[step])}
        if step == 3:
            path = folder / 'step3.safetensors'
            save_file({'flow': flow, 'post': latent}, str(path))
            row.update(path=str(path), sha256=digest(path))
        steps.append(row)
        # The original TF2 protocol restores DQ post only BETWEEN full steps.
        # Never reset a tensor inside the selected block30/31 pair.
        scheduler.latents = exact_native_latent(target['post'], scheduler.latents)
        return value

    observer = NativePairRestoration(runtime, teacher, arm)
    scheduler.step_pre, scheduler.step_post = pre, post
    started = time.monotonic()
    report = {'status': 'RUNNING', 'arm': arm, 'index': index, 'steps': steps,
              'quality_or_speed_acceptance': False}
    atomic_json(folder / 'INPUT.json', report)
    try:
        with torch.inference_mode(), observer:
            pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(folder / 'diagnostic_only.mp4'))
        torch.cuda.synchronize()
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']:
            raise RuntimeError('Native full-four-step linear/QDQ coverage changed')
        verify_runtime(runtime, 'DQ' if arm == 'DQ' else 'UB1')
        if [r['step'] for r in steps] != list(range(4)) or [s['block'] for s in observer.scopes] != [30, 31]:
            raise RuntimeError('Incomplete native step/pair coverage')
        report.update(status='PASS_FIXED_NATIVE_PAIR_RUN', scopes=observer.scopes, counter_delta=delta,
                      elapsed_seconds=time.monotonic() - started,
                      video_sha256=digest(folder / 'diagnostic_only.mp4'),
                      diagnostic_video_is_DQ_by_construction=True)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error), scopes=observer.scopes)
        raise
    finally:
        scheduler.step_pre, scheduler.step_post = old_pre, old_post
        atomic_json(folder / 'INPUT.json', report)
    print(f'R7D1 {arm} input{index:02d} PASS; flowSSE={steps[3]["flow_error"]["squared_error"]:.6f}', flush=True)
    return {'index': index, 'arm': arm, 'path': str(folder / 'INPUT.json'), 'sha256': digest(folder / 'INPUT.json')}


def check_controls(out, identity):
    bindings = {}
    for name, arms in [('control_DQ', ['DQ']), ('control_UB1', ['none', 'none_repeat'])]:
        path = out / name / 'RUN.json'
        report = json.loads(path.read_text())
        if report['status'] != 'PASS_ALL_REGISTERED_RUNS' or report['scientific_identity'] != identity:
            raise RuntimeError('Both complete native controls must pass with identical frozen sources/assets')
        if {(r['index'], r['arm']) for r in report['completed']} != {(i, a) for i in INDICES for a in arms}:
            raise RuntimeError('Incomplete repeated-none or dense identity controls')
        for row in report['completed']:
            if digest(row['path']) != row['sha256']:
                raise RuntimeError('Completed control receipt changed')
            case = json.loads(Path(row['path']).read_text())
            if case['status'] != 'PASS_FIXED_NATIVE_PAIR_RUN':
                raise RuntimeError('Failed native control')
            for step in case['steps']:
                if step.get('path') and digest(step['path']) != step['sha256']:
                    raise RuntimeError('Control flow payload mutated')
        bindings[name] = digest(path)
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['controls', 'interventions'], required=True)
    parser.add_argument('--baseline', choices=['DQ', 'UB1'], default='UB1')
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(ROOT)
    if validate_records(*records) or not method_eligibility(records[0], records[1], METHOD)[0]:
        raise RuntimeError('Fixed diagnostic has not been registered consistently')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'REGISTERED_CONTROLS_THEN_FIXED_ARMS':
        raise RuntimeError('Unexpected diagnostic protocol phase')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    job = f'control_{args.baseline}' if args.phase == 'controls' else f'intervention_shard{args.shard}'
    folder = args.out / job
    folder.mkdir(exist_ok=True)
    lock = (folder / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve completed/failed job; do not overwrite or silently resume')
    torch.set_num_threads(8)
    history = HistoricalSources()
    alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    refs, ref_bindings = references(history, assets)
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': policy_sha, 'original_TF2_reports': ref_bindings}
    prerequisites = check_controls(args.out, identity) if args.phase == 'interventions' else {}
    raw = Path(cache['path']).parents[1] / 'r7d1_cross_layer_transport_20261021' / args.out.name / job
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30:
        raise RuntimeError('Preserve80GiB node reserve plus2GiB diagnostic allowance')
    if args.phase == 'controls':
        arms = ['DQ'] if args.baseline == 'DQ' else ['none', 'none_repeat']
        jobs = [(i, a) for i in INDICES for a in arms]
    else:
        jobs = [(i, a) for j, i in enumerate(INDICES) if j % 2 == args.shard for a in ['A', 'B', 'AB']]
    report = {'status': 'RUNNING', 'job': job, 'scientific_identity': identity, 'jobs': jobs,
              'completed': [], 'control_bindings': prerequisites, 'quality_or_speed_acceptance': False,
              'historical_source_receipts': history.receipts}
    atomic_json(folder / 'RUN.json', report)
    try:
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Incomplete strict raw placeholders')
        if args.phase == 'controls' and args.baseline == 'DQ':
            runtime = adapter.HIF4CacheRuntime(cache['path'])
        else:
            from step_budget_runtime import create_runtime
            runtime = create_runtime(cache['path'], alias)
        report['residency'] = make_resident(runtime)
        teacher = TeacherRecords(cache)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for index, arm in jobs:
                if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                    raise RuntimeError('Frozen experiment source/protocol changed during run')
                row = capture(pipe, runtime, teacher, arm, index, refs, raw / f'input{index:02d}_{arm}')
                if audit['fallback_calls']:
                    raise RuntimeError('Forbidden raw placeholder fallback')
                report['completed'].append(row)
                atomic_json(folder / 'RUN.json', report)
        finally:
            adapter.deactivate_lightx2v_hooks()
        report['status'] = 'PASS_ALL_REGISTERED_RUNS'
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)


if __name__ == '__main__':
    main()
