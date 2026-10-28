"""CPU-only paired CORE completion for six valid videos in two failed parents.

Original parents, GENERATION receipts and all native outputs remain immutable.
This scope creates four paired reports and an explicit six-video reuse manifest;
it performs zero generation, model, tensor fitting or GPU operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HF = Path('/home/xiang/.cache/huggingface')
ID = 'E0_RAW_BF16_VIDEO20_PREFIX_CORE_REPAIR_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_raw_bf16_video20_prefix_core_repair_cpu_20261029/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 600, 'outer_seconds': 900,
             'new_HF_bytes': 128 << 20, 'automatic_retry': False}
LIMITS = {'new_generations': 0, 'model_calls': 0, 'tensor_loads': 0,
          'reused_complete_generations': 6, 'reused_dense_CORE': 2, 'paired_CORE_evaluations': 4}
MODES = ('dense', 'shared', 'step')
STATE_KEYS = {'raw_noise_FP32', 'initial_noise'} | {
    f'{kind}_step{step}' for kind in ('flow', 'latent') for step in range(4)}


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def binding(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    need(path.is_relative_to(base) and path.is_file(), 'Confined file required')
    return {'path': str(path.relative_to(base)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def bound(base, item):
    path = (Path(base)/item['path']).resolve()
    need(binding(path, base) == {k: item[k] for k in ('path', 'bytes', 'sha256')}, 'File binding differs')
    return path


def write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(text)
    temp.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def parents(hf):
    result = []
    for part in (0, 1):
        protocol = ROOT/f'config/e0_raw_bf16_full160_video20_part{part}_20261029_v1.json'
        worker = hf/f'ffnds-derived/e0_raw_bf16_full160_video20_20261029/part{part}/attempt1/worker'
        report = read(worker/'results/RUN.json')
        plan = read(protocol)
        phases = []
        for phase in report['phases']:
            gen = phase['generation_receipt']
            receipt = read(bound(hf, gen))
            video = Path(receipt['video']['path'])
            item = {'generation': gen, 'video': binding(video, hf),
                    'states': receipt['states'], 'native_evidence': receipt['native_operand_transfer_evidence']}
            if phase['mode'] == 'dense':
                item['dense_CORE'] = phase['core_metrics']
            phases.append(item)
        result.append({'part': part, 'case': plan['cases'][0], 'protocol': binding(protocol, ROOT),
                       'RUN': binding(worker/'results/RUN.json', hf),
                       'exit': binding(worker/'EXIT_RECEIPT.json', hf), 'phases': phases})
    return result


def validate_parent(proof, hf, *, verify_files=True):
    plan = read(bound(ROOT, proof['protocol']))
    report = read(bound(hf, proof['RUN']))
    exit_record = read(bound(hf, proof['exit']))
    part = proof['part']
    need(part in (0, 1) and plan['part'] == part and proof['case'] == plan['cases'][0]
         and proof['case']['index'] == part and proof['case']['native_source_ordinal'] == (894, 895)[part],
         'Only original first-source prefix may be reused')
    need(report['id'] == plan['id'] == exit_record['id']
         and report['protocol_sha256'] == exit_record['protocol_sha256'] == proof['protocol']['sha256']
         and exit_record['run_sha256'] == proof['RUN']['sha256']
         and type(exit_record['child_returncode']) is int and exit_record['child_returncode'] == 1 and exit_record['process_exited'] is True
         and exit_record['timeout'] is False and exit_record['automatic_retry'] is False
         and report['status'] == 'FAILED_FULL160_VIDEO20_PART_PRESERVED_NO_RETRY'
         and report['error_type'] == 'AttributeError' and report['error'] == "'NoneType' object has no attribute 'get'"
         and len(report['phases']) == 3 and report['triads'] == [], 'Actual failed-parent identity differs')
    observed = report['operation_ledger']['observed']
    need(observed['video_generations'] == 3 and observed['native_full_model_flows'] == 12
         and observed['native_FFN_calls'] == 480 and observed['core_CPU_reports'] == 2,
         'Completed-prefix actual native counts differ')
    result = []
    for item, phase in zip(proof['phases'], report['phases'], strict=True):
        need(item['generation'] == phase['generation_receipt'], 'Parent phase receipt differs')
        gen = read(bound(hf, item['generation']))
        need(gen['id'] == plan['id'] and gen['protocol_sha256'] == proof['protocol']['sha256']
             and gen['run_id'] == phase['run_id'] and gen['source_index'] == part
             and gen['image_sha256'] == proof['case']['image']['sha256']
             and gen['prompt_sha256'] == proof['case']['prompt_sha256'] and gen['seed'] == 42
             and gen['status'] == 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
             and gen['completed_native_flow_steps'] == 4 and gen['all160_FFN_covered'] is True
             and gen['numeric_path'] == 'BF16_ORIGINAL' and gen['quality_or_speed_acceptance'] is False
             and gen['native_counter_delta'] == phase['native_counter_delta'] == {
                 'linear_forward_calls': 1600, 'activation_qdq_calls': 0, 'weight_quantizations': 0}
             and set(gen['states']) == STATE_KEYS and gen['states'] == item['states']
             and gen['native_operand_transfer_evidence'] == item['native_evidence'], 'Incomplete or mismatched generation')
        need(gen['video']['frames'] == 81 and gen['video']['size'] == [832, 464]
             and gen['video']['fps'] == 16.0 and gen['video']['sha256'] == item['video']['sha256']
             and (hf/item['video']['path']).resolve() == Path(gen['video']['path']).resolve(), 'Video identity differs')
        if verify_files:
            bound(hf, item['video'])
            for state in gen['states'].values():
                bound(hf, state)
            evidence = read(bound(hf, item['native_evidence']))
            need(evidence['all160_FFN_sites'] == [[s, b] for s in range(4) for b in range(40)]
                 and [[v['step'], v['block']] for v in evidence['FFN_boundary_CUDA_by_site']] == evidence['all160_FFN_sites']
                 and len(evidence['prefetch']) == 162, 'All160 native evidence differs')
        result.append((item, gen))
    need([g['mode'] for _, g in result] == list(MODES), 'First triad mode/order differs')
    for key in ('raw_noise_FP32', 'initial_noise'):
        need(len({g['states'][key]['sha256'] for _, g in result}) == 1
             and len({g['states'][key]['raw_tensor_sha256'] for _, g in result}) == 1, 'Real paired noise bytes differ')
    return plan, report, result


def context(plan, parent_run, case, gen, generation_binding, dense, dense_binding):
    need(gen['mode'] != 'dense', 'This producer only computes missing candidate CORE')
    return {'identity': {'source_id': case['image']['sha256'], 'seed': 42, 'expected_frames': 81,
                        'input_prompt_digest': case['prompt_sha256'], 'model_digest': canonical(plan['models']),
                        'protocol_sha256': gen['protocol_sha256'], 'run_sha256': generation_binding['sha256']},
            'reference_identity': {'source_id': dense['image_sha256'], 'seed': dense['seed'],
                                   'input_prompt_digest': dense['prompt_sha256'], 'run_sha256': dense_binding['sha256']},
            'cost': {'generation_wall_seconds': gen['generation_wall_seconds_including_encoders_decode_observers'],
                     'ffn_boundary_seconds': gen['FFN_boundary_CUDA_seconds'], 'setup_seconds': parent_run['raw_model_setup_seconds'],
                     'peak_gpu_memory_mib': gen['peak_cuda_allocated_bytes']/(1 << 20)}}


def validate_plan(plan, frozen=False):
    need(plan['id'] == ID and plan['resources'] == RESOURCES and plan['limits'] == LIMITS
         and plan['output_HF_relative'] == OUTPUT and plan['automatic_retry'] is False
         and [p['part'] for p in plan['parents']] == [0, 1], 'Fixed CPU repair scope differs')
    if frozen:
        need(plan.get('status') == 'FROZEN_REGISTERED' and plan.get('assets_finalized') is True
             and plan.get('independent_review'), 'CPU scope not frozen/reviewed')


def draft(hf):
    files = ['tools/raw_bf16_video20_prefix_core_repair.py', 'tests/test_raw_bf16_video20_prefix_core_repair.py',
             'tools/ffnds_video_metrics.py', 'tools/worker_identity.py', 'tools/durable_worker_job.py']
    return {'id': ID, 'status': 'DRAFT', 'assets_finalized': False, 'resources': RESOURCES, 'limits': LIMITS,
            'automatic_retry': False, 'output_HF_relative': OUTPUT, 'parents': parents(hf),
            'repo_sources': [binding(ROOT/p, ROOT) for p in files],
            'question': 'Complete four missing all81-frame candidate paired CORE reports without new videos or relabelling failed parents',
            'official_quality': 'NOT_RUN', 'human_video_review': 'PENDING', 'quality_or_speed_acceptance': False}


def run(args):
    started = time.monotonic()
    plan = read(args.protocol)
    validate_plan(plan, True)
    digest = sha(args.protocol)
    active = read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    need(active['user_continuation_authorization_20261027']['status'] == 'ACTIVE_CURRENT_USER_DIRECTED_SESSION'
         and len([s for s in scopes if s['id'] == ID and s['protocol_sha256'] == digest
                  and s['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
                  and s['resources'] == RESOURCES and s['limits'] == LIMITS and s['max_gpu_cards'] == 0]) == 1,
         'Unique exact CPU scope not authorized')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 4
         and 'torch' not in sys.modules, 'CPU-only fresh worker required')
    from worker_identity import require_worker
    require_worker()
    out = (args.hf_root/OUTPUT).resolve()
    need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined output required; no retry')
    out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_CPU_PREFIX_CORE_REPAIR_NO_RETRY',
              'automatic_retry': False, 'GPU_calls': 0, 'model_calls': 0, 'new_generations': 0,
              'tensor_loads': 0, 'parents': plan['parents'], 'phases': [],
              'operation_ledger': {'predebited': 0, 'observed': 0}, 'quality_or_speed_acceptance': False,
              'official_quality': 'NOT_RUN', 'human_video_review': 'PENDING'}
    def timeout(*_):
        raise TimeoutError('CPU repair wall600 reached; no retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    write(out/'RUN.json', report)
    try:
        for item in plan['repo_sources']:
            bound(ROOT, item)
        from ffnds_video_metrics import evaluate, validate
        import cv2
        cv2.setNumThreads(4)
        for parent in plan['parents']:
            old_plan, old_run, phases = validate_parent(parent, args.hf_root)
            dense_item, dense = phases[0]
            dense_report = read(bound(args.hf_root, dense_item['dense_CORE']))
            need(not validate(dense_report) and dense_report['candidate']['sha256'] == dense_item['video']['sha256']
                 and dense_report['identity']['run_sha256'] == dense_item['generation']['sha256']
                 and dense_report['reference'] is None, 'Original Dense CORE differs')
            for item, gen in phases:
                phase = {'part': parent['part'], 'mode': gen['mode'], 'source_index': gen['source_index'],
                         'native_source_ordinal': gen['native_source_ordinal'], 'generation_receipt': item['generation'],
                         'video': item['video'], 'states': item['states'], 'native_operand_transfer_evidence': item['native_evidence'],
                         'failed_native_parent_RUN': parent['RUN'], 'failed_native_parent_exit': parent['exit'],
                         'native_generation_status': gen['status'], 'parent_remains_failed': True}
                if gen['mode'] == 'dense':
                    phase.update(core_metrics=dense_item['dense_CORE'], CORE_identity='REUSED_EXISTING_VALID_DENSE_CORE')
                else:
                    ctx = context(old_plan, old_run, parent['case'], gen, item['generation'], dense, dense_item['generation'])
                    report['operation_ledger']['predebited'] += 1
                    write(out/'RUN.json', report)
                    core = evaluate(args.hf_root/item['video']['path'], args.hf_root/dense_item['video']['path'], ctx)
                    report['operation_ledger']['observed'] += 1
                    need(not validate(core), 'Paired CORE invalid')
                    path = out/f"part{parent['part']}_{gen['mode']}_CORE.json"
                    write(path, core)
                    phase.update(core_metrics=binding(path, args.hf_root), CORE_identity='NEW_CPU_PAIRED_CORE_NO_NATIVE_RERUN')
                report['phases'].append(phase)
                write(out/'RUN.json', report)
        need(len(report['phases']) == 6 and report['operation_ledger'] == {'predebited': 4, 'observed': 4}
             and 'torch' not in sys.modules, 'Exact six-video reuse/four-CORE CPU scope differs')
        report['status'] = 'COMPLETE_CPU_PREFIX_CORE_REPAIR_SIX_VALID_GENERATIONS_PARENTS_REMAIN_FAILED'
    except BaseException as error:
        report.update(status='FAILED_CPU_PREFIX_CORE_REPAIR_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        report['wall_seconds'] = time.monotonic()-started
        report['new_output_bytes_before_final_RUN'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        need(report['new_output_bytes_before_final_RUN'] < RESOURCES['new_HF_bytes']-(1 << 20), 'CPU output budget exceeded')
        write(out/'RUN.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'run'))
    parser.add_argument('--hf-root', type=Path, default=HF)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.action == 'draft':
        write(args.out, draft(args.hf_root))
    else:
        run(args)


if __name__ == '__main__':
    main()
