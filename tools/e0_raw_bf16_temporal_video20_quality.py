"""Eight official axes on forty new videos, reusing twenty actual Dense scores.

Numerical metric calls use the previously verified isolated implementation.
No new Dense scoring, generation, fitting or threshold selection occurs here.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
_spec = importlib.util.spec_from_file_location('_ffnds_temporal_official_quality_helpers', ROOT/'tools/e0_raw_bf16_video20_quality.py')
q = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(q)
base, v4 = q.base, q.v4
ID = 'E0_RAW_BF16_VALUE10_DCAC10_VIDEO20_OFFICIAL_QUALITY_20261029'
TEMPLATE = 'config/e0_raw_bf16_video20_quality_20261029_v1.json'
GENERATION = 'config/e0_raw_bf16_temporal_video20_part{part}_20261029_v1.json'
COHORT = q.COHORT
MODES = ('dense', 'value10', 'dcac10')
DIMENSIONS, COMMON = q.DIMENSIONS, q.COMMON
COUNTS = {d: 40 if d in COMMON else 20 for d in DIMENSIONS}
COSTS = {'imaging_quality': {'MUSIQ': 3240}, 'aesthetic_quality': {'CLIP_L14': 120, 'aesthetic_head': 120},
    'motion_smoothness': {'AMT': 1600}, 'dynamic_degree': {'RAFT': 1600, 'RAFT_iterations': 32000},
    'subject_consistency': {'DINO': 3240}, 'background_consistency': {'CLIP_B32': 40},
    'i2v_subject': {'DINO': 1640}, 'i2v_background': {'DreamSim_embed': 1640}}
LIMITS = {'official_dimension_calls': 8, 'new_video_dimension_results': 280,
    'reused_core_reports': 40, 'RAFT_forwards': 1600, 'RAFT_iterations': 32000,
    'video_generations': 0, 'training_updates': 0, 'decompositions': 0, 'CPU_metric_recomputations': 0}
RESOURCES = {'GPU': 1, 'CPU_threads': 2, 'wall_seconds': 3600, 'outer_seconds': 3900,
    'new_HF_bytes': 1 << 29, 'GPU_reserved_bytes': 8 << 30, 'free_disk_bytes': 10 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 600, 'outer_seconds': 900, 'automatic_retry': False}
POLICY = {**q.POLICY, 'arms': list(MODES), 'new_scored_videos': 40, 'reused_Dense_scored_videos': 20,
          'difference': 'MATCHED_R960_VALUE_VS_DC_AC_TARGET_SAME_NEW10_CALIBRATION',
          'statistical_unit': 'TWENTY_EXPOSED_IMAGE_CLUSTERS_SEED42_SOURCE_EQUAL',
          'Dense_metric_identity': 'REUSE_ACTUAL_SAME_CLIPS_SAME_PROTECTED_FUNCTIONS_NO_DENSE_METRIC_REPLAY'}
CPU_STATUS = 'PASS_TEMPORAL40_GENERATION_DENSE20_SCORE_ENVIRONMENT_FULL_BYTE_PROOF'
REQUIRED = {'tools/e0_raw_bf16_temporal_video20_quality.py', 'tests/test_e0_raw_bf16_temporal_video20_quality.py',
            'tools/e0_raw_bf16_temporal_video20.py', 'tools/e0_raw_bf16_video20_quality.py',
            'tools/durable_worker_job.py', 'scripts/worker_wait_uuid_idle_exec.sh', COHORT, TEMPLATE}
verify_import_proof, verify_environment = q.verify_import_proof, q.verify_environment
rows_for_dimension, stat_record = q.rows_for_dimension, q.stat_record


def validate(plan, frozen=False):
    base.require(plan['id'] == ID and plan['resources'] == RESOURCES and plan['limits'] == LIMITS
        and plan['costs'] == COSTS and plan['CPU_preflight_resources'] == CPU_RESOURCES
        and plan['dimensions'] == list(DIMENSIONS) and plan['policy'] == POLICY
        and plan['automatic_retry'] is False and plan['numeric_path'] == 'BF16_ORIGINAL', 'Fixed forty-clip quality scope differs')
    cohort = base.read_bound(ROOT, plan['source_cohort'])
    expected = [q.video_identity(c, m) for c in cohort['sources'] for m in MODES]
    base.require(len(expected) == len(plan['videos']) == 60
        and all(all(v[k] == x for k, x in e.items()) for v, e in zip(plan['videos'], expected))
        and plan['new_videos'] == [v for v in plan['videos'] if v['mode'] != 'dense']
        and len(plan['new_videos']) == 40, 'Exact original source/arm identities required')
    base.require(all(len(rows_for_dimension(plan['new_videos'], d)) == COUNTS[d] for d in DIMENSIONS)
        and REQUIRED <= {b['path'] for b in plan['repo_sources']}, 'Metric eligibility or source closure differs')
    if frozen:
        base.require(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
            and len(plan['parents']) == 2 and all(p['RUN'] and p['exit'] for p in plan['parents'])
            and all(v.get('generation') and v.get('video') and v.get('core') for v in plan['videos']), 'Completed real assets not frozen')


def authorize(plan, sha):
    validate(plan, True); active = base.read(ROOT/'research/ACTIVE_RESEARCH.json')
    base.require(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True, 'Current continuation inactive')
    ss = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    base.require(len(ss) == 1 and ss[0]['protocol_sha256'] == sha
        and ss[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and ss[0]['limits'] == LIMITS and ss[0]['resources'] == RESOURCES and ss[0]['max_gpu_cards'] == 1,
        'Exact new quality scope not registered')
    return ss[0]


def dense_quality(plan, hf):
    proof = plan['reused_Dense_quality']; old = base.read_bound(ROOT, proof['protocol'])
    report = base.read_bound(hf, proof['RUN'])
    base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], old['id'])
    base.require(report['status'] == 'COMPLETE_RAW_BF16_VIDEO20_60_CLIPS_EIGHT_DIMENSIONS_QUALITY_NOT_ACCEPTED'
        and report['protocol_sha256'] == proof['protocol']['sha256']
        and old['source_cohort'] == plan['source_cohort']
        and all(old[k] == plan[k] for k in ('environment', 'import_environment', 'hf_bindings', 'reused_import_proof'))
        and report['predebited'] == report['invoked'] == report['completed'] == q.LIMITS
        and not report['assigned_GPU_resources']['violations']
        and len(report['completed_dimensions']) == 8 and len(report['RAFT_observations']) == 60,
        'Exact actual Dense quality receipt and same environment required')
    return old, report


def verify_generation(plan, hf):
    old, _ = dense_quality(plan, hf)
    _, _, dense_protocols, _ = q.generation_metadata(old, hf)
    dense_rows = [v for v in old['videos'] if v['mode'] == 'dense']
    base.require([v for v in plan['videos'] if v['mode'] == 'dense'] == dense_rows, 'Different reused Dense clips')
    phases = {}; protocols = {}; bank_identity = None
    for proof in plan['parents']:
        p = base.read_bound(ROOT, proof['protocol']); r = base.read_bound(hf, proof['RUN'])
        base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], p['id'])
        base.require(r['status'] == 'COMPLETE_MATCHED_TEMPORAL_REPRESENTATION_TEN_SOURCES_TWENTY_FULL160_VIDEOS'
            and r['protocol_sha256'] == proof['protocol']['sha256']
            and len(r['phases']) == 20 and len(r['pairs']) == 10
            and r['new_Dense_generations'] == r['new_fits'] == 0
            and not r['assigned_GPU_resources']['violations']
            and not r['operation_ledger']['forbidden_attempts']
            and all(r['operation_ledger'][side].get(k, 0) == n for side in ('predebited', 'observed') for k, n in p['limits'].items()),
            'New candidate producer not closed in its full registered scope')
        base.require(p['source_cohort'] == plan['source_cohort'] and r['factor_bank'] == p['factor_bank']
            and (bank_identity is None or bank_identity == p['factor_bank']), 'Candidate cohort/bank changed across parts')
        bank_identity = p['factor_bank']
        for phase in r['phases']:
            key = phase['native_source_ordinal'], phase['mode']
            base.require(key not in phases, 'Duplicate actual generation')
            dense_p = dense_protocols[phase['native_source_ordinal'], 'dense']
            base.require(all(p[k] == dense_p[k] for k in ('models', 'runtime_overlay', 'runtime_versions')),
                         'Candidate/Dense checkpoint or numerical recipe differs')
            phases[key], protocols[key] = phase, proof['protocol']['sha256']
    paths = []
    from ffnds_video_metrics import validate as validate_core
    for v in plan['new_videos']:
        phase = phases[v['source'], v['mode']]
        gen = base.read_bound(hf, v['generation']); registered = base.read_bound(hf, v['registered'])
        base.require(phase['generation_receipt'] == v['generation'] and gen['states'] == phase['states'] == v['states']
            and phase['core_metrics'] == v['core'] and phase['native_operand_transfer_evidence'] == v['native_evidence']
            and gen['protocol_sha256'] == protocols[v['source'], v['mode']]
            and gen['all160_FFN_covered'] is True and gen['completed_native_flow_steps'] == 4
            and gen['status'] == 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
            and all(gen[k] == v[k] for k in ('mode', 'seed', 'source_id', 'image_sha256', 'prompt_sha256')),
            'Native candidate identity differs')
        path = base.bound(hf, v['video']); paths.append(path)
        base.require(base.clip_binding(gen['video'], hf) == v['video']
            and registered['immutable_generation_receipt'] == v['generation']
            and registered['generation_origin'] == 'NEW_ACTUAL_CANDIDATE'
            and registered['decoded_frames'] == gen['video']['frames'] == 81
            and registered['decoded_size'] == gen['video']['size'] == [832, 464]
            and registered['decoded_fps'] == gen['video']['fps'] == 16
            and registered['native_ffn_sites'] == 160 and registered['scheduler_generator_reset'] is True,
            'Real registered full81frame candidate differs')
        core = base.read_bound(hf, v['core']); dense = next(d for d in dense_rows if d['source'] == v['source'])
        native = base.read_bound(hf, v['native_evidence'])
        base.require(native['all160_FFN_sites'] == [[s, b] for s in range(4) for b in range(40)], 'Actual all160 evidence differs')
        base.require(not validate_core(core) and core['candidate']['sha256'] == v['video']['sha256']
            and core['reference']['sha256'] == dense['video']['sha256']
            and core['identity']['run_sha256'] == v['generation']['sha256']
            and core['identity']['source_id'] == v['image_sha256']
            and core['identity']['input_prompt_digest'] == v['prompt_sha256']
            and core['reference_identity']['run_sha256'] == dense['generation']['sha256'], 'Whole CORE reference differs')
        for name in ('raw_noise_FP32', 'initial_noise'):
            base.require(all(v['states'][name][k] == dense['states'][name][k]
                for k in ('bytes', 'sha256', 'raw_tensor_sha256', 'shape', 'dtype')), 'Actual Dense/candidate noise differs')
    base.require(len(phases) == len(set(paths)) == len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 40,
        'Distinct new40 video files required')


def all_byte_bindings(plan):
    rows = [('REPO', b) for b in plan['repo_sources']] + [('REPO', plan['source_cohort'])]
    rows += [('HF', b) for b in plan['hf_bindings']]
    for proof in [*plan['parents'], plan['reused_Dense_quality']]:
        rows += [('REPO', proof['protocol']), ('HF', proof['RUN']), ('HF', proof['exit'])]
    for video in plan['videos']:
        rows += [('HF', video[k]) for k in ('video', 'generation', 'core', 'native_evidence')]
        if video.get('registered'): rows.append(('HF', video['registered']))
        rows += [('HF', b) for b in video['states'].values()]
    rows += [('HF', i['binding']) for i in plan['images']]
    unique = {}
    for root, item in rows:
        b = {k: item[k] for k in ('path', 'bytes', 'sha256')}; key = root, b['path']
        base.require(key not in unique or unique[key] == b, 'Conflicting reused byte identity')
        unique[key] = b
    return [(root, b) for (root, _), b in sorted(unique.items())]


def cpu_report(plan, sha):
    return {'id': ID, 'protocol_sha256': sha, 'status': CPU_STATUS,
        'sources_sha256': base.state_sha(plan['repo_sources']), 'parents': plan['parents'],
        'reused_Dense_quality': plan['reused_Dense_quality'], 'source_cohort': plan['source_cohort'],
        'new_scored_videos': 40, 'reused_Dense_scored_videos': 20,
        'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False, 'model_calls': 0, 'GPU_calls': 0}


def preflight(args):
    base.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 2, 'Fresh CUDA-hidden two-CPU byte worker required')
    plan = base.read(args.protocol); sha = base.sha(args.protocol); authorize(plan, sha); q.verify_worker(plan)
    base.require(not args.out.exists(), 'Fresh byte proof only')
    report = {'id': ID, 'protocol_sha256': sha, 'status': 'STARTED_FORTY_VIDEO_QUALITY_BYTE_PROOF',
              'byte_checks': [], 'predebited_bytes': 0, 'actual_verified_bytes': 0}
    base.write(args.out, report); start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Quality CPU600s exhausted')))
    signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for root, binding in all_byte_bindings(plan):
            report['predebited_bytes'] += binding['bytes']
            path = base.bound(ROOT if root == 'REPO' else args.hf_root, binding)
            report['actual_verified_bytes'] += binding['bytes']
            report['byte_checks'].append({'root': root, 'binding': binding, 'stat': stat_record(path)})
        verify_generation(plan, args.hf_root); verify_import_proof(plan, args.hf_root); verify_environment(plan, args.hf_root)
        base.require(not {'torch', 'numpy'} & set(sys.modules), 'Numerical import during byte proof')
        report.update(cpu_report(plan, sha)); report['byte_checks_sha256'] = base.state_sha(report['byte_checks'])
    except BaseException as error:
        report.update(status='FAILED_QUALITY_CPU_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start; base.write(args.out, report)


def require_cpu(scope, plan, sha, hf):
    proof = scope['cpu_preflight']; report = base.read_bound(hf, proof['RUN'])
    base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], sha, ID)
    base.require(all(report.get(k) == v for k, v in cpu_report(plan, sha).items())
        and report['byte_checks_sha256'] == base.state_sha(report['byte_checks'])
        and [(r['root'], r['binding']) for r in report['byte_checks']] == all_byte_bindings(plan), 'Exited CPU proof differs')
    for row in report['byte_checks']:
        base.require(stat_record((ROOT if row['root'] == 'REPO' else hf)/row['binding']['path']) == row['stat'], 'Byte-proved asset changed')
    return proof


def verify_assets(plan, hf):
    q.verify_worker(plan); verify_generation(plan, hf); verify_import_proof(plan, hf)


def combine_per_video_evidence(plan, hf, dimensions, observations):
    """Compose scalar results without inventing a new official aggregate score."""
    _, old = dense_quality(plan, hf)
    dense_paths = {str((hf/v['video']['path']).resolve()) for v in plan['videos'] if v['mode'] == 'dense'}
    combined = []
    for entry in dimensions:
        dimension = entry['dimension']
        original = next(e for e in old['completed_dimensions'] if e['dimension'] == dimension)
        new_paths = [str((hf/v['video']['path']).resolve()) for v in rows_for_dimension(plan['new_videos'], dimension)]
        new = base.check_result(entry['result'], new_paths, dimension)
        dense = [row for row in original['result'][-1] if row['video_path'] in dense_paths]
        base.require(len(dense) == (20 if dimension in COMMON else 10), 'Reused Dense metric coverage differs')
        combined.append({'dimension': dimension, 'result': [None, dense+new],
                         'composition': 'PER_VIDEO_ROWS_ONLY_NO_NEW_OFFICIAL_AGGREGATE'})
    dense_flow = [r for r in old['RAFT_observations'] if r['video_path'] in dense_paths]
    base.require(len(dense_flow) == 20 and len(observations) == 40, 'Exact old20/new40 RAFT coverage required')
    return combined, dense_flow+observations


def summarize(plan, hf, dimensions, observations):
    combined, flows = combine_per_video_evidence(plan, hf, dimensions, observations)
    q.MODES, q.POLICY = MODES, POLICY  # Private helper namespace; frozen public module unchanged.
    result = q.summarize(plan, hf, combined, flows)
    pairs = result['per_source']
    result.update(new_scored_videos=40, reused_Dense_scored_videos=20,
                  reused_Dense_quality=plan['reused_Dense_quality'], official_aggregate_recomputed=False)
    value = result['source_equal_continuous_RAFT_means']['value10']
    balanced = result['source_equal_continuous_RAFT_means']['dcac10']
    result['same_budget_dcac10_vs_value10'] = {
        'continuous_RAFT_ratio_of_source_equal_means': balanced/value if value else None,
        'per_axis_mean_signed_difference': {d: statistics.fmean(float(s['scores']['dcac10'][d])-float(s['scores']['value10'][d])
             for s in pairs if d in s['scores']['dense']) for d in DIMENSIONS}}
    return result


def draft(hf, bind_completed=False):
    old = base.read(ROOT/TEMPLATE)
    fields = ('environment', 'import_environment', 'hf_bindings', 'directory_fd_audit_policy',
              'activation_bootstrap_policy', 'directory_fd_filelock_probe_source', 'reused_import_proof',
              'execution_private_worker_config_sha256', 'source_cohort', 'images')
    plan = {k: copy.deepcopy(old[k]) for k in fields}
    folder = hf/old['output_HF_relative']
    proof = {'protocol': base.binding(ROOT/TEMPLATE, ROOT), 'RUN': base.binding(folder/'RUN.json', hf),
             'exit': base.binding(folder.parent/'EXIT_RECEIPT.json', hf)}
    cohort = base.read_bound(ROOT, plan['source_cohort']); parents = []; candidates = {}
    for part in (0, 1):
        path = ROOT/GENERATION.format(part=part); producer = base.read(path); folder = hf/producer['output_HF_relative']
        parent = {'part': part, 'protocol': base.binding(path, ROOT), 'RUN': None, 'exit': None}
        if bind_completed:
            parent.update(RUN=base.binding(folder/'RUN.json', hf), exit=base.binding(folder.parent/'EXIT_RECEIPT.json', hf))
            report = base.read_bound(hf, parent['RUN'])
            base.exited(base.read_bound(hf, parent['exit']), parent['RUN'], parent['protocol']['sha256'], producer['id'])
            for phase in report['phases']:
                index = next(i for i, j in enumerate(base.read(folder/'EXECUTED_SOURCES/registered_video_plan.json')['samples']) if j['run_id'] == phase['run_id'])
                candidates[phase['native_source_ordinal'], phase['mode']] = {
                    'video': base.clip_binding(phase['video'], hf), 'generation': phase['generation_receipt'],
                    'registered': base.binding(folder/f'job{index:06d}/RUN.json', hf), 'core': phase['core_metrics'],
                    'native_evidence': phase['native_operand_transfer_evidence'], 'states': phase['states']}
        parents.append(parent)
    dense = {v['source']: v for v in old['videos'] if v['mode'] == 'dense'}
    videos = []
    for case in cohort['sources']:
        videos.append(dense[case['native_source_ordinal']])
        for mode in MODES[1:]:
            videos.append({**q.video_identity(case, mode), **candidates.get((case['native_source_ordinal'], mode), {})})
    paths = {b['path'] for b in old['repo_sources']} | REQUIRED | {GENERATION.format(part=p) for p in (0, 1)}
    plan.update(id=ID, status='DRAFT_NOT_AUTHORIZED', numeric_path='BF16_ORIGINAL', assets_finalized=False,
        automatic_retry=False, dimensions=list(DIMENSIONS), resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES,
        limits=LIMITS, costs=COSTS, policy=POLICY, parents=parents, reused_Dense_quality=proof,
        videos=videos, new_videos=[v for v in videos if v['mode'] != 'dense'],
        output_HF_relative='ffnds-derived/e0_raw_bf16_temporal_video20_quality_20261029/attempt1/worker/results',
        repo_sources=[base.binding(ROOT/p, ROOT) for p in sorted(paths)],
        independent_review='NOT_PERFORMED_AGENT_SERVICE_USAGE_LIMIT')
    validate(plan)
    if bind_completed: verify_generation(plan, hf)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'preflight', 'run'))
    parser.add_argument('--protocol', type=Path); parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path); parser.add_argument('--bind-completed', action='store_true')
    args = parser.parse_args()
    if args.action == 'draft':
        base.require(args.out is not None and not args.out.exists(), 'Fresh draft only'); base.write(args.out, draft(args.hf_root, args.bind_completed))
    elif args.action == 'preflight': preflight(args)
    else: run(args)


def run(args):
    base.precision_environment()
    base.require('torch' not in sys.modules and 'numpy' not in sys.modules,'Fresh isolated quality worker required')
    hf,protocol=args.hf_root.resolve(),args.protocol.resolve();plan,digest=base.read(protocol),base.sha(protocol)
    scope=authorize(plan,digest);cpu=require_cpu(scope,plan,digest,hf);proof=verify_import_proof(plan,hf)
    from worker_identity import require_worker
    require_worker()
    out=(hf/plan['output_HF_relative']).resolve()
    base.require(out.is_relative_to(hf) and out!=hf and not out.exists(),'Fresh confined quality attempt required; no retry')
    out.mkdir(parents=True);base.write(out/'ATTEMPT.json',{'id':ID,'protocol_sha256':digest,'automatic_retry':False})
    started=time.monotonic();monitor=None
    report={'id':ID,'status':'STARTED_TEMPORAL40_EIGHT_AXES_WITH_REUSED_DENSE20','protocol_sha256':digest,
      'numeric_path':'BF16_ORIGINAL','prior_independent_CPU_preflight':cpu,'reused_successful_imports_CPU_preflight':proof,
      'producers':plan['parents'],'reused_Dense_quality':plan['reused_Dense_quality'],
      'reused_Dense_clips':20,'new_candidate_clips':40,
      'core_reports_reused':[v['core'] for v in plan['new_videos']],
      'declared_outer_model_work_budgets':COSTS,'predebited_outer_model_work':{},
      'model_work_accounting':'Original frame/batch budgets; RAFT forwards and iterations observed exactly; other internal work UNKNOWN',
      'opaque_internal_model_forwards':'UNKNOWN','opaque_model_constructions':'UNKNOWN','opaque_checkpoint_loads':'UNKNOWN',
      'completed_dimensions':[],'RAFT_observations':[],'official_module_functions':[],
      'new_scored_videos':40,'independent_image_clusters':20,'official_five_independent_samples':False,
      'temporal_flickering':'NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED','camera_motion':'NOT_RUN_NOT_IN_SCOPE',
      'human_fullplay_review':'NOT_RUN','quality_accepted':False,'native_dir_fd_audit':{'denied_attempts':[]}}
    ledger=base.Ledger(report,limits=LIMITS,persist=lambda:base.write(out/'RUN.json',report))
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        base.require(time.monotonic()-started<=RESOURCES['wall_seconds'],'Quality wall budget exhausted')
        base.require(shutil.disk_usage(out).free>=RESOURCES['free_disk_bytes'],'Quality free disk reserve exceeded')
        base.require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Quality new-output byte budget exceeded')
    def timeout(*_):raise TimeoutError('Fixed3600second temporal quality attempt exhausted; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    try:
        verify_assets(plan,hf);helper,v8=verify_environment(plan,hf);helper.two_cpus();check()
        with ledger('reused_core_reports',40):report['core_reuse_status']='BOUND_ALL81FRAME_REPORTS_NO_RECOMPUTATION'
        env=plan['environment'];base.scope_cache_environment(out)
        report['isolation']=base.configure_import_environment(plan,hf)
        helper.disable_local_ipv6_probe(hf/env['site'],v8)
        numpy=importlib.import_module('numpy');report['pre_Torch_actual_NumPy']=base.module_observation(numpy,'numpy',plan,hf)
        import torch
        report['actual_Torch_origin']=base.module_observation(torch,'torch',plan,hf)
        from startup_gpu_reservation import reserve_before_assets
        from assigned_gpu_resources import AssignedGPUResources
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
          'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        base.require(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Single assigned A100 required')
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir()
        for item in plan['repo_sources']:
            target=snapshot/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(protocol,snapshot/'protocol.json')
        common=importlib.import_module('e0_ub1_consumer_official_quality')
        base.require(Path(common.__file__).resolve()==ROOT/'tools/e0_ub1_consumer_official_quality.py','Frozen helper origin differs')
        binary=Path(shutil.which('nvidia-smi') or '').resolve()
        base.require(str(binary)==env['nvidia_smi']['path'] and base.sha(binary)==env['nvidia_smi']['sha256'],'GPU monitor executable differs')
        source=(hf/env['source_root']).resolve();sys.path.insert(0,str(source))
        from offline_i2v_metric_loaders import submodules
        assets=base.read(ROOT/env['asset_manifest']);modelbase=hf/assets['HF_relative_root'];device=torch.device('cuda:0')
        oldhub=torch.hub.load
        def no_download(*_args,**_kwargs):raise RuntimeError('Network/implicit metric download forbidden')
        torch.hub.load=common.make_local_hub_loader(oldhub,modelbase/'source/dino',{'facebookresearch/dino:main',str(modelbase/'source/dino')})
        def state_load(url,*_args,**kwargs):
            base.require(url=='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth','Unregistered metric checkpoint URL')
            return torch.load(modelbase/'dino_model/dino_vitbase16_pretrain.pth',map_location=kwargs.get('map_location'),weights_only=kwargs.get('weights_only',False))
        torch.hub.load_state_dict_from_url=state_load;torch.hub.download_url_to_file=no_download
        import socket
        socket.socket.connect=no_download;socket.create_connection=no_download
        import subprocess
        native_imports={'denied_attempts':[]};_,_,checked_font=helper.import_audit_guard(out,native_imports)
        subprocess.check_output=checked_font(subprocess.check_output,hf/env['site']/'matplotlib/font_manager.py')
        modules={}
        for dimension in DIMENSIONS:
            modules[dimension],observation=v4.metric_module(dimension,plan,hf)
            report['official_module_functions'].append(observation)
        report['actual_module_origins']=base.actual_import_origins(plan,hf)
        report['actual_distribution_versions']=helper.effective_distributions(base.read(ROOT/v8['lock_path']),hf/env['overlay'])
        report['native_metric_import_audit']=native_imports
        base.require(not native_imports['denied_attempts'],'Protected native import denied an operation')
        def denied(reason):
            def fail():report['native_dir_fd_audit']['denied_attempts'].append(reason);raise RuntimeError(reason)
            return fail
        audit=common.audit_scope(out,AssignedGPUResources.query.__code__,binary,os.environ['FFNDS_PHYSICAL_GPU'])
        sys.addaudithook(v4.v3.normalize_dir_fd_audit(audit,out,report['native_dir_fd_audit'],denied))
        work=out/'work';images=work/'vbench2_beta_i2v/data/crop/16-9';images.mkdir(parents=True)
        for image in plan['images']:(images/image['name']).symlink_to(base.bound(hf,image['binding']))
        os.chdir(work)
        for dimension in DIMENSIONS:
            check();selected=rows_for_dimension(plan['new_videos'],dimension)
            expected_paths=[str(base.bound(hf,x['video'])) for x in selected]
            base.require(len(selected)==COUNTS[dimension],'Forty-clip scene-aware dimension dispatch differs')
            inp=out/(dimension+'_INPUT.json')
            base.write(inp,[{'prompt_en':x['prompt'],'dimension':[dimension],'image_name':x['image_name'],'video_list':[path]} for x,path in zip(selected,expected_paths)])
            report['active_dimension']=dimension
            base.require(dimension not in report['predebited_outer_model_work'],'Duplicate dimension predebit')
            report['predebited_outer_model_work'][dimension]=COSTS[dimension].copy();base.write(out/'RUN.json',report)
            module=modules[dimension]
            base.require(Path(module.__file__).resolve()==source/('vbench2_beta_i2v' if dimension.startswith('i2v_') else 'vbench')/(dimension+'.py'),'Unexpected official module origin')
            if dimension=='i2v_background':
                original_dream=module.dreamsim
                def dream(*args_,**kwargs_):
                    kwargs_.update(cache_dir=str(modelbase/'dreamsim'),device=device);return original_dream(*args_,**kwargs_)
                module.dreamsim=dream
            try:
                # Both the outer call and every intended per-video result are
                # charged before invoking the original evaluator. Failures keep
                # this attempted work and never get a second invocation.
                with ledger('official_dimension_calls'),ledger('new_video_dimension_results',len(selected)):
                    if dimension=='dynamic_degree':
                        with base.raft_observer(module,report['RAFT_observations'],ledger,expected_paths),torch.inference_mode():
                            result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                    elif dimension=='i2v_background':
                        from peft import PeftModel,LoraConfig
                        adapter=modelbase/'dreamsim/ensemble_lora'
                        adapter_sha=next(x['sha256'] for x in plan['hf_bindings'] if x['path'].endswith('/dreamsim/ensemble_lora/adapter_config.json'))
                        with common.dreamsim_adapter_compat(PeftModel,LoraConfig,adapter,adapter_sha),torch.inference_mode():
                            result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                        report['dreamsim_adapter_compatibility']={'config_sha256':adapter_sha,'original_five_lora_keys_only':True,'inactive_extensions_verified':True,'checkpoint_bytes_changed':False}
                    else:
                        with torch.inference_mode():result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                    torch.cuda.synchronize();result=common.plain(result)
                    base.check_result(result,expected_paths,dimension)
                    entry={'dimension':dimension,'result':result,'input_sha256':base.sha(inp)}
                    if dimension=='dynamic_degree':
                        observations=report['RAFT_observations']
                        base.require(len(observations)==40 and {x['video_path'] for x in observations}==set(expected_paths),'Forty actual new RAFT observations required')
                        for row in observations:base.validate_raft(row)
                        base.require(report['completed']['RAFT_forwards']==1600 and report['completed']['RAFT_iterations']==32000,'Actual1600/32000 RAFT work differs')
                        entry['RAFT_observations']=copy.deepcopy(observations)
                    report['completed_dimensions'].append(entry)
                gc.collect();torch.cuda.empty_cache()
                base.require(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'Quality GPU reservation budget exceeded')
            finally:
                if dimension=='i2v_background':module.dreamsim=original_dream
        ledger.finish();base.require(report['predebited_outer_model_work']==COSTS,'Incomplete eight-axis outer budgets')
        base.require(not report['native_dir_fd_audit']['denied_attempts'],'Native directory-FD audit violation')
        report['source_summary']=summarize(plan,hf,report['completed_dimensions'],report['RAFT_observations'])
        report.update(active_dimension=None,status='COMPLETE_TEMPORAL40_EIGHT_AXES_REUSED_DENSE20_QUALITY_NOT_ACCEPTED',peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
        check()
    except BaseException as error:
        report.update(status='FAILED_RAW_QUALITY_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        original_error=sys.exc_info()[1];signal.alarm(0);report['wall_seconds']=time.monotonic()-started
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report()
                base.require(not report['assigned_GPU_resources'].get('violations'),'Quality GPU monitor violation')
            except BaseException as error:
                report.update(status='FAILED_GPU_MONITOR_PRESERVED_NO_RETRY',monitor_error=str(error));base.write(out/'RUN.json',report)
                if original_error is None:raise
        base.write(out/'RUN.json',report)
    return report


if __name__ == '__main__': main()
