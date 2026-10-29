"""Normal free-student early/late original-W2 actions on twelve new images.

Only the fixed W2 action changes. There is no teacher-state replacement, new
fit, temporal cache, feature selection, or online use of future Dense outputs.
The first clip repeats an old all-factor trajectory before any new-source arm.
"""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_s2_depth_private_temporal', ROOT/'tools/e0_raw_bf16_temporal_video20.py')
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
import e0_s2_new12_dense_baseline as acquisition
from raw_bf16_fixed_layer_w2_runtime import FixedLayerW2Runtime
vm, native, io = legacy.vm, legacy.native, legacy.io
MODES = ('value_all', 'value_early10_dense', 'value_late10_dense')
BLOCKS = {MODES[0]: (), MODES[1]: tuple(range(10)), MODES[2]: tuple(range(30, 40))}
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
             'host_available_min_bytes': 128 << 30, 'free_disk_min_bytes': 6 << 30,
             'wall_seconds': 2400, 'outer_seconds': 2700, 'new_HF_bytes': 4 << 30,
             'scope_pair_max_GPUs': 2, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 600,
                 'outer_seconds': 900, 'new_HF_bytes': 8 << 20, 'automatic_retry': False}
DESIGN = {'evidence': 'MULTISOURCE_NORMAL_FREE_STUDENT_FIXED_PHYSICAL_POLICY_CONTRAST',
          'source_unit': 'TWELVE_NEW_DEVELOPMENT_IMAGES_NO_FIT_NO_BLIND_CONFIRMATION',
          'factor_family': 'EXACT_FROZEN_VALUE10_SHARED80_R960', 'rank': 960,
          'modes': list(MODES), 'dense_blocks_by_mode': {k: list(v) for k, v in BLOCKS.items()},
          'action': 'ORIGINAL_W2_ON_OWN_CURRENT_STUDENT_H_AT_FIXED_BLOCKS_ALL_FOUR_STEPS',
          'unchanged': 'ORIGINAL_W1_GELU_B2_GATE_RESIDUAL_ATTENTION_SCHEDULER_NO_TEACHER_STATE_INJECTION',
          'prediction_early': 'EARLY_RESTORATION_CAN_REDUCE_STATE_DIFFERENCES_REACHING_LATER_BLOCKS',
          'prediction_late': 'LATE_READOUT_RECOVERY_CAN_MATTER_MORE_DESPITE_LARGE_OBSERVED_PREFIX_P',
          'competition': 'PROJECTION_FILTERING_NONLINEAR_CONSUMERS_AND_LOCAL_SENSITIVITY_CAN_CHANGE_OR_REVERSE_EARLY_LATE_ORDER',
          'claim_boundary': 'CAUSAL_EFFECT_OF_THESE_COMPLETE_POLICIES_NOT_UNIQUE_PROPAGATION_MECHANISM',
          'actual_Dense_calls_per_mixed_clip': 40, 'actual_factor_sites_per_mixed_clip': 120,
          'prefetch_Dense_early_late': [42, 40], 'prefetch_is_not_equal_cost': True,
          'logical_FFN_MAC_retained_all_factor': (13824*5120+(13824+5120)*960)/(2*13824*5120),
          'logical_FFN_MAC_retained_mixed': .25+.75*(13824*5120+(13824+5120)*960)/(2*13824*5120),
          'terminal': 'ALL81_FRAME_PAIRED_CORE_AND_SAVED_FLOW_LATENT;_SIGNED_STATE_MOTION_VISUAL_GEOMETRY_ANALYSIS_SEPARATELY_REGISTERED',
          'timing': 'OBSERVED_SETUP_FACTOR_PRELOAD_PREFETCH_FFN_FULL_WALL_NOT_FAIR_WARMED_SPEED_CLAIM',
          'official_quality_calls': 0, 'fits': 0, 'independent_FFN_calls': 0,
          'automatic_retry': False, 'quality_or_speed_accepted': False}
NEW_SOURCES = ('tools/e0_s2_new12_fixed_depth_videos.py', 'tests/test_e0_s2_new12_fixed_depth_videos.py',
               'tools/raw_bf16_fixed_layer_w2_runtime.py', 'tests/test_raw_bf16_fixed_layer_w2_runtime.py',
               'tools/e0_s2_causal_worker_proof.py', 'tests/test_e0_s2_causal_worker_proof.py',
               'tools/e0_s2_causal_worker_proof_v2.py', 'tests/test_e0_s2_causal_worker_proof_v2.py',
               'tools/e0_s2_causal_worker_proof_v3.py', 'tests/test_e0_s2_causal_worker_proof_v3.py',
               'tools/e0_s2_causal_worker_proof_v4.py', 'tests/test_e0_s2_causal_worker_proof_v4.py',
               'tools/e0_s2_causal_worker_proof_v5.py', 'tests/test_e0_s2_causal_worker_proof_v5.py',
               'config/e0_s2_causal_idle_a100_master_cpu_20261029_v5.json')


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts')
    return f'E0_S2_NEW12_FIXED_DEPTH_NORMAL_VIDEOS_20261029_PART{part}_V1'


def output(part):
    scope_id(part)
    return f'ffnds-derived/e0_s2_new12_fixed_depth_videos_20261029/part{part}/attempt1/worker/results'


def order(case):
    offset = (case['index']//2) % 3
    return MODES[offset:]+MODES[:offset]


def run_id(case, mode):
    native.need(mode in MODES, 'Unknown fixed W2 action')
    return f's2depth_ord{case["native_source_ordinal"]}_seed42_{mode}'


def expected(mode):
    dense = 4*len(BLOCKS[mode]); loads = dense+(2 if 0 in BLOCKS[mode] else 0)
    return {'dense_compute_sites': dense, 'factor_compute_sites': 160-dense,
            'factor_GEMMs': 2*(160-dense), 'original_W2_prefetches': loads,
            'original_W2_skips': 162-loads}


def limits(part):
    scope_id(part); n = 19
    result = {k: v//2*n for k, v in vm.native_original_limits.items()}
    result.update(original_expert_loads=2, raw_Default_MM_loads=812, buffer_Default_MM_loads=40,
        T5_raw_Default_MM_loads=168, T5_buffer_Default_MM_loads=7,
        core_CPU_reports=18, BF16_factor_GEMM=5120, original_W2_H2D_skips=2586,
        original_W2_H2D_skipped_bytes=2586*vm.ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=80, factor_CPU_tensor_bytes=80*vm.FACTOR_BYTES,
        factor_GPU_family_preloads=1, factor_GPU_H2D_tensors=240,
        factor_GPU_H2D_bytes=80*vm.FACTOR_BYTES,
        prefetch_transfer_observations=19*162, stage_CUDA_event_pairs=19*7,
        state_tensor_D2H=190, state_tensor_D2H_bytes=19*math.prod(native.VIDEO['latent_shape'])*30,
        state_output_writes=190, video_geometry_checks=19,
        matched_reference_noise_byte_checks=36, historical_state_byte_checks=10,
        source_registration_writes=18, reused_Dense_registration_writes=6)
    return result


def parent_proof(path, hf):
    plan = native.read(path); folder = hf/plan['output_HF_relative']
    return {'protocol': native.binding(path, ROOT), 'RUN': native.binding(folder/'RUN.json', hf),
            'exit': native.binding(folder.parent/'EXIT_RECEIPT.json', hf)}


def verify_parent(proof, hf, expected_status):
    plan = native.read(native.bound(ROOT, proof['protocol'])); run = native.read(native.bound(hf, proof['RUN']))
    vm.exited(native.read(native.bound(hf, proof['exit'])), proof['RUN'], proof['protocol']['sha256'], plan['id'])
    native.need(run['status'] == expected_status and not run['assigned_GPU_resources']['violations']
        and run['operation_ledger']['predebited'] == run['operation_ledger']['observed']
        and not run['operation_ledger'].get('forbidden_attempts', []), 'Actual completed parent required')
    return plan, run


def references(plan, hf, payload=False):
    dense_plan, run = verify_parent(plan['Dense_parent'], hf, 'COMPLETE_NEW12_DENSE_BASELINE_PART')
    native.need(dense_plan['models'] == plan['models'] and dense_plan['runtime_overlay'] == plan['runtime_overlay']
        and run['historical_control_all_ten_states_passed'] is True, 'Dense checkpoint/recipe/control differs')
    result = {}
    for case in plan['cases']:
        rows = [r for r in run['phases'] if r['native_source_ordinal'] == case['native_source_ordinal']]
        native.need(len(rows) == 1, 'Exactly one Dense reference')
        phase = rows[0]; gen = native.read(native.bound(hf, phase['generation_receipt']))
        native.need(all(gen[k] == v for k, v in {'mode': 'dense', 'numeric_path': 'BF16_ORIGINAL',
            'source_id': case['source_id'], 'image_sha256': case['image']['sha256'],
            'prompt_sha256': case['prompt_sha256'], 'seed': 42, 'all160_FFN_covered': True,
            'completed_native_flow_steps': 4, 'status': 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'}.items()),
            'Exact Dense source/recipe/video identity differs')
        native.need(gen['protocol_sha256']==plan['Dense_parent']['protocol']['sha256']
            and gen['states']==phase['states'] and gen['video']==phase['video']
            and gen['video']['frames']==81 and gen['video']['fps']==16
            and gen['video']['size']==[832,464] and len(gen['states'])==10
            and gen['scheduler_generator_reset'] is True
            and gen['native_counter_delta']=={'linear_forward_calls':1600,'activation_qdq_calls':0,'weight_quantizations':0},
            'Dense parent/generation/state/video geometry differs')
        video = dict(gen['video']); video['path'] = str(Path(video['path']).relative_to(hf))
        if payload:
            native.bound(hf, video)
            for item in gen['states'].values(): native.bound(hf, item)
        result[case['native_source_ordinal']] = {'video': video, 'generation': phase['generation_receipt'],
            'states': gen['states'], 'protocol_sha256': gen['protocol_sha256']}
    return result


def neutral_reference(plan, hf, payload=False):
    oldplan, old = verify_parent(plan['old_factor_parent'], hf,
        'COMPLETE_MATCHED_TEMPORAL_REPRESENTATION_TEN_SOURCES_TWENTY_FULL160_VIDEOS')
    case = plan['neutral_case']
    rows = [r for r in old['phases'] if r['native_source_ordinal'] == case['native_source_ordinal'] and r['mode'] == 'value10']
    native.need(len(rows) == 1 and oldplan['models'] == plan['models'] and oldplan['factor_bank'] == plan['factor_bank']
        and oldplan['runtime_overlay'] == plan['runtime_overlay'], 'Old neutral factor/recipe differs')
    gen = native.read(native.bound(hf, rows[0]['generation_receipt']))
    native.need(gen['seed'] == 42 and gen['all160_FFN_covered'] is True and len(gen['states']) == 10
        and gen['image_sha256'] == case['image']['sha256'] and gen['prompt_sha256'] == case['prompt_sha256'],
        'Old value10 control identity differs')
    native.need(gen['states']==rows[0]['states'] and gen['video']==rows[0]['video']
        and gen['mode']=='value10' and gen['numeric_path']=='BF16_ORIGINAL'
        and gen['status']=='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
        and gen['protocol_sha256']==plan['old_factor_parent']['protocol']['sha256']
        and gen['source_id']==case['source_id'] and gen['completed_native_flow_steps']==4,
        'Old factor generation/parent identity differs')
    if payload:
        for item in gen['states'].values(): native.bound(hf, item)
    return {'generation': rows[0]['generation_receipt'], 'states': gen['states']}


def factor_rows(plan, hf, payload=False):
    rows = [r for r in legacy.factor_index(plan, hf, False) if r['family'] == 'value10']
    native.need(len(rows) == 80, 'Only complete value10 shared80 bank')
    if payload:
        rows = [{**r, **vm.factor_header(hf/r['binding']['path'], r['tensors'], payload=True)} for r in rows]
        native.need(all(r['sha256'] == r['binding']['sha256'] for r in rows), 'Actual factor payload SHA differs')
    return rows


def draft(part, hf):
    path = ROOT/f'config/e0_s2_new12_dense_baseline_part{part}_20261029_v2.json'
    old = native.read(path); capture = native.read(native.bound(ROOT, old['neutral_parent']))
    inherit = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
               'private_worker_config_sha256', 'execution_private_worker_config_sha256',
               'execution_raw_master_CPU_proof', 'runtime_overlay')
    plan = {k: copy.deepcopy(old[k]) for k in inherit}
    paths = {b['path'] for b in old['repo_sources']} | set(NEW_SOURCES) | {
        str(path.relative_to(ROOT)), legacy.BANK, 'tools/e0_raw_bf16_temporal_video20.py',
        'tools/e0_s2_new12_dense_baseline.py', 'tools/e0_s2_new12_dense_baseline_v2.py'}
    paths.add(f'config/e0_raw_bf16_temporal_video20_part{part}_20261029_v1.json')
    cases = copy.deepcopy(old['cases'][1:]); neutral = copy.deepcopy(old['cases'][0])
    plan.update(id=scope_id(part), part=part, status='DRAFT', assets_finalized=False,
        numeric_path='BF16_ORIGINAL', resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES,
        limits=limits(part), design=DESIGN, cases=cases, neutral_case=neutral,
        source_bank=old['source_bank'], source_cohort=old['source_bank'], factor_bank=capture['factor_bank'],
        Dense_parent=parent_proof(path, hf), old_factor_parent=parent_proof(ROOT/f'config/e0_raw_bf16_temporal_video20_part{part}_20261029_v1.json', hf),
        output_HF_relative=output(part), automatic_retry=False,
        run_ids=[run_id(neutral, MODES[0])]+[run_id(c,m) for c in cases for m in order(c)],
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(paths)])
    worker = native.read(ROOT/'config/e0_s2_causal_idle_a100_master_cpu_20261029_v5.json')
    paths |= {b['path'] for b in worker['repo_sources']}
    plan['repo_sources']=[native.binding(ROOT/p,ROOT) for p in sorted(paths)]
    folder = hf/worker['output_HF_relative']
    plan.update(execution_private_worker_config_sha256=worker['private_worker_config_sha256'],
        execution_raw_master_CPU_proof={'protocol': native.binding(ROOT/'config/e0_s2_causal_idle_a100_master_cpu_20261029_v5.json',ROOT),
            'RUN': native.binding(folder/'RUN.json',hf), 'launch_exit': native.binding(folder.parent/'EXIT_RECEIPT.json',hf)})
    return plan


def validate(plan, frozen=False):
    p = plan['part']; source = native.read(native.bound(ROOT, plan['Dense_parent']['protocol']))
    native.need(plan['id'] == scope_id(p) and plan['resources'] == RESOURCES and plan['limits'] == limits(p)
        and plan['CPU_preflight_resources'] == CPU_RESOURCES and plan['design'] == DESIGN
        and plan['cases'] == source['cases'][1:] and plan['neutral_case'] == source['cases'][0]
        and plan['source_bank'] == source['source_bank'] and plan['output_HF_relative'] == output(p)
        and plan['source_cohort'] == plan['source_bank']
        and plan['numeric_path'] == 'BF16_ORIGINAL' and plan['automatic_retry'] is False
        and plan['run_ids'] == [run_id(plan['neutral_case'], MODES[0])]+[run_id(c,m) for c in plan['cases'] for m in order(c)]
        and len(plan['cases']) == 6 and len(set(plan['run_ids'])) == 19
        and set(NEW_SOURCES) <= {b['path'] for b in plan['repo_sources']}, 'Exact new12 fixed policy scope differs')
    for key in ('models', 'model_view_root', 'model_view_metadata', 'runtime_overlay', 'runtime_versions', 'local_transformers'):
        native.need(plan[key] == source[key], 'Original Dense environment differs')
    worker = native.read(native.bound(ROOT,plan['execution_raw_master_CPU_proof']['protocol']))
    native.need(worker['models']==plan['models'] and worker['private_worker_config_sha256']==plan['execution_private_worker_config_sha256'],
                'Fresh actual-worker CPU proof differs')
    if frozen: native.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True, 'Not frozen')


def authorize(plan, digest):
    validate(plan, True); active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and active['current_session_resource_constraint_20261028']['max_simultaneously_assigned_idle_GPUs'] is None,
        'Current continuation/dynamic authority differs')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(rows) == 1 and rows[0]['protocol_sha256'] == digest
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and rows[0]['resources'] == RESOURCES and rows[0]['limits'] == plan['limits']
        and rows[0]['run_ids'] == plan['run_ids'] and rows[0]['max_gpu_cards'] == 1
        and sum(s.get('assigned_GPU_slots',0) for s in active['experiment_authority']['current_user_authorized_scopes']
            if s.get('id') in {scope_id(0),scope_id(1)}) <= RESOURCES['scope_pair_max_GPUs'],
        'Exact finite two-worker allocation required')
    return rows[0]


def preflight(args):
    start = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol); authorize(plan,digest)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'numpy','torch'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4 and not args.out.exists()
        and args.out.resolve().is_relative_to(args.hf_root.resolve()), 'Fresh confined hidden-CUDA byte worker')
    args.out.mkdir(parents=True); report = {'id':plan['id'], 'protocol_sha256':digest, 'status':'STARTED_BYTE_PROOF'}
    native.write(args.out/'RUN.json',report)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU600s exceeded')));signal.alarm(600)
    try:
        for b in plan['repo_sources']: native.bound(ROOT,b)
        execution_master_adapter(); io.require_execution_master(plan,args.hf_root)
        native.model_metadata(plan);native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
        acquisition.verify_source_provenance(native.read(native.bound(ROOT,plan['source_bank'])),args.hf_root)
        for c in [plan['neutral_case'],*plan['cases']]:native.bound(args.hf_root,c['image'])
        refs = references(plan,args.hf_root,True); neutral = neutral_reference(plan,args.hf_root,True)
        rows = factor_rows(plan,args.hf_root,True)
        native.need(not {'numpy','torch'} & set(sys.modules), 'Numerical import during byte proof')
        report.update(status='PASS_NEW12_FIXED_DEPTH_BYTES_ENV_AND_REFERENCES',source_closure_sha256=native.canonical(plan['repo_sources']),
            factor_bindings=rows,factor_bindings_sha256=native.canonical(rows),references_sha256=native.canonical(refs),
            neutral=neutral,model_calls=0,GPU_calls=0,numerical_imports=False)
    except BaseException as error:
        report.update(status='FAILED_CPU_NO_RETRY',error=str(error),error_type=type(error).__name__);raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-start
        native.need(len(json.dumps(report,ensure_ascii=False).encode())<=CPU_RESOURCES['new_HF_bytes']//2,'CPU proof output exceeded')
        native.write(args.out/'RUN.json',report)


def require_cpu(scope,plan,digest,hf):
    proof=scope['cpu_preflight'];report=native.read(native.bound(hf,proof['RUN']))
    vm.exited(native.read(native.bound(hf,proof['exit'])),proof['RUN'],digest,plan['id'])
    native.need(report['status']=='PASS_NEW12_FIXED_DEPTH_BYTES_ENV_AND_REFERENCES'
        and report['id']==plan['id'] and report['protocol_sha256']==digest
        and report['source_closure_sha256']==native.canonical(plan['repo_sources'])
        and report['factor_bindings_sha256']==native.canonical(report['factor_bindings'])
        and report['model_calls']==report['GPU_calls']==0 and report['numerical_imports'] is False
        and 0<=report['wall_seconds']<=CPU_RESOURCES['wall_seconds'],'Closed CPU proof required')
    for row in report['factor_bindings']:native.need(native.stat_record(hf/row['binding']['path'])==row['stat'],'Factor changed')
    refs=references(plan,hf);native.need(native.canonical(refs)==report['references_sha256'],'Reference changed')
    native.need(neutral_reference(plan,hf)==report['neutral'],'Old factor neutral reference changed')
    return proof,report['factor_bindings'],refs,report['neutral']


def load_cpu_bank(torch,rows,hf,ledger):
    from safetensors.torch import load_file
    result={}
    for row in rows:
        path=hf/row['binding']['path']
        def load():
            native.need(native.stat_record(path)==row['stat'],'Factor stat differs')
            values=load_file(str(path),device='cpu')
            native.need(set(values)==set(vm.SHAPES) and all(v.dtype==torch.bfloat16 and list(v.shape)==vm.SHAPES[k]
                and v.device.type=='cpu' and v.is_contiguous() for k,v in values.items())
                and native.stat_record(path)==row['stat'],'Factor payload layout/stat differs')
            return values
        values=ledger.call('setup',{'factor_CPU_file_loads':1,'factor_CPU_tensor_bytes':vm.FACTOR_BYTES},load)
        for key in legacy.coordinates(row):result[key]=values
    native.need(len(result)==160,'Full160 bank required');return result


_original_geometry=vm.transfer_geometry
def execution_master_adapter():
    import e0_s2_causal_worker_proof_v5 as proof
    io.require_execution_master=proof.require_closed_proof


def transfer_geometry(buffer,destination,block,runtime):
    # A call-local immutable view; never change global runtime mode at compute time.
    view=SimpleNamespace(mode='factor' if runtime.is_factor(block) else 'dense',objects=runtime.objects)
    return _original_geometry(buffer,destination,block,view)


def registration(plan,hf,refs):
    from ffnds_cohort_preflight import audit
    sources=[];jobs=[]
    for c in plan['cases']:
        condition=c['source_id']+':'+c['prompt_sha256']
        sources.append({'source_id':c['source_id'],'image_id':c['image']['sha256'],
            'prompt_id':c['prompt_sha256'],'prompt':c['prompt'],
            'image_path':str((hf/c['image']['path']).resolve()),'image_name':Path(c['image']['path']).name,
            'conditioning_id':condition,'role':'DEVELOPMENT',
            'exposure_history':{'fitting':False,'method_selection':True,'prior_evaluation':True}})
        common={'source_id':c['source_id'],'sample_index':0,'seed':42,
            'initial_noise_id':'raw160_ord'+str(c['native_source_ordinal'])+'_shared_real_seed42',
            'conditioning_id':condition,'scheduler_id':'Wan22StepDistill_shift5_1000_750_500_250',
            'frames':81,'fps':16,'width':832,'height':464}
        dense=refs[c['native_source_ordinal']];gen=native.read(native.bound(hf,dense['generation']))
        jobs.append({**common,'variant':'dense','run_id':gen['run_id'],
            'asset_id':'REUSED_DENSE:'+dense['generation']['sha256']})
        for mode in order(c):
            rid=run_id(c,mode);jobs.append({**common,'variant':mode,'run_id':rid,'asset_id':rid+':new_actual_video'})
    manifest={'schema':'ffnds.cohort.v1','phase':'REGISTERED_REAL_GENERATION_PLAN','mode':'CUSTOM_PAIRED',
        'purpose':'DEVELOPMENT','sources':sources,'samples':jobs,'required_variants':['dense',*MODES],
        'expected_samples_per_source_per_variant':1,'source_bank_id':plan['source_cohort']['sha256'],
        'native_protocol_id':plan['id']}
    native.need(not audit(manifest)['errors'],'Paired fixed policy cohort invalid');return manifest


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(plan,digest)
    proof,rows,refs,neutral=require_cpu(scope,plan,digest,args.hf_root)
    native.need(not {'numpy','torch'} & set(sys.modules),'Fresh numerical process required')
    execution_master_adapter();io.require_execution_master(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    native.model_metadata(plan);local,package_rows=native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions']
        and len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=='1' for k in
        ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'Runtime precision/CPU differs')
    out=(args.hf_root/plan['output_HF_relative']).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=plan['id'],limits(plan['part']),tuple(plan['run_ids']),RESOURCES
    vm.MODES,vm.phase_order,vm.run_id,vm.RESOURCES,vm.transfer_geometry=MODES,order,run_id,RESOURCES,transfer_geometry
    report={'id':plan['id'],'protocol_sha256':digest,'part':plan['part'],'status':'STARTED_FIXED_DEPTH_NORMAL_STUDENT',
        'numeric_path':'BF16_ORIGINAL','CPU_preflight':proof,'design':DESIGN,'phases':[],'triples':[],
        'automatic_retry':False,'quality_or_speed_acceptance':False,'official_quality':'NOT_RUN','new_fits':0}
    monitor=None;counter=io.WrittenBytes(out,RESOURCES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
    for rid in plan['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),128<<20)
    journal=counter.open_journal(out/'OPERATIONS.jsonl');persist=lambda:counter.write_json(out/'RUN.json',report)
    ledger=vm.Ledger(report.setdefault('operation_ledger',{}),lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
    meter=vm.FactorMeter(ledger);current={'rid':None,'skips':[],'dispatch':[]};persist()
    def emit(event):
        if event['event']=='original_W2_H2D_skipped':
            native.need(current['rid'] is not None,'W2 skip outside registered clip')
            ledger.call(current['rid'],{'original_W2_H2D_skips':1,'original_W2_H2D_skipped_bytes':vm.ORIGINAL_W2_BYTES},current['skips'].append,event)
        elif event['event']=='fixed_W2_dispatch_returned':current['dispatch'].append(event)
        else:journal.write(json.dumps({'kind':'runtime_dispatch',**event},separators=(',',':'))+'\n')
    def check():
        if monitor:monitor.check()
        counter.check();available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available>=RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes']
            and time.monotonic()-started<=RESOURCES['wall_seconds'],'Host/disk/wall budget exceeded')
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU40GiB exceeded')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed depth2400s exceeded')));signal.alarm(2400)
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,package_rows),native.NoHiF4();file_guard=io.single_file_bound();file_guard.__enter__()
    try:
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'One assigned A100 required')
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'TF32':False,'FP32_matmul':'highest','BF16_reduced_precision_reduction':True};check()
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();copies=[(ROOT/b['path'],snapshot/b['path'],b) for b in plan['repo_sources']]
        copies.append((args.protocol,snapshot/'protocol.json',native.binding(args.protocol,ROOT)));counter.hold('frozen_metadata',sum(b['bytes'] for _,_,b in copies))
        for src,dst,b in copies:dst.parent.mkdir(parents=True,exist_ok=True);counter.copy_bound(src,dst,b)
        native.need(counter.holds.pop('frozen_metadata')==0,'Source snapshot cost differs')
        with native.OriginalGuard(torch,plan,ledger) as guard:
            at=time.perf_counter();pipe=native.build_pipeline(plan);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-at,original_loader_paths=guard.loaded)
            cpu_bank=load_cpu_bank(torch,rows,args.hf_root,ledger);at=time.perf_counter();gpu_bank=legacy.to_gpu_bank(torch,cpu_bank,ledger,'setup')
            report.update(factor_preload_wall_seconds=time.perf_counter()-at,factor_GPU_bytes=80*vm.FACTOR_BYTES,factor_preloads=1,original_GPU_buffers_retained=True)
            manifest=registration(plan,args.hf_root,refs);counter.write_json(snapshot/'registered_video_plan.json',manifest)
            with FixedLayerW2Runtime(pipe,guard,gpu_bank,factor_forward=vm.core.factor_forward,meter=meter,
                    synchronize=torch.cuda.synchronize,record_stream=lambda t:t.record_stream(torch.cuda.current_stream()),emit=emit) as runtime:
                def phase(case,mode,neutral_clip=False):
                    check();authorize(plan,digest);rid=run_id(case,mode);current.update(rid=rid,skips=[],dispatch=[]);meter.rid=rid
                    runtime.set_dense_blocks(BLOCKS[mode]);runtime.set_mode('factor')
                    before=ledger.record['observed'].get('BF16_factor_GEMM',0)
                    p=vm.native_phase(torch,pipe,guard,runtime,plan,case,mode,out,args.hf_root,digest,report,persist,check,counter,{})
                    exp=expected(mode);calls=current['dispatch'];native.need(len(calls)==160 and {(d['step'],d['block']) for d in calls}=={(s,b) for s in range(4) for b in range(40)},'All160 actual policy dispatches required')
                    native.need(sum(d['kind']=='dense' for d in calls)==exp['dense_compute_sites']
                        and all(d['kind']==('dense' if d['block'] in BLOCKS[mode] else 'factor')
                            and d['expert']==(0 if d['step']<2 else 1) for d in calls)
                        and ledger.record['observed'].get('BF16_factor_GEMM',0)-before==exp['factor_GEMMs']
                        and len(current['skips'])==exp['original_W2_skips']
                        and p['original_W2_H2D_bytes']==exp['original_W2_prefetches']*vm.ORIGINAL_W2_BYTES,'Actual physical policy cost differs')
                    counter.write_json(out/rid/'W2_DISPATCH.json',{'mode':mode,'dense_blocks':list(BLOCKS[mode]),'actual_dispatch':calls,'actual_skips':current['skips'],'expected_and_actual':exp})
                    p.update(fixed_policy=exp,dense_blocks=list(BLOCKS[mode]),W2_dispatch=counter.binding(out/rid/'W2_DISPATCH.json',args.hf_root))
                    if neutral_clip:
                        for name,item in p['states'].items():ledger.call(rid,{'historical_state_byte_checks':1},lambda item=item,name=name:native.need(vm.same_state(item,neutral['states'][name]),'Old all-factor state mismatch; stop before new sources'))
                        report['historical_factor_neutral_all10_passed']=True
                    else:
                        dense=refs[case['native_source_ordinal']];legacy.match_noise(p,dense,ledger)
                        reference_path=native.bound(args.hf_root,dense['video'])
                        reference_stat=native.stat_record(reference_path)
                        for name in ('raw_noise_FP32','initial_noise'):native.bound(args.hf_root,dense['states'][name])
                        legacy.core_report(plan,p,case,dense,args.hf_root,out,counter,ledger,report['raw_model_setup_seconds'])
                        measured=native.read(native.bound(args.hf_root,p['core_metrics']))
                        native.need(measured['reference']['sha256']==dense['video']['sha256']
                            and measured['candidate']['sha256']==p['video']['sha256']
                            and native.stat_record(reference_path)==reference_stat,
                            'Actual decoded CORE video identities changed after CPU proof')
                        index=next(i for i,j in enumerate(manifest['samples']) if j['run_id']==rid);folder=out/f'job{index:06d}';folder.mkdir()
                        ledger.call(rid,{'source_registration_writes':1},counter.write_json,folder/'RUN.json',legacy.registration_record(manifest,index,p,args.hf_root))
                    counter.write_json(out/rid/'PHASE.json',p);persist();check();return p
                phase(plan['neutral_case'],MODES[0],True)
                for c in plan['cases']:
                    dense=refs[c['native_source_ordinal']];old=native.read(native.bound(args.hf_root,dense['generation']));old['generation_receipt']=dense['generation']
                    index=next(i for i,j in enumerate(manifest['samples']) if j['run_id']==old['run_id']);folder=out/f'job{index:06d}';folder.mkdir()
                    ledger.call(run_id(c,order(c)[0]),{'reused_Dense_registration_writes':1},counter.write_json,folder/'RUN.json',legacy.registration_record(manifest,index,old,args.hf_root,reused=True))
                    completed=[phase(c,m)['run_id'] for m in order(c)];report['triples'].append({'source':c['native_source_ordinal'],'order':list(order(c)),'run_ids':completed});persist()
                torch.cuda.synchronize();runtime.bank={};gpu_bank.clear()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified=collect(manifest,out);counter.write_json(out/'VERIFIED_GENERATIONS.json',verified)
            native.need(len(report['phases'])==19 and len(report['triples'])==6 and not counter.holds,'Incomplete fixed depth part')
            report.update(status='COMPLETE_NEW12_FIXED_DEPTH_NORMAL_STUDENT_PART',actual_new_full160_videos=19,
                actual_new_source_candidate_videos=18,actual_reused_Dense_videos=6,actual_flow_steps=76,
                actual_FFN_positions=3040,actual_factor_GEMMs=5120,actual_Dense_W2_compute_sites=480,
                actual_states=190,verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json',args.hf_root),checked_transformers_imports=finder.origins);check()
    except BaseException as error:
        report.update(status='FAILED_FIXED_DEPTH_NO_RETRY',error=str(error),error_type=type(error).__name__);raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources']['violations'],'Final GPU monitor violation')
            except BaseException as error:monitor_error=error;report.update(status='FAILED_FIXED_DEPTH_NO_RETRY',resource_error=str(error))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        for rid in plan['run_ids']:
            path=counter.path(out/rid/'output.mp4')
            if path not in counter.files and path.exists():
                try:counter.observe_external(path,hold=str(path))
                except BaseException as error:counter.accounting_errors.append(str(error))
        report.update(wall_seconds=time.monotonic()-started,output_accounting_checkpoint_before_final_RUN=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',args.hf_root));persist();journal.close();file_guard.__exit__(None,None,None)
        if monitor_error is not None:raise monitor_error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('draft','inspect','preflight','run'))
    parser.add_argument('--part',type=int,choices=(0,1));parser.add_argument('--protocol',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));parser.add_argument('--local-root',type=Path);args=parser.parse_args()
    if args.action=='draft':
        native.need(args.out is not None and not args.out.exists(),'Fresh draft required');native.write(args.out,draft(args.part,args.hf_root))
    elif args.action=='inspect':validate(native.read(args.protocol),True);print('STATIC_ONLY_NOT_MODEL')
    elif args.action=='preflight':preflight(args)
    else:run(args)


if __name__=='__main__':main()
