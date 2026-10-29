"""SM120 BF16 fixed-depth videos; four independent, finite matched shards.

No teacher-state injection or cross-hardware Dense reuse. Two runtime-neutral
controls precede each shard's three newly paired sources. Draft/preflight are
byte-only; native model work requires ACTIVE plus a closed SM120 master proof.
"""
import argparse
import copy
import importlib
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
spec = importlib.util.spec_from_file_location('_sm120_private_fixed_depth', ROOT/'tools/e0_s2_new12_fixed_depth_videos.py')
fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
legacy, vm, native, io, acquisition = fixed.legacy, fixed.vm, fixed.native, fixed.io, fixed.acquisition
from raw_bf16_factor_w2_runtime import FactorW2Runtime
from raw_bf16_fixed_layer_w2_runtime import FixedLayerW2Runtime

MASTER = 'config/e0_s2_sm120_bf16_worker_master_cpu_20261029_v1.json'
SOURCE_CONFIG = 'config/e0_s2_new12_dense_baseline_part{}_20261029_v2.json'
SELF = 'tools/e0_s2_new12_sm120_fixed_depth_videos.py'
TEST = 'tests/test_e0_s2_new12_sm120_fixed_depth_videos.py'
MODES = ('dense', 'value_all', 'value_early10_dense', 'value_late10_dense')
BLOCKS = {'dense': tuple(range(40)), **fixed.BLOCKS}
SHARDS = tuple(tuple(acquisition.ORDINALS[i::4]) for i in range(4))
RESOURCES = {'GPU':1,'CPU_threads':4,'GPU_reserved_max_bytes':40<<30,
             'host_available_min_bytes':128<<30,'free_disk_min_bytes':10<<30,
             'wall_seconds':3600,'outer_seconds':3900,'new_HF_bytes':8<<30,
             'scope_group_max_GPUs':4,'automatic_retry':False}
CPU_RESOURCES = {'GPU':0,'CPU_threads':4,'wall_seconds':600,'outer_seconds':900,
                 'new_HF_bytes':8<<20,'automatic_retry':False}
DESIGN = {'evidence':'SM120_MATCHED_ORIGINAL_BF16_NORMAL_FREE_STUDENT_FIXED_POLICY',
          'source_unit':'TWELVE_EXPOSED_DEVELOPMENT_IMAGES_SEED42_NOT_CONFIRMATION',
          'factor_family':'EXACT_FROZEN_VALUE10_SHARED80_R960','rank':960,
          'modes':list(MODES),'dense_blocks_by_mode':{k:list(v) for k,v in BLOCKS.items()},
          'neutral':'TWO_INDEPENDENT_RUNTIME_CONTEXTS_SAME_SM120_SOURCE_SEED_FULL_VIDEO_AND_TEN_STATE_BYTES',
          'dense_reference':'NEW_SAME_SM120_NORMAL_FORWARD_PER_SOURCE_NO_A100_REUSE',
          'unchanged':'W1_GELU_ATTENTION_SCHEDULER_GATE_B2_NO_TEACHER_STATE_INJECTION',
          'prefetch_dense_early_late':[42,40],'prefetch_is_not_equal_cost':True,
          'claim_boundary':'POLICY_CAUSAL_CONTRAST_NOT_UNIQUE_PROPAGATION_MECHANISM',
          'quality_or_speed_accepted':False,'automatic_retry':False}


def scope_id(part):
    native.need(type(part) is int and part in range(4),'Four independent shards required')
    return f'E0_S2_NEW12_SM120_FIXED_DEPTH_MATCHED_20261029_PART{part}_V1'


def output(part):
    scope_id(part)
    return f'ffnds-derived/e0_s2_new12_sm120_fixed_depth_videos_20261029/part{part}/attempt1/worker/results'


def order(case):
    candidates=MODES[1:]; offset=(case['index']//2)%3
    return ('dense',)+candidates[offset:]+candidates[:offset]


def run_id(case, mode):
    native.need(mode in MODES,'Unsupported mode')
    control=case.get('_control_runtime')
    if control:
        native.need(control in ('original','fixed') and mode=='dense','Bad neutral runtime')
        return f's2sm120_control_ord{case["native_source_ordinal"]}_seed42_{control}_dense'
    return f's2sm120_ord{case["native_source_ordinal"]}_seed42_{mode}'


def control_case(plan, name):
    native.need(name in ('original','fixed'),'Unknown control runtime')
    return {**plan['neutral_case'],'_control_runtime':name}


def expected(mode):
    if mode=='dense':
        return {'dense_compute_sites':160,'factor_compute_sites':0,'factor_GEMMs':0,
                'original_W2_prefetches':162,'original_W2_skips':0}
    return fixed.expected(mode)


def limits(part):
    scope_id(part); n=14
    result={k:(v if k in acquisition.SETUP else v//2*n) for k,v in vm.native_original_limits.items()}
    result.update(core_CPU_reports=9,BF16_factor_GEMM=2400,original_W2_H2D_skips=1212,
        original_W2_H2D_skipped_bytes=1212*vm.ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=80,factor_CPU_tensor_bytes=80*vm.FACTOR_BYTES,
        factor_GPU_family_preloads=1,factor_GPU_H2D_tensors=240,factor_GPU_H2D_bytes=80*vm.FACTOR_BYTES,
        prefetch_transfer_observations=n*162,stage_CUDA_event_pairs=n*7,
        state_tensor_D2H=n*10,state_tensor_D2H_bytes=n*math.prod(native.VIDEO['latent_shape'])*30,
        state_output_writes=n*10,video_geometry_checks=n,matched_reference_noise_byte_checks=18,
        neutral_state_byte_checks=10,neutral_video_byte_checks=1,source_registration_writes=12)
    # Historical comparison is deliberately unmeasured on another GPU architecture.
    result['historical_state_byte_checks']=0
    return result


def execution_master_adapter():
    proof=importlib.import_module('e0_s2_sm120_bf16_worker_proof')
    io.require_execution_master=proof.require_closed_proof


def case_selection(sources, part):
    byordinal={c['native_source_ordinal']:c for p in sources for c in p['cases'][1:]}
    native.need(len(byordinal)==12 and set(byordinal)==set(acquisition.ORDINALS),'Source parent coverage differs')
    return [copy.deepcopy(byordinal[o]) for o in SHARDS[part]]


def draft(part,hf):
    scope_id(part)
    sources=[native.read(ROOT/SOURCE_CONFIG.format(i)) for i in (0,1)]
    first=sources[0]; worker=native.read(ROOT/MASTER)
    for source in sources[1:]:
        for key in ('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','runtime_overlay','source_bank'):
            native.need(source[key]==first[key],'Source parents differ')
    capture=native.read(native.bound(ROOT,first['neutral_parent']))
    native.need(worker['models']==first['models'],'SM120 proof original checkpoint identity differs')
    inherit=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers',
             'private_worker_config_sha256','runtime_overlay')
    plan={k:copy.deepcopy(first[k]) for k in inherit}
    neutral=copy.deepcopy(sources[part%2]['cases'][0]); cases=case_selection(sources,part)
    paths={b['path'] for source in sources for b in source['repo_sources']}
    paths|={b['path'] for b in worker['repo_sources']}
    paths|={SELF,TEST,MASTER,'tools/e0_s2_sm120_bf16_worker_proof.py',
            'tools/e0_s2_new12_fixed_depth_videos.py','tests/test_e0_s2_new12_fixed_depth_videos.py',
            'tools/raw_bf16_fixed_layer_w2_runtime.py','tests/test_raw_bf16_fixed_layer_w2_runtime.py',
            'tools/raw_bf16_factor_w2_runtime.py',legacy.BANK,*[SOURCE_CONFIG.format(i) for i in (0,1)]}
    folder=hf/worker['output_HF_relative']
    plan.update(id=scope_id(part),part=part,status='DRAFT',assets_finalized=False,numeric_path='BF16_ORIGINAL',
        resources=copy.deepcopy(RESOURCES),CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES),
        limits=limits(part),design=copy.deepcopy(DESIGN),cases=cases,neutral_case=neutral,
        source_bank=first['source_bank'],source_cohort=first['source_bank'],factor_bank=capture['factor_bank'],
        source_protocols=[native.binding(ROOT/SOURCE_CONFIG.format(i),ROOT) for i in (0,1)],
        execution_private_worker_config_sha256=worker['private_worker_config_sha256'],
        execution_raw_master_CPU_proof={'protocol':native.binding(ROOT/MASTER,ROOT),
            'RUN':native.binding(folder/'RUN.json',hf),'launch_exit':native.binding(folder.parent/'EXIT_RECEIPT.json',hf)},
        output_HF_relative=output(part),automatic_retry=False,
        run_ids=[run_id(control_case({'neutral_case':neutral},name),'dense') for name in ('original','fixed')]
                +[run_id(c,m) for c in cases for m in order(c)],
        repo_sources=[native.binding(ROOT/p,ROOT) for p in sorted(paths)])
    return plan


def validate(plan,frozen=False):
    part=plan['part']; sources=[native.read(native.bound(ROOT,b)) for b in plan['source_protocols']]
    native.need([b['path'] for b in plan['source_protocols']]==[SOURCE_CONFIG.format(i) for i in (0,1)],'Wrong source protocols')
    native.need(plan['id']==scope_id(part) and plan['resources']==RESOURCES and plan['limits']==limits(part)
        and plan['CPU_preflight_resources']==CPU_RESOURCES and plan['design']==DESIGN
        and plan['cases']==case_selection(sources,part) and plan['neutral_case']==sources[part%2]['cases'][0]
        and plan['source_bank']==sources[0]['source_bank']==sources[1]['source_bank']
        and plan['source_cohort']==plan['source_bank'] and plan['output_HF_relative']==output(part)
        and plan['numeric_path']=='BF16_ORIGINAL' and plan['automatic_retry'] is False
        and plan['run_ids']==[run_id(control_case(plan,n),'dense') for n in ('original','fixed')]
            +[run_id(c,m) for c in plan['cases'] for m in order(c)]
        and len(plan['cases'])==3 and len(set(plan['run_ids']))==14,
        'Exact SM120 finite video scope differs')
    for source in sources:
        for key in ('models','model_view_root','model_view_metadata','runtime_overlay','runtime_versions','local_transformers'):
            native.need(plan[key]==source[key],'Original BF16 recipe differs')
    capture=native.read(native.bound(ROOT,sources[0]['neutral_parent']))
    native.need(plan['factor_bank']==capture['factor_bank'],'Frozen value10 factor bank differs')
    proof=plan['execution_raw_master_CPU_proof'];native.need(proof['protocol']['path']==MASTER,'SM120 master protocol differs')
    worker=native.read(native.bound(ROOT,proof['protocol']))
    native.need(worker['models']==plan['models'] and worker['private_worker_config_sha256']==plan['execution_private_worker_config_sha256'],
        'SM120 full-master proof identity differs')
    paths={b['path'] for source in sources for b in source['repo_sources']} | {b['path'] for b in worker['repo_sources']}
    paths|={SELF,TEST,MASTER,'tools/e0_s2_sm120_bf16_worker_proof.py',
        'tools/e0_s2_new12_fixed_depth_videos.py','tests/test_e0_s2_new12_fixed_depth_videos.py',
        'tools/raw_bf16_fixed_layer_w2_runtime.py','tests/test_raw_bf16_fixed_layer_w2_runtime.py',
        'tools/raw_bf16_factor_w2_runtime.py',legacy.BANK,*[SOURCE_CONFIG.format(i) for i in (0,1)]}
    actual=[b['path'] for b in plan['repo_sources']]
    native.need(len(actual)==len(set(actual)) and set(actual)==paths,'Exact repository source closure required')
    sha=plan['execution_private_worker_config_sha256']
    native.need(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha), 'Execution worker SHA required')
    if frozen:native.need(plan['status']=='FROZEN_REGISTERED' and plan['assets_finalized'] is True,'Protocol not frozen')


def authorize(plan,digest):
    validate(plan,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'Continuation inactive')
    rows=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==plan['id']]
    native.need(len(rows)==1 and rows[0]['protocol_sha256']==digest
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
        and rows[0]['resources']==RESOURCES and rows[0]['limits']==plan['limits']
        and rows[0]['run_ids']==plan['run_ids'] and rows[0]['max_gpu_cards']==1
        and sum(s.get('assigned_GPU_slots',0) for s in active['experiment_authority']['current_user_authorized_scopes']
            if s.get('id') in {scope_id(i) for i in range(4)})<=4,'Exact finite scope/allocation required')
    return rows[0]


def preflight(args):
    start=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol);authorize(plan,digest)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'numpy','torch'} & set(sys.modules)
        and len(os.sched_getaffinity(0))<=4 and not args.out.exists()
        and args.out.resolve().is_relative_to(args.hf_root.resolve()),'Fresh confined hidden-CUDA byte worker')
    args.out.mkdir(parents=True);report={'id':plan['id'],'protocol_sha256':digest,'status':'STARTED_SM120_BYTE_PROOF'}
    native.write(args.out/'RUN.json',report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight budget exceeded')));signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for b in plan['repo_sources']:native.bound(ROOT,b)
        execution_master_adapter();io.require_execution_master(plan,args.hf_root)
        native.model_metadata(plan);native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
        acquisition.verify_source_provenance(native.read(native.bound(ROOT,plan['source_bank'])),args.hf_root)
        for case in [plan['neutral_case'],*plan['cases']]:native.bound(args.hf_root,case['image'])
        rows=fixed.factor_rows(plan,args.hf_root,True)
        native.need(not {'numpy','torch'} & set(sys.modules),'Numerical imports during byte preflight')
        report.update(status='PASS_SM120_FIXED_DEPTH_BYTES_ENV_AND_SOURCE',source_closure_sha256=native.canonical(plan['repo_sources']),
            factor_bindings=rows,factor_bindings_sha256=native.canonical(rows),model_calls=0,GPU_calls=0,numerical_imports=False)
    except BaseException as e:
        report.update(status='FAILED_CPU_NO_RETRY',error=str(e),error_type=type(e).__name__);raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-start
        native.need(len(json.dumps(report,ensure_ascii=False).encode())<=CPU_RESOURCES['new_HF_bytes']//2,'CPU report budget exceeded')
        native.write(args.out/'RUN.json',report)


def require_cpu(scope,plan,digest,hf):
    proof=scope['cpu_preflight'];report=native.read(native.bound(hf,proof['RUN']))
    vm.exited(native.read(native.bound(hf,proof['exit'])),proof['RUN'],digest,plan['id'])
    native.need(report['status']=='PASS_SM120_FIXED_DEPTH_BYTES_ENV_AND_SOURCE' and report['id']==plan['id']
        and report['protocol_sha256']==digest and report['source_closure_sha256']==native.canonical(plan['repo_sources'])
        and report['factor_bindings_sha256']==native.canonical(report['factor_bindings'])
        and report['model_calls']==report['GPU_calls']==0 and report['numerical_imports'] is False
        and 0<=report['wall_seconds']<=CPU_RESOURCES['wall_seconds'],'Closed CPU proof required')
    for row in report['factor_bindings']:native.need(native.stat_record(hf/row['binding']['path'])==row['stat'],'Factor changed')
    return proof,report['factor_bindings']


def transfer_geometry(buffer,destination,block,runtime):
    factor=runtime.is_factor(block) if isinstance(runtime,FixedLayerW2Runtime) else runtime.mode=='factor'
    return fixed._original_geometry(buffer,destination,block,SimpleNamespace(mode='factor' if factor else 'dense',objects=runtime.objects))


def validate_neutral(old,new,ledger):
    native.need(old['run_id']!=new['run_id'] and old['native_source_ordinal']==new['native_source_ordinal']
        and old['mode']==new['mode']=='dense' and old['seed']==new['seed']==42
        and set(old['states'])==set(new['states']) and len(old['states'])==10,'Two complete independent neutral runs required')
    for name,item in old['states'].items():
        ledger.call(new['run_id'],{'neutral_state_byte_checks':1},lambda item=item,name=name:
            native.need(vm.same_state(item,new['states'][name]),'SM120 Dense neutral state mismatch'))
    ledger.call(new['run_id'],{'neutral_video_byte_checks':1},lambda:native.need(
        old['video']['sha256']==new['video']['sha256'] and old['video']['bytes']==new['video']['bytes']
        and old['video']['frames']==new['video']['frames']==81 and old['video']['fps']==new['video']['fps']==16
        and old['video']['size']==new['video']['size']==[832,464],'SM120 Dense neutral video bytes/geometry mismatch'))


def registration(plan,hf):
    from ffnds_cohort_preflight import audit
    sources=[];jobs=[]
    for c in plan['cases']:
        condition=c['source_id']+':'+c['prompt_sha256']
        sources.append({'source_id':c['source_id'],'image_id':c['image']['sha256'],
            'prompt_id':c['prompt_sha256'],'prompt':c['prompt'],'image_path':str((hf/c['image']['path']).resolve()),
            'image_name':Path(c['image']['path']).name,'conditioning_id':condition,'role':'DEVELOPMENT',
            'exposure_history':{'fitting':False,'method_selection':True,'prior_evaluation':True}})
        for mode in order(c):
            rid=run_id(c,mode)
            jobs.append({'source_id':c['source_id'],'variant':mode,'sample_index':0,'seed':42,
                'run_id':rid,'asset_id':rid+':new_actual_video',
                'initial_noise_id':'sm120_ord'+str(c['native_source_ordinal'])+'_shared_real_seed42',
                'conditioning_id':condition,'scheduler_id':'Wan22StepDistill_shift5_1000_750_500_250',
                'frames':81,'fps':16,'width':832,'height':464})
    manifest={'schema':'ffnds.cohort.v1','phase':'REGISTERED_REAL_GENERATION_PLAN',
        'mode':'CUSTOM_PAIRED','purpose':'DEVELOPMENT','sources':sources,'samples':jobs,
        'required_variants':list(MODES),'expected_samples_per_source_per_variant':1,
        'source_bank_id':plan['source_cohort']['sha256'],'native_protocol_id':plan['id']}
    native.need(not audit(manifest)['errors'],'SM120 four-arm cohort invalid')
    return manifest


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(plan,digest)
    proof,rows=require_cpu(scope,plan,digest,args.hf_root)
    native.need(not {'numpy','torch'} & set(sys.modules),'Fresh numerical process required')
    execution_master_adapter();io.require_execution_master(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    for case in [plan['neutral_case'],*plan['cases']]:native.bound(args.hf_root,case['image'])
    native.model_metadata(plan);local,package_rows=native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions']
        and len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=='1' for k in
        ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'Runtime precision/CPU differs')
    out=(args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=plan['id'],limits(plan['part']),tuple(plan['run_ids']),RESOURCES
    vm.MODES,vm.phase_order,vm.run_id,vm.RESOURCES,vm.transfer_geometry=MODES,order,run_id,RESOURCES,transfer_geometry
    report={'id':plan['id'],'protocol_sha256':digest,'part':plan['part'],'status':'STARTED_SM120_FIXED_DEPTH_MATCHED',
        'numeric_path':'BF16_ORIGINAL','CPU_preflight':proof,'design':DESIGN,'phases':[],'paired_sources':[],
        'cross_hardware_neutral_comparison':'NOT_MEASURED','automatic_retry':False,
        'quality_or_speed_acceptance':False,'official_quality':'NOT_RUN','new_fits':0}
    monitor=None;counter=io.WrittenBytes(out,RESOURCES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
    for rid in plan['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),128<<20)
    journal=counter.open_journal(out/'OPERATIONS.jsonl');persist=lambda:counter.write_json(out/'RUN.json',report)
    ledger=vm.Ledger(report.setdefault('operation_ledger',{}),lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
    meter=vm.FactorMeter(ledger);current={'rid':None,'skips':[],'dispatch':[]};persist()
    def emit(event):
        if event['event']=='original_W2_H2D_skipped':
            native.need(current['rid'] is not None,'W2 skip outside registered phase')
            ledger.call(current['rid'],{'original_W2_H2D_skips':1,'original_W2_H2D_skipped_bytes':vm.ORIGINAL_W2_BYTES},current['skips'].append,event)
        elif event['event']=='fixed_W2_dispatch_returned':current['dispatch'].append(event)
        else:journal.write(json.dumps({'kind':'runtime_dispatch',**event},separators=(',',':'))+'\n')
    def check():
        if monitor:monitor.check()
        counter.check();available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available>=RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes']
            and time.monotonic()-started<=RESOURCES['wall_seconds'],'Host/disk/wall budget exceeded')
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU40GiB exceeded')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('SM120 fixed-depth wall budget exceeded')));signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,package_rows),native.NoHiF4();file_guard=io.single_file_bound();file_guard.__enter__()
    try:
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0)
            and 'RTX PRO 6000' in torch.cuda.get_device_name(0),'One assigned RTX PRO 6000 SM120 required')
        report.update(GPU_name=torch.cuda.get_device_name(0),GPU_compute_capability=list(torch.cuda.get_device_capability(0)))
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'TF32':False,'FP32_matmul':'highest','BF16_reduced_precision_reduction':True};check()
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();copies=[(ROOT/b['path'],snapshot/b['path'],b) for b in plan['repo_sources']]
        copies.append((args.protocol,snapshot/'protocol.json',native.binding(args.protocol,ROOT)));counter.hold('frozen_metadata',sum(b['bytes'] for _,_,b in copies))
        for src,dst,b in copies:dst.parent.mkdir(parents=True,exist_ok=True);counter.copy_bound(src,dst,b)
        native.need(counter.holds.pop('frozen_metadata')==0,'Source snapshot cost differs')
        manifest=registration(plan,args.hf_root);counter.write_json(snapshot/'registered_video_plan.json',manifest)
        with native.OriginalGuard(torch,plan,ledger) as guard:
            at=time.perf_counter();pipe=native.build_pipeline(plan);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-at,original_loader_paths=guard.loaded)
            cpu_bank=fixed.load_cpu_bank(torch,rows,args.hf_root,ledger);at=time.perf_counter();gpu_bank=legacy.to_gpu_bank(torch,cpu_bank,ledger,'setup')
            report.update(factor_preload_wall_seconds=time.perf_counter()-at,factor_GPU_bytes=80*vm.FACTOR_BYTES,factor_preloads=1,original_GPU_buffers_retained=True)
            runtime_args=dict(factor_forward=vm.core.factor_forward,meter=meter,synchronize=torch.cuda.synchronize,
                              record_stream=lambda t:t.record_stream(torch.cuda.current_stream()),emit=emit)
            def phase(runtime,case,mode,*,is_old_control=False):
                check();authorize(plan,digest);rid=run_id(case,mode);current.update(rid=rid,skips=[],dispatch=[]);meter.rid=rid
                if is_old_control:
                    runtime.set_mode('dense')
                else:
                    runtime.set_dense_blocks(() if mode=='dense' else BLOCKS[mode]);runtime.set_mode('dense' if mode=='dense' else 'factor')
                before=ledger.record['observed'].get('BF16_factor_GEMM',0)
                p=vm.native_phase(torch,pipe,guard,runtime,plan,case,mode,out,args.hf_root,digest,report,persist,check,counter,{})
                exp=expected(mode);calls=current['dispatch']
                if not is_old_control:
                    native.need(len(calls)==160 and {(d['step'],d['block']) for d in calls}=={(s,b) for s in range(4) for b in range(40)}
                        and sum(d['kind']=='dense' for d in calls)==exp['dense_compute_sites']
                        and all(d['kind']==('dense' if d['block'] in BLOCKS[mode] else 'factor')
                            and d['expert']==(0 if d['step']<2 else 1) for d in calls),'Exact all160 policy dispatch differs')
                native.need(ledger.record['observed'].get('BF16_factor_GEMM',0)-before==exp['factor_GEMMs']
                    and len(current['skips'])==exp['original_W2_skips']
                    and p['original_W2_H2D_bytes']==exp['original_W2_prefetches']*vm.ORIGINAL_W2_BYTES,'Actual physical W2 policy differs')
                if not is_old_control:
                    counter.write_json(out/rid/'W2_DISPATCH.json',{'mode':mode,'dense_blocks':list(BLOCKS[mode]),
                        'actual_dispatch':calls,'actual_skips':current['skips'],'expected_and_actual':exp})
                    p['W2_dispatch']=counter.binding(out/rid/'W2_DISPATCH.json',args.hf_root)
                p.update(fixed_policy=exp,dense_blocks=list(BLOCKS[mode]),runtime_context='ORIGINAL_FACTOR_RUNTIME' if is_old_control else 'FIXED_LAYER_RUNTIME')
                counter.write_json(out/rid/'PHASE.json',p);persist();check();return p
            # Each context restores the original objects before the other is installed.
            with FactorW2Runtime(pipe,guard,gpu_bank,**runtime_args) as old_runtime:
                old=phase(old_runtime,control_case(plan,'original'),'dense',is_old_control=True)
            with FixedLayerW2Runtime(pipe,guard,gpu_bank,**runtime_args) as runtime:
                new=phase(runtime,control_case(plan,'fixed'),'dense')
                validate_neutral(old,new,ledger);report['SM120_runtime_neutral_all10_and_video_passed']=True;persist()
                for c in plan['cases']:
                    dense=None;completed=[]
                    for mode in order(c):
                        p=phase(runtime,c,mode);rid=p['run_id'];completed.append(rid)
                        if mode=='dense':
                            video=dict(p['video']);video['path']=str(Path(video['path']).resolve().relative_to(args.hf_root.resolve()))
                            dense={'video':video,'generation':p['generation_receipt'],'states':p['states'],'protocol_sha256':digest}
                        else:
                            native.need(dense is not None,'Fresh same-device Dense must precede candidate')
                            legacy.match_noise(p,dense,ledger)
                            reference_path=native.bound(args.hf_root,dense['video']);reference_stat=native.stat_record(reference_path)
                            legacy.core_report(plan,p,c,dense,args.hf_root,out,counter,ledger,report['raw_model_setup_seconds'])
                            p['Dense_reference_origin']='NEW_SAME_SM120_SCOPE'
                            measured=native.read(native.bound(args.hf_root,p['core_metrics']))
                            native.need(measured['reference']['sha256']==dense['video']['sha256']
                                and measured['candidate']['sha256']==p['video']['sha256']
                                and native.stat_record(reference_path)==reference_stat,'Same-SM120 CORE identity differs')
                        index=next(i for i,j in enumerate(manifest['samples']) if j['run_id']==rid)
                        folder=out/f'job{index:06d}';folder.mkdir()
                        registered=legacy.registration_record(manifest,index,p,args.hf_root)
                        registered['generation_origin']='NEW_ACTUAL_SAME_SM120_DENSE' if mode=='dense' else 'NEW_ACTUAL_SAME_SM120_CANDIDATE'
                        ledger.call(rid,{'source_registration_writes':1},counter.write_json,folder/'RUN.json',registered)
                        counter.write_json(out/rid/'PHASE.json',p);persist();check()
                    report['paired_sources'].append({'source':c['native_source_ordinal'],'order':list(order(c)),'run_ids':completed});persist()
                torch.cuda.synchronize();runtime.bank={};gpu_bank.clear()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified=collect(manifest,out);counter.write_json(out/'VERIFIED_GENERATIONS.json',verified)
            native.need(len(report['phases'])==14 and len(report['paired_sources'])==3 and not counter.holds,'Incomplete paired shard')
            report.update(status='COMPLETE_SM120_NEW12_FIXED_DEPTH_MATCHED_PART',actual_full160_videos=14,
                actual_neutral_videos=2,actual_new_source_videos=12,actual_flow_steps=56,actual_FFN_positions=2240,
                actual_factor_GEMMs=2400,actual_Dense_W2_compute_sites=1040,actual_states=140,
                verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json',args.hf_root),checked_transformers_imports=finder.origins);check()
    except BaseException as error:
        report.update(status='FAILED_SM120_FIXED_DEPTH_PRESERVED_NO_RETRY',error=str(error),error_type=type(error).__name__);raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources']['violations'],'Final GPU monitor violation')
            except BaseException as error:monitor_error=error;report.update(status='FAILED_SM120_FIXED_DEPTH_PRESERVED_NO_RETRY',resource_error=str(error))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        for rid in plan['run_ids']:
            path=counter.path(out/rid/'output.mp4')
            if path not in counter.files and path.exists():
                try:counter.observe_external(path,hold=str(path))
                except BaseException as error:counter.accounting_errors.append(str(error))
        report.update(wall_seconds=time.monotonic()-started,output_accounting_checkpoint_before_final_RUN=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',args.hf_root));persist();journal.close();file_guard.__exit__(None,None,None)
        if monitor_error is not None:raise monitor_error


def verify_closed(args):
    hf=args.hf_root.resolve();plan=native.read(args.protocol);validate(plan,True);digest=native.sha(args.protocol)
    folder=(hf/plan['output_HF_relative']).resolve()
    native.need(args.run.resolve()==folder/'RUN.json' and args.exit.resolve()==folder.parent/'EXIT_RECEIPT.json',
        'Closed RUN/exit must be the registered attempt')
    binding=native.binding(args.run,hf);report=native.read(args.run)
    vm.exited(native.read(args.exit),binding,digest,plan['id'])
    native.need(report.get('id')==plan['id'] and report.get('protocol_sha256')==digest
        and report.get('status')=='COMPLETE_SM120_NEW12_FIXED_DEPTH_MATCHED_PART'
        and report.get('SM120_runtime_neutral_all10_and_video_passed') is True
        and report.get('GPU_compute_capability')==[12,0]
        and 'RTX PRO 6000' in report.get('GPU_name','')
        and report.get('assigned_GPU_resources',{}).get('sampled_isolation_pass') is True
        and not report['assigned_GPU_resources'].get('violations')
        and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds'], 'Closed native evidence differs')
    for side in ('predebited','observed'):
        native.need(all(report['operation_ledger'][side].get(k,0)==v for k,v in plan['limits'].items()),'Closed operation budget differs')
    native.need(not report['operation_ledger'].get('forbidden_attempts')
        and [p['run_id'] for p in report['phases']]==plan['run_ids']
        and len(report['paired_sources'])==3,'Closed phase coverage differs')
    class ReadOnlyLedger:
        def call(self,_rid,_cost,fn):return fn()
    validate_neutral(report['phases'][0],report['phases'][1],ReadOnlyLedger())
    for p in report['phases']:
        gen=native.read(native.bound(hf,p['generation_receipt']))
        native.need(gen['run_id']==p['run_id'] and gen['protocol_sha256']==digest
            and gen['states']==p['states'] and gen['video']==p['video'] and gen['all160_FFN_covered'] is True
            and gen['completed_native_flow_steps']==4 and len(gen['states'])==10,'Actual generated phase differs')
        path=Path(p['video']['path']).resolve();native.need(path.is_relative_to(hf),'Video escaped HF')
        video={**p['video'],'path':str(path.relative_to(hf))};native.bound(hf,video)
        if p.get('runtime_context')=='FIXED_LAYER_RUNTIME':
            dispatch=native.read(native.bound(hf,p['W2_dispatch']))
            native.need(dispatch['expected_and_actual']==expected(p['mode'])
                and len(dispatch['actual_dispatch'])==160 and len(dispatch['actual_skips'])==expected(p['mode'])['original_W2_skips']
                and {(d['step'],d['block']) for d in dispatch['actual_dispatch']}=={(step,block) for step in range(4) for block in range(40)}
                and all(d['kind']==('dense' if d['block'] in BLOCKS[p['mode']] else 'factor')
                        and d['expert']==(0 if d['step']<2 else 1) for d in dispatch['actual_dispatch']),'Closed dispatch differs')
    for case in plan['cases']:
        source=[p for p in report['phases'][2:] if p['native_source_ordinal']==case['native_source_ordinal']]
        native.need([p['mode'] for p in source]==list(order(case)),'Complete source arms/order differ')
        dense=source[0]
        for candidate in source[1:]:
            native.need(candidate.get('Dense_reference_origin')=='NEW_SAME_SM120_SCOPE', 'Wrong Dense reference origin')
            for name in ('raw_noise_FP32','initial_noise'):
                native.need(vm.same_state(candidate['states'][name],dense['states'][name]),'Closed paired noise differs')
            core=native.read(native.bound(hf,candidate['core_metrics']))
            native.need(core['candidate']['sha256']==candidate['video']['sha256']
                and core['reference']['sha256']==dense['video']['sha256']
                and core['identity']['run_sha256']==candidate['generation_receipt']['sha256']
                and core['reference_identity']['run_sha256']==dense['generation_receipt']['sha256'], 'Closed paired CORE identity differs')
    verified=native.read(native.bound(hf,report['verified_generations']))
    native.need(verified['phase']=='COMPLETE_REGISTERED_GENERATION_VERIFIED' and len(verified['samples'])==12
        and {r['run_id'] for r in verified['samples']}==set(plan['run_ids'][2:]),
        'Four-arm verified generation registration incomplete')
    return {'id':plan['id'],'status':'PASS_CLOSED_SM120_FIXED_DEPTH_SHARD','RUN':binding,
        'exit':native.binding(args.exit,hf),'GPU_calls':0,'model_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('draft','inspect','preflight','run','verify-closed'))
    parser.add_argument('--part',type=int,choices=tuple(range(4)));parser.add_argument('--protocol',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));parser.add_argument('--local-root',type=Path)
    parser.add_argument('--run',type=Path);parser.add_argument('--exit',type=Path);args=parser.parse_args()
    if args.action=='draft':
        native.need(args.part is not None and args.out is not None and not args.out.exists(),'Fresh draft/part required');native.write(args.out,draft(args.part,args.hf_root))
    elif args.action=='inspect':validate(native.read(args.protocol),True);print('STATIC_ONLY_NOT_MODEL')
    elif args.action=='preflight':preflight(args)
    elif args.action=='run':run(args)
    else:print(json.dumps(verify_closed(args),indent=2))

if __name__=='__main__':main()
