"""Registered V3 block-prefix replay with CPU-batched control verification.

The V2 full parent must be exited, hash-bound and closed at a block boundary.
Original W2/factor BF16 and typed postwrite semantics come from frozen V1 sources.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import e0_s2_spatial_prefix_replay as v1
native = v1.native

BASE = 'E0_S2_SPATIAL52_SAME_H_BATCHED_PREFIX_REPLAY_V3_20261029'
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 24 << 30,
    'host_available_min_bytes': 64 << 30, 'free_disk_min_bytes': 4 << 30,
    'wall_seconds': 3600, 'outer_seconds': 3900, 'new_HF_bytes': 512 << 20,
    'packet_full_SHA_stream_bytes_max': 40 << 30, 'automatic_retry': False}
TOLERANCE = {'relative': 1e-7, 'absolute': 1e-9, 'phase_wrapped_absolute': 1e-8,
             'near_reference_energy': 1e-20, 'near_threshold_policy': 'RECORD_AND_FAIL_ANY_MISMATCH_NO_EPS_CHANGE'}
CPU_RESOURCES = {'GPU':0,'CPU_threads':4,'wall_seconds':300,'outer_seconds':360,'new_HF_bytes':16 << 20}
EXTRA_SOURCES = {'tools/e0_s2_spatial_prefix_replay_batched_v3.py',
                 'tests/test_e0_s2_spatial_prefix_replay_batched_v3.py',
                 'tools/raw_bf16_spatial_prefix_torch_batched.py',
                 'tests/test_raw_bf16_spatial_prefix_torch_batched.py'}


def output(part):
    return f'ffnds-derived/e0_s2_spatial52_same_h_batched_prefix_replay_v3_20261029/part{part}/attempt1/worker/results'


def need(ok, reason):
    if not ok: raise ValueError(reason)


def work_limits(remaining, controls):
    """Exact planned operation counts; block-level weights are reused within a block."""
    positions = list(remaining) + list(controls)
    need(len(set(positions)) == len(positions), 'Duplicate workload position')
    need(len(controls) in (2, 3), 'Require 2 or 3 controls')
    blocks = len({x[0] for x in positions}); n = len(positions)
    return {'raw_headers': 1, 'raw_W2_tensor_reads': blocks, 'raw_bias_tensor_reads': blocks,
        'raw_H2D_tensors': 2*blocks, 'raw_H2D_bytes': blocks*(5120*13824+5120)*2,
        'factor_file_reads': 2*blocks, 'factor_tensor_SHA_checks': 6*blocks,
        'factor_H2D_tensors': 6*blocks, 'factor_H2D_bytes': 2*blocks*v1.core.vm.FACTOR_BYTES,
        'packet_reads': 3*n, 'packet_tensor_SHA_checks': 27*n, 'row_coordinate_checks': 3*n,
        'original_u_byte_neutrals': 3*n, 'h_H2D_tensors': 3*n,
        'h_H2D_bytes': 3*n*1092*13824*2, 'BF16_original_W2_GEMM': 3*n,
        'BF16_factor_GEMM': 8*n, 'predicted_Y_D2H_tensors': 7*n,
        'predicted_Y_D2H_bytes': 7*n*1092*5120*2, 'counterfactual_postwrite': 7*n,
        'rowLN_calls': 10*n, 'gY_FP64_products': 10*n, 'temporal_comparisons': 6*n,
        'native_anchor_bridges': 6*n, 'rFFT_calls': 32*n, 'prefix_input_comparisons': 2*n,
        'source_position_records': 2*n, 'full_model_calls': 0, 'W1_calls': 0, 'fits': 0,
        'new_videos': 0, 'official_quality_calls': 0}


def parent_limits(n):
    need(type(n) is int and 0 < n < 160 and n % 4 == 0, 'Parent must end at a nonempty complete block prefix')
    positions = [(block, source, step) for block in range(n//4) for source in range(2) for step in range(2)]
    # Operation counts are identical to V3 except there are no comparison controls.
    controls = positions[:2]; nblocks=n//4
    out=work_limits(positions[2:], controls)
    need(len({x[0] for x in positions}) == nblocks, 'Parent block accounting differs')
    return {key: value for key,value in out.items() if value}


def canonical_positions(part, sources):
    need(part in (0,1) and len(sources)==2 and len(set(sources))==2, 'Invalid expert/source identities')
    return [(block, source, step) for block in range(40) for source in sources for step in (2*part,2*part+1)]


def construct_workload(part, sources, completed, control_indices=(0, 1)):
    parent_limits(completed)
    ordered=canonical_positions(part,sources)
    need(len(control_indices) in (2,3) and len(set(control_indices)) == len(control_indices)
         and all(type(i) is int and 0 <= i < completed for i in control_indices), 'Invalid frozen control indices')
    controls=[ordered[i] for i in control_indices]; remaining=ordered[completed:]
    return {'inherited': ordered[:completed], 'controls': controls, 'remaining': remaining,
            'limits': work_limits(remaining, controls)}


def merge_records(part, sources, inherited, completed, added):
    """Merge only after independent parent/control verification; no duplicate override."""
    audit_records(inherited,part,sources,completed)
    remaining=canonical_positions(part,sources)[completed:]
    need(len(added)==2*len(remaining),'Incomplete remaining records')
    for index,(block,source,step) in enumerate(remaining):
        for j,family in enumerate(v1.FAMILIES):
            row=added[2*index+j]
            need((row.get('block'),row.get('source_id'),row.get('step'),row.get('family')) ==
                 (block,source,step,family),'Remaining position order or identity differs')
    merged=list(inherited)+list(added)
    need(len(merged)==320,'Expected all 160 positions with two families')
    return merged


def cpu_batched_pair(replay_arrays, native_arrays, masks, *, spectrum):
    """CPU ndarray adapter for 4-corner view and native bridge; one materialization.

    Operand order: replay T,S,CS,CT; native T,CS. This function does not meter
    operations, execute model code or control resource budgets.
    """
    import torch
    import raw_bf16_spatial_prefix_torch_batched as batched
    need(len(replay_arrays)==4 and len(native_arrays)==2,'Expected four replay and two native arrays')
    replay=[torch.from_numpy(x) for x in replay_arrays]
    native_inputs=[torch.from_numpy(x) for x in native_arrays]
    session=batched.Session()
    tree={'view':session.temporal_views(*replay,masks,spectrum=spectrum),
          'bridge':session.native_rounding_bridge(*native_inputs,replay[0],replay[2],masks,spectrum=spectrum)}
    return session.materialize(tree)


def audit_records(records, part, sources, completed):
    expected = canonical_positions(part, sources)[:completed]
    need(len(records) == 2*completed, 'Exactly two candidate records per completed position required')
    for i, (block, source, step) in enumerate(expected):
        for j, family in enumerate(v1.FAMILIES):
            row=records[2*i+j]
            need((row.get('block'),row.get('source_id'),row.get('step'),row.get('family')) ==
                 (block,source,step,family), 'Wrong source, step, family, or execution-order prefix')
    return expected


def compare_records(expected, actual, tolerance=TOLERANCE):
    """Compare every field; null/bool exactly, floats with explicit tolerances.

    Near-zero reference energies are annotated, but never silently exempted.
    """
    failures=[]; near=[]; stats={'numbers':0,'max_absolute':0.0,'max_scaled_error':0.0}
    def visit(a,b,path,near_context=False):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a)!=set(b): failures.append((path,'key mismatch'));return
            e=a.get('reference_energy',a.get('native_reference_energy'))
            nearby=near_context or (type(e) in (float,int) and math.isfinite(e) and e<=tolerance['near_reference_energy'])
            if nearby and not near_context: near.append(path)
            for k in a:visit(a[k],b[k],path+'/'+k,nearby)
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):failures.append((path,'array mismatch'));return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i),near_context)
        elif a is None or isinstance(a,(str,bool)):
            if type(a)!=type(b) or a!=b: failures.append((path,'identity/null/bool mismatch'))
        elif type(a) in (int,float):
            if type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b):
                failures.append((path,'nonfinite or type mismatch'));return
            if type(a) is int:
                if type(b) is not int or a!=b:failures.append((path,'integer mismatch'))
                return
            delta=abs(a-b)
            if path.endswith('candidate_phase_radians'):
                delta=abs(math.atan2(math.sin(a-b),math.cos(a-b))); allowed=tolerance['phase_wrapped_absolute']
            else: allowed=tolerance['absolute']+tolerance['relative']*abs(a)
            stats['numbers']+=1;stats['max_absolute']=max(stats['max_absolute'],delta)
            stats['max_scaled_error']=max(stats['max_scaled_error'],delta/allowed)
            if delta>allowed:failures.append((path,'numerical tolerance exceeded'))
        else: failures.append((path,'unsupported field type'))
    visit(expected,actual,'$')
    return {'passed':not failures,'failures':failures,'near_reference_paths':near,'numeric_summary':stats}


def verify_parent(hf, proof):
    """Require actual exited V2, full file SHA/receipt and exactly closed whole-block ledger."""
    hf=Path(hf).resolve(); plan=native.read(native.bound(ROOT,proof['protocol']))
    run=native.read(native.bound(hf,proof['RUN'])); receipt=native.read(native.bound(hf,proof['exit']))
    need(plan['id'].startswith('E0_S2_SPATIAL52_SAME_H_LOCAL_PREFIX_REPLAY_LAUNCH_V2_20261029_PART')
         and run['id']==receipt['id']==plan['id'] and run['protocol_sha256']==receipt['protocol_sha256']==proof['protocol']['sha256']
         and receipt['run_sha256']==proof['RUN']['sha256'] and receipt['process_exited'] is True
         and type(receipt['child_returncode']) is int and receipt['child_returncode'] == 1
         and receipt.get('timeout') is False and receipt['automatic_retry'] is False
         and run['status']=='FAILED_LOCAL_PREFIX_REPLAY_PRESERVED_NO_RETRY'
         and run.get('error_type') == 'ValueError' and run.get('error') == 'Exact replay not registered'
         and run.get('assigned_GPU_resources', {}).get('violations') == [],
         'V2 parent must be actually exited and failed, with immutable status and receipt')
    need(plan['id'] == 'E0_S2_SPATIAL52_SAME_H_LOCAL_PREFIX_REPLAY_LAUNCH_V2_20261029_PART'+str(plan['part']),
         'Parent exact expert identity differs')
    completed=run['completed_positions']; expected=parent_limits(completed)
    need(run['operation_ledger']['predebited']==run['operation_ledger']['observed']==expected,
         'Parent operations are not exactly a completed whole-block prefix')
    positions_file=native.bound(hf,run['positions']); operations_file=native.bound(hf,run['operations'])
    records=[json.loads(line) for line in positions_file.read_text().splitlines()]
    totals={}
    for line in operations_file.read_text().splitlines():
        event=json.loads(line);need(event.get('kind')=='predebit','Unknown operation journal entry')
        for key,count in event['costs'].items():totals[key]=totals.get(key,0)+count
    need(totals==expected,'Parent journal count mismatch')
    sources=[case['source_id'] for case in plan['input_index']['cases']]
    positions=audit_records(records,plan['part'],sources,completed)
    return {'completed_positions':completed,'positions':positions,'records':records,'part':plan['part'],
            'source_order':sources,'parent_plan':plan,'parent_report':run}


def make_unbound(part):
    need(part in (0,1),'Invalid part')
    return {'id':BASE+'_PART'+str(part),'status':'DRAFT_NOT_BOUND_NOT_EXECUTABLE',
        'parent':'NOT_BOUND_REQUIRES_ACTUAL_V2_RUN_EXIT_FULL_SHA_AND_COMPLETE_BLOCK_PREFIX',
        'resources':RESOURCES,'control_tolerance':TOLERANCE,'controls':'TWO_OR_THREE_AFTER_PARENT_EXIT',
        'remaining':'NOT_DETERMINABLE_UNTIL_PARENT_EXIT','native_runner':'IMPLEMENTED_NOT_RUN',
        'cpu_preflight':'IMPLEMENTED_NOT_RUN','registration':'REQUIRES_BOUND_PARENT_AND_INDEPENDENT_REVIEW'}


def draft(hf,part,proof=None):
    if proof is None: return make_unbound(part)
    parent=verify_parent(hf,proof); need(parent['part']==part,'Parent expert differs')
    plan0=parent['parent_plan']; completed=parent['completed_positions']
    indices=(0,1,completed-1); work=construct_workload(part,parent['source_order'],completed,indices)
    paths={b['path'] for b in plan0['repo_sources']} | EXTRA_SOURCES
    for binding in plan0['repo_sources']:native.bound(ROOT,binding)
    return {'id':BASE+'_PART'+str(part),'part':part,'status':'DRAFT_NOT_AUTHORIZED','assets_finalized':False,
        'automatic_retry':False,'parent':proof,'parent_completed_positions':completed,
        'parent_input_index_sha256':plan0['input_index_sha256'],'source_order':parent['source_order'],
        'control_indices':list(indices),'controls':[list(x) for x in work['controls']],
        'remaining':[list(x) for x in work['remaining']],'limits':work['limits'],'resources':RESOURCES,
        'CPU_preflight_resources':CPU_RESOURCES,'control_tolerance':TOLERANCE,'session_max_GPUs':2,
        'output_HF_relative':output(part),'repo_sources':[native.binding(ROOT/p,ROOT) for p in sorted(paths)],
        'design':'ORIGINAL_BF16_W2_AND_FACTOR_TYPED_POSTWRITE_WITH_BATCHED_TORCH_CPU_STATISTICS',
        'independent_review':None}


def validate(plan,frozen=False):
    part=plan['part']; n=plan['parent_completed_positions']; indices=[0,1,n-1]
    work=construct_workload(part,plan['source_order'],n,indices)
    need(plan['id']==BASE+'_PART'+str(part) and plan['resources']==RESOURCES
         and plan['CPU_preflight_resources']==CPU_RESOURCES and plan['control_tolerance']==TOLERANCE
         and plan['session_max_GPUs']==2 and plan['control_indices']==indices
         and plan['controls']==[list(x) for x in work['controls']]
         and plan['remaining']==[list(x) for x in work['remaining']] and plan['limits']==work['limits']
         and plan['output_HF_relative']==output(part) and plan['automatic_retry'] is False
         and EXTRA_SOURCES <= {b['path'] for b in plan['repo_sources']},'V3 scope differs')
    if frozen:need(plan['status']=='FROZEN_REGISTERED' and plan['assets_finalized'] is True
                   and bool(plan.get('independent_review')),'V3 scope not frozen/reviewed')


def authorize(plan,sha):
    validate(plan,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==plan['id']]
    need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True
         and len(scopes)==1 and scopes[0].get('protocol_sha256')==sha
         and scopes[0].get('limits')==plan['limits'] and scopes[0].get('resources')==RESOURCES
         and scopes[0].get('max_gpu_cards')==1 and scopes[0].get('session_max_GPUs')==2
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'), 'V3 ACTIVE scope differs')
    return scopes[0]


def validate_parent_and_sources(plan,hf):
    parent=verify_parent(hf,plan['parent']); parent_plan=parent['parent_plan']
    need(parent['part']==plan['part'] and parent['completed_positions']==plan['parent_completed_positions']
         and parent['source_order']==plan['source_order']
         and parent_plan['input_index_sha256']==plan['parent_input_index_sha256']
         and native.canonical(parent_plan['input_index'])==plan['parent_input_index_sha256'], 'V3 parent identity changed')
    expected={b['path'] for b in parent_plan['repo_sources']} | EXTRA_SOURCES
    need({b['path'] for b in plan['repo_sources']}==expected,'V3 repo closure differs')
    new_bindings={b['path']:b for b in plan['repo_sources']}
    need(all(new_bindings[b['path']]==b for b in parent_plan['repo_sources']), 'V2 frozen source changed')
    for binding in plan['repo_sources']:native.bound(ROOT,binding)
    v1.core.io.require_execution_master(parent_plan,hf)
    need(v1.collect_index(parent_plan['parents'],hf,plan['part'])==parent_plan['input_index'], 'Captured source index changed')
    for row in parent_plan['input_index']['factors']:
        need(native.stat_record(hf/row['binding']['path'])==row['stat'], 'Factor stat changed')
    return parent


def preflight(args):
    started=time.monotonic(); plan=native.read(args.protocol); sha=native.sha(args.protocol); authorize(plan,sha)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'} & set(sys.modules)
         and len(os.sched_getaffinity(0))<=4 and args.out is not None and not args.out.exists()
         and args.out.resolve().is_relative_to(args.hf_root.resolve()),'Fresh confined CPU-only preflight required')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('V3 preflight timeout')));signal.alarm(300)
    report={'id':plan['id'],'protocol_sha256':sha,'status':'STARTED_V3_METADATA_PREFLIGHT'};native.write(args.out,report)
    try:
        parent=validate_parent_and_sources(plan,args.hf_root)
        report.update(status='PASS_V3_CLOSED_PREFIX_METADATA_ONLY',parent_completed_positions=parent['completed_positions'],
            source_closure_sha256=native.canonical(plan['repo_sources']),parent_proof_sha256=native.canonical(plan['parent']),
            model_calls=0,GPU_calls=0,packet_value_bytes_read=0)
    except BaseException as error:
        report.update(status='FAILED_V3_CPU_PREFLIGHT_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;native.write(args.out,report)


class Meter:
    def __init__(self,limits,report,emit):self.limits,self.report,self.emit=limits,report,emit;report.update(predebited={},observed={})
    def work(self,costs,fn,*args,**kwargs):
        need(all(k in self.limits and type(n) is int and n>0 and self.report['predebited'].get(k,0)+n<=self.limits[k]
                 for k,n in costs.items()),'V3 operation budget exceeded')
        for k,n in costs.items():self.report['predebited'][k]=self.report['predebited'].get(k,0)+n
        self.emit({'kind':'predebit','costs':costs});result=fn(*args,**kwargs)
        for k,n in costs.items():self.report['observed'][k]=self.report['observed'].get(k,0)+n
        return result
    def call(self,key,fn,*args,**kwargs):return self.work({key:1},fn,*args,**kwargs)
    def finish(self):
        need(all(self.report[side].get(k,0)==n for side in ('predebited','observed') for k,n in self.limits.items()),'V3 incomplete work ledger')


def run(args):
    started=time.monotonic();hf=args.hf_root.resolve();plan=native.read(args.protocol);sha=native.sha(args.protocol)
    scope=authorize(plan,sha);proof=scope['cpu_preflight'];cpu=native.read(native.bound(hf,proof['RUN']))
    v1.core.vm.exited(native.read(native.bound(hf,proof['exit'])),proof['RUN'],sha,plan['id'])
    need(cpu['status']=='PASS_V3_CLOSED_PREFIX_METADATA_ONLY'
         and cpu['source_closure_sha256']==native.canonical(plan['repo_sources'])
         and cpu['parent_proof_sha256']==native.canonical(plan['parent']),'Exited V3 CPU gate differs')
    need(not {'torch','numpy'} & set(sys.modules) and len(os.sched_getaffinity(0))<=4,'Fresh numeric worker required')
    parent=validate_parent_and_sources(plan,hf);parent_plan=parent['parent_plan'];index=parent_plan['input_index']
    need({k:importlib.metadata.version(k) for k in parent_plan['runtime_versions']}==parent_plan['runtime_versions'],'Runtime versions differ')
    out=(hf/plan['output_HF_relative']).resolve();need(out.is_relative_to(hf) and not out.exists(),'Fresh confined V3 output required');out.mkdir(parents=True)
    report={'id':plan['id'],'protocol_sha256':sha,'status':'STARTED_V3_CONTROL_REPLAY',
        'parent':plan['parent'],'parent_completed_positions':parent['completed_positions'],'part':plan['part'],
        'controls_completed':0,'remaining_completed':0,'controls_all_passed':False,'packet_bytes_read':0,
        'raw_tensor_IO':{},'raw_tensors':[],'CPU_preflight':proof,'automatic_retry':False,
        'model_calls':0,'new_videos':0,'fits':0,'quality_or_speed_accepted':False}
    monitor=None;counter=v1.core.io.WrittenBytes(out,RESOURCES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
    journal=counter.open_journal(out/'OPERATIONS.jsonl');records=counter.open_journal(out/'POSITIONS.jsonl')
    controls=counter.open_journal(out/'CONTROLS.jsonl');merged=counter.open_journal(out/'MERGED_POSITIONS.jsonl')
    meter=Meter(plan['limits'],report.setdefault('operation_ledger',{}),lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
    persist=lambda:counter.write_json(out/'RUN.json',report)
    persist()
    def check():
        if monitor:monitor.check()
        counter.check()
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        need(time.monotonic()-started<=RESOURCES['wall_seconds'] and available>=RESOURCES['host_available_min_bytes']
             and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes'],'V3 wall/host/disk budget exceeded')
        if 'torch' in sys.modules:need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'V3 GPU memory exceeded')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('V3 3600s exhausted')));signal.alarm(3600)
    try:
        import torch
        reservation,monitor,report['CUDA_reservation']=v1.spectrum.reserved_startup.reserve_before_assets(torch)
        import numpy as np
        from safetensors.torch import load_file
        need(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'One assigned A100 required')
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        need(not any(os.environ.get(k)=='1' for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'TF32 override')
        for item in plan['repo_sources']+[native.binding(args.protocol,ROOT)]:
            target=out/'EXECUTED_SOURCES'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            counter.hold('source',item['bytes']);counter.copy_bound(ROOT/item['path'],target,item,hold='source')
            need(counter.holds.pop('source')==0,'V3 source snapshot accounting differs')
        master=native.read(native.bound(hf,parent_plan['execution_raw_master_CPU_proof']['RUN']))
        kind='high' if plan['part']==0 else 'low';model=next(m for m in parent_plan['models'] if m['kind']==kind)
        master_stat=next(m['stat'] for m in master['model_payloads'] if m['kind']==kind)
        header=meter.call('raw_headers',lambda:native.raw_header_check(native.header(model['path'])))
        need(header['header_sha256']==model['header_sha256'],'Original W2 header differs')
        factors={(r['family'],r['block']):r for r in index['factors']};cases={c['source_id']:c for c in index['cases']}
        parent_by_key={(r['block'],r['source_id'],r['step'],r['family']):r for r in parent['records']}
        new_records=[]
        for phase,positions in (('control',[tuple(x) for x in plan['controls']]),('remaining',[tuple(x) for x in plan['remaining']])):
            if phase=='remaining':
                need(report['controls_completed']==3 and report['controls_all_passed'],'All controls must pass first')
                report['status']='STARTED_V3_REMAINING_REPLAY';persist()
            for block in sorted({x[0] for x in positions}):
                check();authorize(plan,sha)
                weight,wr=meter.work({'raw_W2_tensor_reads':1,'raw_H2D_tensors':1,'raw_H2D_bytes':5120*13824*2},
                    v1.spectrum.read_raw_tensor,torch,model,header,f'blocks.{block}.ffn.2.weight',[5120,13824],master_stat,report['raw_tensor_IO'])
                bias,br=meter.work({'raw_bias_tensor_reads':1,'raw_H2D_tensors':1,'raw_H2D_bytes':5120*2},
                    v1.spectrum.read_raw_tensor,torch,model,header,f'blocks.{block}.ffn.2.bias',[5120],master_stat,report['raw_tensor_IO'])
                report['raw_tensors'].extend([wr,br]);banks={}
                for family in v1.FAMILIES:
                    row=factors[family,block];path=hf/row['binding']['path']
                    need(row['tensors']['right.bias']['raw_tensor_sha256']==br['raw_tensor_sha256']
                         and native.stat_record(path)==row['stat'],'Candidate factor/bias differs')
                    bank=meter.call('factor_file_reads',load_file,str(path),device='cpu')
                    need(native.stat_record(path)==row['stat'] and set(bank)==set(row['tensors']),'Factor changed')
                    for name,tensor in bank.items():
                        expected=row['tensors'][name]
                        meter.call('factor_tensor_SHA_checks',lambda tensor=tensor,expected=expected:
                            need(tensor.dtype==torch.bfloat16 and list(tensor.shape)==expected['shape']
                                 and hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()==expected['raw_tensor_sha256'],'Factor SHA differs'))
                    banks[family]=meter.work({'factor_H2D_tensors':3,'factor_H2D_bytes':v1.core.vm.FACTOR_BYTES},
                        lambda:{k:t.to('cuda') for k,t in bank.items()})
                for _,source_id,step in (p for p in positions if p[0]==block):
                    check();case=cases[source_id];masks=v1.geometry.spatial_masks(case['geometry']);packets={};h_gpu={};entries={}
                    for mode in v1.core.MODES:
                        entry=next(e for e in case['arms'][mode] if e['step']==step and e['block']==block)
                        raw=meter.call('packet_reads',v1.stream_packet,hf,entry['payload'],report)
                        packet=meter.work({'packet_tensor_SHA_checks':9,'row_coordinate_checks':1},v1.data.decode_packet,
                            raw,entry,case['geometry']['selected_rows'],np,v1.core.SHAPES,whole_SHA_already_stream_verified=True)
                        meter.call('original_u_byte_neutrals',v1.typed.neutral_postwrite,packet,np)
                        packets[mode]=packet;entries[mode]=entry;del raw
                        h_gpu[mode]=meter.work({'h_H2D_tensors':1,'h_H2D_bytes':1092*13824*2},
                            lambda:torch.frombuffer(bytearray(v1.typed.bf16_bytes(packet['h'],np)),dtype=torch.bfloat16).reshape(1092,13824).to('cuda'))
                    predicted={};owners={}
                    for mode in v1.core.MODES:
                        key='raw_'+mode;owners[key]=mode;predicted[key]=meter.call('BF16_original_W2_GEMM',torch.addmm,bias,h_gpu[mode],weight.T)
                    for family in v1.FAMILIES:
                        for owner in ('dense',family):
                            key=family+'_'+owner;owners[key]=owner
                            predicted[key]=v1.spectrum.core.factor_forward(h_gpu[owner],banks[family],meter)
                    ys={};gy={};us={};ln={}
                    for key,value in predicted.items():
                        y_cpu=meter.work({'predicted_Y_D2H_tensors':1,'predicted_Y_D2H_bytes':1092*5120*2},lambda:value.detach().cpu())
                        ys[key]=(y_cpu.view(torch.uint16).numpy().astype(np.uint32)<<16).view(np.float32);packet=packets[owners[key]]
                        gy[key]=meter.call('gY_FP64_products',lambda:ys[key].astype(np.float64)*packet['gate'].astype(np.float64))
                        _,us[key]=meter.call('counterfactual_postwrite',v1.typed.postwrite,packet['r'],ys[key],packet['gate'],np)
                        ln[key]=meter.call('rowLN_calls',v1.typed.row_ln,us[key],np)
                    native_gy={};native_ln={}
                    for mode,packet in packets.items():
                        native_gy[mode]=meter.call('gY_FP64_products',lambda packet=packet:packet['Y'].astype(np.float64)*packet['gate'].astype(np.float64))
                        native_ln[mode]=meter.call('rowLN_calls',v1.typed.row_ln,packet['u'],np)
                    for family in v1.FAMILIES:
                        keys=('raw_dense','raw_'+family,family+'_'+family,family+'_dense')
                        result={'source_id':case['source_id'],'source':case['source'],'step':step,'block':block,
                            'family':family,'expert':kind,'original_three_u_byte_neutrals':True,
                            'gate_shift_scale_equal':{n:entries['dense']['tensor_metadata'][n]['raw_tensor_sha256']==entries[family]['tensor_metadata'][n]['raw_tensor_sha256'] for n in ('gate','shift','scale')},
                            'M1092_vs_fullM_Y_relative_SSE':{'raw_teacher':v1.geometry.relative_sse(ys['raw_dense'],packets['dense']['Y'],np),
                                'candidate_student':v1.geometry.relative_sse(ys[family+'_'+family],packets[family]['Y'],np)},
                            'same_first_site_V1_h_raw_equal':{n:np.array_equal(packets['dense'][n],packets[family][n]) for n in ('V1','h')} if step==0 and block==0 else None}
                        result['observed_prefix_input_relative_SSE']=meter.call('prefix_input_comparisons',v1.prefix_inputs,packets['dense'],packets[family],masks,np)
                        for target,arrays in (('gY',gy),('u',us),('rowLN_u',ln)):
                            native_arrays=native_gy if target=='gY' else native_ln if target=='rowLN_u' else {m:p['u'] for m,p in packets.items()}
                            replay=[arrays[k].reshape(21,52,5120) for k in keys]
                            natives=[native_arrays['dense'].reshape(21,52,5120),native_arrays[family].reshape(21,52,5120)]
                            costs={'temporal_comparisons':1,'native_anchor_bridges':1}
                            if target=='gY':costs['rFFT_calls']=16
                            pair=meter.work(costs,cpu_batched_pair,replay,natives,masks,spectrum=target=='gY')
                            result[target]=pair['view'];result.setdefault('native_fullM_anchor_bridge',{})[target]=pair['bridge']
                        if phase=='control':
                            comparison=compare_records(parent_by_key[(block,source_id,step,family)],result)
                            meter.call('source_position_records',lambda:controls.write(json.dumps({'position':[block,source_id,step],
                                'family':family,'comparison':comparison,'record':result},separators=(',',':'),allow_nan=False)+'\n'))
                            need(comparison['passed'],'V3 control numeric mismatch')
                        else:
                            meter.call('source_position_records',lambda:records.write(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'))
                            new_records.append(result)
                    if phase=='control':report['controls_completed']+=1
                    else:report['remaining_completed']+=1
                    persist();del packets,h_gpu,predicted,ys,gy,us,ln,native_gy,native_ln
                del weight,bias,banks
            if phase=='control':report['controls_all_passed']=True;persist()
        torch.cuda.synchronize();meter.finish()
        need(report['remaining_completed']==len(plan['remaining']) and report['controls_completed']==3 and not counter.holds,'V3 incomplete positions')
        all_records=merge_records(plan['part'],plan['source_order'],parent['records'],parent['completed_positions'],new_records)
        for row in all_records:merged.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        report.update(status='COMPLETE_V3_MERGED_160_POSITIONS',actual_independent_images=2,merged_records=len(all_records))
    except BaseException as error:
        report.update(status='FAILED_V3_REPLAY_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report()
                need(not report['assigned_GPU_resources']['violations'],'V3 GPU resource violation')
            except BaseException as error:
                monitor_error=error;report.update(status='FAILED_V3_REPLAY_PRESERVED_NO_RETRY',resource_error=str(error))
        report.update(wall_seconds=time.monotonic()-started,output_accounting=counter.summary(),
            positions=counter.binding(out/'POSITIONS.jsonl',hf),controls=counter.binding(out/'CONTROLS.jsonl',hf),
            merged_positions=counter.binding(out/'MERGED_POSITIONS.jsonl',hf),operations=counter.binding(out/'OPERATIONS.jsonl',hf))
        persist();records.close();controls.close();merged.close();journal.close()
        if monitor_error:raise monitor_error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('draft','audit-parent','preflight','run'))
    parser.add_argument('--part',type=int,choices=(0,1));parser.add_argument('--proof',type=Path)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'))
    args=parser.parse_args()
    if args.action=='draft':
        result=draft(args.hf_root,args.part,json.loads(args.proof.read_text()) if args.proof else None)
        if args.out:
            need(not args.out.exists(),'Fresh V3 draft required');args.out.parent.mkdir(parents=True,exist_ok=True);native.write(args.out,result)
        else:print(json.dumps(result,indent=2))
    elif args.action=='audit-parent':
        data=verify_parent(args.hf_root,json.loads(args.proof.read_text()))
        print(json.dumps({'status':'PARENT_METADATA_VERIFIED_ONLY','completed_positions':data['completed_positions'],
                          'part':data['part'],'source_order':data['source_order']},indent=2))
    elif args.action=='preflight':preflight(args)
    else:run(args)


if __name__=='__main__':main()
