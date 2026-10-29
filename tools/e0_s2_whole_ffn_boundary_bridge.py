"""SM120 whole-FFN native pre-GELU boundary bridge; finite paired replay.

Only existing exited Dense/value_all parent trajectories on sources 911/456 are
eligible. Each four-step native generation is repeated once with a passive tap.
No model execution occurs on draft or byte-only CPU preflight.
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
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_whole_ffn_bridge_sm120_parent',ROOT/'tools/e0_s2_new12_sm120_fixed_depth_videos.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
vm,native,io,legacy,fixed=parent.vm,parent.native,parent.io,parent.legacy,parent.fixed
ID='E0_S2_WHOLE_FFN_NATIVE_BOUNDARY_BRIDGE_20261029'
OUTPUT='ffnds-derived/e0_s2_whole_ffn_boundary_bridge_20261029/attempt1/worker/results'
PARENTS=('config/e0_s2_new12_sm120_fixed_depth_part0_20261029_v1.json',
         'config/e0_s2_new12_sm120_fixed_depth_part2_20261029_v1.json')
SELF='tools/e0_s2_whole_ffn_boundary_bridge.py'
TEST='tests/test_e0_s2_whole_ffn_boundary_bridge.py'
ORDINALS=(911,456);MODES=('dense','value_all')
SITES=((0,0),(1,20),(2,20),(3,39))
TAUS=(0,1,4,7,10,13,16,20); XY=((7,13),(7,39),(21,13),(21,39))
ROWS=tuple(t*29*52+y*52+x for t in TAUS for y,x in XY)
SHAPES={'V1':[32,5120],'a':[32,13824],'h':[32,13824],'Y':[32,5120],
        'r':[32,5120],'u':[32,5120],'gate':[5120],'rows':[32]}
PACKET_BYTES=32*(4*5120+2*13824)*2+5120*2+32*8
RESOURCES={'GPU':1,'CPU_threads':4,'GPU_reserved_max_bytes':40<<30,'host_available_min_bytes':128<<30,
           'free_disk_min_bytes':10<<30,'wall_seconds':3600,'outer_seconds':3900,
           'new_HF_bytes':1<<30,'automatic_retry':False}
CPU_RESOURCES={'GPU':0,'CPU_threads':4,'wall_seconds':900,'outer_seconds':1020,
               'new_HF_bytes':8<<20,'automatic_retry':False}
DESIGN={'evidence':'NATIVE_PRE_GELU_FULL_M_SELECTED_ROWS_NUMERICAL_BRIDGE_NOT_GENERALIZATION',
        'source_ordinals':list(ORDINALS),'modes':list(MODES),'sites':[list(x) for x in SITES],
        'tau':list(TAUS),'xy':[list(x) for x in XY],'rows':list(ROWS),
        'preactivation':'ORIGINAL_FULL_M_FFN0_RETURN_POST_BIAS_BEFORE_TANH_GELU',
        'sampling':'INDEX_SELECT_ONLY_SAME_NATIVE_STREAM_NO_SMALL_M_RECOMPUTE',
        'reference':'SAME_SM120_PARENT_EACH_ARM_ALL10_STATES_AND_VIDEO_BYTES',
        'online_change':'NONE_PASSIVE_CAPTURE_ONLY','fitting':False,'automatic_retry':False}


def run_id(case,mode):
    native.need(mode in MODES and case['native_source_ordinal'] in ORDINALS,'Unregistered bridge source/mode')
    return f'whole_ffn_bridge_ord{case["native_source_ordinal"]}_seed42_{mode}'


def limits():
    result={k:(v if k in parent.acquisition.SETUP else v//2*4) for k,v in vm.native_original_limits.items()}
    result.update(core_CPU_reports=0,BF16_factor_GEMM=640,original_W2_H2D_skips=324,
        original_W2_H2D_skipped_bytes=324*vm.ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=80,factor_CPU_tensor_bytes=80*vm.FACTOR_BYTES,
        factor_GPU_family_preloads=1,factor_GPU_H2D_tensors=240,factor_GPU_H2D_bytes=80*vm.FACTOR_BYTES,
        prefetch_transfer_observations=4*162,stage_CUDA_event_pairs=4*7,
        state_tensor_D2H=40,state_tensor_D2H_bytes=4*math.prod(native.VIDEO['latent_shape'])*30,
        state_output_writes=40,video_geometry_checks=4,row_index_uploads=1,row_index_upload_bytes=32*8,
        raw_FFN_conditioning_captures=16,operand_row_selections=16*6,actual_grid_geometry_checks=16,
        raw_postwrite_packets=16,packet_tensor_D2H=16*8,packet_tensor_D2H_bytes=16*PACKET_BYTES,
        packet_payload_validations=16,packet_safetensor_writes=16,packet_metadata_writes=16,
        parent_state_byte_checks=40,parent_video_byte_checks=4)
    return result


def parent_proof(path,hf):
    p=native.read(path);folder=hf/p['output_HF_relative']
    return {'protocol':native.binding(path,ROOT),'RUN':native.binding(folder/'RUN.json',hf),
            'exit':native.binding(folder.parent/'EXIT_RECEIPT.json',hf)}


def check_parents(proofs,hf,*,payload=False):
    from types import SimpleNamespace
    native.need(len(proofs)==2, 'Both closed SM120 parent proofs required')
    cases=[];controls={};plans=[]
    for part,ordinal,proof in zip((0,2),ORDINALS,proofs):
        native.need(proof['protocol']['path']==PARENTS[0 if part==0 else 1],'Wrong SM120 parent protocol')
        p=native.read(native.bound(ROOT,proof['protocol']));r=native.read(native.bound(hf,proof['RUN']))
        vm.exited(native.read(native.bound(hf,proof['exit'])),proof['RUN'],proof['protocol']['sha256'],p['id'])
        parent.verify_closed(SimpleNamespace(protocol=ROOT/proof['protocol']['path'],hf_root=Path(hf),
            run=Path(hf)/proof['RUN']['path'],exit=Path(hf)/proof['exit']['path']))
        native.need(p['part']==part and r['part']==part and r['id']==p['id']
            and r['protocol_sha256']==proof['protocol']['sha256']
            and r['status']=='COMPLETE_SM120_NEW12_FIXED_DEPTH_MATCHED_PART'
            and r['SM120_runtime_neutral_all10_and_video_passed'] is True
            and r['actual_full160_videos']==14 and len(r['phases'])==14
            and all(r['operation_ledger'][side].get(k,0)==n for side in ('predebited','observed') for k,n in p['limits'].items())
            and not r['operation_ledger']['forbidden_attempts'] and not r['assigned_GPU_resources']['violations'],
            'Complete exited 14-phase SM120 parent required')
        case=next(c for c in p['cases'] if c['native_source_ordinal']==ordinal);cases.append(case);plans.append(p)
        for mode in MODES:
            selected=[x for x in r['phases'][2:] if x['native_source_ordinal']==ordinal and x['mode']==mode]
            native.need(len(selected)==1,'Unique actual parent arm required')
            phase=selected[0];gen=native.read(native.bound(hf,phase['generation_receipt']))
            native.need(gen['states']==phase['states'] and gen['video']==phase['video']
                and gen['run_id']==phase['run_id'] and gen['protocol_sha256']==proof['protocol']['sha256']
                and gen['source_id']==case['source_id'] and gen['image_sha256']==case['image']['sha256']
                and gen['prompt_sha256']==case['prompt_sha256'] and gen['mode']==mode and gen['seed']==42
                and gen['all160_FFN_covered'] is True and gen['completed_native_flow_steps']==4
                and gen['scheduler_generator_reset'] is True and len(gen['states'])==10
                and gen['video']['frames']==81 and gen['video']['size']==[832,464] and gen['video']['fps']==16,
                'Parent generation identity differs')
            v=vm.native.binding(Path(gen['video']['path']),hf)
            native.need(v['sha256']==gen['video']['sha256'] and v['bytes']==gen['video']['bytes'],'Parent video SHA differs')
            if payload:
                native.bound(hf,v)
                for state in gen['states'].values():native.bound(hf,state)
            controls[ordinal,mode]={'generation':phase['generation_receipt'],'states':gen['states'],'video':v}
    for key in ('models','model_view_root','model_view_metadata','runtime_overlay','runtime_versions','local_transformers',
                'factor_bank','execution_raw_master_CPU_proof','execution_private_worker_config_sha256','source_bank'):
        native.need(plans[0][key]==plans[1][key],'SM120 parent recipe/worker/factor differs: '+key)
    return plans,cases,controls


def draft(hf):
    proofs=[parent_proof(ROOT/p,hf) for p in PARENTS]
    plans,cases,_=check_parents(proofs,hf);old=plans[0]
    inherit=('models','model_view_root','model_view_metadata','runtime_overlay','runtime_versions','local_transformers',
             'factor_bank','execution_raw_master_CPU_proof','execution_private_worker_config_sha256',
             'private_worker_config_sha256','source_bank','source_cohort')
    paths={b['path'] for p in plans for b in p['repo_sources']} | {SELF,TEST,*PARENTS,'tools/e0_s2_new12_sm120_fixed_depth_videos.py'}
    plan={k:copy.deepcopy(old[k]) for k in inherit}
    plan.update(id=ID,status='DRAFT',assets_finalized=False,numeric_path='BF16_ORIGINAL',
        resources=copy.deepcopy(RESOURCES),CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES),limits=limits(),
        design=copy.deepcopy(DESIGN),parents=proofs,cases=cases,output_HF_relative=OUTPUT,automatic_retry=False,
        run_ids=[run_id(c,mode) for c in cases for mode in MODES],
        repo_sources=[native.binding(ROOT/p,ROOT) for p in sorted(paths)])
    return plan


def validate(plan,frozen=False):
    native.need(plan['id']==ID and plan['resources']==RESOURCES and plan['CPU_preflight_resources']==CPU_RESOURCES
        and plan['limits']==limits() and plan['design']==DESIGN and plan['output_HF_relative']==OUTPUT
        and plan['numeric_path']=='BF16_ORIGINAL' and plan['automatic_retry'] is False
        and [c['native_source_ordinal'] for c in plan['cases']]==list(ORDINALS)
        and plan['run_ids']==[run_id(c,m) for c in plan['cases'] for m in MODES]
        and len(set(plan['run_ids']))==4 and len(plan['parents'])==2,'Fixed bridge scope differs')
    parents=[native.read(native.bound(ROOT,p['protocol'])) for p in plan['parents']]
    native.need([p['protocol']['path'] for p in plan['parents']]==list(PARENTS),'Parent protocols differ')
    for p in parents:
        for key in ('models','model_view_root','model_view_metadata','runtime_overlay','runtime_versions','local_transformers',
                    'factor_bank','execution_raw_master_CPU_proof','execution_private_worker_config_sha256','source_bank','source_cohort'):
            native.need(plan[key]==p[key],'Original parent recipe differs: '+key)
    native.need(plan['cases']==[next(c for c in p['cases'] if c['native_source_ordinal']==o) for p,o in zip(parents,ORDINALS)],'Source cases differ')
    paths={b['path'] for p in parents for b in p['repo_sources']} | {SELF,TEST,*PARENTS,'tools/e0_s2_new12_sm120_fixed_depth_videos.py'}
    actual=[b['path'] for b in plan['repo_sources']]
    native.need(len(actual)==len(set(actual)) and set(actual)==paths,'Exact source closure differs')
    if frozen:native.need(plan['status']=='FROZEN_REGISTERED' and plan['assets_finalized'] is True,'Not frozen')


def authorize(plan,digest):
    validate(plan,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID]
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and len(scopes)==1 and scopes[0].get('protocol_sha256')==digest
        and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
        and scopes[0].get('limits')==limits() and scopes[0].get('resources')==RESOURCES
        and scopes[0].get('run_ids')==plan['run_ids'] and scopes[0].get('max_gpu_cards')==1,'Exact bridge ACTIVE scope required')
    return scopes[0]


def canonical_controls(controls):
    return [{'ordinal':o,'mode':mode,**controls[o,mode]} for o in ORDINALS for mode in MODES]


def preflight(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol);authorize(plan,digest)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'numpy','torch'} & set(sys.modules)
        and len(os.sched_getaffinity(0))<=4 and not args.out.exists() and args.out.resolve().is_relative_to(args.hf_root.resolve()),'Fresh hidden-CUDA CPU proof required')
    args.out.parent.mkdir(parents=True,exist_ok=True);report={'id':ID,'protocol_sha256':digest,'status':'STARTED_BRIDGE_BYTES'};native.write(args.out,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bridge CPU preflight exhausted')));signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for b in plan['repo_sources']:native.bound(ROOT,b)
        parent.execution_master_adapter();io.require_execution_master(plan,args.hf_root)
        native.model_metadata(plan);native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
        plans,cases,controls=check_parents(plan['parents'],args.hf_root)
        native.need(cases==plan['cases'],'Source controls changed')
        for case in cases:native.bound(args.hf_root,case['image'])
        rows=fixed.factor_rows(plan,args.hf_root,True)
        control_stats=[]
        for control in canonical_controls(controls):
            for item in [control['video'],*control['states'].values()]:
                path=args.hf_root/item['path'];before=native.stat_record(path)
                native.bound(args.hf_root,item)
                native.need(native.stat_record(path)==before,'Parent control changed while hashing')
                control_stats.append({'path':item['path'],'stat':before})
        native.need(not {'numpy','torch'} & set(sys.modules),'Numerical imports during byte preflight')
        report.update(status='PASS_WHOLE_FFN_BRIDGE_BYTES_NO_MODEL_NO_GPU',control_stats=control_stats,source_closure_sha256=native.canonical(plan['repo_sources']),
            factors=rows,factors_sha256=native.canonical(rows),controls=canonical_controls(controls),
            controls_sha256=native.canonical(canonical_controls(controls)),model_calls=0,GPU_calls=0,numerical_imports=False)
    except BaseException as e:report.update(status='FAILED_CPU_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:signal.alarm(0);report['wall_seconds']=time.monotonic()-started;native.write(args.out,report)


def require_cpu(scope,plan,digest,hf):
    proof=scope['cpu_preflight'];report=native.read(native.bound(hf,proof['RUN']))
    vm.exited(native.read(native.bound(hf,proof['exit'])),proof['RUN'],digest,ID)
    native.need(report['id']==ID and report['protocol_sha256']==digest and report['status']=='PASS_WHOLE_FFN_BRIDGE_BYTES_NO_MODEL_NO_GPU'
        and report['source_closure_sha256']==native.canonical(plan['repo_sources'])
        and report['factors_sha256']==native.canonical(report['factors']) and report['controls_sha256']==native.canonical(report['controls'])
        and report['model_calls']==report['GPU_calls']==0 and report['numerical_imports'] is False
        and 0<=report['wall_seconds']<=CPU_RESOURCES['wall_seconds'],'Exited CPU byte proof differs')
    for row in report['factors']:native.need(native.stat_record(hf/row['binding']['path'])==row['stat'],'Factor changed after CPU proof')
    _,cases,controls=check_parents(plan['parents'],hf)
    native.need(cases==plan['cases'] and canonical_controls(controls)==report['controls'],'Parent controls changed')
    expected_paths=[item['path'] for control in canonical_controls(controls) for item in [control['video'],*control['states'].values()]]
    native.need([row['path'] for row in report['control_stats']]==expected_paths, 'Control byte-proof closure differs')
    for row in report['control_stats']:native.need(native.stat_record(hf/row['path'])==row['stat'], 'Parent control changed after CPU proof')
    return proof,report['factors'],controls


class BridgeTap(vm.LightTap):
    """Capture selected rows at original W1 return, GELU input, W2 and postwrite."""
    def __init__(self,*args,indices,packet_writer,**kwargs):
        super().__init__(*args,**kwargs)
        self.indices,self.packet_writer=indices,packet_writer
        self.pending=None;self.captured=set()

    def select(self,name,value):
        torch=self.guard.torch;p=self.pending
        width=13824 if name in ('a','h') else 5120
        native.need(p is not None and name not in p['values'] and value.dtype==torch.bfloat16
            and list(value.shape)==[31668,width],'Original full-M boundary shape/dtype differs: '+name)
        p['source_layout'][name]={'shape':list(value.shape),'dtype':'bfloat16','stride':list(value.stride())}
        p['values'][name]=self.guard.ledger.call(self.rid,{'operand_row_selections':1},value.index_select,0,self.indices)

    def mm_call(self,mm,value,meta,original):
        slot=self.mapping.get(id(mm));suffix=slot[2] if slot is not None and slot[0]=='block' else None
        if self.pending is not None and suffix in ('ffn.0','ffn.2'):
            native.need(self.block==self.pending['expert_block'],'FFN boundary escaped capture site')
            if suffix=='ffn.0':
                torch=self.guard.torch;bias=getattr(mm,'bias',None)
                native.need(torch.is_tensor(bias) and bias.dtype==torch.bfloat16 and bias.device.type=='cuda'
                    and bias.device.index in (None,0) and tuple(bias.shape)==(13824,)
                    and meta.get('bias_stride') is not None and list(bias.stride())==list(meta['bias_stride'])==[1],
                    'Original W1 bias must be actual BF16 CUDA 13824-vector')
                self.pending['source_layout']['W1_bias']={'shape':list(bias.shape),'dtype':'bfloat16',
                    'device':str(bias.device),'stride':list(bias.stride()),'identity':'ACTUAL_FFN0_MM_BIAS'}
            self.select('V1' if suffix=='ffn.0' else 'h',value)
        result=super().mm_call(mm,value,meta,original)
        if self.pending is not None and suffix=='ffn.0':self.select('a',result)
        if self.pending is not None and suffix=='ffn.2':self.select('Y',result);self.pending['Y_object']=result
        return result

    def install(self):
        super().install()
        for expert,model in enumerate(self.runner.model.model):
            infer=model.transformer_infer;old_ffn,old_write=infer.infer_ffn,infer.post_process
            def ffn(phase,x,attn,shift,scale,gate=None,_expert=expert,_infer=infer,_fn=old_ffn):
                site=(self.step(_expert),int(_infer.block_idx))
                if site not in SITES:return _fn(phase,x,attn,shift,scale,gate)
                native.need(self.pending is None and site not in self.captured and self.block==(_expert,site[1])
                    and gate is not None and gate.numel()==5120,'Duplicate/wrong selected FFN site')
                values=self.guard.ledger.call(self.rid,{'raw_FFN_conditioning_captures':1},
                    lambda:{'gate':gate.reshape(-1).detach().clone()})
                self.pending={'step':site[0],'block':site[1],'expert_block':(_expert,site[1]),'values':values,
                              'source_layout':{},'x_object':x,'gate_object':gate,'ready':False}
                result=_fn(phase,x,attn,shift,scale,gate)
                native.need(result is self.pending.get('Y_object') and set(self.pending['values'])=={'gate','V1','a','h','Y'},
                    'Original FFN selected operands not paired')
                self.pending['ready']=True;return result
            def write(x,y,gate,pre_infer_out=None,_expert=expert,_fn=old_write):
                p=self.pending
                if p is None:return _fn(x,y,gate,pre_infer_out)
                native.need(p['ready'] and self.block==p['expert_block'] and self.step(_expert)==p['step']
                    and x is p['x_object'] and y is p['Y_object'] and gate is p['gate_object'],'Postwrite identity differs')
                self.select('r',x);result=_fn(x,y,gate,pre_infer_out)
                native.need(result is x,'Original BF16 inplace postwrite differs');self.select('u',result)
                def grid():
                    value=list(pre_infer_out.grid_sizes.tuple)
                    native.need(value==[21,29,52],'Actual latent row geometry differs');return value
                actual_grid=self.guard.ledger.call(self.rid,{'actual_grid_geometry_checks':1},grid)
                self.guard.ledger.call(self.rid,{'raw_postwrite_packets':1},self.packet_writer,p,actual_grid)
                self.captured.add((p['step'],p['block']));self.pending=None;return result
            self.stack.enter_context(native.patch_method(infer,'infer_ffn',ffn))
            self.stack.enter_context(native.patch_method(infer,'post_process',write))

    def finish(self):
        super().finish()
        native.need(self.pending is None and self.captured==set(SITES),'Four actual selected boundary sites required')


def verify_neutral(phase,reference,ledger,hf):
    native.need(set(phase['states'])==set(reference['states']) and len(phase['states'])==10,'Ten actual states required')
    for name,item in phase['states'].items():
        ledger.call(phase['run_id'],{'parent_state_byte_checks':1},lambda name=name,item=item:
            native.need(vm.same_state(item,reference['states'][name]),'Same-SM120 observer state mismatch: '+name))
    ledger.call(phase['run_id'],{'parent_video_byte_checks':1},lambda:native.need(
        phase['video']['sha256']==reference['video']['sha256'] and phase['video']['bytes']==reference['video']['bytes'],
        'Same-SM120 observer video SHA mismatch'))


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(plan,digest)
    proof,rows,controls=require_cpu(scope,plan,digest,args.hf_root)
    native.need(not {'numpy','torch'} & set(sys.modules),'Fresh numeric worker required')
    parent.execution_master_adapter();io.require_execution_master(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    for case in plan['cases']:native.bound(args.hf_root,case['image'])
    native.model_metadata(plan);local,package_rows=native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions']
        and len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=='1' for k in
        ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'Runtime precision/CPU differs')
    out=(args.hf_root/OUTPUT).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=ID,limits(),tuple(plan['run_ids']),RESOURCES
    vm.run_id,vm.RESOURCES,vm.transfer_geometry=run_id,RESOURCES,parent.transfer_geometry
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NATIVE_WHOLE_FFN_BRIDGE','numeric_path':'BF16_ORIGINAL',
        'CPU_preflight':proof,'design':DESIGN,'phases':[],'packets':[],'automatic_retry':False,
        'quality_or_speed_acceptance':False,'official_quality':'NOT_RUN','training_updates':0}
    monitor=None;counter=io.WrittenBytes(out,RESOURCES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
    for rid in plan['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),128<<20)
    journal=counter.open_journal(out/'OPERATIONS.jsonl');persist=lambda:counter.write_json(out/'RUN.json',report)
    ledger=vm.Ledger(report.setdefault('operation_ledger',{}),lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
    meter=vm.FactorMeter(ledger);current={'rid':None,'skips':[],'dispatch':[]};persist()
    def emit(event):
        if event['event']=='original_W2_H2D_skipped':
            native.need(current['rid'] is not None,'Skip outside phase')
            ledger.call(current['rid'],{'original_W2_H2D_skips':1,'original_W2_H2D_skipped_bytes':vm.ORIGINAL_W2_BYTES},current['skips'].append,event)
        elif event['event']=='fixed_W2_dispatch_returned':current['dispatch'].append(event)
        else:journal.write(json.dumps({'kind':'runtime_dispatch',**event},separators=(',',':'))+'\n')
    def check():
        if monitor:monitor.check()
        counter.check();available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available>=RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes']
            and time.monotonic()-started<=RESOURCES['wall_seconds'],'Host/disk/wall budget exceeded')
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU memory exceeded')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bridge wall budget exhausted')));signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,package_rows),native.NoHiF4();file_guard=io.single_file_bound();file_guard.__enter__()
    old_tap=vm.LightTap
    try:
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        from safetensors.torch import save_file
        from ub1_consumer_runtime import tensor_sha
        reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0)
            and 'RTX PRO 6000' in torch.cuda.get_device_name(0),'One assigned SM120 required')
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report.update(GPU_name=torch.cuda.get_device_name(0),GPU_compute_capability=[12,0],
                      runtime_precision={'TF32':False,'FP32_matmul':'highest','BF16_reduced_precision_reduction':True});check()
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();copies=[(ROOT/b['path'],snapshot/b['path'],b) for b in plan['repo_sources']]
        copies.append((args.protocol,snapshot/'protocol.json',native.binding(args.protocol,ROOT)));counter.hold('frozen_metadata',sum(b['bytes'] for _,_,b in copies))
        for src,dst,b in copies:dst.parent.mkdir(parents=True,exist_ok=True);counter.copy_bound(src,dst,b)
        native.need(counter.holds.pop('frozen_metadata')==0,'Snapshot byte reserve differs')
        index=ledger.call('setup',{'row_index_uploads':1,'row_index_upload_bytes':len(ROWS)*8},lambda:torch.tensor(ROWS,dtype=torch.int64,device='cuda'))
        with native.OriginalGuard(torch,plan,ledger) as guard:
            at=time.perf_counter();pipe=native.build_pipeline(plan);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-at,original_loader_paths=guard.loaded)
            cpu_bank=fixed.load_cpu_bank(torch,rows,args.hf_root,ledger);at=time.perf_counter();gpu_bank=legacy.to_gpu_bank(torch,cpu_bank,ledger,'setup')
            report.update(factor_preload_wall_seconds=time.perf_counter()-at,factor_GPU_bytes=80*vm.FACTOR_BYTES,original_GPU_buffers_retained=True)
            runtime_args=dict(factor_forward=vm.core.factor_forward,meter=meter,synchronize=torch.cuda.synchronize,
                record_stream=lambda t:t.record_stream(torch.cuda.current_stream()),emit=emit)
            def packet(p,grid):
                native.need(set(p['values'])==set(SHAPES)-{'rows'} and grid==[21,29,52],'Boundary packet fields differ')
                values={**p['values'],'rows':index};cpu={};counter.reserve_payload(PACKET_BYTES)
                for name,value in values.items():
                    cpu[name]=ledger.call(current['rid'],{'packet_tensor_D2H':1,'packet_tensor_D2H_bytes':value.numel()*value.element_size()},lambda value=value:value.detach().cpu().contiguous())
                def verify():
                    native.need(all(list(t.shape)==SHAPES[k] and t.dtype==(torch.int64 if k=='rows' else torch.bfloat16)
                        and bool(torch.isfinite(t).all()) for k,t in cpu.items()) and cpu['rows'].tolist()==list(ROWS), 'Boundary packet shape/dtype/coordinates differ')
                    return {k:{'shape':list(t.shape),'dtype':str(t.dtype).removeprefix('torch.'),'raw_tensor_sha256':tensor_sha(t)} for k,t in cpu.items()}
                meta=ledger.call(current['rid'],{'packet_payload_validations':1},verify)
                folder=out/current['rid'];stem=f'step{p["step"]}_block{p["block"]:02d}';path=folder/(stem+'.safetensors')
                try:ledger.call(current['rid'],{'packet_safetensor_writes':1},save_file,cpu,str(path))
                finally:
                    if path.exists():counter.observe_external(path)
                record={'schema':'ffnds.whole_ffn.native_boundary_bridge.v1','id':ID,'protocol_sha256':digest,
                    'run_id':current['rid'],'source_ordinal':current['ordinal'],'mode':current['mode'],
                    'source_id':current['case']['source_id'],'image_sha256':current['case']['image']['sha256'],
                    'prompt_sha256':current['case']['prompt_sha256'],'seed':42,
                    'source_state':'teacher_input' if current['mode']=='dense' else 'student_input',
                    'step':p['step'],'block':p['block'],'expert':'high' if p['step']<2 else 'low',
                    'grid':grid,'population_rows':31668,'rows':list(ROWS),'tau':list(TAUS),'xy':[list(x) for x in XY],
                    'boundary':'ORIGINAL_FULL_M_POST_W1_BIAS_PRE_GELU_SELECTED_ROWS',
                    'Y_boundary':y_boundary(current['mode']),
                    'source_layout':p['source_layout'],
                    'tensor_metadata':meta,'payload':counter.binding(path,args.hf_root)}
                mp=folder/(stem+'.json');ledger.call(current['rid'],{'packet_metadata_writes':1},counter.write_json,mp,record)
                report['packets'].append({'run_id':current['rid'],'step':p['step'],'block':p['block'],
                                          'metadata':counter.binding(mp,args.hf_root)});persist();check()
            vm.LightTap=lambda *a,**kw:BridgeTap(*a,indices=index,packet_writer=packet,**kw)
            with parent.FixedLayerW2Runtime(pipe,guard,gpu_bank,**runtime_args) as runtime:
                for case in plan['cases']:
                    for mode in MODES:
                        check();authorize(plan,digest);rid=run_id(case,mode);current.update(rid=rid,ordinal=case['native_source_ordinal'],mode=mode,case=case,skips=[],dispatch=[]);meter.rid=rid
                        runtime.set_dense_blocks(());runtime.set_mode('dense' if mode=='dense' else 'factor')
                        before=ledger.record['observed'].get('BF16_factor_GEMM',0)
                        phase=vm.native_phase(torch,pipe,guard,runtime,plan,case,mode,out,args.hf_root,digest,report,persist,check,counter,{})
                        exp=parent.expected(mode);calls=current['dispatch']
                        native.need(len(calls)==160 and {(x['step'],x['block']) for x in calls}=={(s,b) for s in range(4) for b in range(40)}
                            and all(x['kind']==('dense' if mode=='dense' else 'factor') and x['expert']==(0 if x['step']<2 else 1) for x in calls)
                            and ledger.record['observed'].get('BF16_factor_GEMM',0)-before==exp['factor_GEMMs']
                            and len(current['skips'])==exp['original_W2_skips']
                            and phase['original_W2_H2D_bytes']==exp['original_W2_prefetches']*vm.ORIGINAL_W2_BYTES,'Actual runtime dispatch differs')
                        verify_neutral(phase,controls[case['native_source_ordinal'],mode],ledger,args.hf_root)
                        phase['parent_control']=controls[case['native_source_ordinal'],mode];phase['all10_states_and_video_byte_neutral']=True
                        counter.write_json(out/rid/'W2_DISPATCH.json',{'mode':mode,'actual_dispatch':calls,'actual_skips':current['skips'],'expected_and_actual':exp})
                        phase['W2_dispatch']=counter.binding(out/rid/'W2_DISPATCH.json',args.hf_root)
                        counter.write_json(out/rid/'PHASE.json',phase);persist();check()
                torch.cuda.synchronize();runtime.bank={};gpu_bank.clear()
            ledger.finish();native.need(len(report['phases'])==4 and len(report['packets'])==16 and not counter.holds,'Incomplete bridge scope')
            report.update(status='COMPLETE_NATIVE_WHOLE_FFN_BOUNDARY_BRIDGE',actual_full160_videos=4,actual_FFN_sites=640,
                actual_boundary_packets=16,actual_states=40,checked_transformers_imports=finder.origins);check()
    except BaseException as e:
        report.update(status='FAILED_BOUNDARY_BRIDGE_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        vm.LightTap=old_tap;signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources']['violations'],'GPU monitor violation')
            except BaseException as e:monitor_error=e;report.update(status='FAILED_BOUNDARY_BRIDGE_PRESERVED_NO_RETRY',resource_error=str(e))
        for finder_item in (finder,no_hif4):
            if finder_item in sys.meta_path:sys.meta_path.remove(finder_item)
        for rid in plan['run_ids']:
            path=counter.path(out/rid/'output.mp4')
            if path not in counter.files and path.exists():
                try:counter.observe_external(path,hold=str(path))
                except BaseException as e:counter.accounting_errors.append(str(e))
        report.update(wall_seconds=time.monotonic()-started,output_accounting_checkpoint_before_final_RUN=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',args.hf_root));persist();journal.close();file_guard.__exit__(None,None,None)
        if monitor_error is not None:raise monitor_error


def verify_packet_header(path):
    path=Path(path)
    with path.open('rb') as f:
        raw=f.read(8);native.need(len(raw)==8,'Short safetensors header')
        length=struct.unpack('<Q',raw)[0];native.need(0<length<=1<<20,'Invalid safetensors header length')
        header=json.loads(f.read(length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    native.need(set(tensors)==set(SHAPES),'Packet header tensor names differ')
    ranges=[]
    for name,shape in SHAPES.items():
        row=tensors[name];dtype='I64' if name=='rows' else 'BF16'
        native.need(row['shape']==shape and row['dtype']==dtype,'Packet header shape/dtype differs')
        lo,hi=row['data_offsets'];native.need(type(lo) is int and type(hi) is int and 0<=lo<=hi
            and hi-lo==math.prod(shape)*(8 if name=='rows' else 2),'Packet header byte extent differs')
        ranges.append((lo,hi))
    cursor=0
    for lo,hi in sorted(ranges):native.need(lo==cursor,'Packet header offsets overlap or gap');cursor=hi
    native.need(cursor==PACKET_BYTES and path.stat().st_size==8+length+cursor,'Packet payload size differs')


def y_boundary(mode):
    native.need(mode in ('dense','value_all'),'Unexpected bridge mode')
    return 'ORIGINAL_DENSE_W2_OUTPUT' if mode=='dense' else 'ACTUAL_STUDENT_FACTOR_W2_OUTPUT_NOT_ORIGINAL_DENSE_W2'


def packet_geometry_matches(packet):
    return (packet.get('step') in (0,1,2,3) and packet.get('grid')==[21,29,52] and packet.get('population_rows')==31668
            and packet.get('tau')==list(TAUS) and packet.get('xy')==[list(x) for x in XY]
            and packet.get('expert')==('high' if packet.get('step') in (0,1) else 'low')
            and packet.get('rows')==list(ROWS))


def verify_closed(args):
    hf=args.hf_root.resolve();plan=native.read(args.protocol);validate(plan,True);digest=native.sha(args.protocol);out=(hf/OUTPUT).resolve()
    native.need(args.run.resolve()==out/'RUN.json' and args.exit.resolve()==out.parent/'EXIT_RECEIPT.json','Wrong bridge RUN/exit path')
    binding=native.binding(args.run,hf);report=native.read(args.run)
    vm.exited(native.read(args.exit),binding,digest,ID)
    native.need(report.get('id')==ID and report.get('protocol_sha256')==digest
        and report.get('status')=='COMPLETE_NATIVE_WHOLE_FFN_BOUNDARY_BRIDGE'
        and report.get('GPU_compute_capability')==[12,0] and 'RTX PRO 6000' in report.get('GPU_name','')
        and not report['assigned_GPU_resources']['violations'] and report['assigned_GPU_resources']['sampled_isolation_pass'] is True
        and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds']
        and all(report['operation_ledger'][side].get(k,0)==n for side in ('predebited','observed') for k,n in limits().items())
        and not report['operation_ledger']['forbidden_attempts']
        and [p['run_id'] for p in report['phases']]==plan['run_ids']
        and report.get('actual_full160_videos')==4 and report.get('actual_FFN_sites')==640
        and report.get('actual_boundary_packets')==16 and report.get('actual_states')==40
        and report['output_accounting_checkpoint_before_final_RUN']['peak_tracked_bytes_plus_reserved_and_atomic_temp']<=RESOURCES['new_HF_bytes']
        and report['output_accounting_checkpoint_before_final_RUN']['unconsumed_reserved_bytes']==0
        and not report['output_accounting_checkpoint_before_final_RUN']['accounting_errors']
        and len(report['packets'])==16,'Closed bridge execution differs')
    _,_,controls=require_cpu({'cpu_preflight':report['CPU_preflight']},plan,digest,hf)
    _,cases,_=check_parents(plan['parents'],hf)
    native.need(cases==plan['cases'],'Parent lineage differs')
    class ReadOnlyLedger:
        def call(self,_rid,_cost,fn):return fn()
    for phase in report['phases']:
        ref=controls[phase['native_source_ordinal'],phase['mode']]
        native.need(phase['all10_states_and_video_byte_neutral'] is True and phase['parent_control']==ref,'Phase neutral identity differs')
        verify_neutral(phase,ref,ReadOnlyLedger(),hf)
        gen=native.read(native.bound(hf,phase['generation_receipt']))
        native.need(gen['run_id']==phase['run_id'] and gen['states']==phase['states'] and gen['video']==phase['video']
            and gen['all160_FFN_covered'] is True and gen['protocol_sha256']==digest,'Bridge generation differs')
        for item in phase['states'].values():native.bound(hf,item)
        actual=phase['video'];p=Path(actual['path']).resolve();native.need(p.is_relative_to(hf),'Video outside HF')
        native.bound(hf,{'path':str(p.relative_to(hf)),'bytes':actual['bytes'],'sha256':actual['sha256']})
        dispatch=native.read(native.bound(hf,phase['W2_dispatch']))
        native.need(dispatch['mode']==phase['mode'] and dispatch['expected_and_actual']==parent.expected(phase['mode'])
            and len(dispatch['actual_dispatch'])==160 and len(dispatch['actual_skips'])==parent.expected(phase['mode'])['original_W2_skips']
            and {(d['step'],d['block']) for d in dispatch['actual_dispatch']}=={(s,b) for s in range(4) for b in range(40)}
            and all(d['kind']==('dense' if phase['mode']=='dense' else 'factor') and d['expert']==(0 if d['step']<2 else 1)
                for d in dispatch['actual_dispatch']),'Closed dispatch differs')
    native.need({(p['run_id'],p['step'],p['block']) for p in report['packets']}==
        {(rid,s,b) for rid in plan['run_ids'] for s,b in SITES},'Packet site coverage differs')
    for item in report['packets']:
        p=native.read(native.bound(hf,item['metadata']))
        native.need(p['id']==ID and p['protocol_sha256']==digest and p['run_id']==item['run_id']
            and (p['step'],p['block'])==(item['step'],item['block']) and packet_geometry_matches(p)
            and p['rows']==list(ROWS) and p['boundary']=='ORIGINAL_FULL_M_POST_W1_BIAS_PRE_GELU_SELECTED_ROWS'
            and set(p['tensor_metadata'])==set(SHAPES)
            and all(p['tensor_metadata'][k]['shape']==shape
                and p['tensor_metadata'][k]['dtype']==('int64' if k=='rows' else 'bfloat16')
                and len(p['tensor_metadata'][k]['raw_tensor_sha256'])==64 for k,shape in SHAPES.items())
            and all(p['source_layout'][k]['shape']==[31668,13824 if k in ('a','h') else 5120]
                and p['source_layout'][k]['dtype']=='bfloat16' for k in ('V1','a','h','Y','r','u'))
            and p['source_layout']['W1_bias']['shape']==[13824]
            and p['source_layout']['W1_bias']['dtype']=='bfloat16'
            and p['source_layout']['W1_bias']['device'] in ('cuda','cuda:0')
            and p['source_layout']['W1_bias']['stride']==[1]
            and p['source_layout']['W1_bias']['identity']=='ACTUAL_FFN0_MM_BIAS'
            and p['Y_boundary']==y_boundary(p['mode'])
            and any(p['source_ordinal']==c['native_source_ordinal'] and p['mode'] in MODES
                and p['run_id']==run_id(c,p['mode']) and p['source_id']==c['source_id']
                and p['image_sha256']==c['image']['sha256'] and p['prompt_sha256']==c['prompt_sha256']
                and p['seed']==42 and p['source_state']==('teacher_input' if p['mode']=='dense' else 'student_input')
                for c in plan['cases']), 'Packet metadata differs')
        verify_packet_header(native.bound(hf,p['payload']))
    return {'id':ID,'status':'PASS_CLOSED_NATIVE_BOUNDARY_BRIDGE','RUN':binding,
            'exit':native.binding(args.exit,hf),'model_calls':0,'GPU_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','inspect','preflight','run','verify-closed'))
    p.add_argument('--protocol',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));p.add_argument('--local-root',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--exit',type=Path);a=p.parse_args()
    if a.action=='draft':
        native.need(a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);native.write(a.out,draft(a.hf_root))
    elif a.action=='inspect':validate(native.read(a.protocol));print('STATIC_ONLY_NO_MODEL')
    elif a.action=='preflight':preflight(a)
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
