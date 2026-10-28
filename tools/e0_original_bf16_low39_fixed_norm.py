"""Single-source transfer: fixed step norm gains in eight raw-BF16 flows.

Reuses frozen original loading/capture helpers. No old module globals/files are
changed; no basis fit, model training, video decoding or official scores run.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
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
if str(ROOT/'tools') not in sys.path:sys.path.insert(0,str(ROOT/'tools'))
import e0_original_bf16_low39_four_corner as prior
import raw_bf16_fixed_norm as core
native,geometry_core=prior.native,prior.core
ID='E0_ORIGINAL_BF16_LOW39_FIXED_NORM_TRANSFER_20261028'
PARENT='config/e0_original_bf16_low39_four_corner_20261028_v1.json'
PARENT_SHA='fcd50986dc650e80abc0a1a665fca973b92af940fa6b34bf6615e1c78f322d4f'
PARENT_RUN_SHA='768a5218da2ac17914ffd7e6a3e1623adf848be679a71a941634596c2d1c0088'
PARENT_EXIT_SHA='49e536c4725c4f525ffaece08d5daf23ac0d1781def107c1df8c9325613ae35b'
NORM_PROTOCOL='config/e0_raw_norm_rows_20261028_v1.json'
NORM_PROTOCOL_SHA='c4c0b464ef9cd67b6227b9b0a9a28bd59705678820e411d92e6d3179e75ee49f'
NORM_RUN_SHA='6e265ed29bb6a018e8fe2312259d29a9f12f6e26d38d7c6040870422ffe43a40'
KAPPAS={'2':1.039908414775213,'3':1.020973593727097}
OPERATIONS=(('DD2',2),('DD3',3),('SD2',2),('SD3',3),('CD2',2),('CD3',3),('DC3',3),('CC3',3))
RUN_IDS=tuple('raw_fixed_norm_source12_'+op for op,_ in OPERATIONS)
RESOURCES={**prior.RESOURCES,'wall_seconds':900,'outer_seconds':1200}
SHAPE=prior.SHAPE
M,D,H=prior.M,prior.D,prior.H
REQUIRED={'tools/raw_bf16_fixed_norm.py','tools/e0_original_bf16_low39_fixed_norm.py','tests/test_original_bf16_low39_fixed_norm.py'}


def limits():
    result={k:0 for k in prior.limits()}
    result.update(original_expert_loads=2,raw_Default_MM_loads=812,buffer_Default_MM_loads=40,
      T5_raw_Default_MM_loads=168,T5_buffer_Default_MM_loads=7,T5_Default_MM_calls=168,T5_checkpoint_block_buffer_loads=24,
      input_encoder_requests=1,T5_positive_prompt_encodes=1,VAE_encodes=1,scheduler_prepare=1,FP32_initial_noise_draws=1,
      scheduler_step_pre=8,scheduler_step_post=8,native_full_model_flows=8,native_blocks=320,native_FFN_calls=320,
      native_FFN_linears=635,FFN_CUDA_event_pairs=320,native_non_FFN_linears=2560,
      pre_Default_MM_calls=40,head_Default_MM_calls=8,patch_Conv3D_calls=8,pre_infer_calls=8,final_LayerNorm_calls=8,post_unpatchify_calls=8,
      original_checkpoint_block_buffer_loads=328,offload_buffer_resets=8,raw_noise_byte_controls=1,
      input_state_uploads=8,input_state_upload_bytes=8*math.prod(SHAPE)*2,row_index_uploads=8,row_index_upload_bytes=8*256*8,
      control_pack_loads=6,input_after_step_pre_byte_checks=8,actual_head_conditioning_captures=8,conditioning_equal_checks=6,
      native_boundary_capture_cases=8,teacher_full_Y_references_retained=3,candidate_overridden_W2_dispatches=5,
      candidate_coefficient_GEMMs=5,candidate_output_GEMMs=5,projection_H2D=3,projection_H2D_bytes=(H*512+D*512+D)*2,
      geometry_BF16_gain_add=2,geometry_head_build=2,FP64_stat_matrix_products=8,FP64_q_variance_cases=8,
      native_head_decomposition_cases=6,fullM_gated_Y_diagnostic_cases=6,native_scheduler_rounding_cases=8,
      flow_four_corner_decompositions=1,same_input_prefix_byte_checks=5,post_error_identity_cases=1,flow_capture_pack_writes=8,source_conditioning_setups=1)
    result.update(prior_capture_pack_loads=4,projection_pack_reload_checks=1,original_raw_flow_post_byte_checks=4,
      prior_parent_flow_post_byte_checks=8,normalization_storage_identity_checks=8,normalization_identity_noops=5,
      conditioning_normalization_gain_calls=3,norm_gain_BF16_to_FP32_casts=3,norm_gain_FP32_multiplies=3,norm_gain_FP32_to_BF16_copies=3,
      effective_normalizer_cases=3,raw_effective_normalizer_diagnostic_cases=8)
    return result


class Ledger(prior.Ledger):
    @contextmanager
    def work(self,key,*,rid='setup',costs=None):
        costs={key:1} if costs is None else costs
        native.need(rid=='setup' or rid in RUN_IDS,'Unregistered fixed-norm operation')
        native.need(all(k in limits() and type(n) is int and n>0 and self.report['predebited'].get(k,0)+n<=limits()[k] for k,n in costs.items()),'Fixed-norm budget exceeded')
        for k,n in costs.items():self.report['predebited'][k]=self.report['predebited'].get(k,0)+n
        self.journal({'kind':'reserved_before_call','run_id':rid,'costs':costs})
        yield
        for k,n in costs.items():self.report['observed'][k]=self.report['observed'].get(k,0)+n
        self.journal({'kind':'observed_return','run_id':rid,'costs':costs})
    def finish(self):
        native.need(not self.report['forbidden_attempts'] and all(self.report[p].get(k,0)==n for p in ('predebited','observed') for k,n in limits().items()),'Incomplete fixed-norm budget')


def kappa_for(op,step):return KAPPAS[str(step)] if op.startswith('C') else 1.
def mode_for(op):return 'D' if op.startswith('D') else 'S'


def validate(plan,frozen=False):
    native.need(plan.get('id')==ID and plan.get('numeric_path')=='BF16_ORIGINAL' and plan.get('limits')==limits()
      and plan.get('resources')==RESOURCES and plan.get('run_ids')==list(RUN_IDS) and plan.get('kappas')==KAPPAS
      and plan.get('parameters')==geometry_core.P and plan.get('runtime_overlay')==native.recipe()
      and plan.get('fitted_source')==0 and plan.get('transfer_source')==12 and plan.get('automatic_retry') is False,'Fixed norm scope differs')
    native.need(len(plan['cases'])==1 and plan['cases'][0]['index']==12 and set(plan['cases'][0]['rows'])=={'2','3'},'Single exposed transfer source12 required')
    for s in (2,3):
        rows=plan['cases'][0]['rows'][str(s)]
        native.need(len(rows)==len(set(rows))==256 and all(type(i) is int and 0<=i<M for i in rows),'Original fixed row coordinates required')
    native.need(plan['prior']['protocol']['path']==PARENT and plan['prior']['protocol']['sha256']==PARENT_SHA
      and plan['prior']['RUN']['sha256']==PARENT_RUN_SHA and plan['prior']['exit']['sha256']==PARENT_EXIT_SHA
      and plan['norm_fit']['protocol']['path']==NORM_PROTOCOL and plan['norm_fit']['protocol']['sha256']==NORM_PROTOCOL_SHA
      and plan['norm_fit']['RUN']['sha256']==NORM_RUN_SHA,'Actual parent/fit identity differs')
    sources=[b['path'] for b in plan['repo_sources']]
    native.need(REQUIRED<=set(sources) and len(sources)==len(set(sources)),'Fixed-norm source closure incomplete')
    path=Path(plan['output_HF_relative'])
    native.need(not path.is_absolute() and path.parts[:1]==('ffnds-derived',) and '..' not in path.parts,'Output escaped HF')
    if frozen:native.need(plan.get('assets_finalized') is True and plan.get('binding_status')=='BOUND_READY_FOR_REGISTRATION','Fixed-norm draft NOT_BOUND')


def exited(receipt,run,protocol,id):
    native.need(receipt.get('id')==id and receipt.get('protocol_sha256')==protocol and receipt.get('run_sha256')==run['sha256']
      and receipt.get('process_exited') is True and type(receipt.get('child_returncode')) is int and receipt['child_returncode']==0
      and receipt.get('timeout') is False and receipt.get('automatic_retry') is False,'Actual exited success/RUN SHA differs')


def verify_lineage(plan,hf,stat_check=True):
    old=native.read(native.bound(ROOT,plan['prior']['protocol']));run=native.read(native.bound(hf,plan['prior']['RUN']))
    prior.validate(old,True);exited(native.read(native.bound(hf,plan['prior']['exit'])),plan['prior']['RUN'],PARENT_SHA,prior.ID)
    native.need(run.get('status')=='COMPLETE_RAW_BF16_LOW39_R512_TWELVE_FULL_FLOWS_NO_VIDEO' and run.get('id')==prior.ID
      and run.get('protocol_sha256')==PARENT_SHA and run.get('all_four_neutral_flow_and_post_controls_passed') is True
      and not run['operation_ledger']['forbidden_attempts'] and not run.get('assigned_GPU_resources',{}).get('violations')
      and all(run['operation_ledger'][p].get(k,0)==n for p in ('predebited','observed') for k,n in prior.limits().items()),'Parent twelve flows incomplete')
    inherited=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256',
      'parent_protocol','parent_RUN','parent_exit','raw_master_CPU_proof','runtime_overlay')
    native.need(all(plan[k]==old[k] for k in inherited) and plan['cases']==[c for c in old['cases'] if c['index']==12],'Original raw sources/assets changed')
    prior.verify_parent(plan,hf,stat_check=stat_check)
    expected={p['operation']:p for p in run['flows'] if p['source_index']==12 and p['operation'] in ('DD2','DD3','SD2','SD3')}
    native.need(set(expected)=={'DD2','DD3','SD2','SD3'} and plan['parent_captures']=={op:{'capture':p['capture'],'tensor_metadata':p['tensor_metadata']} for op,p in expected.items()}
      and plan['projection']==run['weight_space_geometry'],'Prior native controls/projection identity differs')
    fit=native.read(native.bound(hf,plan['norm_fit']['RUN']));fit_protocol=native.read(native.bound(ROOT,plan['norm_fit']['protocol']))
    exited(native.read(native.bound(hf,plan['norm_fit']['exit'])),plan['norm_fit']['RUN'],NORM_PROTOCOL_SHA,fit_protocol['id'])
    native.need(fit.get('status')=='COMPLETE_SAVED_ROWS_DEVELOPMENT' and fit.get('protocol_sha256')==NORM_PROTOCOL_SHA
      and fit.get('parent_RUN_sha256')==PARENT_RUN_SHA and fit.get('automatic_retry') is False
      and all(fit[p]==fit['limits']==fit_protocol['limits'] for p in ('predebited','observed')),'Original fixed kappa fit not complete')
    folds=[f for f in fit['folds'] if f['train_source']==0 and f['heldout_source']==12]
    native.need(len(folds)==2 and {str(f['step']):f['heldout']['kappa'] for f in folds}==KAPPAS,'No new/refitted kappa or held-source feedback allowed')
    return old,run,fit


def check_pack(hf,binding,tensors):
    path=native.bound(hf,binding)
    with path.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0];native.need(0<header_size<=1<<20,'Pack header size invalid')
        header=json.loads(stream.read(header_size));base=8+header_size
        native.need(set(header)-{'__metadata__'}==set(tensors),'Saved pack tensors differ')
        for name,meta in tensors.items():
            item=header[name];dtype={'bfloat16':'BF16','float32':'F32','int64':'I64'}[meta['dtype']]
            begin,end=item['data_offsets'];size=math.prod(meta['shape'])*{'BF16':2,'F32':4,'I64':8}[dtype]
            native.need(item['shape']==meta['shape'] and item['dtype']==dtype and 0<=begin<end and end-begin==size,'Saved pack shape/dtype/size differs')
            stream.seek(base+begin);data=stream.read(size)
            native.need(len(data)==size and hashlib.sha256(data).hexdigest()==meta['raw_tensor_sha256'],'Saved pack raw tensor SHA differs')
    return path.stat().st_size


def verify_small_assets(plan,hf):
    for b in plan['repo_sources']:native.bound(ROOT,b)
    for v in plan['parent_captures'].values():check_pack(hf,v['capture'],v['tensor_metadata'])
    check_pack(hf,plan['projection']['export'],plan['projection']['tensors'])
    for b in prior.control_bindings(plan).values():prior.check_state_file(hf,b)
    native.bound(hf,plan['cases'][0]['image'])


def cpu_identity(plan,digest):
    return {'id':ID,'protocol_sha256':digest,'sources_sha256':native.canonical(plan['repo_sources']),
      'parent_RUN_sha256':PARENT_RUN_SHA,'norm_fit_RUN_sha256':NORM_RUN_SHA,'raw_master_CPU_proof':plan['raw_master_CPU_proof'],
      'status':'PASS_FIXED_NORM_PARENT_CONTROLS_PACK_AND_RAW_MASTER_STAT_NO_NUMERIC_IMPORTS',
      'original_control_files':len(prior.control_bindings(plan)),'prior_capture_files':4,'projection_pack_files':1,
      'all_CUDA_devices_hidden':True,'Torch_or_NumPy_imported':False,'model_calls':0,'GPU_calls':0,'raw_master_payload_bytes_rehashed':0}


def preflight(args):
    start=time.monotonic();native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'numpy' not in sys.modules,'Fresh hidden-CUDA stdlib CPU proof required')
    plan=native.read(args.protocol);validate(plan,True);native.need(len(os.sched_getaffinity(0))<=4,'CPU affinity exceeds4')
    verify_lineage(plan,args.hf_root);verify_small_assets(plan,args.hf_root);native.model_metadata(plan)
    native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    result=cpu_identity(plan,native.sha(args.protocol));result['wall_seconds']=time.monotonic()-start
    native.need(result['wall_seconds']<=300 and 'torch' not in sys.modules and 'numpy' not in sys.modules,'CPU proof wall/import gate failed')
    return result


def authorize(plan,digest):
    validate(plan,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');policy=active.get('resource_scheduling_policy_20261028',{})
    native.need(active.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True
      and policy.get('mode')=='DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
      and active.get('current_session_resource_constraint_20261028',{}).get('max_simultaneously_assigned_idle_GPUs')==2,'Current session authority differs')
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    native.need(len(scopes)==1 and scopes[0].get('protocol_sha256')==digest and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and scopes[0].get('limits')==limits() and scopes[0].get('resources')==RESOURCES and scopes[0].get('run_ids')==list(RUN_IDS)
      and scopes[0].get('max_gpu_cards')==1,'Unique fixed-norm ACTIVE registration required')
    return scopes[0]


def require_cpu(scope,plan,digest,hf):
    proof=scope.get('cpu_preflight');native.need(isinstance(proof,dict) and {'RUN','launch_exit'}<=set(proof),'Independent CPU proof NOT_BOUND')
    result=native.read(native.bound(hf,proof['RUN']))
    native.need(all(result.get(k)==v for k,v in cpu_identity(plan,digest).items()) and 0<=result.get('wall_seconds',-1)<=300,'Fixed-norm byte CPU proof differs')
    exited(native.read(native.bound(hf,proof['launch_exit'])),proof['RUN'],digest,ID);return proof


class FlowTap(prior.FlowTap):
    def __init__(self,*args,kappa=1.,**kwargs):
        super().__init__(*args,**kwargs);self.kappa=kappa
    def install(self):
        # Patch the native callable first. The inherited outer capture keeps
        # seeing the original infer_non_blocks caller and original raw u.
        norm=self.runner.model.model[1].transformer_weights.norm
        old=norm.apply
        def gain(value):
            normalized=old(value)
            return core.normalized_gain(self.guard.torch,normalized,value,self.kappa,self.work)
        self.stack.enter_context(native.patch_method(norm,'apply',gain))
        super().install()


def load_prior_controls(torch,plan,hf,ledger):
    from safetensors.torch import load_file
    result={}
    for op,item in plan['parent_captures'].items():
        def load(item=item):
            values=load_file(str(native.bound(hf,item['capture'])),device='cpu')
            native.need(set(values)==set(item['tensor_metadata']) and all(list(v.shape)==item['tensor_metadata'][k]['shape']
              and str(v.dtype).removeprefix('torch.')==item['tensor_metadata'][k]['dtype']
              and prior.tensor_sha(v)==item['tensor_metadata'][k]['raw_tensor_sha256'] for k,v in values.items()),'Prior CPU capture payload differs')
            return values
        result[op]=ledger.call('setup',{'prior_capture_pack_loads':1},load)
    return result


def load_projection(torch,plan,hf,ledger):
    from safetensors.torch import load_file
    expected=plan['projection']['tensors']
    def load():
        values=load_file(str(native.bound(hf,plan['projection']['export'])),device='cpu')
        native.need(set(values)=={'B','Phi','b2'} and all(v.dtype==torch.bfloat16 and list(v.shape)==expected[k]['shape']
          and prior.tensor_sha(v)==expected[k]['raw_tensor_sha256'] for k,v in values.items()),'Actual parent BF16 projection reload differs')
        return values
    values=ledger.call('setup',{'projection_pack_reload_checks':1},load);pack={}
    for k,v in values.items():pack[k]=ledger.call('setup',{'projection_H2D':1,'projection_H2D_bytes':v.numel()*v.element_size()},v.to,'cuda',non_blocking=False)
    return pack


def require_neutral(torch,actual,expected):
    native.need(actual.dtype==expected.dtype and actual.shape==expected.shape
      and torch.equal(actual.detach().cpu().view(torch.uint8),expected.view(torch.uint8)),'Neutral flow/post raw bytes mismatch')


def single_flow(torch,pipe,guard,case,op,step,input_cpu,geometry,pack,controls,parent_controls,context,out,hf,
                report,persist,check,target=None,Y_target=None,prefix_reference=None):
    from safetensors.torch import save_file
    rid='raw_fixed_norm_source12_'+op;ledger=guard.ledger;work=lambda key:ledger.work(key,rid=rid)
    kappa=kappa_for(op,step)
    phase={'run_id':rid,'source_index':12,'operation':op,'step':step,'mode':mode_for(op),'status':'STARTED_NO_RETRY',
      'prefix':'CORRECTED_C_STEP2_POST' if op in ('DC3','CC3') else 'HISTORICAL_RAW_DENSE',
      'kappa_sigmaD_over_sigmaS':kappa,'deployed_FP32_reciprocal_alpha':core.gain_scalar(kappa),
      'V1_role':'RAW_PRE_W1_BF16_WITH_NO_Q1','hidden_role':'RAW_GELU_TANH_WITH_NO_Q2',
      'online_teacher_values_used':False}
    report['flows'].append(phase);persist();scheduler=pipe.runner.scheduler
    pipe.runner.input_info,pipe.runner.inputs=context
    def upload():
        native.need(input_cpu.dtype==torch.bfloat16 and list(input_cpu.shape)==SHAPE,'Original BF16 input geometry differs')
        return input_cpu.to('cuda',non_blocking=False)
    before=ledger.call(rid,{'input_state_uploads':1,'input_state_upload_bytes':math.prod(SHAPE)*2},upload);scheduler.latents=before
    rows=ledger.call(rid,{'row_index_uploads':1,'row_index_upload_bytes':256*8},torch.tensor,case['rows'][str(step)],dtype=torch.int64,device='cuda')
    tap=FlowTap(pipe,guard,rid,step,mode_for(op),rows,geometry,pack,Y_target,kappa=kappa)
    started=time.perf_counter()
    try:
        ledger.call(rid,{'scheduler_step_pre':1},scheduler.step_pre,step_index=step)
        def same_input():
            native.need(scheduler.latents.dtype==torch.bfloat16 and list(scheduler.latents.shape)==SHAPE
              and prior.tensor_sha(scheduler.latents)==prior.tensor_sha(input_cpu),'step_pre changed frozen input')
            phase['input_after_step_pre']={'shape':SHAPE,'dtype':'bfloat16','raw_tensor_sha256':prior.tensor_sha(scheduler.latents)}
        ledger.call(rid,{'input_after_step_pre_byte_checks':1},same_input)
        with work('native_boundary_capture_cases'),tap,torch.inference_mode():pipe.runner.model.infer(pipe.runner.inputs)
        flow=scheduler.noise_pred
        native.need(flow.dtype==torch.float32 and list(flow.shape)==SHAPE,'Actual full native flow differs')
        ledger.call(rid,{'scheduler_step_post':1},scheduler.step_post);post=scheduler.latents
        if op in ('DD2','DD3'):
            for actual,key in ((flow,'flow'+str(step)),(post,'post'+str(step))):
                ledger.call(rid,{'original_raw_flow_post_byte_checks':1},require_neutral,torch,actual,controls[key])
            phase['original_raw_flow_and_post_bitwise_passed']=True
        if op in ('DD2','DD3','SD2','SD3'):
            for actual,key in ((flow,'flow'),(post,'latents')):
                ledger.call(rid,{'prior_parent_flow_post_byte_checks':1},require_neutral,torch,actual,parent_controls[op][key])
            phase['prior_parent_flow_and_post_bitwise_passed']=True
        sigma=scheduler.sigmas[step].item();sn=scheduler.sigmas[step+1].item() if step<3 else None
        rounding=geometry_core.scheduler_residual(torch,before,flow,post,sigma,sn,work)
        phase['scheduler_rounding']={k:v for k,v in rounding.items() if not torch.is_tensor(v)}
        effective,phase['normalizer_diagnostics']=core.effective_statistics(torch,tap.stats,kappa,work)
        if target is not None:
            phase['head_diagnostics_vs_same_step_Dense_teacher_state']=geometry_core.head_terms(torch,tap.head,target['head'],effective,target['stats'],work)
            phase['head_diagnostics_normalizer_identity']='Effective sigma = raw_u_sigma / deployed_FP32_alpha; q is raw_u q, not fitted; native remainder includes gain FP32/BF16 rounding'
        phase.update(tap.diagnostics)
        if prefix_reference is not None:
            def check_prefix():
                native.need(all(torch.equal(tap.capture[k].view(torch.uint8),prefix_reference['capture'][k].view(torch.uint8)) for k in ('V1','r')),'Same-input prefix/V1/r changed')
            ledger.call(rid,{'same_input_prefix_byte_checks':1},check_prefix)
        def save():
            path=out/(rid+'.safetensors');payload={'flow':flow,'latents':post,'head':tap.head,'rows':rows,**tap.capture}
            cpu={k:v.detach().cpu().contiguous() for k,v in payload.items()}
            native.need(not path.exists() and all(bool(torch.isfinite(v).all()) for v in cpu.values()),'Nonfinite capture or output overwrite')
            save_file(cpu,str(path));phase['capture']={'root':'hf',**native.binding(path,hf)}
            phase['tensor_metadata']={k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.'),'raw_tensor_sha256':prior.tensor_sha(v)} for k,v in cpu.items()}
            return cpu['latents']
        post_cpu=ledger.call(rid,{'flow_capture_pack_writes':1},save);torch.cuda.synchronize();tap.finish()
        phase.update(status='COMPLETE_FULL40_FIXED_NORM_FLOW_NOT_VIDEO',wall_seconds=time.perf_counter()-started,
          original_MM_operand_metadata=tap.shapes,FFN_CUDA_seconds=sum(a.elapsed_time(b) for a,b in tap.events)/1000)
        result={'flow':flow,'post':post,'post_cpu':post_cpu,'head':tap.head,'stats':tap.stats,'effective_stats':effective,
          'Y':tap.Y_full,'capture':tap.capture,'rounding':rounding,'input':before}
        check();return result
    except BaseException as error:
        phase.update(status='FAILED_FIXED_NORM_FLOW_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:persist()


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol)
    scope=authorize(plan,digest);cpu=require_cpu(scope,plan,digest,args.hf_root)
    native.need('torch' not in sys.modules and 'numpy' not in sys.modules,'Fresh native worker required')
    verify_lineage(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    native.model_metadata(plan)
    local,rows=native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions'],'Actual package versions differ')
    from worker_identity import require_worker,private_config_path
    require_worker();native.need(native.sha(private_config_path()) in plan['private_worker_config_sha256'],'Worker identity differs')
    native.need(len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=='1' for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'CPU affinity/TF32 environment differs')
    out=(args.hf_root/plan['output_HF_relative']).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined fixed-norm output required')
    out.mkdir(parents=True);report={'id':ID,'protocol_sha256':digest,'numeric_path':'BF16_ORIGINAL','status':'STARTED_FIXED_EIGHT_FLOW_NO_RETRY',
      'CPU_preflight':cpu,'automatic_retry':False,'flows':[],'fitted_kappas':KAPPAS,'fitted_source':0,'native_transfer_source':12,
      'independent_native_transfer_image_clusters':1,'source_role':'PREVIOUSLY_EXPOSED_DEVELOPMENT_TRANSFER_NOT_NEW_HELDOUT',
      'training_updates':0,'video_generations':0,'official_quality_calls':0,'decompositions':0,
      'prior':plan['prior'],'norm_fit':plan['norm_fit'],'original_prefetch_cost_retained':True,'quality_or_speed_accepted':False}
    persist=lambda:native.write(out/'RUN.json',report);journal=(out/'OPERATIONS.jsonl').open('x',buffering=1)
    ledger=Ledger(report.setdefault('operation_ledger',{}),lambda event:journal.write(json.dumps(event,separators=(',',':'))+'\n'))
    persist();monitor=None
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(free>=RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes'],'Host/disk reserves exceeded')
        native.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Fixed-norm new-output budget exceeded')
        native.need(time.monotonic()-started<=RESOURCES['wall_seconds'],'Fixed-norm wall900 exhausted')
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU reserved32GiB exceeded')
    def timeout(*_):raise TimeoutError('Fixed norm wall900 exhausted; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,rows),native.NoHiF4()
    try:
        check();snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir()
        for item in plan['repo_sources']:
            destination=snapshot/item['path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],destination)
        shutil.copyfile(args.protocol,snapshot/'protocol.json')
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,report['CUDA_reservation']=reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Single A100 required')
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'matmul_TF32':False,'float32_matmul_precision':'highest','bf16_reduced_precision_reduction':True}
        controls=prior.load_controls(torch,plan,args.hf_root,ledger)[12];parent_controls=load_prior_controls(torch,plan,args.hf_root,ledger)
        torch.cuda.reset_peak_memory_stats()
        with native.OriginalGuard(torch,plan,ledger) as guard:
            begun=time.perf_counter();pipe=native.build_pipeline(plan);report['raw_model_setup_seconds']=time.perf_counter()-begun
            case=plan['cases'][0];context=prior.prepare_source(pipe,guard,case,controls,args.hf_root);geometry,data={},{}
            def flow(op,step,input_cpu,pack=None,target=None,Y_target=None,prefix_reference=None):
                result=single_flow(torch,pipe,guard,case,op,step,input_cpu,geometry,pack,controls,parent_controls,context,out,args.hf_root,
                  report,persist,check,target,Y_target,prefix_reference);data[op]=result;return result
            for step in (2,3):flow('DD'+str(step),step,controls['input'+str(step)])
            pack=load_projection(torch,plan,args.hf_root,ledger)
            for step in (2,3):
                target=data['DD'+str(step)];flow('SD'+str(step),step,controls['input'+str(step)],pack,target,target['Y'],target)
            report['all_four_Dense_and_bare_neutral_flow_post_controls_passed']=True;persist();check()
            for step in (2,3):
                target=data['DD'+str(step)];flow('CD'+str(step),step,controls['input'+str(step)],pack,target,target['Y'],target)
            target=data['DD3'];own_input=data['CD2']['post_cpu']
            flow('DC3',3,own_input,None,target,target['Y'])
            paid_teacher=data['DC3'];flow('CC3',3,own_input,pack,target,paid_teacher['Y'],paid_teacher)
            group=[data[op] for op in ('DD3','CD3','DC3','CC3')]
            values=geometry_core.four_corner(torch,*(x['flow'] for x in group),ledger.work)
            values.update(source_index=12,step=3,delta_sigma=.625,**prior.post_identity(torch,group,ledger));report['four_corner']=values
            report['historical_bare_four_corner_reference']=plan['prior']
            ledger.finish();check();report.update(status='COMPLETE_RAW_BF16_FIXED_NORM_SOURCE12_EIGHT_FLOWS_NO_VIDEO',checked_transformers_imports=finder.origins,
              original_loader_paths=guard.loaded,teacher_full_Y_retained_bytes=3*M*D*2,peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status='FAILED_FIXED_NORM_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources'].get('violations'),'Final GPU resource violation')
            except BaseException as error:
                monitor_error=error;report.update(status='FAILED_FIXED_NORM_PRESERVED_NO_RETRY',resource_error=str(error))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        report['wall_seconds']=time.monotonic()-started;persist();journal.close()
        if monitor_error is not None and sys.exc_info()[0] is None:raise monitor_error
    return report


def draft(hf):
    hf=Path(hf);old=native.read(ROOT/PARENT);folder=hf/old['output_HF_relative'];run=native.read(folder/'RUN.json')
    norm=hf/'ffnds-derived/e0_raw_norm_rows_20261028/attempt1/worker'
    plan={k:copy.deepcopy(old[k]) for k in ('parent_protocol','parent_RUN','parent_exit','raw_master_CPU_proof','models','model_view_root','model_view_metadata',
      'runtime_overlay','runtime_versions','local_transformers','private_worker_config_sha256','parameters')}
    plan.update(id=ID,status='DRAFT_FIXED_STEP_NORM_TRANSFER_NO_MODEL_GPU',numeric_path='BF16_ORIGINAL',assets_finalized=False,
      binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING',automatic_retry=False,resources=RESOURCES,limits=limits(),run_ids=list(RUN_IDS),
      cases=[copy.deepcopy(c) for c in old['cases'] if c['index']==12],kappas=KAPPAS,fitted_source=0,transfer_source=12,
      prior={'protocol':native.binding(ROOT/PARENT,ROOT),'RUN':native.binding(folder/'RUN.json',hf),'exit':native.binding(folder.parent/'EXIT_RECEIPT.json',hf)},
      norm_fit={'protocol':native.binding(ROOT/NORM_PROTOCOL,ROOT),'RUN':native.binding(norm/'results/RUN.json',hf),'exit':native.binding(norm/'EXIT_RECEIPT.json',hf)},
      projection=run['weight_space_geometry'],parent_captures={p['operation']:{'capture':p['capture'],'tensor_metadata':p['tensor_metadata']} for p in run['flows'] if p['source_index']==12 and p['operation'] in ('DD2','DD3','SD2','SD3')},
      output_HF_relative='ffnds-derived/e0_original_bf16_low39_fixed_norm_transfer_20261028/attempt1/worker/results',
      sequence='DD2 DD3 original+prior byte neutral; SD2 SD3 prior byte neutral; CD2 CD3 teacher-state; DC3 paid teacher then CC3 on CD2 post',
      online_graph='Full original W1/GELU; frozen native BF16 B/Phi/b2; fresh native LN result -> FP32 1/kappa gain -> BF16 copyback -> original modulation/shift/head bias',
      normalizer_identity='Raw postwrite u unchanged; diagnostics distinguish raw-u sigma from effective sigma=(1/FP32(1/kappa))*raw_sigma; native remainder contains new gain rounding',
      source_identity='Source0 kappa fit inherited; source12 single already exposed development transfer image cluster; one seed42; 256 audit rows nested',
      raw_master_reuse='Prior actual CPU raw69GB proof plus unchanged device/inode/size/mtime/ctime; no new raw model payload hash',
      cost_identity='All original raw W2 checkpoint offload/preload cost retained; no system speed or full80 claim',
      gain_arithmetic={'nonidentity_calls':3,'BF16_to_FP32_casts':3,'FP32_inplace_multiplies':3,'FP32_to_BF16_copybacks':3,'identity_noops':5,
        'FP32_temporary_max_bytes':M*D*4,'unfused_payload_read_write_lower_bytes_per_nonidentity_call':M*D*20},
      CPU_preflight_resources={'GPU':0,'CPU_threads':4,'wall_seconds':300,'outer_seconds':360,'new_raw_master_hash_bytes':0},
      new_HF_payload_estimate_bytes=8*(math.prod(SHAPE)*6+M*64*2+4*256*D*2+3*D*2+256*8))
    plan['repo_sources']=[native.binding(ROOT/p,ROOT) for p in sorted({b['path'] for b in old['repo_sources']}|REQUIRED)]
    validate(plan);verify_lineage(plan,hf,stat_check=False);return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('draft','inspect','preflight','run'))
    p.add_argument('--protocol',type=Path);p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--local-root',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();args.hf_root=args.hf_root.resolve()
    if args.command=='draft':
        native.need(args.out is not None and not args.out.exists(),'Fresh draft output required');plan=draft(args.hf_root);native.write(args.out,plan)
        print(json.dumps({'id':ID,'binding_status':plan['binding_status'],'model_calls':0}))
    elif args.command=='inspect':
        plan=native.read(args.protocol);validate(plan);print(json.dumps({'id':ID,'limits':limits(),'resources':RESOURCES,'model_calls':0},indent=2))
    elif args.command=='preflight':
        native.need(args.out is not None and args.local_root is not None,'Fresh CPU output/local package required')
        out=args.out.resolve();native.need(out.is_relative_to(args.hf_root) and not out.exists(),'Fresh confined CPU output required')
        out.mkdir(parents=True);report={'id':ID,'protocol_sha256':native.sha(args.protocol),'status':'STARTED_BYTE_PREFLIGHT_NO_RETRY'}
        native.write(out/'RUN.json',report)
        try:report=preflight(args)
        except BaseException as error:report.update(status='FAILED_BYTE_PREFLIGHT_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
        finally:native.write(out/'RUN.json',report)
        print(json.dumps({'status':report['status'],'model_calls':0}))
    else:
        native.need(args.local_root is not None,'Exact local package required');print(json.dumps({'status':run(args)['status']}))


if __name__=='__main__':main()
