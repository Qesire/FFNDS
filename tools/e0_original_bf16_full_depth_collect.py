"""Two independent raw-BF16 data parts: one old neutral and three calibration sources.

Each part collects real V1/h/Y/r/u/gate at every 4x40 native site. It retains
all original model/encoder/offload work; no fit, student, QDQ or score runs.
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
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:sys.path.insert(0,str(ROOT/'tools'))
import e0_original_bf16_videos as native
import raw_bf16_full_depth_sources as sources
from ffnds_experiment_plans import row_draw

BASE_ID='E0_ORIGINAL_BF16_FULL_DEPTH_CAL6_DEV2_20261028'
PARENT='config/e0_original_bf16_videos_20261028_v1.json'
PARENT_SHA='2f5352e91a877e85c0c0e88d8c216f74f5124693161ad412b7226aff73c69122'
PARENT_RUN_SHA='59e99d08d206aac4ffabb11453e4b6b27449d046f07d14027d5e65e9644064c9'
PARENT_EXIT_SHA='c718ec457194d2e48deed47c9ccb18502ddb44a37355a6f4874867bd4c493b04'
COHORT='config/e0_original_bf16_full_depth_sources_20261028_draft.json'
M,D,H,ROWS=31668,5120,13824,256
SHAPE=native.VIDEO['latent_shape']
ROW_SHAPES={'V1':[ROWS,D],'h':[ROWS,H],'Y':[ROWS,D],'r':[ROWS,D],'u':[ROWS,D],
            'gate':[D],'shift':[D],'scale':[D],'rows':[ROWS]}
PACKET_BYTES=ROWS*(4*D+H)*2+3*D*2+ROWS*8
RESOURCES={'GPU':1,'CPU_threads':4,'GPU_reserved_max_bytes':32<<30,'host_available_min_bytes':128<<30,
  'wall_seconds':1200,'outer_seconds':1500,'new_HF_bytes':12<<30,'free_disk_min_bytes':24<<30,
  'session_max_GPUs':2,'repository_GPU_policy':'DYNAMIC_MAX_ACTUALLY_CALLABLE','automatic_retry':False}
REQUIRED={'tools/e0_original_bf16_full_depth_collect.py','tests/test_e0_original_bf16_full_depth_collect.py',
  'tools/raw_bf16_full_depth_sources.py','tests/test_raw_bf16_full_depth_sources.py',*sources.BANKS.values()}


def scope_id(part):
    native.need(type(part) is int and part in (0,1),'Exactly two independent parts');return BASE_ID+'_PART'+str(part)
def run_id(case):return 'raw_full_depth_ord'+str(case['native_source_ordinal'])+'_seed42_dense'


def limits():
    result={k:0 for k in native.LIMITS}
    result.update(original_expert_loads=2,raw_Default_MM_loads=812,buffer_Default_MM_loads=40,T5_raw_Default_MM_loads=168,T5_buffer_Default_MM_loads=7,
      T5_Default_MM_calls=672,T5_checkpoint_block_buffer_loads=96,video_generations=3,input_encoder_requests=4,T5_positive_prompt_encodes=4,
      VAE_encodes=4,VAE_decodes=3,core_CPU_reports=3,scheduler_prepare=4,FP32_initial_noise_draws=4,scheduler_step_pre=16,scheduler_step_post=16,
      native_full_model_flows=16,native_blocks=640,native_FFN_calls=640,native_FFN_linears=1280,FFN_CUDA_event_pairs=640,
      native_non_FFN_linears=5120,pre_Default_MM_calls=80,head_Default_MM_calls=16,patch_Conv3D_calls=16,pre_infer_calls=16,
      final_LayerNorm_calls=16,post_unpatchify_calls=16,original_checkpoint_block_buffer_loads=648,offload_buffer_resets=8,
      raw_FP32_noise_files=4,initial_BF16_noise_files=4,native_FP32_flow_files=16,native_BF16_post_files=16)
    result.update(native_prefix_replays_no_decode=1,row_index_uploads=4,row_index_upload_bytes=4*4*40*ROWS*8,
      raw_FFN_conditioning_captures=640,operand_row_selections=640*5,raw_postwrite_packets=640,
      packet_tensor_D2H=640*len(ROW_SHAPES),packet_tensor_D2H_bytes=640*PACKET_BYTES,
      packet_safetensor_writes=640,packet_metadata_writes=640,packet_payload_validations=640,actual_grid_geometry_checks=640,
      neutral_state_byte_checks=10,state_tensor_D2H=40,state_tensor_D2H_bytes=4*math.prod(SHAPE)*(4+2+4*4+4*2),
      video_geometry_checks=3,source_registration_writes=3,packet_output_byte_reservations=640,state_output_byte_reservations=40)
    return result


class Ledger:
    def __init__(self,report,run_ids,journal=lambda _:None):
        self.report,self.run_ids,self.journal=report,set(run_ids),journal
        report.update(predebited={},observed={},forbidden_attempts=[])
    @contextmanager
    def work(self,key,*,rid='setup',costs=None):
        costs={key:1} if costs is None else costs
        native.need(rid=='setup' or rid in self.run_ids,'Unregistered full-depth source')
        native.need(all(k in limits() and type(n) is int and n>0 and self.report['predebited'].get(k,0)+n<=limits()[k] for k,n in costs.items()),'Full-depth operation budget exceeded')
        for k,n in costs.items():self.report['predebited'][k]=self.report['predebited'].get(k,0)+n
        self.journal({'kind':'reserved_before_call','run_id':rid,'costs':costs});yield
        for k,n in costs.items():self.report['observed'][k]=self.report['observed'].get(k,0)+n
        self.journal({'kind':'observed_return','run_id':rid,'costs':costs})
    def call(self,rid,costs,fn,*args,**kwargs):
        with self.work('',rid=rid,costs=costs):return fn(*args,**kwargs)
    def forbidden(self,key):
        def fail(*_a,**_kw):
            self.report['forbidden_attempts'].append(key);self.journal({'kind':'forbidden_before_dispatch','operation':key})
            raise RuntimeError('Outside fixed raw capture scope: '+key)
        return fail
    def finish(self):
        native.need(not self.report['forbidden_attempts'] and all(self.report[p].get(k,0)==n for p in ('predebited','observed') for k,n in limits().items()),'Incomplete full-depth exact budget')


def drawings(ordinal):
    return {str(s):{str(b):row_draw(M,ROWS,base_seed=42,source_index=ordinal,step=s,block=b,state_pair=True)[0]
      for b in range(40)} for s in range(4)}


def validate(plan,frozen=False):
    native.need(plan.get('id')==scope_id(plan.get('part')) and plan.get('numeric_path')=='BF16_ORIGINAL'
      and plan.get('limits')==limits() and plan.get('resources')==RESOURCES and plan.get('runtime_overlay')==native.recipe()
      and plan.get('automatic_retry') is False,'Fixed full-depth identity differs')
    cases=plan['cases'];native.need(len(cases)==4 and [c['role'] for c in cases]==['development','calibration','calibration','calibration']
      and len({c['image_sha256'] for c in cases})==len({c['native_source_ordinal'] for c in cases})==4
      and all(c['part']==plan['part'] and c['rows']==drawings(c['native_source_ordinal']) for c in cases),'Versioned source/row geometry differs')
    native.need(cases[0]['consumer20_index']==(0 if plan['part']==0 else 12) and plan['run_ids']==[run_id(c) for c in cases], 'Old neutral locator and native ordinal confused')
    native.need(plan['parent']['protocol']['path']==PARENT and plan['parent']['protocol']['sha256']==PARENT_SHA
      and plan['parent']['RUN']['sha256']==PARENT_RUN_SHA and plan['parent']['exit']['sha256']==PARENT_EXIT_SHA,'Original raw parent identity differs')
    paths=[b['path'] for b in plan['repo_sources']];native.need(REQUIRED<=set(paths) and len(paths)==len(set(paths)),'Full-depth source closure incomplete')
    out=Path(plan['output_HF_relative']);native.need(not out.is_absolute() and out.parts[:1]==('ffnds-derived',) and '..' not in out.parts,'Output escaped HF')
    if frozen:native.need(plan.get('assets_finalized') is True and plan.get('binding_status')=='BOUND_READY_FOR_REGISTRATION','Full-depth draft NOT_BOUND')


def exited(value,run,sha,id):
    native.need(value.get('id')==id and value.get('protocol_sha256')==sha and value.get('run_sha256')==run['sha256']
      and value.get('process_exited') is True and type(value.get('child_returncode')) is int and value['child_returncode']==0
      and value.get('timeout') is False and value.get('automatic_retry') is False,'Actual independent exit0 identity differs')


def verify_lineage(plan,hf,*,stat_check=True):
    old=native.read(native.bound(ROOT,plan['parent']['protocol']));run=native.read(native.bound(hf,plan['parent']['RUN']))
    native.validate(old,True);exited(native.read(native.bound(hf,plan['parent']['exit'])),plan['parent']['RUN'],PARENT_SHA,native.ID)
    native.need(run.get('status')=='COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING' and run.get('protocol_sha256')==PARENT_SHA
      and not run['operation_ledger']['forbidden_attempts'] and all(run['operation_ledger'][p].get(k,0)==n for p in ('predebited','observed') for k,n in native.LIMITS.items()),'Original raw baseline incomplete')
    cohort=native.read(native.bound(ROOT,plan['source_cohort']));sources.validate_sources(cohort,hf)
    wanted=[s for s in cohort['sources'] if s['part']==plan['part']]
    for case,source in zip(plan['cases'],wanted):
        native.need(all(case[k]==source[k] for k in source) and case['image_sha256']==source['image']['sha256']
          and case['index']==source['consumer20_index'] and case['conditioning_id']==source['source_id']+':native_crop_condition','Full-depth source provenance differs')
    inherited=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256')
    native.need(all(plan[k]==old[k] for k in inherited) and plan['raw_master_CPU_proof']==run['CPU_preflight'],'Raw master/model runtime identity differs')
    if stat_check:native.require_CPU({'cpu_preflight':plan['raw_master_CPU_proof']},old,PARENT_SHA,hf)
    case=plan['cases'][0];phase=next(p for p in run['phases'] if p['source_index']==case['consumer20_index'])
    native.need(case['controls']==phase['states'] and case['reused_video']==phase['video'] and case['source_id']==phase['source_id'],'Old raw neutral/source controls differ')
    return cohort,old,run


def cpu_identity(plan,digest):
    return {'id':plan['id'],'protocol_sha256':digest,'status':'PASS_RAW_FULL_DEPTH_SOURCE_ROWS_CONTROL_BYTES_AND_REUSED_MASTER_STAT_NO_NUMERIC_IMPORTS',
      'part':plan['part'],'sources_sha256':native.canonical(plan['repo_sources']),'source_cohort_sha256':plan['source_cohort']['sha256'],
      'raw_master_CPU_proof':plan['raw_master_CPU_proof'],'controls':10,'sources':4,'native_row_sites_planned':640,
      'all_CUDA_devices_hidden':True,'Torch_or_NumPy_imported':False,'model_calls':0,'GPU_calls':0,'raw_master_payload_bytes_rehashed':0}


def preflight(args):
    started=time.monotonic();native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'Fresh hidden CUDA stdlib preflight required')
    plan=native.read(args.protocol);validate(plan,True);native.need(len(os.sched_getaffinity(0))<=4,'Preflight affinity exceeds4')
    verify_lineage(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    for case in plan['cases']:native.bound(args.hf_root,case['image'])
    for b in plan['cases'][0]['controls'].values():native.bound(args.hf_root,b)
    native.model_metadata(plan);native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    result=cpu_identity(plan,native.sha(args.protocol));result['wall_seconds']=time.monotonic()-started
    native.need(result['wall_seconds']<=300 and not {'torch','numpy'}&set(sys.modules),'CPU proof wall/import gate differs');return result


def authorize(plan,digest):
    validate(plan,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');policy=active.get('resource_scheduling_policy_20261028',{})
    native.need(active.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True
      and policy.get('mode')=='DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
      and active.get('current_session_resource_constraint_20261028',{}).get('max_simultaneously_assigned_idle_GPUs')==2,'Current session authority differs')
    matches=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==plan['id']]
    native.need(len(matches)==1 and matches[0].get('protocol_sha256')==digest and matches[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and matches[0].get('limits')==limits() and matches[0].get('resources')==RESOURCES and matches[0].get('run_ids')==plan['run_ids']
      and matches[0].get('max_gpu_cards')==1,'Unique independent full-depth part registration required');return matches[0]


def require_cpu(scope,plan,digest,hf):
    proof=scope.get('cpu_preflight');native.need(isinstance(proof,dict) and {'RUN','launch_exit'}<=set(proof),'Independent CPU part proof NOT_BOUND')
    run=native.read(native.bound(hf,proof['RUN']));native.need(all(run.get(k)==v for k,v in cpu_identity(plan,digest).items()) and 0<=run.get('wall_seconds',-1)<=300,'Independent byte CPU proof differs')
    exited(native.read(native.bound(hf,proof['launch_exit'])),proof['RUN'],digest,plan['id']);return proof


def check_packet_metadata(metadata,rows):
    native.need(set(metadata)==set(ROW_SHAPES) and len(rows)==len(set(rows))==ROWS and all(type(i) is int and 0<=i<M for i in rows),'Packet coordinates/tensor fields differ')
    native.need(all(metadata[k]['shape']==shape and metadata[k]['dtype']==('int64' if k=='rows' else 'bfloat16')
      and len(metadata[k]['raw_tensor_sha256'])==64 and all(c in '0123456789abcdef' for c in metadata[k]['raw_tensor_sha256']) for k,shape in ROW_SHAPES.items()),'Raw packet shape/dtype/hash differs')


def reserve_output_bytes(out,additional):
    # 1MiB covers this small Safetensors header, JSON record and operation log.
    # This is a CPU filesystem check only, not repeated model/GPU authorization.
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
    native.need(used+additional+(1<<20)<=RESOURCES['new_HF_bytes']
      and shutil.disk_usage(out).free-additional-(1<<20)>=RESOURCES['free_disk_min_bytes'],'Output reservation exceeds fixed HF/disk budget')


class CollectorTap(native.NativeTap):
    def __init__(self,pipe,guard,rid,save,check,case,row_indices,packet_writer):
        super().__init__(pipe,guard,rid,save,check)
        self.case,self.row_indices,self.packet_writer=case,row_indices,packet_writer
        self.pending=None;self.seen=set()
    def select(self,name,value):
        torch=self.guard.torch;p=self.pending
        native.need(p is not None and name not in p['values'] and value.dtype==torch.bfloat16
          and list(value.shape)==[M,H if name=='h' else D],'Raw full-M operand boundary differs')
        p['source_layout'][name]={'shape':list(value.shape),'dtype':'bfloat16','stride':list(value.stride())}
        p['values'][name]=self.guard.ledger.call(self.rid,{'operand_row_selections':1},value.index_select,0,p['rows'])
    def mm_call(self,mm,value,meta,original):
        slot=self.mapping.get(id(mm));suffix=slot[2] if slot is not None and slot[0]=='block' else None
        if suffix in ('ffn.0','ffn.2'):
            native.need(self.pending is not None and self.block==self.pending['expert_block'],'FFN MM outside row-capture transaction')
            self.select('V1' if suffix=='ffn.0' else 'h',value)
        result=super().mm_call(mm,value,meta,original)
        if suffix=='ffn.2':self.select('Y',result);self.pending['Y_object']=result
        return result
    def install(self):
        super().install()
        for expert,model in enumerate(self.runner.model.model):
            infer=model.transformer_infer;old_ffn,old_write=infer.infer_ffn,infer.post_process
            def ffn(phase,x,attn,shift,scale,gate=None,_expert=expert,_infer=infer,_fn=old_ffn):
                step=self.step(_expert);block=int(_infer.block_idx);site=(step,block)
                native.need(self.pending is None and site not in self.seen and self.block==(_expert,block)
                  and gate is not None and gate.numel()==shift.numel()==scale.numel()==D,'Duplicate/wrong FFN capture site or conditioning')
                def condition():
                    return {'gate':gate.reshape(-1).detach(),'shift':shift.reshape(-1).detach(),'scale':scale.reshape(-1).detach()}
                values=self.guard.ledger.call(self.rid,{'raw_FFN_conditioning_captures':1},condition)
                self.pending={'step':step,'block':block,'expert_block':(_expert,block),'rows':self.row_indices[step,block],
                  'values':values,'source_layout':{},'x_object':x,'gate_object':gate,'ready':False}
                result=_fn(phase,x,attn,shift,scale,gate)
                native.need(result is self.pending.get('Y_object') and set(self.pending['values'])=={'gate','shift','scale','V1','h','Y'},'Actual raw FFN output identity differs')
                self.pending['ready']=True;return result
            def write(x,y,gate,pre_infer_out=None,_expert=expert,_fn=old_write):
                p=self.pending
                native.need(p is not None and p['ready'] and self.block==p['expert_block'] and self.step(_expert)==p['step']
                  and x is p['x_object'] and y is p['Y_object'] and gate is p['gate_object'],'Postwrite not paired to its actual FFN')
                self.select('r',x)
                result=_fn(x,y,gate,pre_infer_out)
                native.need(result is x,'Original equal-dtype BF16 postwrite changed return identity')
                self.select('u',result)
                def geometry():
                    grid=list(pre_infer_out.grid_sizes.tuple)
                    native.need(len(grid)==3 and all(type(i) is int and i>0 for i in grid) and math.prod(grid)==M,'Actual native grid/population differs')
                    return grid
                grid=self.guard.ledger.call(self.rid,{'actual_grid_geometry_checks':1},geometry)
                self.guard.ledger.call(self.rid,{'raw_postwrite_packets':1},self.packet_writer,p,grid)
                self.seen.add((p['step'],p['block']));self.pending=None
                return result
            self.stack.enter_context(native.patch_method(infer,'infer_ffn',ffn));self.stack.enter_context(native.patch_method(infer,'post_process',write))
    def finish(self):
        super().finish();native.need(self.pending is None and self.seen=={(s,b) for s in range(4) for b in range(40)},'Incomplete full160 raw operands')


def registration(plan,hf):
    from ffnds_cohort_preflight import audit
    registered_sources,samples=[],[]
    for c in plan['cases'][1:]:
        rid=run_id(c)
        registered_sources.append({'source_id':c['source_id'],'image_id':c['image_sha256'],'prompt_id':c['prompt_sha256'],'prompt':c['prompt'],
          'image_path':str((hf/c['image']['path']).resolve()),'image_name':Path(c['image']['path']).name,'conditioning_id':c['conditioning_id'],
          'role':'CALIBRATION','exposure_history':{'fitting':False,'method_selection':True,'prior_evaluation':True}})
        samples.append({'source_id':c['source_id'],'variant':'dense','run_id':rid,'asset_id':'video:'+rid,'seed':42,'sample_index':0,
          'conditioning_id':c['conditioning_id'],'initial_noise_id':plan['id']+':'+c['source_id'],
          'scheduler_id':'Wan22StepDistillScheduler:shift5:sigma1,.9375,.8333333134651184,.625:high,high,low,low',
          **{k:native.VIDEO[k] for k in ('frames','fps','width','height')},'status':'PLANNED'})
    result={'schema':'ffnds.cohort.v1','mode':'CUSTOM_PAIRED','experiment':plan['id'],'phase':'REGISTERED_REAL_GENERATION_PLAN',
      'sources':registered_sources,'samples':samples,'calibration':[],'required_variants':['dense']}
    checked=audit(result);native.need(not checked['errors'] and checked['summary']['independent_image_clusters']==3,'Raw calibration video registration invalid')
    return result


def native_phase(torch,pipe,guard,plan,case,out,hf,digest,report,persist,check):
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    rid=run_id(case);folder=out/rid;folder.mkdir()
    phase={'id':plan['id'],'protocol_sha256':digest,'run_id':rid,'numeric_path':'BF16_ORIGINAL','status':'STARTED_NO_RETRY',
      'source_index':case['consumer20_index'],'native_source_ordinal':case['native_source_ordinal'],'source_id':case['source_id'],
      'image_sha256':case['image_sha256'],'prompt_sha256':case['prompt_sha256'],'conditioning_id':case['conditioning_id'],
      'role':case['role'],'history':case['history'],'mode':'dense','source_state':'teacher_input','seed':42,'states':{},'packets':[],
      'quality_or_speed_acceptance':False,'observer_cost_is_included_in_capture_wall':True}
    report['phases'].append(phase);persist();ledger=guard.ledger
    def save(value,name,key):
        native.need(name not in phase['states'],'Duplicate native state output')
        expected=torch.float32 if key in ('raw_FP32_noise_files','native_FP32_flow_files') else torch.bfloat16
        def output():
            native.need(value.dtype==expected and list(value.shape)==SHAPE,'Actual original state geometry/dtype differs')
            ledger.call(rid,{'state_output_byte_reservations':1},reserve_output_bytes,out,value.numel()*value.element_size())
            cpu=ledger.call(rid,{'state_tensor_D2H':1,'state_tensor_D2H_bytes':value.numel()*value.element_size()},lambda:value.detach().cpu().contiguous())
            native.need(bool(torch.isfinite(cpu).all()),'Nonfinite raw state')
            path=folder/(name+'.safetensors');native.need(not path.exists(),'State overwrite forbidden')
            save_file({'flow' if key=='native_FP32_flow_files' else 'latents':cpu},str(path))
            item={'root':'hf',**native.binding(path,hf),'raw_tensor_sha256':tensor_sha(cpu),'shape':list(cpu.shape),'dtype':str(cpu.dtype).removeprefix('torch.')}
            if case['role']=='development':
                def neutral():
                    ref=case['controls'][name]
                    native.need(item['dtype']==ref['dtype'] and item['shape']==ref['shape'] and item['raw_tensor_sha256']==ref['raw_tensor_sha256'],'Old raw flow/post/noise neutral mismatch')
                ledger.call(rid,{'neutral_state_byte_checks':1},neutral)
            phase['states'][name]=item;persist();return item
        return ledger.call(rid,{key:1},output)
    index=ledger.call(rid,{'row_index_uploads':1,'row_index_upload_bytes':4*40*ROWS*8},torch.tensor,
      [[case['rows'][str(s)][str(b)] for b in range(40)] for s in range(4)],dtype=torch.int64,device='cuda')
    def packet(p,grid):
        native.need(set(p['values'])==set(ROW_SHAPES)-{'rows'},'Missing actual captured operand')
        ledger.call(rid,{'packet_output_byte_reservations':1},reserve_output_bytes,out,PACKET_BYTES)
        values={**p['values'],'rows':p['rows']};cpu={}
        for k,value in values.items():
            cpu[k]=ledger.call(rid,{'packet_tensor_D2H':1,'packet_tensor_D2H_bytes':value.numel()*value.element_size()},lambda value=value:value.detach().cpu().contiguous())
        row_ids=case['rows'][str(p['step'])][str(p['block'])]
        def validate_payload():
            native.need(all(bool(torch.isfinite(v).all()) for v in cpu.values()),'Nonfinite raw row operand')
            meta={k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.'),'raw_tensor_sha256':tensor_sha(v)} for k,v in cpu.items()}
            check_packet_metadata(meta,row_ids)
            native.need(cpu['rows'].tolist()==row_ids,'Saved row payload differs from native-ordinal draw')
            return meta
        meta=ledger.call(rid,{'packet_payload_validations':1},validate_payload)
        name='step'+str(p['step'])+'_block'+f"{p['block']:02d}";path=folder/(name+'.safetensors')
        native.need(not path.exists(),'Raw row pack overwrite forbidden')
        ledger.call(rid,{'packet_safetensor_writes':1},save_file,cpu,str(path))
        record={'schema':'ffnds.raw_bf16.full_depth.operands.v1','source_id':case['source_id'],'image_sha256':case['image_sha256'],
          'prompt_sha256':case['prompt_sha256'],'consumer20_index':case['consumer20_index'],'native_source_ordinal':case['native_source_ordinal'],
          'seed':42,'role':case['role'],'history':case['history'],'numeric_path':'BF16_ORIGINAL','source_state':'teacher_input',
          'step':p['step'],'block':p['block'],'expert':'high' if p['step']<2 else 'low','grid':grid,'population_rows':M,
          'selected_rows':row_ids,'row_draw_identity':'ffnds.row_draw:Python PRNG native_source_ordinal seed42 state_pair=True',
          'full_operand_layout':p['source_layout'],'tensor_metadata':meta,'payload':{'root':'hf',**native.binding(path,hf)},
          'tensor_boundaries':{'V1':'original modulated LN before full raw W1','h':'original tanh GELU output / raw W2 input',
            'Y':'raw full W2 output including original bias','r':'post-attention residual immediately before FFN gated write',
            'u':'original BF16 post_process output; captured after native inplace add','gate':'actual per-channel FFN gate','shift':'actual FFN AdaLN shift','scale':'actual FFN AdaLN scale'},
          'decomposition_or_fit':False,'protocol_sha256':digest,'run_id':rid}
        case_path=folder/(name+'.json');ledger.call(rid,{'packet_metadata_writes':1},native.write,case_path,record)
        phase['packets'].append({'step':p['step'],'block':p['block'],'metadata':native.binding(case_path,hf)})
    pipe.runner.scheduler.generator=None
    tap=CollectorTap(pipe,guard,rid,save,check,case,index,packet);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        if case['role']=='development':
            from lightx2v.utils.input_info import init_empty_input_info,update_input_info_from_dict
            from lightx2v.utils.utils import seed_all
            def replay():
                runner=pipe.runner;info=init_empty_input_info('i2v',pipe.support_tasks)
                update_input_info_from_dict(info,{'seed':42,'prompt':case['prompt'],'negative_prompt':'',
                  'image_path':str((hf/case['image']['path']).resolve()),'save_result_path':'UNUSED_NO_NEW_DEVELOPMENT_VIDEO'})
                seed_all(42);runner.input_info=info
                with tap,torch.inference_mode():
                    runner.inputs=runner.run_input_encoder();runner.init_run()
                    native.need(runner.video_segment_num==1,'Original one-segment runner required')
                    runner.init_run_segment(0);runner.run_segment(0)
                    runner.end_run()
                torch.cuda.synchronize();tap.finish()
            ledger.call(rid,{'native_prefix_replays_no_decode':1},replay)
            phase['all_four_original_flow_and_post_bytes_passed']=True;phase['video_reused']=case['reused_video']
        else:
            def generate():
                with tap,torch.inference_mode():
                    pipe.generate(seed=42,image_path=str((hf/case['image']['path']).resolve()),prompt=case['prompt'],negative_prompt='',save_result_path=str(folder/'output.mp4'))
                torch.cuda.synchronize();tap.finish()
            ledger.call(rid,{'video_generations':1},generate)
            from run_ub1_consumer_videos import inspect_video
            phase['video']=ledger.call(rid,{'video_geometry_checks':1},inspect_video,folder/'output.mp4',native.VIDEO)
        native.need(len(phase['packets'])==160 and set(phase['states'])=={'raw_noise_FP32','initial_noise',*[f'flow_step{s}' for s in range(4)],*[f'latent_step{s}' for s in range(4)]},'Incomplete source raw states/positions')
        phase.update(status='COMPLETE_RAW_BF16_TEACHER_STATE_FULL160',completed_native_flow_steps=4,native_ffn_sites=160,
          native_counter_delta={'linear_forward_calls':len(tap.lines),'activation_qdq_calls':0,'weight_quantizations':0},
          scheduler_generator_reset=True,timetable=tap.timetable,Default_MM_actual_operand_metadata=tap.shapes,
          source_capture_wall_seconds=time.perf_counter()-started,FFN_boundary_CUDA_seconds=sum(a.elapsed_time(b) for a,b in tap.events)/1000,
          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        native.need(phase['peak_cuda_reserved_bytes']<=RESOURCES['GPU_reserved_max_bytes'],'Full-depth GPU peak32GiB exceeded')
        generation=folder/'GENERATION.json';native.write(generation,phase);phase['generation_receipt']=native.binding(generation,hf)
        return phase
    except BaseException as error:
        phase.update(status='FAILED_SOURCE_FULL_DEPTH_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:native.write(folder/'PHASE.json',phase);persist()


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol)
    scope=authorize(plan,digest);cpu=require_cpu(scope,plan,digest,args.hf_root)
    native.need(not {'torch','numpy'}&set(sys.modules),'Fresh model worker before numerical imports required')
    verify_lineage(plan,args.hf_root)
    for b in plan['repo_sources']:native.bound(ROOT,b)
    native.model_metadata(plan);local,rows=native.verify_local_package(plan,{'repo':ROOT,'hf':args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions'],'Actual native package versions differ')
    from worker_identity import require_worker,private_config_path
    require_worker();native.need(native.sha(private_config_path()) in plan['private_worker_config_sha256'],'Private worker identity differs')
    native.need(len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=='1' for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'CPU affinity/TF32 override differs')
    out=(args.hf_root/plan['output_HF_relative']).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined part output required; no retry')
    out.mkdir(parents=True);report={'id':plan['id'],'protocol_sha256':digest,'numeric_path':'BF16_ORIGINAL','part':plan['part'],
      'status':'STARTED_RAW_FULL_DEPTH_PART_NO_RETRY','automatic_retry':False,'CPU_preflight':cpu,'phases':[],
      'source_cohort':plan['source_cohort'],'native_part_image_clusters':4,'new_calibration_image_clusters':3,'old_development_image_clusters':1,
      'complete_cohort_claimed':False,'cohort_completion':'PART_ONLY_REQUIRES_TWO_INDEPENDENT_ACTUAL_EXIT0_RECEIPTS',
      'state_identity':'Teacher states only; no student states captured','source_history':'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT',
      'official_quality':'NOT_RUN','training_updates':0,'decompositions':0,'quality_or_speed_acceptance':False,
      'capture_runtime_identity':'Native generation plus row capture/D2H/serialization/CPU checks; not a throughput measurement'}
    persist=lambda:native.write(out/'RUN.json',report);journal=(out/'OPERATIONS.jsonl').open('x',buffering=1)
    ledger=Ledger(report.setdefault('operation_ledger',{}),plan['run_ids'],lambda event:journal.write(json.dumps(event,separators=(',',':'))+'\n'))
    persist();monitor=None
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available>=RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RESOURCES['free_disk_min_bytes'],'Host/disk resource reserve exceeded')
        native.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Part HF output limit12GiB exceeded')
        native.need(time.monotonic()-started<=RESOURCES['wall_seconds'],'Full-depth part wall1200 exhausted')
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'Part GPU reserved32GiB exceeded')
    def timeout(*_):raise TimeoutError('Fixed full-depth part wall1200 exhausted; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,rows),native.NoHiF4()
    try:
        check();snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir()
        for item in plan['repo_sources']:
            destination=snapshot/item['path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],destination)
        shutil.copyfile(args.protocol,snapshot/'protocol.json')
        shutil.copyfile(ROOT/plan['source_cohort']['path'],snapshot/'source_cohort.json')
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,report['CUDA_reservation']=reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Single assigned A100 required')
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'matmul_TF32':False,'float32_matmul_precision':'highest','bf16_reduced_precision_reduction':True}
        with native.OriginalGuard(torch,plan,ledger) as guard:
            begun=time.perf_counter();pipe=native.build_pipeline(plan);setup_seconds=time.perf_counter()-begun
            report.update(raw_model_setup_seconds=setup_seconds,original_loader_paths=guard.loaded,
              BF16_matrix_residency='Original raw CPU masters and four CUDA block buffers; original offload including W2 unchanged')
            registered=registration(plan,args.hf_root);native.write(snapshot/'registered_video_plan.json',registered);persist();check()
            # The first source is the old dev neutral for this independent part.
            # It must pass all four full-flow/post checks before any new source.
            for index,case in enumerate(plan['cases']):
                native.need(index==0 or report.get('local_development_neutral_passed') is True,'Calibration capture before local neutral')
                phase=native_phase(torch,pipe,guard,plan,case,out,args.hf_root,digest,report,persist,check)
                if index==0:
                    native.need(phase['all_four_original_flow_and_post_bytes_passed'] is True,'Local dev neutral did not pass')
                    report['local_development_neutral_passed']=True;persist();check();continue
                from ffnds_video_metrics import evaluate,validate as validate_core
                import cv2
                cv2.setNumThreads(4)
                identity={'source_id':case['image_sha256'],'seed':42,'expected_frames':81,'input_prompt_digest':case['prompt_sha256'],
                  'model_digest':native.canonical(plan['models']),'protocol_sha256':digest,'run_sha256':phase['generation_receipt']['sha256']}
                cost={'generation_wall_seconds':phase['source_capture_wall_seconds'],'ffn_boundary_seconds':phase['FFN_boundary_CUDA_seconds'],
                  'setup_seconds':setup_seconds,'peak_gpu_memory_mib':phase['peak_cuda_allocated_bytes']/(1<<20)}
                core=ledger.call(run_id(case),{'core_CPU_reports':1},evaluate,Path(phase['video']['path']),None,{'identity':identity,'cost':cost})
                native.need(not validate_core(core),'New81frame CORE incomplete');core_path=out/run_id(case)/'CORE_METRICS.json';native.write(core_path,core)
                phase['core_metrics']=native.binding(core_path,args.hf_root)
                def save_registration():
                    folder=out/f'job{index-1:06d}';folder.mkdir()
                    native.write(folder/'RUN.json',{'schema':'ffnds.registered.native.video.v1','status':'COMPLETE_NATIVE_GENERATION_ONLY',
                      'job':registered['samples'][index-1],'source':registered['sources'][index-1],'native_ffn_sites':160,
                      'scheduler_generator_reset':True,'native_ffn_counters':phase['native_counter_delta'],
                      'actual_initial_noise_file':str(args.hf_root/phase['states']['initial_noise']['path']),
                      'video_path':phase['video']['path'],'decoded_frames':81,'decoded_size':[832,464],'decoded_fps':16,
                      'immutable_generation_receipt':phase['generation_receipt'],'numeric_path':'BF16_ORIGINAL'})
                ledger.call(run_id(case),{'source_registration_writes':1},save_registration);persist();check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified=collect(registered,out);native.write(out/'VERIFIED_GENERATIONS.json',verified)
            report.update(status='COMPLETE_RAW_BF16_FULL_DEPTH_PART_FOUR_SOURCES',verified_cohort=native.binding(out/'VERIFIED_GENERATIONS.json',args.hf_root),
              checked_transformers_imports=finder.origins,captured_native_sites=sum(len(p['packets']) for p in report['phases']),
              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            native.need(report['captured_native_sites']==640,'Part capture total differs');check()
    except BaseException as error:
        report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources'].get('violations'),'Final GPU monitor violations')
            except BaseException as error:
                monitor_error=error;report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY',resource_error=str(error))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        report['wall_seconds']=time.monotonic()-started;persist();journal.close()
        if monitor_error is not None and sys.exc_info()[0] is None:raise monitor_error
    return report


def draft(hf,part,cohort_path=COHORT):
    hf=Path(hf);old=native.read(ROOT/PARENT);folder=hf/old['output_HF_relative'];run=native.read(folder/'RUN.json')
    cohort=native.read(ROOT/cohort_path);sources.validate_sources(cohort,hf);cases=[]
    for source in cohort['sources']:
        if source['part']!=part:continue
        case={**copy.deepcopy(source),'index':source['consumer20_index'],'image_sha256':source['image']['sha256'],
          'conditioning_id':source['source_id']+':native_crop_condition','rows':drawings(source['native_source_ordinal'])}
        if case['role']=='development':
            phase=next(p for p in run['phases'] if p['source_index']==case['consumer20_index'])
            case.update(controls=phase['states'],reused_video=phase['video'])
        cases.append(case)
    plan={k:copy.deepcopy(old[k]) for k in ('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay')}
    plan.update(id=scope_id(part),part=part,status='DRAFT_BOUND_RAW_DATA_COLLECTOR_NO_MODEL_GPU',numeric_path='BF16_ORIGINAL',assets_finalized=False,
      binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING',automatic_retry=False,limits=limits(),resources=RESOURCES,cases=cases,run_ids=[run_id(c) for c in cases],
      parent={'protocol':native.binding(ROOT/PARENT,ROOT),'RUN':native.binding(folder/'RUN.json',hf),'exit':native.binding(folder.parent/'EXIT_RECEIPT.json',hf)},
      raw_master_CPU_proof=run['CPU_preflight'],source_cohort=native.binding(ROOT/cohort_path,ROOT),
      output_HF_relative=f'ffnds-derived/e0_original_bf16_full_depth_collect_20261028/part{part}/attempt1/worker/results',
      source_state='teacher_input',student_state_data='NOT_COLLECTED',source_history='All source images historically exposed in consumer20; calibration means first raw collection in this pilot',
      source_split='6calibration+2development image clusters total; this independent part has1dev then3cal; no blind confirmation',
      row_draw_identity='Python ffnds.row_draw base_seed42, source_index=native_source_ordinal, step/block, state_pair=True; consumer20_index only locates prior raw dev controls',
      tensor_contract={'schema':'ffnds.raw_bf16.full_depth.operands.v1','row_shapes':ROW_SHAPES,'rows':ROWS,'population':M,'slots':4*40,'numeric_path':'BF16_ORIGINAL','Q1':'NONE','Q2':'NONE'},
      neutral_policy='Each independent part first reproduces its own old dev four raw flow+post bytes and raw/initial noise; stop on failure. No peer barrier; no complete-cohort claim without both actual exited parts.',
      row_tensor_payload_bytes_per_part=640*PACKET_BYTES,state_tensor_payload_bytes_per_part=4*math.prod(SHAPE)*(4+2+4*4+4*2),
      single_packet_tensor_payload_bytes=PACKET_BYTES,packet_memory_identity='One site is flushed before the next; CPU/GPU copies and serialization buffers may coexist, total peak is measured by actual native monitors',
      CPU_preflight_resources={'GPU':0,'CPU_threads':4,'wall_seconds':300,'outer_seconds':360,'new_raw_master_hash_bytes':0},
      new_full_model_decompositions=0,new_full_model_training=0,official_quality='NOT_RUN',quality_or_speed_accepted=False)
    plan['repo_sources']=[native.binding(ROOT/p,ROOT) for p in sorted({b['path'] for b in old['repo_sources']}|REQUIRED)]
    validate(plan);verify_lineage(plan,hf,stat_check=False);return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('draft','inspect','preflight','run'))
    p.add_argument('--part',type=int);p.add_argument('--protocol',type=Path);p.add_argument('--hf-root',type=Path,required=True)
    p.add_argument('--local-root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--cohort',default=COHORT)
    args=p.parse_args();args.hf_root=args.hf_root.resolve()
    if args.command=='draft':
        native.need(args.part in (0,1) and args.out is not None and not args.out.exists(),'Part and fresh draft output required')
        value=draft(args.hf_root,args.part,args.cohort);native.write(args.out,value);print(json.dumps({'id':value['id'],'status':value['status'],'model_calls':0}))
    elif args.command=='inspect':
        value=native.read(args.protocol);validate(value);print(json.dumps({'id':value['id'],'part':value['part'],'limits':limits(),'resources':RESOURCES,'model_calls':0},indent=2))
    elif args.command=='preflight':
        native.need(args.out is not None and args.local_root is not None,'Fresh CPU output and local package required')
        out=args.out.resolve();native.need(out.is_relative_to(args.hf_root) and not out.exists(),'Fresh confined CPU output required')
        out.mkdir(parents=True);value={'id':native.read(args.protocol)['id'],'protocol_sha256':native.sha(args.protocol),'status':'STARTED_BYTE_PREFLIGHT_NO_RETRY'};native.write(out/'RUN.json',value)
        try:value=preflight(args)
        except BaseException as error:value.update(status='FAILED_BYTE_PREFLIGHT_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
        finally:native.write(out/'RUN.json',value)
        print(json.dumps({'status':value['status'],'model_calls':0}))
    else:
        native.need(args.local_root is not None,'Exact local package required');print(json.dumps({'status':run(args)['status']}))


if __name__=='__main__':main()
