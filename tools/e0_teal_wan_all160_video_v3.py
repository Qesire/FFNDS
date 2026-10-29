"""V3 two-at-a-time Wan BF16 full-160 TEAL-style video quality diagnostic.

Fixed thresholds originate solely in the separately exited six-source CPU
calibration. This eager mask retains the actual dense W2 and is not a speed or
sparse-kernel experiment. No model work on import or threshold export.
"""
import argparse, copy, importlib.util, json, math, os, signal, sys, time, types, shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_teal_video_private_sm120',ROOT/'tools/e0_s2_new12_sm120_fixed_depth_videos.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
vm,native,io,legacy=base.vm,base.native,base.io,base.legacy
from raw_bf16_factor_w2_runtime import FactorW2Runtime
SELF='tools/e0_teal_wan_all160_video_v3.py';TEST='tests/test_e0_teal_wan_all160_video_v3.py'
V2_SOURCE='tools/e0_teal_wan_all160_video.py'
V2_CONFIG='config/e0_teal_wan_all160_video_part{}_20261029_pending_review_v2.json'
GROUP_ID='E0_TEAL_WAN_V3_NATIVE_TWO_MODEL_GROUP_20261029'
HOST_FORECAST_PER_MODEL=96<<30
CPU_PROTOCOL='config/e0_teal_wan_all160_cpu_local_20261029_pending_review_v1.json'
CPU_OUTPUT='ffnds-derived/e0_teal_wan_all160_cpu_local_20261029/attempt1/worker/results'
THRESHOLD_ARTIFACT='config/e0_teal_wan_all160_calibration_thresholds_20261029_v1.json'
SOURCE_CONFIG='config/e0_s2_new12_dense_baseline_part{}_20261029_v2.json'
MASTER_TEMPLATE='config/e0_s2_new12_fixed_depth_part0_20261029_v1.json'
ORDINALS=(1040,979,456,923);MODES=('dense','threshold25','threshold50');RATES=(0.25,0.5)
RESOURCES={'GPU':1,'CPU_threads':4,'GPU_reserved_max_bytes':40<<30,'host_available_min_bytes':128<<30,
 'free_disk_min_bytes':6<<30,'native_group_id':GROUP_ID,'native_group_max_concurrent_models':2,
 'admission_free_host_bytes':(128+2*96)<<30,'forecast_per_model_bytes':HOST_FORECAST_PER_MODEL,'wall_seconds':3600,'outer_seconds':3900,'new_HF_bytes':1<<30,'automatic_retry':False}
CPU_RESOURCES={'GPU':0,'CPU_threads':4,'wall_seconds':600,'outer_seconds':900,'new_HF_bytes':8<<20,'automatic_retry':False}
DESIGN={'evidence':'QUALITY_DIAGNOSTIC_NO_PHYSICAL_SAVING','sources':list(ORDINALS),'modes':list(MODES),
 'thresholds':'six calibration sources only, actual exited CPU CDF output, fixed before generation',
 'action':'original BF16 W1/GELU -> h abs<=threshold zero -> full original dense W2/bias -> original gate/residual',
 'bypass':'same-hardware independent old/new dense contexts; all ten state bytes and full video bytes',
 'sources_role':'four historically exposed development sources; not confirmation',
 'mask_counts':'full original h tensor per site, total/zero-before/deleted/nonzero-deleted/tie',
 'video_geometry':[81,832,464,16],'seed':42,'extra_native_FFN':0,'training':0,'decomposition':0,
 'new_weight_artifacts':0,'official_quality':False,'physical_sparse_W2':False,'CPU_preflight':'SAME_REVIEWED_PART_STATE_MACHINE_BYTE_ONLY_ZERO_GPU_MODEL_WITH_OWN_SUPERVISOR_AND_ACTUAL_EXIT',
 'session_max_GPUs':4,'native_group_max_models':2,'resource_observation':'separate persisted host/disk/wall values',
 'CPU_affinity_by_part':[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],'automatic_retry':False}


def scope_id(part):
    native.need(type(part) is int and part in range(4),'Four explicit shards required')
    return f'E0_TEAL_WAN_ALL160_VIDEO_V3_20261029_PART{part}'

def outpath(part):return f'ffnds-derived/e0_teal_wan_all160_video_v3_20261029/part{part}/attempt1/worker/results'

def rid(case,mode):
    native.need(mode in MODES,'Unknown mode');control=case.get('_control')
    native.need(control in (None,'old','new') and (control is None or mode=='dense'),'Bad control')
    return f'teal_v3_video_ord{case["native_source_ordinal"]}_seed42_{control+"_" if control else ""}{mode}'

def control(plan,which):return {**plan['neutral_case'],'_control':which}

def threshold_rows(reports):
    result=[]
    for report in reports:
        rows=report['records'];native.need(len(rows)==2 and {r['source_ordinal'] for r in rows}=={894,256},'CPU development identity differs')
        a,b=rows;native.need((a['step'],a['block'],a['expert'])==(b['step'],b['block'],b['expert']),'CPU paired site differs')
        thresholds=[]
        for rate in RATES:
            p=a['rates'][str(rate)]['threshold_calibration'];q=b['rates'][str(rate)]['threshold_calibration']
            native.need(p==q and p['target_rate']==rate and type(p['threshold']) in (float,int)
                and math.isfinite(p['threshold']) and p['threshold']>=0 and p['count']==6*256*13824,'Calibration threshold identity differs')
            thresholds.append(p['threshold'])
        result.append({'step':a['step'],'block':a['block'],'expert':a['expert'],'thresholds':thresholds})
    native.need(len(result)==160 and {(x['step'],x['block']) for x in result}=={(s,b) for s in range(4) for b in range(40)}
        and all(x['expert']==('high' if x['step']<2 else 'low') and x['thresholds'][0]<=x['thresholds'][1] for x in result),'Threshold coverage/order differs')
    return sorted(result,key=lambda x:(x['step'],x['block']))


def export_thresholds(hf,out):
    spec=importlib.util.spec_from_file_location('_teal_cpu_export_parent',ROOT/'tools/e0_teal_wan_all160_cpu_local.py')
    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
    folder=hf/CPU_OUTPUT;protocol=ROOT/CPU_PROTOCOL;run=folder/'RUN.json';exit=folder.parent/'EXIT_RECEIPT.json'
    closed=cpu.verify_closed(SimpleNamespace(hf_root=hf,protocol=protocol,run=run,exit=exit));parent=native.read(run)
    rows=threshold_rows([native.read(native.bound(hf,b)) for b in parent['site_reports']])
    artifact={'schema':'ffnds.teal.wan.calibration.thresholds.v1','source':'SIX_CALIBRATION_SOURCES_ONLY','rates':list(RATES),
      'CPU_parent':closed,'CPU_protocol':native.binding(protocol,ROOT),'CPU_manifest':native.read(protocol)['manifest'],
      'threshold_rows':rows,'no_development_fit':True}
    native.need(not out.exists(),'Fresh threshold artifact required');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(artifact,f,indent=2,allow_nan=False);f.write('\n')
    return native.binding(out,ROOT)


def validate_thresholds(a):
    native.need(a['schema']=='ffnds.teal.wan.calibration.thresholds.v1' and a['source']=='SIX_CALIBRATION_SOURCES_ONLY'
       and a['rates']==list(RATES) and a['no_development_fit'] is True and a['CPU_parent']['status']=='PASS_CLOSED_TEAL_CPU_LOCAL'
       and a['CPU_protocol']['path']==CPU_PROTOCOL,'Threshold artifact lineage differs')
    rows=a['threshold_rows'];native.need(len(rows)==160 and {(x['step'],x['block']) for x in rows}=={(s,b) for s in range(4) for b in range(40)}
      and all(x['expert']==('high' if x['step']<2 else 'low') and len(x['thresholds'])==2 and all(type(z) in (int,float) and math.isfinite(z) and z>=0 for z in x['thresholds'])
      and x['thresholds'][0]<=x['thresholds'][1] for x in rows),'Threshold artifact coverage differs')
    return {(x['step'],x['block']):x['thresholds'] for x in rows}


def mask_values(torch,value,threshold):
    native.need(bool(torch.isfinite(value).all()),'Nonfinite native h')
    absolute=value.abs();dropped=absolute<=threshold
    before_zero=int((value==0).sum().item());count=int(dropped.sum().item());ties=int((absolute==threshold).sum().item())
    masked=value.masked_fill(dropped,0)
    return masked,{'total':value.numel(),'selected':count,'preexisting_zero':before_zero,
                   'selected_nonzero':count-before_zero,'threshold_ties':ties}


def validate_action_coverage(mode,events):
    native.need(mode in MODES,'Mask action mode differs')
    native.need((mode=='dense' and not events) or (mode!='dense' and len(events)==160
      and {(x['step'],x['block']) for x in events}=={(a,b) for a in range(4) for b in range(40)}),
      'Mask all160 coverage differs')


class TealDenseRuntime(FactorW2Runtime):
    """Mask actual original W2 input; original dense loading and W2 unchanged."""
    def __init__(self,*args,thresholds,ledger,**kwargs):
        super().__init__(*args,**kwargs);self.thresholds=thresholds;self.ledger=ledger;self.mask_mode='dense';self.events=[];self.sites=set();self.rid=None
    def set_mask_mode(self,mode,rid):
        native.need(mode in MODES and self.guard.active is None,'Mask policy requires phase boundary')
        self.set_mode('dense');self.mask_mode=mode;self.rid=rid;self.events=[];self.sites=set()
    def __enter__(self):
        super().__enter__()
        try:
            for mm,expert in self.objects.values():
                old=mm.apply;runtime=self
                def apply(obj,value,*,_old=old,_expert=expert):
                    active=runtime.guard.active;native.need(active is not None and active.block is not None,'Mask escaped native FFN')
                    expert_now,block=active.block;step=active.step(expert_now)
                    native.need(expert_now==_expert==(0 if step<2 else 1) and 0<=block<40
                      and value.dtype==runtime.guard.torch.bfloat16 and list(value.shape)==[31668,13824],'Mask source shape/expert differs')
                    if runtime.mask_mode=='dense':result=_old(value)
                    else:
                        native.need((step,block) not in runtime.sites,'Duplicate mask site before operation')
                        runtime.sites.add((step,block))
                        torch=runtime.guard.torch;threshold=runtime.thresholds[step,block][0 if runtime.mask_mode=='threshold25' else 1]
                        masked,counts=runtime.ledger.call(runtime.rid,{'mask_calls':1},mask_values,torch,value,threshold)
                        result=_old(masked);runtime.events.append({'step':step,'block':block,'expert':expert_now,'threshold':threshold,**counts})
                    return result
                self._install(mm,'apply',apply)
        except BaseException:
            super().__exit__(None,None,None);raise
        return self


def limits():
    result={k:(v if k in base.acquisition.SETUP else v//2*5) for k,v in vm.native_original_limits.items()}
    result.update(core_CPU_reports=0,BF16_factor_GEMM=0,original_W2_H2D_skips=0,original_W2_H2D_skipped_bytes=0,
        factor_CPU_file_loads=0,factor_CPU_tensor_bytes=0,factor_GPU_family_preloads=0,factor_GPU_H2D_tensors=0,factor_GPU_H2D_bytes=0,
        prefetch_transfer_observations=5*162,stage_CUDA_event_pairs=5*7,
        state_tensor_D2H=50,state_tensor_D2H_bytes=5*math.prod(native.VIDEO['latent_shape'])*30,
        state_output_writes=50,video_geometry_checks=5,matched_reference_noise_byte_checks=4,
        neutral_state_byte_checks=10,neutral_video_byte_checks=1,source_registration_writes=3,
        historical_state_byte_checks=0,mask_calls=320)
    return result


def source_inputs():
    sources=[native.read(ROOT/SOURCE_CONFIG.format(i)) for i in (0,1)]
    pool={c['native_source_ordinal']:c for p in sources for c in p['cases'][1:]}
    native.need(len(pool)==12 and set(ORDINALS)<=set(pool),'Twelve-source identity differs')
    return sources,pool


def original_dense_parent(hf,sources,template,ordinal):
    proofs=[];matched=[]
    for i,source in enumerate(sources):
        proof=base.fixed.parent_proof(ROOT/SOURCE_CONFIG.format(i),hf)
        parent,run=base.fixed.verify_parent(proof,hf,'COMPLETE_NEW12_DENSE_BASELINE_PART')
        native.need(parent==source and run.get('historical_control_all_ten_states_passed') is True
          and run.get('actual_full160_videos')==7 and run.get('actual_FFN_positions')==1120
          and all(parent[k]==template[k] for k in ('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','runtime_overlay')),
          'Original Dense parent recipe/control differs')
        for case in parent['cases'][1:]:
            if case['native_source_ordinal']!=ordinal:continue
            selected=[x for x in run['phases'][1:] if x['native_source_ordinal']==ordinal]
            native.need(len(selected)==1,'Original Dense source not unique');phase=selected[0]
            gen=native.read(native.bound(hf,phase['generation_receipt']))
            native.need(gen['run_id']==phase['run_id'] and gen['protocol_sha256']==proof['protocol']['sha256']
              and gen['source_id']==case['source_id'] and gen['image_sha256']==case['image']['sha256']
              and gen['prompt_sha256']==case['prompt_sha256'] and gen['seed']==42 and gen['mode']=='dense'
              and gen['states']==phase['states'] and gen['video']==phase['video'] and gen['all160_FFN_covered'] is True
              and gen['completed_native_flow_steps']==4 and gen['scheduler_generator_reset'] is True and len(gen['states'])==10
              and gen['video']['frames']==81 and gen['video']['size']==[832,464] and gen['video']['fps']==16,'Original Dense generation differs')
            v=Path(gen['video']['path']).resolve();native.need(v.is_relative_to(hf.resolve()),'Parent video escaped HF')
            native.bound(hf,{'path':str(v.relative_to(hf.resolve())),'bytes':gen['video']['bytes'],'sha256':gen['video']['sha256']})
            matched.append({'generation':phase['generation_receipt'],'video_sha256':gen['video']['sha256'],
              'raw_noise_sha256':gen['states']['raw_noise_FP32']['sha256'],'initial_noise_sha256':gen['states']['initial_noise']['sha256'],
              'source_ordinal':ordinal,'recorded_GPU_peak_reserved_bytes':phase['peak_cuda_reserved_bytes'],
              'recorded_GPU_peak_allocated_bytes':phase['peak_cuda_allocated_bytes'],'parent_part':i})
        proofs.append(proof)
    native.need(len(matched)==1,'One distinct actual Dense source reference required')
    return proofs,matched[0]


def prior_failed_parts(hf):
    proofs=[]
    for part in range(4):
        path=ROOT/V2_CONFIG.format(part);old=native.read(path);folder=Path(hf)/old['output_HF_relative']
        proof={'protocol':native.binding(path,ROOT),'RUN':native.binding(folder/'RUN.json',hf),
               'exit':native.binding(folder.parent/'EXIT_RECEIPT.json',hf)}
        run=native.read(native.bound(hf,proof['RUN']));receipt=native.read(native.bound(hf,proof['exit']))
        native.need(old['id']==f'E0_TEAL_WAN_ALL160_VIDEO_20261029_PART{part}' and old['part']==part
          and run['id']==old['id']==receipt['id'] and run['protocol_sha256']==receipt['protocol_sha256']==proof['protocol']['sha256']
          and receipt['run_sha256']==proof['RUN']['sha256'] and receipt['process_exited'] is True
          and type(receipt['child_returncode']) is int and receipt['child_returncode']==1 and receipt['timeout'] is False
          and receipt['automatic_retry'] is False and run['status']=='FAILED_TEAL_VIDEO_PRESERVED_NO_RETRY'
          and run['error_type']=='ValueError' and run['error']=='Host/disk/wall budget exceeded'
          and len(run['phases'])==0 and run.get('raw_model_setup_seconds',0)>0,'Actual V2 failed receipt required')
        proofs.append(proof)
    return proofs


def resource_observation(mem_available,disk_free,elapsed,*,admission=False):
    snapshot={'host_available_bytes':mem_available,'disk_free_bytes':disk_free,'elapsed_seconds':elapsed,
      'host_available_min_bytes':RESOURCES['host_available_min_bytes'],'disk_free_min_bytes':RESOURCES['free_disk_min_bytes'],
      'wall_limit_seconds':RESOURCES['wall_seconds'],'admission_required_bytes':RESOURCES['admission_free_host_bytes'] if admission else None}
    if admission and mem_available<RESOURCES['admission_free_host_bytes']:reason='ADMISSION_HOST_FORECAST_BELOW_REQUIRED'
    elif mem_available<RESOURCES['host_available_min_bytes']:reason='HOST_AVAILABLE_BELOW_MINIMUM'
    elif disk_free<RESOURCES['free_disk_min_bytes']:reason='DISK_FREE_BELOW_MINIMUM'
    elif elapsed>RESOURCES['wall_seconds']:reason='WALL_SECONDS_EXCEEDED'
    else:reason=None
    return snapshot,reason


def draft(part,hf):
    scope_id(part);sources,pool=source_inputs();template=native.read(ROOT/MASTER_TEMPLATE)
    threshold=native.binding(ROOT/THRESHOLD_ARTIFACT,ROOT);a=native.read(native.bound(ROOT,threshold));validate_thresholds(a)
    # Recheck the real CPU parent now; never fabricate threshold identity.
    spec=importlib.util.spec_from_file_location('_teal_cpu_draft_parent',ROOT/'tools/e0_teal_wan_all160_cpu_local.py')
    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);folder=hf/CPU_OUTPUT
    closed=cpu.verify_closed(SimpleNamespace(hf_root=hf,protocol=ROOT/CPU_PROTOCOL,run=folder/'RUN.json',exit=folder.parent/'EXIT_RECEIPT.json'))
    native.need(a['CPU_parent']==closed and a['CPU_protocol']==native.binding(ROOT/CPU_PROTOCOL,ROOT),'Artifact not derived from actual closed CPU scope')
    parent=native.read(folder/'RUN.json')
    actual=threshold_rows([native.read(native.bound(hf,b)) for b in parent['site_reports']])
    native.need(a['threshold_rows']==actual and a['CPU_manifest']==native.read(ROOT/CPU_PROTOCOL)['manifest'],'Artifact thresholds do not match closed CPU outputs')
    keys=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay',
          'execution_private_worker_config_sha256','execution_raw_master_CPU_proof')
    plan={k:copy.deepcopy(template[k]) for k in keys};native.need(template['models']==sources[0]['models']==sources[1]['models'],'Original models differ')
    prior_failures=prior_failed_parts(hf)
    case=copy.deepcopy(pool[ORDINALS[part]]);neutral=copy.deepcopy(sources[part%2]['cases'][0])
    parent_proofs,reference=original_dense_parent(hf,sources,template,ORDINALS[part])
    paths={b['path'] for x in sources for b in x['repo_sources']}|{b['path'] for b in template['repo_sources']}|{SELF,TEST,CPU_PROTOCOL,THRESHOLD_ARTIFACT,MASTER_TEMPLATE,'tools/e0_teal_wan_all160_cpu_local.py','tests/test_e0_teal_wan_all160_cpu_local.py',base.SELF,base.TEST,'tools/raw_bf16_factor_w2_runtime.py','tools/durable_worker_job.py','tools/worker_identity.py','scripts/worker_teal_assigned_uuid_exec.sh','scripts/worker_teal_v3_group_uuid_exec.sh',V2_SOURCE,*[V2_CONFIG.format(i) for i in range(4)],*[SOURCE_CONFIG.format(i) for i in (0,1)]}
    plan.update(id=scope_id(part),part=part,status='DRAFT_NOT_REGISTERED',assets_finalized=False,review=None,numeric_path='BF16_ORIGINAL',
      design=copy.deepcopy(DESIGN),resources=copy.deepcopy(RESOURCES),CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES),limits=limits(),
      execution_monitor_binary={'path':'/usr/bin/nvidia-smi','bytes':Path('/usr/bin/nvidia-smi').stat().st_size,'sha256':native.sha('/usr/bin/nvidia-smi')},
      threshold_artifact=threshold,CPU_parent=closed,prior_v2_failures=prior_failures,original_dense_parents=parent_proofs,original_dense_reference=reference,
      case=case,neutral_case=neutral,source_bank=sources[0]['source_bank'],
      source_protocols=[native.binding(ROOT/SOURCE_CONFIG.format(i),ROOT) for i in (0,1)],output_HF_relative=outpath(part),automatic_retry=False,
      run_ids=[rid(control({'neutral_case':neutral},i),'dense') for i in ('old','new')]+[rid(case,m) for m in MODES],
      repo_sources=[native.binding(ROOT/p,ROOT) for p in sorted(paths)])
    return plan


def fingerprint(p):return native.canonical({k:v for k,v in p.items() if k not in ('status','assets_finalized','review')})

def validate(p,hf,frozen=False):
    expected=draft(p['part'],hf);expected['status']=p.get('status');expected['assets_finalized']=p.get('assets_finalized');expected['review']=p.get('review')
    native.need(p==expected,'Exact finite video scope differs')
    if frozen:native.need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True and p['review'] is None,'Not frozen awaiting registration')


def check_group_active(scopes,admission=True):
    if not admission:return
    group=[x for x in scopes if x.get('native_group_id')==GROUP_ID and x.get('execution_status')=='RUNNING_FIXED_SCOPE']
    native.need(all(type(x.get('native_group_active_slots')) is int and x['native_group_active_slots'] in (0,1) for x in group)
       and sum(x['native_group_active_slots'] for x in group)<=2,'V3 native group ACTIVE allocation exceeds two')

def authorize(p,digest,hf,admission=False):
    validate(p,hf,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']]
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(rows)==1
      and rows[0].get('protocol_sha256')==digest and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and rows[0].get('resources')==RESOURCES and rows[0].get('limits')==limits() and rows[0].get('run_ids')==p['run_ids']
      and rows[0].get('max_gpu_cards')==1 and rows[0].get('session_max_GPUs')==4,'Exact independent ACTIVE scope required')
    review=native.read(native.bound(ROOT,rows[0]['review']))
    native.need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==p['id']
      and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE'
      and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id']
      and review.get('responses_to_review') and review.get('adversarial_counterexample')
      and review.get('approval_scope_fingerprint')==fingerprint(p)
      and review.get('cpu_preflight_spec_sha256')==rows[0].get('cpu_preflight_spec_sha256')
      and review.get('gpu_launch_spec_sha256')==rows[0].get('gpu_launch_spec_sha256')
      and review.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'Independent exact recheck required')
    for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,review[k])
    scope=rows[0]
    check_group_active(active['experiment_authority']['current_user_authorized_scopes'],admission)
    for key in ('cpu_preflight_spec_sha256','gpu_launch_spec_sha256'):
        v=scope.get(key);native.need(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef'),'Missing private launch SHA')
    return scope


def verify_job(args,p,digest,scope,kind):
    native.need(args.job_spec is not None and native.sha(args.job_spec)==scope['cpu_preflight_spec_sha256' if kind=='preflight' else 'gpu_launch_spec_sha256'],'Private job SHA differs')
    job=native.read(args.job_spec);hf=args.hf_root.resolve();out=(hf/p['output_HF_relative']).resolve();part=p['part'];aff=DESIGN['CPU_affinity_by_part'][part]
    native.need(job.get('id')==p['id'] and job.get('protocol_sha256')==digest and Path(job['protocol_path']).resolve()==args.protocol.resolve()
      and Path(job['cwd']).resolve()==ROOT and Path(job['hf_root']).resolve()==hf,'Private job scope differs')
    argv=job['argv'];native.need(len(argv)>1 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Private profile absent')
    profile=Path(argv[1].split('=',1)[1]);native.need(profile.is_absolute() and native.sha(profile)==p['execution_private_worker_config_sha256'],'Private worker identity differs')
    env=['env',argv[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin']
    cpu=['taskset','-c',f'{aff[0]}-{aff[-1]}',str(ROOT/'.venv/bin/python'),'-B',str(ROOT/SELF)]
    tail=['--protocol',str(args.protocol.resolve()),'--hf-root',str(hf),'--local-root',str(args.local_root.resolve()),'--job-spec',str(args.job_spec.resolve())]
    if kind=='preflight':
        folder=out.parent.parent/'cpu_preflight'/'worker';result=folder/'results'
        expected=env+['CUDA_VISIBLE_DEVICES=']+cpu+['preflight','--out',str(result)]+tail
        native.need(args.out.resolve()==result and Path(job['output_dir']).resolve()==folder and Path(job['run_json']).resolve()==result/'RUN.json'
          and job['external_timeout_seconds']==CPU_RESOURCES['outer_seconds'],'CPU preflight job paths differ')
    else:
        native.need(len(argv)>len(env)+4 and argv[len(env)+2]==str(part),'Private GPU physical index differs')
        uuid=argv[len(env)+3];group_slot=argv[len(env)+4];native.need(group_slot in ('0','1') and group_slot==os.environ.get('FFNDS_TEAL_V3_GROUP_SLOT') and uuid.startswith('GPU-') and uuid==os.environ.get('CUDA_VISIBLE_DEVICES')
          and os.environ.get('FFNDS_PHYSICAL_GPU')==str(part),'Actual UUID assignment differs')
        expected=env+['bash',str(ROOT/'scripts/worker_teal_v3_group_uuid_exec.sh'),str(part),uuid,group_slot]+cpu+['run']+tail
        native.need(Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json'
          and job['external_timeout_seconds']==RESOURCES['outer_seconds'] and shutil.which('nvidia-smi')==p['execution_monitor_binary']['path']
          and Path(p['execution_monitor_binary']['path']).stat().st_size==p['execution_monitor_binary']['bytes']
          and native.sha(p['execution_monitor_binary']['path'])==p['execution_monitor_binary']['sha256'],'GPU monitor/supervisor differs')
    native.need(argv==expected,'Exact child executable chain differs')


def preflight(args):
    start=time.monotonic();p=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(p,digest,args.hf_root)
    verify_job(args,p,digest,scope,'preflight')
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules)
      and sorted(os.sched_getaffinity(0))==DESIGN['CPU_affinity_by_part'][p['part']] and not args.out.exists() and args.out.resolve().is_relative_to(args.hf_root.resolve()),'Fresh CPU byte scope required')
    args.out.mkdir(parents=True);report={'id':p['id'],'protocol_sha256':digest,'status':'STARTED_CPU_NO_RETRY'};native.write(args.out/'RUN.json',report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight wall')));signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for x in p['repo_sources']:native.bound(ROOT,x)
        import e0_s2_causal_worker_proof_v5 as proof
        proof.require_closed_proof(p,args.hf_root);native.model_metadata(p)
        native.verify_local_package(p,{'repo':ROOT,'hf':args.hf_root},args.local_root)
        for c in (p['case'],p['neutral_case']):native.bound(args.hf_root,c['image'])
        native.need(not {'torch','numpy'}&set(sys.modules),'Numerical preflight import differs')
        report.update(status='PASS_TEAL_VIDEO_BYTE_PREFLIGHT',source_closure_sha256=native.canonical(p['repo_sources']),
            thresholds_sha256=p['threshold_artifact']['sha256'],job_spec_sha256=scope['cpu_preflight_spec_sha256'],model_calls=0,GPU_calls=0,numerical_imports=False)
    except BaseException as e:report.update(status='FAILED_CPU_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:signal.alarm(0);report['wall_seconds']=time.monotonic()-start;native.write(args.out/'RUN.json',report)


def require_cpu(scope,p,digest,hf):
    proof=scope['cpu_preflight'];r=native.read(native.bound(hf,proof['RUN']))
    receipt=native.read(native.bound(hf,proof['exit']));vm.exited(receipt,proof['RUN'],digest,p['id'])
    native.need(receipt.get('spec_sha256')==scope['cpu_preflight_spec_sha256'] and r.get('job_spec_sha256')==scope['cpu_preflight_spec_sha256'],'CPU supervisor identity differs')
    native.need(r['status']=='PASS_TEAL_VIDEO_BYTE_PREFLIGHT' and r['id']==p['id'] and r['protocol_sha256']==digest
      and r['source_closure_sha256']==native.canonical(p['repo_sources']) and r['thresholds_sha256']==p['threshold_artifact']['sha256']
      and r['model_calls']==r['GPU_calls']==0 and r['numerical_imports'] is False and 0<=r['wall_seconds']<=CPU_RESOURCES['wall_seconds'],'Actual CPU proof missing')
    return proof


def registration(p,hf):
    from ffnds_cohort_preflight import audit
    c=p['case'];condition=c['source_id']+':'+c['prompt_sha256']
    source={'source_id':c['source_id'],'image_id':c['image']['sha256'],'prompt_id':c['prompt_sha256'],'prompt':c['prompt'],
      'image_path':str((hf/c['image']['path']).resolve()),'image_name':Path(c['image']['path']).name,'conditioning_id':condition,
      'role':'DEVELOPMENT','exposure_history':{'fitting':False,'method_selection':True,'prior_evaluation':True}}
    jobs=[{'source_id':c['source_id'],'variant':mode,'sample_index':0,'seed':42,'run_id':rid(c,mode),
      'asset_id':rid(c,mode)+':new_actual_video','initial_noise_id':'teal_ord'+str(c['native_source_ordinal'])+'_shared_seed42',
      'conditioning_id':condition,'scheduler_id':'Wan22StepDistill_shift5_1000_750_500_250',
      'frames':81,'fps':16,'width':832,'height':464} for mode in MODES]
    result={'schema':'ffnds.cohort.v1','phase':'REGISTERED_REAL_GENERATION_PLAN','mode':'CUSTOM_PAIRED','purpose':'DEVELOPMENT',
      'sources':[source],'samples':jobs,'required_variants':list(MODES),'expected_samples_per_source_per_variant':1,
      'source_bank_id':p['source_bank']['sha256'],'native_protocol_id':p['id']}
    native.need(not audit(result)['errors'],'Paired registration invalid');return result


def run(args):
    import shutil
    start=time.monotonic();hf=args.hf_root.resolve();p=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(p,digest,hf,admission=True)
    verify_job(args,p,digest,scope,'run');native.need(scope.get('execution_status')=='RUNNING_FIXED_SCOPE' and scope.get('native_group_id')==GROUP_ID and scope.get('native_group_active_slots')==1,'V3 running active group slot required');cpu=require_cpu(scope,p,digest,hf);native.need(not {'torch','numpy'}&set(sys.modules),'Fresh numerical process required')
    import e0_s2_causal_worker_proof_v5 as proof
    proof.require_closed_proof(p,hf)
    for b in p['repo_sources']:native.bound(ROOT,b)
    for c in (p['case'],p['neutral_case']):native.bound(hf,c['image'])
    native.model_metadata(p);local,package_rows=native.verify_local_package(p,{'repo':ROOT,'hf':hf},args.local_root)
    native.need({k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in p['runtime_versions']}==p['runtime_versions']
      and sorted(os.sched_getaffinity(0))==DESIGN['CPU_affinity_by_part'][p['part']] and not any(os.environ.get(k)=='1' for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),'Runtime version/precision differs')
    thresholds=validate_thresholds(native.read(native.bound(ROOT,p['threshold_artifact'])))
    out=(hf/p['output_HF_relative']).resolve();native.need(out.is_relative_to(hf) and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=p['id'],limits(),tuple(p['run_ids']),RESOURCES
    vm.MODES,vm.run_id,vm.RESOURCES,vm.transfer_geometry=MODES,rid,RESOURCES,base.transfer_geometry
    report={'id':p['id'],'protocol_sha256':digest,'part':p['part'],'status':'STARTED_TEAL_VIDEO_NO_RETRY','numeric_path':'BF16_ORIGINAL',
      'CPU_preflight':cpu,'prior_v2_failures':p['prior_v2_failures'],'job_spec_sha256':scope['gpu_launch_spec_sha256'],'design':DESIGN,'phases':[],'automatic_retry':False,'quality_or_speed_acceptance':False,'official_quality':'NOT_RUN'}
    monitor=None;counter=io.WrittenBytes(out,RESOURCES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
    for name in p['run_ids']:counter.hold(str(counter.path(out/name/'output.mp4')),32<<20)
    journal=counter.open_journal(out/'OPERATIONS.jsonl');persist=lambda:counter.write_json(out/'RUN.json',report)
    ledger=vm.Ledger(report.setdefault('operation_ledger',{}),lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
    attempt_peak={'bytes':0}
    def check():
        if monitor:monitor.check()
        counter.check()
        outside=sum(f.stat().st_size for f in out.parent.parent.rglob('*') if f.is_file() and not f.resolve().is_relative_to(out))
        accounted=counter.summary()['peak_tracked_bytes_plus_reserved_and_atomic_temp']+outside+(1<<20)
        attempt_peak['bytes']=max(attempt_peak['bytes'],accounted)
        native.need(accounted<=RESOURCES['new_HF_bytes'],'Whole attempt output budget exceeded')
        snapshot,reason=resource_observation(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
          shutil.disk_usage(out).free,time.monotonic()-start)
        report['last_resource_observation']=snapshot
        if reason:
            report['resource_failure']={'reason':reason,**snapshot};persist();raise ValueError(reason)
        if 'torch' in sys.modules:native.need(sys.modules['torch'].cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU budget exceeded')
    def deadline(*_):
        try:
            snap,_=resource_observation(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
              shutil.disk_usage(out).free,time.monotonic()-start)
            report['resource_failure']={'reason':'WALL_SECONDS_EXCEEDED',**snap}
        except Exception as exc:report['resource_failure']={'reason':'WALL_SECONDS_EXCEEDED','snapshot_error':str(exc)}
        raise TimeoutError('WALL_SECONDS_EXCEEDED')
    persist();signal.signal(signal.SIGALRM,deadline);signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,package_rows),native.NoHiF4();file_guard=io.single_file_bound();file_guard.__enter__()
    try:
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        snapshot,reason=resource_observation(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
          shutil.disk_usage(out).free,time.monotonic()-start,admission=True)
        report['admission_observation']=snapshot;persist()
        if reason:report['resource_failure']={'reason':reason,**snapshot};persist();raise ValueError(reason)
        reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'Assigned A100 GPU required')
        report.update(GPU_name=torch.cuda.get_device_name(0),GPU_compute_capability=list(torch.cuda.get_device_capability(0)))
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True;check()
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();copies=[(ROOT/b['path'],snapshot/b['path'],b) for b in p['repo_sources']]
        copies.append((args.protocol,snapshot/'protocol.json',native.binding(args.protocol,ROOT)));counter.hold('frozen_metadata',sum(b['bytes'] for _,_,b in copies))
        for src,dst,b in copies:dst.parent.mkdir(parents=True,exist_ok=True);counter.copy_bound(src,dst,b)
        native.need(counter.holds.pop('frozen_metadata')==0,'Source snapshot mismatch')
        manifest=registration(p,hf);counter.write_json(snapshot/'registered_video_plan.json',manifest)
        with native.OriginalGuard(torch,p,ledger) as guard:
            at=time.perf_counter();pipe=native.build_pipeline(p);torch.cuda.synchronize()
            report.update(raw_model_setup_seconds=time.perf_counter()-at,original_loader_paths=guard.loaded)
            runtime_args=dict(factor_forward=vm.core.factor_forward,meter=vm.FactorMeter(ledger),synchronize=torch.cuda.synchronize,
                              record_stream=lambda t:t.record_stream(torch.cuda.current_stream()))
            def phase(runtime,case,mode):
                check();authorize(p,digest,hf);name=rid(case,mode)
                if isinstance(runtime,TealDenseRuntime):runtime.set_mask_mode(mode,name)
                else:runtime.set_mode('dense')
                value=vm.native_phase(torch,pipe,guard,runtime,p,case,mode,out,hf,digest,report,persist,check,counter,{})
                native.need(value['original_W2_H2D_bytes']==162*vm.ORIGINAL_W2_BYTES,'Full original W2 prefetch differs')
                if isinstance(runtime,TealDenseRuntime):
                    events=runtime.events
                    validate_action_coverage(mode,events)
                    counter.write_json(out/name/'MASK_ACTIONS.json',{'mode':mode,'events':events,'threshold_artifact':p['threshold_artifact']})
                    value['mask_actions']=counter.binding(out/name/'MASK_ACTIONS.json',hf)
                counter.write_json(out/name/'PHASE.json',value);persist();check();return value
            with FactorW2Runtime(pipe,guard,{},**runtime_args) as old_runtime:
                old=phase(old_runtime,control(p,'old'),'dense')
            with TealDenseRuntime(pipe,guard,{},thresholds=thresholds,ledger=ledger,**runtime_args) as runtime:
                new=phase(runtime,control(p,'new'),'dense');base.validate_neutral(old,new,ledger)
                report['same_hardware_old_new_all10_and_video_neutral']=True;persist()
                dense=None
                for mode in MODES:
                    phase_result=phase(runtime,p['case'],mode)
                    if mode=='dense':dense=phase_result
                    else:
                        native.need(dense is not None,'Actual same-source Dense required');legacy.match_noise(phase_result,dense,ledger)
                    index=MODES.index(mode);folder=out/f'job{index:06d}';folder.mkdir()
                    registered=legacy.registration_record(manifest,index,phase_result,hf)
                    ledger.call(phase_result['run_id'],{'source_registration_writes':1},counter.write_json,folder/'RUN.json',registered)
                    counter.write_json(out/phase_result['run_id']/'PHASE.json',phase_result);persist()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified=collect(manifest,out);counter.write_json(out/'VERIFIED_GENERATIONS.json',verified)
            native.need(len(report['phases'])==5 and not counter.holds,'Incomplete five-video shard')
            report.update(status='COMPLETE_TEAL_VIDEO_V3_QUALITY_DIAGNOSTIC',actual_full160_videos=5,actual_neutral_videos=2,
              actual_study_videos=3,actual_states=50,actual_mask_sites=320,verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json',hf),
              checked_transformers_imports=finder.origins);check()
    except BaseException as e:report.update(status='FAILED_TEAL_VIDEO_V3_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:monitor.close();report['assigned_GPU_resources']=monitor.report();native.need(not report['assigned_GPU_resources']['violations'],'Final GPU monitor violation')
            except BaseException as e:monitor_error=e;report.update(status='FAILED_TEAL_VIDEO_V3_PRESERVED_NO_RETRY',resource_error=str(e))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        for name in p['run_ids']:
            path=counter.path(out/name/'output.mp4')
            if path not in counter.files and path.exists():
                try:counter.observe_external(path,hold=str(path))
                except BaseException as e:counter.accounting_errors.append(str(e))
        report.update(wall_seconds=time.monotonic()-start,whole_attempt_peak_accounted_plus_receipt_reserve=attempt_peak['bytes'],
          output_accounting_checkpoint_before_final_RUN=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',hf));persist();journal.close();file_guard.__exit__(None,None,None)
        if monitor_error is not None:raise monitor_error


def verify_closed(args):
    hf=args.hf_root.resolve();p=native.read(args.protocol);validate(p,hf,True);digest=native.sha(args.protocol);out=hf/p['output_HF_relative']
    native.need(args.run.resolve()==(out/'RUN.json').resolve() and args.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'Wrong actual exit/run')
    binding=native.binding(args.run,hf);r=native.read(args.run);receipt=native.read(args.exit);vm.exited(receipt,binding,digest,p['id'])
    native.need(receipt.get('spec_sha256')==r['job_spec_sha256'],'Actual GPU supervisor receipt differs')
    native.need(r['id']==p['id'] and r['protocol_sha256']==digest and r['status']=='COMPLETE_TEAL_VIDEO_V3_QUALITY_DIAGNOSTIC'
      and r['prior_v2_failures']==p['prior_v2_failures'] and r['same_hardware_old_new_all10_and_video_neutral'] is True and r['actual_full160_videos']==5
      and r['actual_study_videos']==3 and r['actual_states']==50 and r['actual_mask_sites']==320
      and r['GPU_compute_capability']==[8,0] and 'A100' in r['GPU_name'] and r['assigned_GPU_resources']['sampled_isolation_pass'] is True
      and not r['assigned_GPU_resources']['violations'] and 0<=r['wall_seconds']<=RESOURCES['wall_seconds']
      and all(r['operation_ledger'][side].get(k,0)==v for side in ('predebited','observed') for k,v in limits().items())
      and not r['operation_ledger']['forbidden_attempts'] and [x['run_id'] for x in r['phases']]==p['run_ids']
      and r['output_accounting_checkpoint_before_final_RUN']['peak_tracked_bytes_plus_reserved_and_atomic_temp']<=RESOURCES['new_HF_bytes']
      and r['output_accounting_checkpoint_before_final_RUN']['unconsumed_reserved_bytes']==0
      and r['whole_attempt_peak_accounted_plus_receipt_reserve']<=RESOURCES['new_HF_bytes']
      and sum(f.stat().st_size for f in out.parent.parent.rglob('*') if f.is_file())<=RESOURCES['new_HF_bytes'],'Closed execution differs')
    class ReadOnly:
        def call(self,_rid,_cost,fn):return fn()
    base.validate_neutral(r['phases'][0],r['phases'][1],ReadOnly());thresholds=validate_thresholds(native.read(native.bound(ROOT,p['threshold_artifact'])))
    for ix,phase in enumerate(r['phases']):
        expected_mode=('dense','dense',*MODES)[ix]
        expected_case=p['neutral_case'] if ix<2 else p['case']
        native.need(phase['mode']==expected_mode and phase['native_source_ordinal']==expected_case['native_source_ordinal']
          and phase['source_id']==expected_case['source_id'] and phase['image_sha256']==expected_case['image']['sha256']
          and phase['prompt_sha256']==expected_case['prompt_sha256'] and phase['seed']==42
          and phase['status']=='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
          and phase['all160_FFN_covered'] is True and phase['completed_native_flow_steps']==4 and phase['scheduler_generator_reset'] is True
          and ('mask_actions' in phase)==(ix>=1),'Phase mode/source/action identity differs')
        gen=native.read(native.bound(hf,phase['generation_receipt']))
        native.need(gen['run_id']==phase['run_id'] and gen['protocol_sha256']==digest and gen['states']==phase['states']
          and gen['video']==phase['video'] and gen['all160_FFN_covered'] is True and len(gen['states'])==10
          and gen['mode']==expected_mode and gen['source_id']==expected_case['source_id'] and gen['seed']==42
          and gen['video']['frames']==81 and gen['video']['size']==[832,464] and gen['video']['fps']==16,'Generation identity differs')
        for item in phase['states'].values():native.bound(hf,item)
        video=phase['video'];vpath=Path(video['path']).resolve();native.need(vpath.is_relative_to(hf),'Video escaped HF')
        native.bound(hf,{'path':str(vpath.relative_to(hf)),'bytes':video['bytes'],'sha256':video['sha256']})
        if 'mask_actions' in phase:
            actions=native.read(native.bound(hf,phase['mask_actions']));events=actions['events'];mode=phase['mode']
            validate_action_coverage(mode,events)
            native.need(actions['threshold_artifact']==p['threshold_artifact'] and actions['mode']==mode
              and (not events if mode=='dense' else len(events)==160 and {(x['step'],x['block']) for x in events}==set(thresholds)
                and all(x['expert']==(0 if x['step']<2 else 1) and x['threshold']==thresholds[x['step'],x['block']][0 if mode=='threshold25' else 1]
                  and x['total']==31668*13824 and 0<=x['preexisting_zero']<=x['selected']<=x['total']
                  and x['selected_nonzero']==x['selected']-x['preexisting_zero'] and 0<=x['threshold_ties']<=x['selected'] for x in events)),'Mask action identity differs')
    dense=r['phases'][2]
    for phase in r['phases'][3:]:
        for k in ('raw_noise_FP32','initial_noise'):native.need(vm.same_state(phase['states'][k],dense['states'][k]),'Candidate noise differs')
    verified=native.read(native.bound(hf,r['verified_generations']))
    native.need(verified['phase']=='COMPLETE_REGISTERED_GENERATION_VERIFIED' and len(verified['samples'])==3
      and {x['run_id'] for x in verified['samples']}==set(p['run_ids'][2:]),'Registered generations differ')
    return {'id':p['id'],'status':'PASS_CLOSED_TEAL_VIDEO_V3_QUALITY_DIAGNOSTIC','RUN':binding,'exit':native.binding(args.exit,hf)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('export-thresholds','draft','inspect','preflight','run','verify-closed'))
    ap.add_argument('--part',type=int);ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path)
    ap.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));ap.add_argument('--local-root',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
    if a.action=='export-thresholds':
        native.need(a.out is not None and a.out.resolve()==ROOT/THRESHOLD_ARTIFACT,'Fixed threshold artifact path required');print(json.dumps(export_thresholds(a.hf_root,a.out),indent=2))
    elif a.action=='draft':native.need(a.out is not None and not a.out.exists(),'Fresh protocol required');native.write(a.out,draft(a.part,a.hf_root))
    elif a.action=='inspect':validate(native.read(a.protocol),a.hf_root);print('STATIC_ONLY_NO_MODEL')
    elif a.action=='preflight':preflight(a)
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))
if __name__=='__main__':main()
