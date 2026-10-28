"""Official eight-axis evaluation of two existing original-BF16 Dense clips.

Uses the successful protected v4 environment/helpers and its exited imports
proof. This independent two-clip loop never calls the frozen six-clip run.
Draft, inspect and byte proof import only stdlib and bound helper source.
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

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_ffnds_successful_quality_v4_for_raw',ROOT/'tools/e0_compact6016_consumer_quality_v4.py')
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
base=v4.base
ID='E0_ORIGINAL_BF16_CONSUMER00_12_OFFICIAL_QUALITY_20261028'
PARENT='config/e0_original_bf16_videos_20261028_v1.json'
PARENT_SHA='2f5352e91a877e85c0c0e88d8c216f74f5124693161ad412b7226aff73c69122'
PARENT_RUN_SHA='59e99d08d206aac4ffabb11453e4b6b27449d046f07d14027d5e65e9644064c9'
PARENT_EXIT_SHA='c718ec457194d2e48deed47c9ccb18502ddb44a37355a6f4874867bd4c493b04'
METRIC_PROTOCOL='config/e0_compact6016_consumer_quality_20261028_v4.json'
METRIC_SHA='23f56ab7ae27230e2eb5f4cb813c9eef0bc1ec021bb1d753345b128152e86b50'
IMPORT_RUN_SHA='ab4db392b93ba899d984700840f051241acd1bb9ba53bdf012b79613fdebda93'
IMPORT_EXIT_SHA='e33ecde28249663b8722f4a170791ef4a33a0b62ec6605b9de7ba24331050e03'
DIMENSIONS=base.DIMENSIONS
COMMON=base.COMMON
COSTS={'imaging_quality':{'MUSIQ':162},'aesthetic_quality':{'CLIP_L14':6,'aesthetic_head':6},
       'motion_smoothness':{'AMT':80},'dynamic_degree':{'RAFT':80,'RAFT_iterations':1600},
       'subject_consistency':{'DINO':162},'background_consistency':{'CLIP_B32':2},
       'i2v_subject':{'DINO':82},'i2v_background':{'DreamSim_embed':82}}
LIMITS={'official_dimension_calls':8,'new_video_dimension_results':14,'reused_core_reports':2,
        'RAFT_forwards':80,'RAFT_iterations':1600,'video_generations':0,'training_updates':0,
        'decompositions':0,'CPU_metric_recomputations':0}
RESOURCES={'GPU':1,'session_max_GPUs':2,'repository_GPU_policy':'DYNAMIC_MAX_ACTUALLY_CALLABLE',
           'CPU_threads':2,'wall_seconds':600,'outer_seconds':900,'new_HF_bytes':1<<29,
           'GPU_reserved_bytes':8<<30,'free_disk_bytes':10<<30,'automatic_retry':False}
CPU_STATUS='PASS_TWO_RAW_BF16_QUALITY_BYTES_REUSED_IMPORTS_NO_NUMERIC_IMPORTS'
REQUIRED={'tools/e0_original_bf16_consumer_quality.py','tests/test_e0_original_bf16_consumer_quality.py'}


def validate(plan,frozen=False):
    base.require(plan.get('id')==ID and plan.get('numeric_path')=='BF16_ORIGINAL'
      and plan.get('dimensions')==list(DIMENSIONS) and plan.get('limits')==LIMITS
      and plan.get('costs')==COSTS and plan.get('resources')==RESOURCES
      and plan.get('automatic_retry') is False,'Fixed raw quality scope differs')
    base.require(len(plan.get('videos',[]))==2 and [(v['source'],v['mode']) for v in plan['videos']]==[(0,'dense'),(12,'dense')]
      and len(plan['images'])==2 and {x['source'] for x in plan['images']}=={0,12}
      and len({x['binding']['sha256'] for x in plan['images']})==2,'Two distinct raw Dense sources required')
    base.require(plan['parent']['protocol']['path']==PARENT and plan['parent']['protocol']['sha256']==PARENT_SHA
      and plan['parent']['RUN']['sha256']==PARENT_RUN_SHA and plan['parent']['exit']['sha256']==PARENT_EXIT_SHA,'Raw parent identity differs')
    proof=plan['reused_import_proof']
    base.require(proof['protocol']['path']==METRIC_PROTOCOL and proof['protocol']['sha256']==METRIC_SHA
      and proof['RUN']['sha256']==IMPORT_RUN_SHA and proof['exit']['sha256']==IMPORT_EXIT_SHA,'Successful v4 import proof required')
    base.require(plan.get('directory_fd_audit_policy')==v4.v3.POLICY and plan.get('activation_bootstrap_policy')==v4.POLICY,
      'Original successful v4 audit/bootstrap policy differs')
    paths=[b['path'] for b in plan['repo_sources']]
    base.require(len(paths)==len(set(paths)) and REQUIRED<=set(paths),'Immutable quality source closure incomplete')
    base.require(not any(k.startswith('failed_') or k=='legacy' for k in plan),'Unrelated failed/legacy scope was copied')
    out=Path(plan['output_HF_relative'])
    base.require(not out.is_absolute() and '..' not in out.parts and out.parts[:1]==('ffnds-derived',),'Output escaped HF')
    if frozen:base.require(plan.get('assets_finalized') is True and plan.get('binding_status')=='BOUND_READY_FOR_REGISTRATION','Raw quality draft NOT_BOUND')


def verify_import_proof(plan,hf):
    """Verify the successful old closure, without following failed predecessors."""
    proof=plan['reused_import_proof'];old=base.read_bound(ROOT,proof['protocol'])
    report=base.read_bound(hf,proof['RUN']);exit=base.read_bound(hf,proof['exit'])
    base.exited(exit,proof['RUN'],METRIC_SHA,v4.ID)
    base.require(report.get('id')==v4.ID and report.get('protocol_sha256')==METRIC_SHA
      and report.get('status')==base.IMPORT_STATUS and report.get('sources_sha256')==base.state_sha(old['repo_sources'])
      and report.get('import_environment_sha256')==base.state_sha(plan['import_environment'])
      and all(plan[k]==old[k] for k in ('environment','import_environment','hf_bindings','directory_fd_audit_policy','activation_bootstrap_policy','directory_fd_filelock_probe_source')),
      'Successful import proof/environment identity differs')
    base.require(report.get('all_CUDA_devices_hidden') is True and report.get('CUDA_initialized') is False
      and report.get('denied_attempts')==[] and report.get('predebited_official_module_imports')==report.get('completed_official_module_imports')==8
      and report.get('stateless_activation_wrapper_constructions')==8 and report.get('stateless_pool_wrapper_constructions')==1
      and all(report.get(k)==0 for k in ('GPU_calls','checkpoint_loads','model_constructions','parameter_creations','buffer_creations','model_forwards','official_scores')),
      'Actual imports-only zero-model/CUDA evidence differs')
    base.require(report.get('limits')==base.IMPORT_LIMITS and report.get('resources')==base.IMPORT_RESOURCES
      and report.get('isolation',{}).get('flags')=={'isolated':1,'no_site':1,'ignore_environment':1}
      and report['isolation'].get('project_site_fallback') is False,'Actual protected import isolation differs')
    expected={name:{'origin':str((ROOT if b['root']=='REPO' else Path(hf))/b['path']),
        'sha256':b['sha256'],'distribution':b['distribution'],'version':b['version']}
        for name,b in plan['import_environment']['modules'].items()}
    base.require(report.get('module_origins')==expected and [f['dimension'] for f in report.get('functions',[])]==list(DIMENSIONS),
      'Actual module origin/version or eight-function coverage differs')
    for b in old['repo_sources']:base.bound(ROOT,b)
    for b in plan['import_environment']['modules'].values():base.bound(ROOT if b['root']=='REPO' else hf,b)
    for f in report['functions']:
        package='vbench2_beta_i2v' if f['dimension'].startswith('i2v_') else 'vbench'
        path=Path(hf)/plan['environment']['source_root']/package/(f['dimension']+'.py')
        base.require(f['module_path']==str(path.resolve()) and f['module_sha256']==base.sha(path),'Original metric function SHA differs')
    v8=base.read(ROOT/plan['environment']['v8_protocol']);activation=v8['stateless_activation_allowlist']
    origin=(Path(hf)/plan['environment']['site']/activation['site_relative']).resolve()
    expected_activation=[dict(x,module='transformers.activations',source_sha256=activation['source_sha256']) for x in activation['instances']]
    base.require(report.get('stateless_activation_wrappers')==expected_activation
      and report.get('activation_bootstrap')=={'module':'transformers.activations','origin':str(origin),'sha256':activation['source_sha256'],'before':0,'after':8},'Successful ordered activation bootstrap differs')
    normalized=report.get('dir_fd_normalization',{})
    base.require(normalized.get('policy')==v4.v3.POLICY and type(normalized.get('normalized_events')) is int
      and normalized['normalized_events']==len(normalized.get('events',[]))>0,'Actual successful directory-FD cleanup proof absent')
    return proof


def verify_environment(plan,hf):
    helper,v8=v4.v3._verify_environment(plan,hf)
    v4.v3.verify_probe_source(plan,hf)
    return v4.EnvironmentProxy(v4.v3.GuardedEnvironment(helper),v8,Path(hf)/plan['environment']['site']),v8


def verify_generation(plan,hf):
    parent=base.read_bound(ROOT,plan['parent']['protocol']);run=base.read_bound(hf,plan['parent']['RUN'])
    base.exited(base.read_bound(hf,plan['parent']['exit']),plan['parent']['RUN'],PARENT_SHA,parent['id'])
    base.require(run.get('id')==parent['id']=='E0_ORIGINAL_BF16_CONSUMER00_12_VIDEOS_20261028'
      and run.get('status')=='COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING'
      and run.get('protocol_sha256')==PARENT_SHA and run.get('numeric_path')=='BF16_ORIGINAL'
      and not run.get('assigned_GPU_resources',{}).get('violations')
      and not run['operation_ledger']['forbidden_attempts']
      and all(run['operation_ledger'][phase].get(k,0)==n for phase in ('predebited','observed') for k,n in parent['limits'].items()),'Original BF16 producer not complete')
    base.require(len(run['phases'])==2 and {p['source_index'] for p in run['phases']}=={0,12},'Two actual raw phases required')
    paths=[]
    for v in plan['videos']:
        phase=next(p for p in run['phases'] if p['source_index']==v['source'])
        case=next(c for c in parent['cases'] if c['index']==v['source'])
        generation=base.read_bound(hf,v['generation']);registered=base.read_bound(hf,v['registered'])
        base.require(phase['generation_receipt']==v['generation'] and phase['core_metrics']==v['core']
          and phase['status']==generation['status']=='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
          and phase['completed_native_flow_steps']==generation['completed_native_flow_steps']==4
          and all(v[k]==case[k]==phase[k]==generation[k] for k in ('source_id','image_sha256','prompt_sha256','prompt') if k!='prompt')
          and v['prompt']==case['prompt'] and hashlib.sha256(v['prompt'].encode()).hexdigest()==v['prompt_sha256']
          and generation['numeric_path']=='BF16_ORIGINAL' and generation['protocol_sha256']==PARENT_SHA
          and generation['seed']==v['seed']==42,'Raw generation/source identity differs')
        base.require(registered['status']=='COMPLETE_NATIVE_GENERATION_ONLY' and registered['numeric_path']=='BF16_ORIGINAL'
          and registered['job']['run_id']==generation['run_id']==phase['run_id']
          and registered['immutable_generation_receipt']==v['generation'] and registered['native_ffn_sites']==160
          and registered['scheduler_generator_reset'] is True,'Raw registered generation differs')
        path=base.bound(hf,v['video']);paths.append(path)
        base.require(v['video']==base.clip_binding(generation['video'],hf)==base.clip_binding(phase['video'],hf)
          and generation['video']['frames']==registered['decoded_frames']==81
          and generation['video']['size']==registered['decoded_size']==[832,464]
          and generation['video']['fps']==registered['decoded_fps']==16
          and Path(registered['video_path']).resolve()==path,'Raw video bytes/geometry differ')
        core=base.read_bound(hf,v['core'])
        from ffnds_video_metrics import validate as validate_core
        base.require(not validate_core(core) and core['candidate']['sha256']==v['video']['sha256']
          and core['identity']['source_id']==case['image_sha256'] and core['identity']['seed']==42
          and core['identity']['expected_frames']==81 and core['identity']['input_prompt_digest']==case['prompt_sha256']
          and core['identity']['protocol_sha256']==PARENT_SHA and core['identity']['run_sha256']==v['generation']['sha256'],
          'Raw all81frame core/source binding differs')
        image=next(i for i in plan['images'] if i['source']==v['source'])
        base.require(case['image'].get('root')=='hf' and image['binding']=={k:case['image'][k] for k in ('path','bytes','sha256')} and image['binding']['sha256']==v['image_sha256']
          and image['name']==v['image_name'],'Original raw input image differs')
        base.bound(hf,image['binding'])
    base.require(len(set(paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==2,'Aliased raw videos rejected')
    return run


def verify_assets(plan,hf):
    for b in plan['repo_sources']:base.bound(ROOT,b)
    verify_generation(plan,hf);verify_import_proof(plan,hf)
    for b in plan['hf_bindings']:base.bound(hf,b)


def authorize(plan,digest):
    validate(plan,True);active=base.read(ROOT/'research/ACTIVE_RESEARCH.json')
    user=active.get('user_continuation_authorization_20261027',{});policy=active.get('resource_scheduling_policy_20261028',{})
    base.require(user.get('status')=='ACTIVE_CURRENT_USER_DIRECTED_SESSION' and user.get('new_scope_registration_allowed') is True
      and policy.get('mode')=='DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
      and active.get('current_session_resource_constraint_20261028',{}).get('max_simultaneously_assigned_idle_GPUs')==2,'Current session authority differs')
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    base.require(len(scopes)==1 and scopes[0].get('protocol_sha256')==digest and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and scopes[0].get('limits')==LIMITS and scopes[0].get('resources')==RESOURCES and scopes[0].get('max_gpu_cards')==1,'Unique raw quality registration required')
    return scopes[0]


def cpu_report(plan,digest):
    return {'id':ID,'protocol_sha256':digest,'status':CPU_STATUS,'sources_sha256':base.state_sha(plan['repo_sources']),
      'import_environment_sha256':base.state_sha(plan['import_environment']),'reused_import_RUN_sha256':IMPORT_RUN_SHA,
      'parent_RUN_sha256':PARENT_RUN_SHA,'new_scored_videos':2,'reused_core_reports':2,'all_CUDA_devices_hidden':True,
      'NumPy_or_Torch_imported':False,'model_calls':0,'GPU_calls':0}


def preflight(protocol,hf):
    base.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'numpy' not in sys.modules,'Fresh CUDA-hidden byte-only worker required')
    base.precision_environment();plan=base.read(protocol);validate(plan,True)
    verify_assets(plan,hf);verify_environment(plan,hf)
    base.require('torch' not in sys.modules and 'numpy' not in sys.modules,'Numerical imports in byte-only proof')
    return cpu_report(plan,base.sha(protocol))


def require_cpu(scope,plan,digest,hf):
    proof=scope.get('cpu_preflight');base.require(isinstance(proof,dict) and {'RUN','launch_exit'}<=set(proof),'Independent byte CPU proof NOT_BOUND')
    report=base.read_bound(hf,proof['RUN'])
    base.require(all(report.get(k)==v for k,v in cpu_report(plan,digest).items()),'Raw quality CPU proof differs')
    base.exited(base.read_bound(hf,proof['launch_exit']),proof['RUN'],digest,ID)
    return proof


def summarize(plan,hf,dimensions,observations):
    """Source is the observational unit; pair windows describe the same clip."""
    scores={str(v['source']):{} for v in plan['videos']}
    for entry in dimensions:
        dimension=entry['dimension']
        selected=base.rows_for_dimension(plan['videos'],dimension)
        lookup={str(base.bound(hf,v['video'])):str(v['source']) for v in selected}
        rows=base.check_result(entry['result'],list(lookup),dimension)
        for row in rows:scores[lookup[row['video_path']]][dimension]=row['video_results']
    motion={}
    for v in plan['videos']:
        path=str(base.bound(hf,v['video']))
        row=next(x for x in observations if x['video_path']==path)
        base.validate_raft(row)
        motion[str(v['source'])]=base.raft_summary(row['scores'])
    return {'per_source':scores,'continuous_RAFT_per_source':motion,
      'source_equal_common6_means':{d:statistics.fmean(scores[str(v['source'])][d] for v in plan['videos']) for d in COMMON},
      'i2v_subject_source0':scores['0']['i2v_subject'],'i2v_background_source12':scores['12']['i2v_background'],
      'independent_image_clusters':2,'samples_per_source':1,'shared_seed':42,
      'source_exposure':'Previously exposed consumer20 development indices 0/12; no source selection by this score',
      'estimand':'These two original-BF16 clips; continuous motion and binary dynamic degree separately',
      'format_comparison':'No paired same-recipe intervention here; numerical-path differences do not isolate quantization causation',
      'official_five_independent_samples':False,'quality_accepted':False}


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
    report={'id':ID,'status':'STARTED_TWO_ORIGINAL_BF16_CLIPS_QUALITY','protocol_sha256':digest,
      'numeric_path':'BF16_ORIGINAL','prior_independent_CPU_preflight':cpu,'reused_successful_imports_CPU_preflight':proof,
      'producer':plan['parent'],'core_reports_reused':[v['core'] for v in plan['videos']],
      'declared_outer_model_work_budgets':COSTS,'predebited_outer_model_work':{},
      'model_work_accounting':'Original frame/batch budgets; RAFT forwards and iterations observed exactly; other internal work UNKNOWN',
      'opaque_internal_model_forwards':'UNKNOWN','opaque_model_constructions':'UNKNOWN','opaque_checkpoint_loads':'UNKNOWN',
      'completed_dimensions':[],'RAFT_observations':[],'official_module_functions':[],
      'new_scored_videos':2,'independent_image_clusters':2,'official_five_independent_samples':False,
      'temporal_flickering':'NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED','camera_motion':'NOT_RUN_NOT_IN_SCOPE',
      'human_fullplay_review':'NOT_RUN','quality_accepted':False,'native_dir_fd_audit':{'denied_attempts':[]}}
    ledger=base.Ledger(report,limits=LIMITS,persist=lambda:base.write(out/'RUN.json',report))
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        base.require(time.monotonic()-started<=RESOURCES['wall_seconds'],'Quality wall budget exhausted')
        base.require(shutil.disk_usage(out).free>=RESOURCES['free_disk_bytes'],'Quality free disk reserve exceeded')
        base.require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Quality new-output byte budget exceeded')
    def timeout(*_):raise TimeoutError('Fixed600second raw quality attempt exhausted; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    try:
        verify_assets(plan,hf);helper,v8=verify_environment(plan,hf);helper.two_cpus();check()
        with ledger('reused_core_reports',2):report['core_reuse_status']='BOUND_ALL81FRAME_REPORTS_NO_RECOMPUTATION'
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
            check();selected=base.rows_for_dimension(plan['videos'],dimension)
            expected_paths=[str(base.bound(hf,x['video'])) for x in selected]
            base.require(len(selected)==(2 if dimension in COMMON else 1),'Two-clip dimension dispatch differs')
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
                        base.require(len(observations)==2 and {x['video_path'] for x in observations}==set(expected_paths),'Two raw RAFT observations required')
                        for row in observations:base.validate_raft(row)
                        base.require(report['completed']['RAFT_forwards']==80 and report['completed']['RAFT_iterations']==1600,'Actual80/1600 RAFT work differs')
                        entry['RAFT_observations']=copy.deepcopy(observations)
                    report['completed_dimensions'].append(entry)
                gc.collect();torch.cuda.empty_cache()
                base.require(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'Quality GPU reservation budget exceeded')
            finally:
                if dimension=='i2v_background':module.dreamsim=original_dream
        ledger.finish();base.require(report['predebited_outer_model_work']==COSTS,'Incomplete eight-axis outer budgets')
        base.require(not report['native_dir_fd_audit']['denied_attempts'],'Native directory-FD audit violation')
        report['source_summary']=summarize(plan,hf,report['completed_dimensions'],report['RAFT_observations'])
        report.update(active_dimension=None,status='COMPLETE_TWO_ORIGINAL_BF16_CLIPS_EIGHT_DIMENSIONS_QUALITY_NOT_ACCEPTED',peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
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


def make_draft(hf):
    """Small-metadata extraction. No metric weights or numerical imports."""
    hf=Path(hf).resolve();old=base.read(ROOT/METRIC_PROTOCOL);parent=base.read(ROOT/PARENT)
    rawout=hf/parent['output_HF_relative'];rawrun=base.read(rawout/'RUN.json')
    proof=hf/'ffnds-derived/e0_compact6016_quality_isolated_cpu_20261028/v4/imports/worker'
    videos=[]
    for i,case in enumerate(parent['cases']):
        phase=next(p for p in rawrun['phases'] if p['source_index']==case['index'])
        image=next(x for x in old['images'] if x['source']==case['index'])
        videos.append({'source':case['index'],'mode':'dense','seed':42,
          **{k:case[k] for k in ('source_id','image_sha256','prompt_sha256','prompt')},'image_name':image['name'],
          'video':base.clip_binding(phase['video'],hf),'generation':phase['generation_receipt'],
          'registered':base.binding(rawout/f'job{i:06d}/RUN.json',hf),'core':phase['core_metrics']})
    plan={k:copy.deepcopy(old[k]) for k in ('environment','import_environment','hf_bindings','directory_fd_audit_policy','activation_bootstrap_policy','directory_fd_filelock_probe_source')}
    plan.update(id=ID,numeric_path='BF16_ORIGINAL',status='DRAFT_BOUND_TWO_RAW_VIDEOS_QUALITY',
      assets_finalized=False,binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING',automatic_retry=False,
      dimensions=list(DIMENSIONS),limits=LIMITS,costs=COSTS,resources=RESOURCES,videos=videos,
      images=[copy.deepcopy(x) for x in old['images'] if x['source'] in (0,12)],
      parent={'protocol':base.binding(ROOT/PARENT,ROOT),'RUN':base.binding(rawout/'RUN.json',hf),'exit':base.binding(rawout.parent/'EXIT_RECEIPT.json',hf)},
      reused_import_proof={'protocol':base.binding(ROOT/METRIC_PROTOCOL,ROOT),'RUN':base.binding(proof/'imports/RUN.json',hf),'exit':base.binding(proof/'EXIT_RECEIPT.json',hf)},
      output_HF_relative='ffnds-derived/e0_original_bf16_consumer00_12_quality_20261028/attempt1/results',
      estimation={'independent_image_clusters':2,'shared_seed':42,'samples_per_source':1,'exposure':'consumer20 exposed development indices0/12',
        'new_video_results':14,'official_five_independent_samples':False,'human_fullplay_review':'NOT_RUN','quality_accepted':False,
        'cross_format_causal_effect':'NOT_IDENTIFIED; baseline own quality is measured first'},
      pending_freeze='Independent software review, frozen protocol and new separately exited stdlib byte CPU proof; reuse exact successful v4 imports proof only')
    plan['repo_sources']=[base.binding(ROOT/p,ROOT) for p in sorted({b['path'] for b in old['repo_sources']}|REQUIRED)]
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('draft','inspect','preflight','run'))
    parser.add_argument('--protocol',type=Path);parser.add_argument('--hf-root',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='draft':
        base.require(args.hf_root is not None and args.out is not None and not args.out.exists(),'Fresh draft output required')
        plan=make_draft(args.hf_root);validate(plan)
        with args.out.open('x') as stream:json.dump(plan,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        print(json.dumps({'id':ID,'binding_status':plan['binding_status'],'model_calls_executed':0}))
    elif args.command=='inspect':
        base.require(args.protocol is not None,'Protocol required');plan=base.read(args.protocol);validate(plan)
        print(json.dumps({'id':ID,'binding_status':plan['binding_status'],'limits':LIMITS,'model_calls_executed':0},indent=2))
    elif args.command=='preflight':
        base.require(args.protocol is not None and args.hf_root is not None and args.out is not None and not args.out.exists(),'Frozen protocol and fresh CPU proof path required')
        report=preflight(args.protocol,args.hf_root.resolve())
        with args.out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'status':report['status'],'model_calls':0}))
    else:
        base.require(args.protocol is not None and args.hf_root is not None,'Frozen protocol and HF root required')
        report=run(args);print(json.dumps({'status':report['status'],'completed':report['completed']}))


if __name__=='__main__':main()
