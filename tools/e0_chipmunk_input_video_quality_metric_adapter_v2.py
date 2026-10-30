"""Chipmunk P64 lineage and immutable scope adapter for existing official TEAL metric engines.

Only actual exited parent and exited parent verifier can produce a draft. The
original metric engines remain unchanged; metric calls and budgets are theirs.
"""
import copy,importlib.util
from pathlib import Path
import e0_chipmunk_input_video_quality_quality_lineage as line
ROOT=Path(__file__).resolve().parents[1]
BASE='config/e0_raw_bf16_temporal_video20_quality_20261029_v1.json'
BANK='config/e0_s2_new12_development_source_bank_20261029_v1.json'
PRIOR_SCORE='research/20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.json'
ADAPTER='tools/e0_chipmunk_input_video_quality_metric_adapter_v2.py';TEST='tests/test_e0_chipmunk_input_video_quality_metric_adapter_v2.py';FREEZER='tools/e0_chipmunk_input_video_quality_metric_freeze_v2.py';READ_BUDGET='tools/e0_sharq_wan_metric_read_budget.py';READ_TEST='tests/test_e0_sharq_wan_metric_read_budget.py';VERIFY='tools/e0_chipmunk_input_video_quality_metric_verify_v2.py';VERIFY_TEST='tests/test_e0_chipmunk_input_video_quality_metric_adapter_v2.py'

def install(kind):
 if kind not in ('quality','raft'):raise ValueError('Invalid metric kind')
 original='tools/e0_chipmunk_input_video_quality_quality_engine_v2.py' if kind=='quality' else 'tools/e0_chipmunk_input_video_quality_raft_engine_v2.py'
 launcher='tools/e0_chipmunk_input_video_quality_official_quality_v2.py' if kind=='quality' else 'tools/e0_chipmunk_input_video_quality_continuous_raft_v2.py'
 s=importlib.util.spec_from_file_location('_sharq_metric_original_'+kind,ROOT/original);engine=importlib.util.module_from_spec(s);s.loader.exec_module(engine)
 base=engine.base;need=base.require
 execution_spec=importlib.util.spec_from_file_location('_sharq_execution_'+kind,ROOT/'tools/e0_chipmunk_input_video_quality_execution.py');execution=importlib.util.module_from_spec(execution_spec);execution_spec.loader.exec_module(execution);engine.execution=execution
 engine.SELF=launcher;engine.SHARDS=tuple((x,) for x in line.ORDINALS);engine.MODES=line.MODES
 if kind=='quality':engine.motion.MODES=line.MODES;engine.motion.SHARDS=engine.SHARDS
 engine.STATUS='COMPLETE_CHIPMUNK_INPUT_VIDEO_'+('EIGHT_AXIS' if kind=='quality' else 'CONTINUOUS_RAFT')+'_NOT_ACCEPTED'
 engine.CPU_STATUS='PASS_CHIPMUNK_INPUT_VIDEO_'+kind.upper()+'_BYTE_PREFLIGHT_NO_NUMPY_TORCH_GPU_MODEL'
 def identity(shard):
  need(type(shard) is int and shard in (0,1,2,3),'Invalid shard');return 'E0_CHIPMUNK_INPUT_VIDEO_V2_'+kind.upper()+'_20261030_PART'+str(shard)
 def output(shard):return 'ffnds-derived/e0_chipmunk_input_video_quality_v2_'+kind+'_20261030/part'+str(shard)+'/attempt1/worker/results'
 def draft(hf,shard):
  identity(shard);parent,rows=line.parent_data(hf,shard);template=base.read(ROOT/BASE);prior_path=('config/e0_teal_wan_quality_v3_part0_20261029_pending_review_v1.json' if kind=='quality' else 'config/e0_teal_wan_raft_v3_part1_20261029_pending_review_v1.json');prior=base.read(ROOT/prior_path);generation=base.read_bound(ROOT,parent['protocol'])
  paths={x['path'] for x in prior['repo_sources']}|{x['path'] for x in generation['repo_sources']}|{original,launcher,'tools/e0_chipmunk_input_video_quality_raft_engine_v2.py',ADAPTER,TEST,FREEZER,READ_BUDGET,READ_TEST,VERIFY,VERIFY_TEST,'tools/e0_chipmunk_input_video_quality_quality_lineage.py','tests/test_e0_chipmunk_input_video_quality_quality_lineage.py','tools/e0_chipmunk_p64_factorial_quality_lineage.py','tests/test_e0_chipmunk_p64_factorial_quality_lineage.py',BASE,BANK,prior_path,line.parent_path(shard),line.parent_entry(shard)['owner_source']['path']}|execution.HELPERS|{'tools/e0_chipmunk_input_video_quality_monitor_v2.py','tests/test_e0_chipmunk_input_video_quality_monitor_v2.py'}|{'tools/e0_chipmunk_input_video_quality_metric_jobs_v2.py','tests/test_e0_chipmunk_input_video_quality_metric_adapter_v2.py','tests/test_e0_chipmunk_input_video_quality_entry_v2.py','tools/e0_chipmunk_p64_quality_lineage.py','tests/test_e0_chipmunk_p64_quality_lineage.py','tools/e0_chipmunk_p64_factorial_quality_aggregate.py','tests/test_e0_chipmunk_p64_factorial_quality_aggregate.py'}
  p={k:copy.deepcopy(template[k]) for k in ('environment','import_environment','hf_bindings','reused_import_proof','directory_fd_audit_policy','activation_bootstrap_policy','directory_fd_filelock_probe_source')}
  p.update(id=identity(shard),shard=shard,status='DRAFT_NOT_AUTHORIZED',binding_status='BOUND_DRAFT_NOT_REGISTERED',assets_finalized=False,automatic_retry=False,limits=copy.deepcopy(engine.LIMITS),resources=copy.deepcopy(engine.RESOURCES),CPU_preflight_resources=copy.deepcopy(engine.CPU_RESOURCES),parents=[parent],videos=rows,repo_sources=[base.binding(ROOT/x,ROOT) for x in sorted(paths)],template=base.binding(ROOT/BASE,ROOT),execution_monitor_binary=execution.monitor_binding(base),execution_private_worker_config_sha256=execution.PROFILE_SHA,execution_device={'name_contains':'A100','cuda_capability':[8,0],'session_max_GPUs':4,'physical_gpu_index':execution.physical_gpu(kind,shard)},monitor_policy={'query_timeout_seconds':30,'sampling_interval_seconds':5,'close_join_seconds':35,'fail_on_foreign_or_query_error':True,'diagnostics':'QUERY_LABEL_ELAPSED_ERROR_NO_UUID'},lineage_verifier=line.contract(),read_budget={'per_phase_sha_bytes_max':16<<30,'journal_bytes_max':16<<20,'worker_extra_bytes_max':64<<20,'worker_extra_enforcement':'SOFT_SAMPLED_AT_SHA_EVENTS','unit':'PER_FILE_SHA_PREDEBIT_AND_COMPLETE','small_metadata_and_native_internal_model_loads':'SEPARATELY_UNKNOWN'},output_HF_relative=output(shard),estimand='Four exposed development image sources, one seed each; all three evaluated videos are newly generated. No heldout or human acceptance.')
  if kind=='quality':
   case=rows[0]['case'];source={'source_id':case['source_id'],'image':case['image'],'prompt':case['prompt'],'scene_stratum':case.get('stratum',case.get('case_role','DEVELOPMENT_SOURCE'))};p.update(costs=copy.deepcopy(engine.COSTS),dimensions=list(engine.DIMENSIONS),source_bank=base.binding(ROOT/line.parent_path(shard),ROOT),images=[{'ordinal':line.ORDINALS[shard],'name':str(line.ORDINALS[shard])+'.png','binding':case['image'],'scene_stratum':source['scene_stratum']}],policy={'all_eight_dimensions_each_clip':True,'I2V_axes':'DEVELOPMENT_DIAGNOSTIC_NO_OFFICIAL_STRATUM','continuous_RAFT':'SEPARATE_REGISTERED_PRODUCER','temporal_flickering':'NOT_RUN_NINTH_AXIS','aggregation':'SOURCE_EQUAL_PAIRED_FRESH_SAME_DEVICE_DENSE','official_five_independent_samples':False})
  else:
   p.update(raft_assets=[b for b in template['hf_bindings'] if '/installed_source_view/' in b['path'] or b['path'].endswith(engine.WEIGHT_SUFFIX)],sampling={'videos':3,'frames_per_video':81,'fps':16,'frame_stride':2,'pairs_per_video':40,'RAFT_iterations_per_pair':20})
  unique=all_bindings(p);video_bytes=sum(v['video']['bytes'] for v in rows);p['known_hash_work']={'preflight_unique_files':len(unique),'preflight_unique_bytes':sum(b['bytes'] for _,b in unique),'preflight_extra_video_hash_calls':3,'preflight_extra_video_bytes':video_bytes,'GPU_extra_video_hash_calls':6,'GPU_extra_video_bytes':2*video_bytes,'GPU_i2v_image_hash_calls':2 if kind=='quality' else 0,'GPU_i2v_image_bytes':2*case['image']['bytes'] if kind=='quality' else 0,'additional_SHA_model_source_snapshot_calls':'ACTUALLY_JOURNALED_COUNTS_NOT_PREASSERTED'}
  return p
 def validate(p,frozen=False):
  shard=p['shard'];need(p['id']==identity(shard) and p['limits']==engine.LIMITS and p['resources']==engine.RESOURCES and p['CPU_preflight_resources']==engine.CPU_RESOURCES and p['automatic_retry'] is False and p['output_HF_relative']==output(shard) and p['lineage_verifier']==line.contract() and p['read_budget']=={'per_phase_sha_bytes_max':16<<30,'journal_bytes_max':16<<20,'worker_extra_bytes_max':64<<20,'worker_extra_enforcement':'SOFT_SAMPLED_AT_SHA_EVENTS','unit':'PER_FILE_SHA_PREDEBIT_AND_COMPLETE','small_metadata_and_native_internal_model_loads':'SEPARATELY_UNKNOWN'} and p['execution_device']=={'name_contains':'A100','cuda_capability':[8,0],'session_max_GPUs':4,'physical_gpu_index':execution.physical_gpu(kind,shard)},'fixed scope differs')
  need(len(p['parents'])==1 and p['parents'][0]['protocol']['path']==line.parent_path(shard) and [(x['ordinal'],x['mode']) for x in p['videos']]==[(line.ORDINALS[shard],m) for m in line.MODES],'parent coverage differs')
  template=base.read(ROOT/BASE);need(all(p[k]==template[k] for k in ('environment','import_environment','hf_bindings','reused_import_proof','directory_fd_audit_policy','activation_bootstrap_policy','directory_fd_filelock_probe_source')),'frozen environment differs')
  prior_path=('config/e0_teal_wan_quality_v3_part0_20261029_pending_review_v1.json' if kind=='quality' else 'config/e0_teal_wan_raft_v3_part1_20261029_pending_review_v1.json');prior=base.read(ROOT/prior_path);generation=base.read_bound(ROOT,p['parents'][0]['protocol']);required={x['path'] for x in prior['repo_sources']}|{x['path'] for x in generation['repo_sources']}|{original,launcher,'tools/e0_chipmunk_input_video_quality_raft_engine_v2.py',ADAPTER,TEST,FREEZER,READ_BUDGET,READ_TEST,VERIFY,VERIFY_TEST,'tools/e0_chipmunk_input_video_quality_quality_lineage.py','tests/test_e0_chipmunk_input_video_quality_quality_lineage.py','tools/e0_chipmunk_p64_factorial_quality_lineage.py','tests/test_e0_chipmunk_p64_factorial_quality_lineage.py',BASE,BANK,prior_path,line.parent_path(shard),line.parent_entry(shard)['owner_source']['path']}|execution.HELPERS|{'tools/e0_chipmunk_input_video_quality_monitor_v2.py','tests/test_e0_chipmunk_input_video_quality_monitor_v2.py'}|{'tools/e0_chipmunk_input_video_quality_metric_jobs_v2.py','tests/test_e0_chipmunk_input_video_quality_metric_adapter_v2.py','tests/test_e0_chipmunk_input_video_quality_entry_v2.py','tools/e0_chipmunk_p64_quality_lineage.py','tests/test_e0_chipmunk_p64_quality_lineage.py','tools/e0_chipmunk_p64_factorial_quality_aggregate.py','tests/test_e0_chipmunk_p64_factorial_quality_aggregate.py'};paths=[x['path'] for x in p['repo_sources']];need(len(paths)==len(set(paths)) and set(paths)==required,'source closure differs')
  if kind=='raft':need(p['sampling']=={'videos':3,'frames_per_video':81,'fps':16,'frame_stride':2,'pairs_per_video':40,'RAFT_iterations_per_pair':20} and p['raft_assets']==[b for b in template['hf_bindings'] if '/installed_source_view/' in b['path'] or b['path'].endswith(engine.WEIGHT_SUFFIX)] and len([b for b in p['raft_assets'] if b['path'].endswith(engine.WEIGHT_SUFFIX) and b['sha256']==engine.WEIGHT_SHA])==1,'RAFT asset differs')
  if kind=='quality':
   case=p['videos'][0]['case'];source={'source_id':case['source_id'],'image':case['image'],'prompt':case['prompt'],'scene_stratum':case.get('stratum',case.get('case_role','DEVELOPMENT_SOURCE'))};need(p['source_bank']==base.binding(ROOT/line.parent_path(shard),ROOT) and p['images']==[{'ordinal':line.ORDINALS[shard],'name':str(line.ORDINALS[shard])+'.png','binding':case['image'],'scene_stratum':source['scene_stratum']}] and source['source_id']==case['source_id'] and source['image']['sha256']==case['image']['sha256'] and source['prompt']==case['prompt'],'quality image identity differs')

  need(p.get('monitor_policy')=={'query_timeout_seconds':30,'sampling_interval_seconds':5,'close_join_seconds':35,'fail_on_foreign_or_query_error':True,'diagnostics':'QUERY_LABEL_ELAPSED_ERROR_NO_UUID'},'monitor policy differs')
  need(p['execution_private_worker_config_sha256']==execution.PROFILE_SHA and p['execution_monitor_binary']==execution.monitor_binding(base),'worker identity differs')
  if kind=='quality':need(p['costs']==engine.COSTS and p['dimensions']==list(engine.DIMENSIONS) and len(p['images'])==1 and p['policy']=={'all_eight_dimensions_each_clip':True,'I2V_axes':'DEVELOPMENT_DIAGNOSTIC_NO_OFFICIAL_STRATUM','continuous_RAFT':'SEPARATE_REGISTERED_PRODUCER','temporal_flickering':'NOT_RUN_NINTH_AXIS','aggregation':'SOURCE_EQUAL_PAIRED_FRESH_SAME_DEVICE_DENSE','official_five_independent_samples':False},'quality dimensions/policy differ')
  unique=all_bindings(p);video_bytes=sum(v['video']['bytes'] for v in p['videos']);image_bytes=p['images'][0]['binding']['bytes'] if kind=='quality' else 0;expected_hash={'preflight_unique_files':len(unique),'preflight_unique_bytes':sum(b['bytes'] for _,b in unique),'preflight_extra_video_hash_calls':3,'preflight_extra_video_bytes':video_bytes,'GPU_extra_video_hash_calls':6,'GPU_extra_video_bytes':2*video_bytes,'GPU_i2v_image_hash_calls':2 if kind=='quality' else 0,'GPU_i2v_image_bytes':2*image_bytes,'additional_SHA_model_source_snapshot_calls':'ACTUALLY_JOURNALED_COUNTS_NOT_PREASSERTED'};need(p['known_hash_work']==expected_hash,'declared hash work differs')
  if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['binding_status']=='BOUND_READY_FOR_REGISTRATION' and p['assets_finalized'] is True,'scope not frozen')
 def all_bindings(p):
  b=[('REPO',x) for x in p['repo_sources']]+[('HF',x) for x in (p['hf_bindings'] if kind=='quality' else p['raft_assets'])]+[(x['root'],x) for x in p['import_environment']['modules'].values()]+[('REPO',p['reused_import_proof']['protocol'])]+[('HF',p['reused_import_proof'][k]) for k in ('RUN','exit')]+[('HF',p['parents'][0][k]) for k in ('RUN','exit','verification_RUN','verification_exit')]
  for v in p['videos']:b.extend([('HF',v['video']),('HF',v['generation'])]+[('HF',x) for x in v['states'].values()])
  if kind=='quality':b.extend([('HF',x['binding']) for x in p['images']])
  unique={}
  for root,x in b:
   key=(root,x['path']);row={k:x[k] for k in ('path','bytes','sha256')};need(key not in unique or unique[key]==row,'binding conflict');unique[key]=row
  return [(k[0],v) for k,v in sorted(unique.items())]
 def verify_lineage(p,hf,ledger=None):
  need(ledger is not None,'verifier ledger required');rows=line.verify_child(p['parents'][0],hf,p['shard'],ledger);need(rows==p['videos'],'parent lineage changed');paths=[base.bound(hf,x['video']) for x in rows];need(len(set(paths))==len({(x.stat().st_dev,x.stat().st_ino) for x in paths})==3,'video alias')
 engine.identity=identity;engine.draft=draft;engine.validate=validate;engine.all_bindings=all_bindings;engine.verify_lineage=verify_lineage
 # Activate only after exact ACTIVE, private spec, worker and CPU checks; the
 # sidecar lives in the already-created supervisor worker directory.
 original_job=execution.verify_job;current={'guard':None,'snapshot_base':None,'snapshot_sha':None}
 def checked_job(args,plan,scope,base_module,kind_arg,phase,source,resources):
  result=original_job(args,plan,scope,base_module,kind_arg,phase,source,resources)
  from e0_sharq_wan_metric_read_budget import HashReads
  worker=(args.out.parent.parent if phase=='preflight' else (args.hf_root/plan['output_HF_relative']).parent)
  guard=HashReads(base_module,worker/'HASH_READS.jsonl',phase);guard.__enter__();current['guard']=guard
  import e0_teal_output_budget_v3 as snapshots
  current['snapshot_base']=snapshots.old.base;current['snapshot_sha']=snapshots.old.base.sha;snapshots.old.base.sha=guard.sha
  return result
 execution.verify_job=checked_job
 for action in ('preflight','run'):
  old=getattr(engine,action)
  def wrapped(args,_old=old,_action=action):
   try:return _old(args)
   finally:
    guard=current['guard']
    if guard is not None:
     import sys,hashlib
     typ,value,tb=sys.exc_info();current['snapshot_base'].sha=current['snapshot_sha'];failure=None
     try:guard.__exit__(typ,value,tb)
     except BaseException as error:failure=error
     current['guard']=None
     out=args.out if _action=='preflight' else args.hf_root/base.read(args.protocol)['output_HF_relative']/'RUN.json'
     if out.is_file():
      report=base.read(out);journal=guard.path;data=journal.read_bytes();report['sha_read_ledger']={'journal':{'path':str(journal.resolve().relative_to(args.hf_root.resolve())),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},'predebited_bytes':guard.predebited,'completed_bytes':guard.completed,'inflight':guard.inflight,'scope':'whole-file SHA through metric base and immutable snapshot base; small metadata and model-loader internal I/O separately UNKNOWN'};
      if failure:report.update(status='FAILED_HASH_LEDGER_PRESERVED_NO_RETRY',error_type=type(failure).__name__,error=str(failure)[:500])
      base.write(out,report)
     if failure:raise failure
  setattr(engine,action,wrapped)
 old_cpu=engine.require_cpu
 def checked_cpu(scope,plan,digest,hf):
  proof=old_cpu(scope,plan,digest,hf)
  from e0_sharq_wan_metric_read_budget import require_complete
  require_complete(base,hf,base.read_bound(hf,proof['RUN']));return proof
 engine.require_cpu=checked_cpu
 old_closed=engine.verify_closed
 def authorized_closed(args):
  need(getattr(engine,'_authorized_verifier',False),'independent registered verifier required')
  from e0_sharq_wan_metric_read_budget import require_complete
  require_complete(base,args.hf_root,base.read(args.run));return old_closed(args)
 engine.verify_closed=authorized_closed
 return engine
