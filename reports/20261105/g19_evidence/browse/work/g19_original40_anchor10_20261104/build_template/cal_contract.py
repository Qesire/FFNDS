"""Original CAL receipt, coverage and pairing contract; Python standard library only."""
import hashlib,json,math,re
from pathlib import Path,PurePosixPath

SOURCES=('chip10_02','chip10_04');BRANCHES=('conditional','unconditional');EXPERTS=('high','low')
REMOTE='/public/home/scc/pb23061276/projects/wan-quant/work/g19_original40_anchor10_20261104/'
BUILD_CONFIG={'sources':list(SOURCES),'seed':42,'experts':list(EXPERTS),'layers':list(range(40)),
 'branches':list(BRANCHES),'steps':40,'rows':64,'anchor_interval':10,'expert_boundary_forces_anchor':True,
 'fit':'RATIO_OF_EQUAL_SOURCE_EQUAL_BRANCH_EQUAL_NONANCHOR_STEP_TOKEN_MOMENTS',
 'zero_denominator_alpha':0,'alpha_dtype':'F64','weight_load':'ORIGINAL_F32_TO_NATIVE_BF16',
 'matrix_construction':'F64_PRODUCT_THEN_F32_THEN_BF16','matrix_layout':'input_output',
 'matrix_storage':'BF16_RAW_UINT16','diagnostic_measure':'CURRENT_TARGET_GATE'}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def write_json(p,obj):
 with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def digest(value):
 if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('invalid SHA')
 return value
def allowed(path):
 p=PurePosixPath(path)
 if not p.is_absolute() or not str(p).startswith(REMOTE) or '..' in p.parts:raise ValueError('CAL path outside original40 round')
 return Path(path)
def pair_plan(dispatch):
 if len(dispatch)!=40 or [x['step'] for x in dispatch]!=list(range(40)):raise ValueError('dispatch coverage')
 anchors=[];pairs=[];previous=None;anchor=None
 for x in dispatch:
  step=x['step'];expert=x['expert'];t=x['timestep']
  if expert not in EXPERTS or not isinstance(t,(int,float)) or not math.isfinite(t) or expert!=('high' if t>=900 else 'low'):raise ValueError('native expert boundary')
  if step and not t<dispatch[step-1]['timestep']:raise ValueError('timetable not strictly decreasing')
  if step%10==0 or expert!=previous:
   anchor=step;anchors.append({'step':step,'expert':expert,'reason':'global_multiple_of_10' if step%10==0 else 'expert_boundary'})
  else:
   offset=step-anchor
   if not 1<=offset<=9:raise ValueError('anchor age')
   pairs.append({'expert':expert,'anchor_step':anchor,'step':step,'offset':offset,
    'anchor_timestep':dispatch[anchor]['timestep'],'target_timestep':t,
    'time_delta':dispatch[anchor]['timestep']-t})
  previous=expert
 for pair in pairs:pair['window_nonanchor_steps']=sum(x['anchor_step']==pair['anchor_step'] for x in pairs)
 if not all(any(p['expert']==e for p in pairs) for e in EXPERTS):raise ValueError('both experts require nonanchor observations')
 return anchors,pairs

def validate_cal(scope,review,start,end,events,weight_ids,effective,source,scope_hash):
 if source not in SOURCES or scope.get('phase')!='ORIGINAL40_CALIBRATION' or scope.get('seed')!=42 or scope.get('run_ids')!=[source]:raise ValueError('CAL source/phase')
 if review.get('status')!='ACCEPTED' or review.get('scope_sha256')!=scope_hash or not review.get('reviewer') or review.get('reviewer') in scope.get('authors',[]):raise ValueError('CAL preexecution review')
 if start['status']!='STARTED' or start['scope'].get('_verified_scope_sha256')!=scope_hash or {k:v for k,v in start['scope'].items() if k!='_verified_scope_sha256'}!=scope:raise ValueError('CAL START identity')
 if end['status']!='SUCCESS' or end.get('exception') is not None or end['result']['source_id']!=source or end['result']['samples']!=3200:raise ValueError('complete CAL required')
 config=scope['required_runtime_config']
 expected={'model_cls':'wan2.2_moe','infer_steps':40,'enable_cfg':True,'sample_guide_scale':[3.5,3.5],'sample_shift':5.,'boundary':.9}
 if any(config.get(k)!=v for k,v in expected.items()) or any(effective.get(k)!=v for k,v in config.items()):raise ValueError('original40 effective configuration')
 if not effective.get('model_path')==scope['base_model_dir'] or 'distill' in scope['base_model_dir'].lower():raise ValueError('original base model')
 dispatch=end['result']['expert_dispatch'];anchors,pairs=pair_plan(dispatch)
 if len(end['result']['latent_digests'])!=40 or [x['step'] for x in end['result']['latent_digests']]!=list(range(40)):raise ValueError('CAL latent coverage')
 times=[x for x in events if x.get('event')=='TIMESTEPS']
 if len(times)!=1 or times[0]['values']!=[x['timestep'] for x in dispatch]:raise ValueError('actual timetable identity')
 captures=[x for x in events if x.get('event')=='CAL_W1_SAMPLES'];expected_sites={(s,dispatch[s]['expert'],b,l) for s in range(40) for b in BRANCHES for l in range(40)}
 if len(captures)!=3200 or {(x['step'],x['expert'],x['branch'],x['layer']) for x in captures}!=expected_sites:raise ValueError('capture coverage')
 for x in captures:
  allowed(x['file']);digest(x['sha256'])
  if x['rows']!=64 or len(x['token_grid'])!=3 or min(x['token_grid'])<=0 or x['timestep']!=dispatch[x['step']]['timestep']:raise ValueError('capture geometry')
  expected_path=scope['output_dir']+f"/samples/{x['expert']}/{x['branch']}/{x['layer']:02d}/{x['step']:02d}.pt"
  if x['file']!=expected_path:raise ValueError('capture path')
 specs={x['path']:x for x in scope['original_weight_files']}
 if len(specs)!=12 or set(weight_ids)!=set(specs):raise ValueError('twelve original weight shards required')
 for path,ident in weight_ids.items():
  digest(ident['sha256']);spec=specs[path]
  if any(ident[k]!=spec[k] for k in ('size','mtime_ns','header_sha256')):raise ValueError('weight identity and frozen headers disagree')
 return captures,anchors,pairs

def verify_inputs(manifest):
 if manifest.get('schema')!='ORIGINAL40_CAL_INPUTS_V1' or set(manifest['sources'])!=set(SOURCES):raise ValueError('build input schema/sources')
 runs={};common=None;configs=[]
 for source,entry in manifest['sources'].items():
  rec=entry['receipts']
  if set(rec)!={'scope','review','root_self_audit','start','exit','progress','weight_identities','effective_config','cohort','result_review'}:raise ValueError('receipt inventory')
  for name,item in rec.items():
   p=allowed(item['path']);digest(item['sha256'])
   if sha(p)!=item['sha256']:raise ValueError('CAL receipt changed '+name)
  scope,review,start,end,weights,effective=(read(rec[k]['path']) for k in ('scope','review','start','exit','weight_identities','effective_config'))
  result_review=read(rec['result_review']['path'])
  if result_review.get('status')!='ACCEPTED' or not result_review.get('reviewer') or result_review['reviewer'] in scope.get('authors',[]):raise ValueError('independent CAL result review required')
  for key in ('scope','start','exit','progress','weight_identities','effective_config'):
   if result_review.get(key+'_sha256')!=rec[key]['sha256']:raise ValueError('CAL result review identity '+key)
  if review['root_self_audit_sha256']!=rec['root_self_audit']['sha256'] or end['result']['progress_sha256']!=rec['progress']['sha256'] or scope['manifest_sha256']!=rec['cohort']['sha256']:raise ValueError('CAL receipt chain')
  rows=[r for r in read(rec['cohort']['path']) if r['run_id']==source]
  if len(rows)!=1 or rows[0]['seed']!=42 or rows[0]!=entry['source_row']:raise ValueError('CAL source row')
  events=[json.loads(x) for x in Path(rec['progress']['path']).read_text(encoding='utf8').splitlines() if x.strip()]
  captures,anchors,pairs=validate_cal(scope,review,start,end,events,weights,effective,source,rec['scope']['sha256'])
  compact=[{k:x[k] for k in ('step','expert','branch','layer','timestep','file','sha256','rows','token_grid')} for x in captures]
  if compact!=entry['captures'] or anchors!=entry['anchors'] or pairs!=entry['pairs']:raise ValueError('frozen sample or pair plan changed')
  if common is not None and weights!=common:raise ValueError('CAL sources used different original payloads')
  common=weights;configs.append((scope['base_model_dir'],scope['required_runtime_config'],scope['base_assets_sha256'],scope['external_sources_sha256'],scope['python_source_roots']))
  runs[source]={'scope':scope,'weights':weights,'entry':entry,'anchors':anchors,'pairs':pairs,
   'captures':{(x['expert'],x['branch'],x['layer'],x['step']):x for x in captures}}
 if configs[0]!=configs[1] or common!=manifest['original_weight_identities']:raise ValueError('original model identity mismatch')
 if len({manifest['sources'][s]['source_row']['image_sha256'] for s in SOURCES})!=2:raise ValueError('duplicate CAL source image')
 return runs,common
