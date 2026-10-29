"""Four video factorial parent identity. Draft requires both actual closed parents."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_p64_quality_lineage as prior
MODES=('none','high_only','low_only','both');ORDINALS=(898,96)
CONFIGS=tuple(f'config/e0_chipmunk_p64_factorial_video_20261030_part{i}_frozen.json' for i in (0,1))
OWNER='tools/e0_chipmunk_p64_factorial_owner.py'
need=prior.native.need

def assemble(part,new_phases,old_rows,case,generation):
 need(type(part) is int and part in (0,1),'part differs')
 need([p['mode'] for p in new_phases]==list(MODES[:3]) and [r['mode'] for r in old_rows]==list(prior.MODES),'parent modes differ')
 old_dense,both=old_rows
 for old in old_rows:
  if 'case' in old:
   need(all(old['case'][key]==case[key] for key in ('source_id','prompt','image')),'paired source image/prompt differs')
 need(old_dense['ordinal']==both['ordinal']==ORDINALS[part] and old_dense['video']['sha256']==new_phases[0]['video']['sha256'] and old_dense['video']['bytes']==new_phases[0]['video']['bytes'],'NONE differs from old Dense bytes')
 for phase in new_phases:
  for name in ('raw_noise_FP32','initial_noise'):
   need(all(phase['states'][name][k]==old_dense['states'][name][k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype')),'paired noise differs')
 rows=[]
 for phase in new_phases:
  video=phase['video'];path=Path(video['path']).resolve();hf=Path('/home/xiang/.cache/huggingface').resolve();need(path.is_relative_to(hf),'video outside HF')
  rows.append({'ordinal':ORDINALS[part],'mode':phase['mode'],'case':case,'video':{'path':str(path.relative_to(hf)),'bytes':video['bytes'],'sha256':video['sha256']},'generation':generation,'states':phase['states'],'run_id':phase['run_id']})
 rows.append({**both,'mode':'both','reused_previous_video':True})
 need(len({r['video']['path'] for r in rows})==4,'video paths aliased')
 return rows

def parent_data(hf,part):
 need(type(part) is int and part in (0,1),'part differs');hf=Path(hf).resolve();need(hf==Path('/home/xiang/.cache/huggingface').resolve(),'HF differs')
 spec=importlib.util.spec_from_file_location('_factorial_quality_bound_owner',ROOT/OWNER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
 q=ROOT/CONFIGS[part];p=prior.native.read(q);owner.validate(p,True);v=owner.inspect_verification(SimpleNamespace(protocol=q));need(v['status']=='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_P64_FACTORIAL_VIDEO','actual independent verifier required')
 out=hf/p['output_HF_relative'];run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';rb=prior.native.binding(run,hf);e=prior.native.read(ex);owner.vm.exited(e,rb,prior.native.sha(q),p['id']);r=prior.native.read(run)
 need(r.get('status')=='COMPLETE_CHIPMUNK_P64_FACTORIAL_VIDEO_EXIT_PENDING' and r.get('source_ordinal')==ORDINALS[part] and r.get('none_matches_previous_dense') is True and r.get('none_video_byte_equal') is True,'parent completion differs')
 old=prior.parent_data(hf,part);need(p['closed_v2']==old,'previous parent identity differs');oldrows=prior.closed_rows(old,hf,part)
 rows=assemble(part,r['phases'],oldrows,p['case'],rb)
 return {'protocol':prior.native.binding(q,ROOT),'RUN':rb,'exit':prior.native.binding(ex,hf),'verification_RUN':v['verification_RUN'],'verification_exit':v['verification_exit'],'previous_parent':old},rows

def parent_path(part):
 need(type(part) is int and part in (0,1),'part differs');return CONFIGS[part]

def parent_entry(part):
 return {'protocol':prior.native.binding(ROOT/parent_path(part),ROOT),'owner_source':prior.native.binding(ROOT/OWNER,ROOT)}

def closed_rows(proof,hf,part):
 actual,rows=parent_data(hf,part);need(actual==proof,'actual parent identity changed');return rows

def visual_rows(proof,hf,part):
 return [r for r in closed_rows(proof,hf,part) if r['mode'] in ('high_only','low_only')]

def reused_visual_reference(hf,part,rows):
 # Verify the already-closed producer and match its two video byte identities.
 from types import SimpleNamespace
 import e0_chipmunk_p64_metric_verify as old_verify
 q=ROOT/f'config/e0_chipmunk_p64_visual_20261030_part{part}_v1.json'
 p=prior.native.read(q);proof=old_verify.inspect(SimpleNamespace(kind='visual',protocol=q,hf_root=Path(hf)))
 need(proof['status']=='PASS_ACTUAL_EXIT_CHIPMUNK_P64_QUALITY_CLOSED','old visual verifier missing')
 out=Path(hf)/p['output_HF_relative'];run=prior.native.read(out/'RUN.json');manifest=prior.native.bound(Path(hf),run['manifest']);m=prior.native.read(manifest)
 old={v['mode']:v['video']['sha256'] for v in m['videos']};new={v['mode']:v['video']['sha256'] for v in rows}
 need(old=={'original_dense':new['none'],'global_p64':new['both']},'old visual video SHA differs')
 return {'protocol':prior.native.binding(q,ROOT),'manifest':prior.native.binding(manifest,Path(hf)),'verification_RUN':proof['verification_RUN'],'verification_exit':proof['verification_exit'],'reused_modes':{'none':'original_dense','both':'global_p64'},'new_decodes':0}

def contract():
 py=ROOT/'.venv/bin/python';need(py.is_file(),'byte verifier interpreter missing')
 return {'python_path':str(py),'python_sha256':prior.native.sha(py),'source':'tools/e0_chipmunk_p64_factorial_quality_lineage.py','argv':[str(py),'-I','-B',str(ROOT/'tools/e0_chipmunk_p64_factorial_quality_lineage.py'),'--verify-child'],'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,'child_CPUs':'inherited_from_metric_worker','child_sha_accounting':'nested parent identity hashes outside parent HashReads journal; actual calls and bytes UNKNOWN','CUDA_VISIBLE_DEVICES':'','numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}

def verify_child(proof,hf,part,ledger):
 import json,os,subprocess
 c=contract();request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part};env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')};env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def call():
  result=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=300,check=False);need(result.returncode==0,'parent byte verifier failed: '+result.stderr[-2000:]);r=json.loads(result.stdout);need(r.get('request')==request and r.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and r.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'child identity differs');return r['rows']
 return ledger.call(call)

if __name__=='__main__':
 import json,os,sys
 need(sys.argv[1:]==['--verify-child'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'child isolation differs');request=json.load(sys.stdin);proof,rows=parent_data(request['hf'],request['part']);need(proof==request['proof'] and not {'torch','numpy'}&set(sys.modules),'parent identity or numerical import differs');print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))
