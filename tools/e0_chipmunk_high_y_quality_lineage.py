"""Three-arm lineage: real Y-only parent must have exited and independently verified."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_p64_factorial_quality_lineage as factorial
prior=factorial.prior;need=prior.native.need
MODES=('none','high_p64','high_y_reuse');ORDINALS=(898,96)
CONFIGS=tuple(f'config/e0_chipmunk_high_y_reuse_video_20261030_part{i}_pending_review_v2.json' for i in (0,1))
OWNER='tools/e0_chipmunk_high_y_reuse_owner.py'
def assemble(part,phases,oldrows,case,generation,hf):
 need(type(part) is int and part in (0,1),'invalid part')
 need([x['mode'] for x in phases]==['none','high_y_cache_dense','high_y_reuse'],'new phases differ')
 old={x['mode']:x for x in oldrows};need(set(old)==set(factorial.MODES),'old modes differ')
 for key in ('source_id','prompt','image'):need(old['none']['case'][key]==case[key],'paired source differs')
 none=phases[0];need(all(none['video'][k]==old['none']['video'][k] for k in ('bytes','sha256')),'NONE differs from closed Dense')
 for phase in phases:
  for name in ('raw_noise_FP32','initial_noise'):
   need(all(phase['states'][name][k]==old['none']['states'][name][k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype')),'initial noise differs')
 def row(phase,mode):
  v=phase['video'];p=Path(v['path']).resolve();need(p.is_relative_to(hf) and p.is_file() and p.stat().st_size==v['bytes'],'video path differs')
  return {'ordinal':ORDINALS[part],'mode':mode,'case':case,'video':{'path':str(p.relative_to(hf)),'bytes':v['bytes'],'sha256':v['sha256']},'generation':generation,'states':phase['states'],'run_id':phase['run_id']}
 rows=[row(none,'none'),{**old['high_only'],'mode':'high_p64','reused_previous_video':True},row(phases[2],'high_y_reuse')]
 need(len({x['video']['path'] for x in rows})==3,'aliased video paths');return rows

def parent_data(hf,part):
 need(type(part) is int and part in (0,1),'invalid part');hf=Path(hf).resolve();need(hf==Path('/home/xiang/.cache/huggingface').resolve(),'HF differs')
 spec=importlib.util.spec_from_file_location('_high_y_bound_owner',ROOT/OWNER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
 q=ROOT/CONFIGS[part];p=prior.native.read(q);owner.validate(p,True);v=owner.inspect_verification(SimpleNamespace(protocol=q));need(v['status']=='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_HIGH_Y_REUSE_VIDEO','actual Y verifier required')
 out=hf/p['output_HF_relative'];run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';rb=prior.native.binding(run,hf);owner.vm.exited(prior.native.read(ex),rb,prior.native.sha(q),p['id']);r=prior.native.read(run)
 need(r.get('status')=='COMPLETE_CHIPMUNK_HIGH_Y_REUSE_VIDEO_EXIT_PENDING' and r.get('source_ordinal')==ORDINALS[part] and r.get('dense_neutral_matches_previous') is True and r.get('neutral_video_checks')=={'none':True,'high_y_cache_dense':True},'Y parent completion differs')
 oldproof,oldrows=factorial.parent_data(hf,part);rows=assemble(part,r['phases'],oldrows,p['case'],rb,hf)
 return {'protocol':prior.native.binding(q,ROOT),'RUN':rb,'exit':prior.native.binding(ex,hf),'verification_RUN':v['verification_RUN'],'verification_exit':v['verification_exit'],'high_p64_parent':oldproof},rows

def parent_path(part):
 need(type(part) is int and part in (0,1),"part differs");return CONFIGS[part]
def parent_entry(part):
 return {"protocol":prior.native.binding(ROOT/parent_path(part),ROOT),"owner_source":prior.native.binding(ROOT/OWNER,ROOT)}
def closed_rows(proof,hf,part):
 actual,rows=parent_data(hf,part);need(actual==proof,"parent identity differs");return rows
def visual_rows(proof,hf,part):
 return [r for r in closed_rows(proof,hf,part) if r["mode"]=="high_y_reuse"]

def contract():
 py=ROOT/'.venv/bin/python';need(py.is_file(),'byte verifier interpreter missing')
 return {'python_path':str(py),'python_sha256':prior.native.sha(py),'source':'tools/e0_chipmunk_high_y_quality_lineage.py','argv':[str(py),'-I','-B',str(ROOT/'tools/e0_chipmunk_high_y_quality_lineage.py'),'--verify-child'],'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,'child_CPUs':'inherited_from_metric_worker','child_sha_accounting':'nested parent identity hashes outside parent HashReads journal; actual calls and bytes UNKNOWN','CUDA_VISIBLE_DEVICES':'','numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}

def verify_child(proof,hf,part,ledger):
 import json,os,subprocess
 c=contract();request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part};env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')};env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def call():
  result=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=300,check=False);need(result.returncode==0,'parent byte verifier failed: '+result.stderr[-2000:]);r=json.loads(result.stdout);need(r.get('request')==request and r.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and r.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'child identity differs');return r['rows']
 return ledger.call(call)

if __name__=='__main__':
 import json,os,sys
 need(sys.argv[1:]==['--verify-child'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'child isolation differs');request=json.load(sys.stdin);proof,rows=parent_data(request['hf'],request['part']);need(proof==request['proof'] and not {'torch','numpy'}&set(sys.modules),'parent identity or numerical import differs');print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))
