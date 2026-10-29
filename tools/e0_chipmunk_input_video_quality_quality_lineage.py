"""Require actual source-specific five-phase producer plus independent verifier."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_p64_factorial_quality_lineage as factorial
prior=factorial.prior;need=prior.native.need
MODES=('none','high_h_readonly_p64','high_x_recompute_p64');ORDINALS=(898,96,1040,979)
CONFIGS=tuple(f'config/e0_chipmunk_input_video_20261030_part{i}_pending_review_v1.json' for i in range(4));OWNER='tools/e0_chipmunk_input_video_owner.py'
def parent_path(part):need(type(part) is int and part in range(4),'invalid part');return CONFIGS[part]
def parent_entry(part):return {'protocol':prior.native.binding(ROOT/parent_path(part),ROOT),'owner_source':prior.native.binding(ROOT/OWNER,ROOT)}
def parent_data(hf,part):
 need(type(part) is int and part in range(4),'invalid part');hf=Path(hf).resolve();need(hf==Path('/home/xiang/.cache/huggingface').resolve(),'HF differs');q=ROOT/parent_path(part);need(q.is_file(),'parent not frozen/closed');p=prior.native.read(q)
 spec=importlib.util.spec_from_file_location('_input_video_bound_owner',ROOT/OWNER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);owner.validate(p,True);v=owner.inspect_verification(SimpleNamespace(protocol=q));need(v['status']=='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO','actual parent verifier required')
 out=hf/p['output_HF_relative'];run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';rb=prior.native.binding(run,hf);owner.vm.exited(prior.native.read(ex),rb,prior.native.sha(q),p['id']);r=prior.native.read(run);need(r.get('status')=='COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING' and r.get('source_ordinal')==ORDINALS[part] and r.get('cache_dense_matches_current_none') is True and [x['mode'] for x in r['phases']]==list(owner.driver.MODES),'parent coverage differs')
 rows=[]
 for i in (0,3,4):
  phase=r['phases'][i];vpath=Path(phase['video']['path']).resolve();need(vpath.is_relative_to(hf) and vpath.is_file() and vpath.stat().st_size==phase['video']['bytes'],'video path differs');rows.append({'ordinal':ORDINALS[part],'mode':phase['mode'],'case':p['case'],'video':{'path':str(vpath.relative_to(hf)),'bytes':phase['video']['bytes'],'sha256':phase['video']['sha256']},'generation':rb,'states':phase['states'],'run_id':phase['run_id']})
 need(len({x['video']['path'] for x in rows})==3,'video paths alias');return {'protocol':prior.native.binding(q,ROOT),'RUN':rb,'exit':prior.native.binding(ex,hf),'verification_RUN':v['verification_RUN'],'verification_exit':v['verification_exit']},rows
def closed_rows(proof,hf,part):
 actual,rows=parent_data(hf,part);need(actual==proof,'parent identity differs');return rows
def visual_rows(proof,hf,part):return closed_rows(proof,hf,part)
def contract():
 py=ROOT/'.venv/bin/python';need(py.is_file(),'interpreter missing');return {'python_path':str(py),'python_sha256':prior.native.sha(py),'source':'tools/e0_chipmunk_input_video_quality_quality_lineage.py','argv':[str(py),'-I','-B',str(ROOT/'tools/e0_chipmunk_input_video_quality_quality_lineage.py'),'--verify-child'],'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,'child_CPUs':'inherited_from_metric_worker','child_sha_accounting':'nested parent identity hashes outside parent HashReads journal; actual calls and bytes UNKNOWN','CUDA_VISIBLE_DEVICES':'','numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}
def verify_child(proof,hf,part,ledger):
 c=contract();request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part};env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')};env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def call():
  r=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=300,check=False);need(r.returncode==0,'parent byte verifier failed: '+r.stderr[-2000:]);v=json.loads(r.stdout);need(v.get('request')==request and v.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and v.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'child differs');return v['rows']
 return ledger.call(call)
if __name__=='__main__':
 need(sys.argv[1:]==['--verify-child'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'child isolation differs');request=json.load(sys.stdin);proof,rows=parent_data(request['hf'],request['part']);need(proof==request['proof'] and not {'torch','numpy'}&set(sys.modules),'parent identity differs');print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))
