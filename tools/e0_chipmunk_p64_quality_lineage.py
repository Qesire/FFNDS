"""P64 actual parent + independent artifact-verifier lineage; no tensors/GPU."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_wan_global_p64_owner as owner
native=owner.native
ORDINALS=(898,96);MODES=('original_dense','global_p64')
PARENT_PATHS=({'protocol': {'path': 'config/e0_chipmunk_global_p64_video_v2_20261030_part0_pending_review_v4.json', 'bytes': 237484, 'sha256': '2f9a00897eed35184700e4ec415f1add68aad220242b887216a9563e84f5027d'}, 'owner_source': {'path': 'tools/e0_chipmunk_wan_global_p64_owner_v2.py', 'bytes': 37322, 'sha256': 'fea1d6767ef40dbc701a1f45dee685a70d8f0181d173ff2d016f4a2508b5d6bd'}}, {'protocol': {'path': 'config/e0_chipmunk_global_p64_video_v2_20261030_part1_pending_review_v4.json', 'bytes': 237616, 'sha256': 'b2595eb1d1240bfa43ee6f1ad82f7641dbd0c1e24b2545ae3fe89fe61208394c'}, 'owner_source': {'path': 'tools/e0_chipmunk_wan_global_p64_owner_v2.py', 'bytes': 37322, 'sha256': 'fea1d6767ef40dbc701a1f45dee685a70d8f0181d173ff2d016f4a2508b5d6bd'}})  # Actual independently closed V2 producer identity.
def parent_entry(part):
 native.need(PARENT_PATHS is not None and type(part) is int and part in (0,1),'final producer protocol NOT_BOUND')
 entry=PARENT_PATHS[part];native.need(isinstance(entry,dict) and set(entry)=={'protocol','owner_source'},'explicit parent mapping required')
 for key in ('protocol','owner_source'):
  binding=entry[key];native.need(isinstance(binding,dict) and set(binding)=={'path','bytes','sha256'},'parent binding format');native.bound(ROOT,binding)
 return entry

def parent_path(part):return parent_entry(part)['protocol']['path']
def parent_owner(part):
 entry=parent_entry(part);path=native.bound(ROOT,entry['owner_source']);spec=importlib.util.spec_from_file_location('_chipmunk_p64_bound_owner_'+str(part),path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def check_verification(result,parent_id):
 native.need(result.get('status')=='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_P64_VIDEO_V2' and result.get('id')==parent_id,'actual V2 exited verifier required')

def parent_data(hf,part):
 entry=parent_entry(part);actual_owner=parent_owner(part);protocol=ROOT/entry['protocol']['path'];p=native.read(protocol);actual_owner.validate(p,True)
 out=Path(hf)/p['output_HF_relative'];v=out.parent.parent/'verification/worker/results/RUN.json';vx=v.parent.parent/'EXIT_RECEIPT.json';run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json'
 x=actual_owner.inspect_verification(SimpleNamespace(protocol=protocol));check_verification(x,p['id'])
 return {'protocol':native.binding(protocol,ROOT),'RUN':native.binding(run,hf),'exit':native.binding(ex,hf),'verification_RUN':native.binding(v,hf),'verification_exit':native.binding(vx,hf)}
def paired_initial_noise(phases):
 keys=('raw_noise_FP32','initial_noise'); fields=('sha256','bytes','raw_tensor_sha256','shape','dtype')
 for k in keys:
  native.need(all(phases[0]['states'][k][f]==phases[3]['states'][k][f] for f in fields),'actual paired initial noise differs')

def closed_rows(proof,hf,part):
 hf=Path(hf).resolve();native.need(proof==parent_data(hf,part),'parent identity changed');p=native.read(native.bound(ROOT,proof['protocol']));r=native.read(native.bound(hf,proof['RUN']));e=native.read(native.bound(hf,proof['exit']));v=native.read(native.bound(hf,proof['verification_RUN']))
 actual_owner=parent_owner(part);actual_owner.vm.exited(e,proof['RUN'],proof['protocol']['sha256'],p['id']);native.need(r['status']=='COMPLETE_CHIPMUNK_P64_VIDEO_V2_EXIT_PENDING' and r['source_ordinal']==ORDINALS[part] and r['seed']==42 and r['same_device_BF16_neutral'] is True and v['RUN']==proof['RUN'] and v['exit']==proof['exit'] and v['parent_scope_id']==p['id'] and v['predebited_bytes']==v['completed_bytes'] and v['inflight']=='NONE','producer or verifier identity')
 phases=r['phases'];native.need([x['mode'] for x in phases]==list(actual_owner.driver.MODES),'phase identity');paired_initial_noise(phases);rows=[]
 for idx in (0,3):
  phase=phases[idx];video=phase['video'];path=Path(video['path']).resolve();native.need(path.is_relative_to(hf) and path.is_file() and path.stat().st_size==video['bytes'],'missing parent video');binding={'path':str(path.relative_to(hf)),'bytes':video['bytes'],'sha256':video['sha256']};rows.append({'ordinal':ORDINALS[part],'mode':phase['mode'],'case':p['case'],'video':binding,'generation':proof['RUN'],'states':phase['states'],'run_id':phase['run_id']})
 native.need(len({x['video']['path'] for x in rows})==2,'aliased videos');return rows

def contract():
 py=ROOT/'.venv/bin/python';native.need(py.is_file(),'byte verifier interpreter missing')
 return {'python_path':str(py),'python_sha256':native.sha(py),'source':'tools/e0_chipmunk_p64_quality_lineage.py','argv':[str(py),'-I','-B',str(ROOT/'tools/e0_chipmunk_p64_quality_lineage.py'),'--verify-child'],'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,'child_CPUs':'inherited_from_metric_worker','child_sha_accounting':'nested parent protocol and receipt hashes occur inside child, outside parent HashReads journal; actual calls and bytes UNKNOWN','CUDA_VISIBLE_DEVICES':'','numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}

def verify_child(proof,hf,part,ledger):
 import os,subprocess
 c=contract();request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part};env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')};env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def call():
  result=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=300,check=False);native.need(result.returncode==0,'parent byte verifier failed: '+result.stderr[-2000:]);r=json.loads(result.stdout);native.need(r.get('request')==request and r.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and r.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'child identity differs');return r['rows']
 return ledger.call(call)

if __name__=='__main__':
 import os
 native.need(sys.argv[1:]==['--verify-child'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'child isolation differs');request=json.load(sys.stdin);rows=closed_rows(request['proof'],Path(request['hf']),request['part']);native.need(not {'torch','numpy'}&set(sys.modules),'child numerical import');print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))
