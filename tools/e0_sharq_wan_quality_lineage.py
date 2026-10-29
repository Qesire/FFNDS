"""Actual exited SharQ full160 + independently exited verifier lineage.

No model/GPU import, no implicit artifact rehash. The consuming scope must
predebit and verify each video/state/image byte binding before metric work.
"""
import json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_sharq_wan_all160_video_owner_v2 as owner
native=owner.native
ORDINALS=(1040,979);MODES=('dense','dense32','sharq')
PARENT='config/e0_sharq_wan_all160_video_v2_part{}_20261030_frozen_pending_v3.json'

def parent_data(hf,part):
 native.need(type(part) is int and part in (0,1),'invalid parent part')
 protocol=ROOT/PARENT.format(part);p=native.read(protocol);owner.validate(p,True)
 out=Path(hf)/p['output_HF_relative'];verification=out.parent.parent/'verification/worker/results/RUN.json';vx=verification.parent.parent/'EXIT_RECEIPT.json';run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json'
 result=owner.inspect_verification(SimpleNamespace(protocol=protocol));native.need(result['status']=='PASS_ACTUAL_EXIT_CLOSED_SHARQ_WAN_ALL160_VIDEO','actual exited verifier required')
 return {'protocol':native.binding(protocol,ROOT),'RUN':native.binding(run,hf),'exit':native.binding(ex,hf),'verification_RUN':native.binding(verification,hf),'verification_exit':native.binding(vx,hf)}

def closed_rows(proof,hf,part):
 hf=Path(hf).resolve();native.need(proof==parent_data(hf,part),'parent identity changed');p=native.read(native.bound(ROOT,proof['protocol']));r=native.read(native.bound(hf,proof['RUN']));e=native.read(native.bound(hf,proof['exit']));v=native.read(native.bound(hf,proof['verification_RUN']))
 native.need(v['RUN']==proof['RUN'] and v['exit']==proof['exit'] and v['parent_scope_id']==p['id'] and v['completed_bytes']==v['predebited_bytes'] and v['inflight']=='NONE','verification did not bind actual generation')
 owner.evidence.verify_record(p,r,e,proof['RUN']['sha256'],proof['protocol']['sha256'])
 phases=r['phases'];native.need([x['mode'] for x in phases]==['bf16','bf16_cache','dense32','sharq'] and r.get('current_phase',{}).get('status')=='EVIDENCE_COMPLETE' and r.get('current_state') is None,'V2 persisted phase identity differs');native.need(all(all(r['operation_ledger'][side].get(k)==n for k,n in (('sharq_stream_bridge_calls',800),('sharq_bridge_event_records',1600),('sharq_bridge_event_waits',1600))) for side in ('predebited','observed')),'V2 bridge ledger differs');rows=[]
 for mode,index in zip(MODES,(0,2,3)):
  phase=phases[index];video=phase['video'];path=Path(video['path']).resolve();native.need(path.is_relative_to(hf) and path.is_file() and path.stat().st_size==video['bytes'],'missing parent video')
  binding={'path':str(path.relative_to(hf)),'bytes':video['bytes'],'sha256':video['sha256']}
  rows.append({'ordinal':ORDINALS[part],'mode':mode,'case':p['case'],'video':binding,'generation':proof['RUN'],'states':phase['states'],'run_id':phase['run_id']})
 native.need(len({x['video']['path'] for x in rows})==3,'aliased videos');return rows

def contract():
 py=ROOT/'.venv/bin/python';native.need(py.is_file(),'byte verifier interpreter missing')
 return {'python_path':str(py),'python_sha256':native.sha(py),'source':'tools/e0_sharq_wan_quality_lineage.py','argv':[str(py),'-I','-B',str(ROOT/'tools/e0_sharq_wan_quality_lineage.py'),'--verify-child'],'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,'child_CPUs':'inherited_from_metric_worker','child_sha_accounting':'nested parent protocol and receipt hashes occur inside child, outside parent HashReads journal; actual calls and bytes UNKNOWN','CUDA_VISIBLE_DEVICES':'','numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}

def verify_child(proof,hf,part,ledger):
 import os,subprocess
 c=contract();request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part};env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')};env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def call():
  result=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=300,check=False);native.need(result.returncode==0,'parent byte verifier failed: '+result.stderr[-2000:]);r=json.loads(result.stdout);native.need(r.get('request')==request and r.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and r.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'child identity differs');return r['rows']
 return ledger.call(call)

if __name__=='__main__':
 import os
 native.need(sys.argv[1:]==['--verify-child'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'child isolation differs');request=json.load(sys.stdin);rows=closed_rows(request['proof'],Path(request['hf']),request['part']);native.need(not {'torch','numpy'}&set(sys.modules),'child numerical import');print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))
