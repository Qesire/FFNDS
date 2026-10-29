"""Build private exact jobs from a closed-parent frozen quality protocol; never launch."""
import argparse,hashlib,json,os
from pathlib import Path
import e0_chipmunk_p64_quality_execution as execution
ROOT=Path(__file__).resolve().parents[1]
HF=Path('/home/xiang/.cache/huggingface')
PROFILE=Path('/home/xiang/.config/ffnds/worker_causal_idle_a100_20261029.json')
SOURCES={'quality':'tools/e0_chipmunk_p64_official_quality.py','raft':'tools/e0_chipmunk_p64_continuous_raft.py','visual':'tools/e0_chipmunk_p64_visual_cpu.py'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(condition,message):
 if not condition:raise ValueError(message)
def jobs(kind,plan,protocol,profile,uuid,job_paths):
 need(kind in SOURCES and plan['status']=='FROZEN_PENDING_REGISTRATION' and plan['assets_finalized'] is True,'actual frozen parent-bound plan required')
 need(profile.is_absolute() and sha(profile)==execution.PROFILE_SHA and plan['execution_private_worker_config_sha256']==execution.PROFILE_SHA,'profile identity differs')
 part=plan.get('part',plan.get('shard'));need(type(part) is int and part in (0,1),'part differs')
 if kind!='visual':need(plan['execution_device']['physical_gpu_index']==execution.physical_gpu(kind,part) and isinstance(uuid,str) and uuid.startswith('GPU-'),'private GPU assignment differs')
 out=HF/plan['output_HF_relative'];digest=sha(protocol);src=ROOT/SOURCES[kind];cpus=execution.affinity(kind,part);cpuarg=str(cpus[0]) if len(cpus)==1 else f'{cpus[0]}-{cpus[-1]}'
 common=['env','FFNDS_WORKER_CONFIG='+str(profile),f'OMP_NUM_THREADS={len(cpus)}',f'MKL_NUM_THREADS={len(cpus)}',f'OPENBLAS_NUM_THREADS={len(cpus)}','PATH=/usr/bin:/bin']
 python=ROOT/'.venv/bin/python' if kind=='visual' else HF/plan['import_environment']['python_HF_relative'];flags=['-I','-B'] if kind=='visual' else ['-I','-S','-B'];child=['taskset','-c',cpuarg,str(python),*flags,str(src)]
 def spec(phase,argv,worker,seconds,identifier=None):return {'id':identifier or plan['id'],'protocol_path':str(protocol),'protocol_sha256':digest,'cwd':str(ROOT),'argv':argv,'output_dir':str(worker),'hf_root':str(HF),'run_json':str(worker/'results/RUN.json'),'external_timeout_seconds':seconds}
 def tail(job):return ['--protocol',str(protocol),'--hf-root',str(HF),'--job-spec',str(job)]
 result={}
 if kind=='visual':
  worker=out.parent;result['run']=spec('run',common+['CUDA_VISIBLE_DEVICES=']+child+['run']+tail(job_paths['run']),worker,plan['resources']['outer_seconds'])
 else:
  worker=out.parent.parent/'cpu_preflight/worker';result['cpu']=spec('cpu',common+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(worker/'results/RUN.json')]+tail(job_paths['cpu']),worker,plan['CPU_preflight_resources']['outer_seconds'])
  worker=out.parent;result['gpu']=spec('gpu',common+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),str(execution.physical_gpu(kind,part)),uuid]+child+['run']+tail(job_paths['gpu']),worker,plan['resources']['outer_seconds'])
 worker=out.parent.parent/'verification/worker';vcpu=cpus[0];verify_python=ROOT/'.venv/bin/python' if kind=='visual' else python;vflags=['-I','-B'] if kind=='visual' else ['-I','-S','-B'];vcommon=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(vcpu),str(verify_python),*vflags,str(ROOT/'tools/e0_chipmunk_p64_metric_verify.py'),'verify','--kind',kind];result['verify']=spec('verify',vcommon+tail(job_paths['verify']),worker,1920,plan['id']+'_VERIFY')
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--kind',choices=tuple(SOURCES),required=True);a.add_argument('--protocol',type=Path,required=True);a.add_argument('--gpu-uuid-file',type=Path);a.add_argument('--private-dir',type=Path,required=True);x=a.parse_args();protocol=x.protocol.resolve();plan=json.loads(protocol.read_text());
 import e0_chipmunk_p64_metric_adapter as adapter
 engine=adapter.install(x.kind) if x.kind!='visual' else __import__('e0_chipmunk_p64_visual_cpu');engine.validate(plan,True) if x.kind!='visual' else engine.validate(plan,HF,True)
 part=plan.get('part',plan.get('shard'));parent=plan['parent'] if x.kind=='visual' else plan['parents'][0];need(adapter.line.parent_data(HF,part)==parent and adapter.line.closed_rows(parent,HF,part)==plan['videos'],'actual closed parent changed')
 uuid=x.gpu_uuid_file.read_text().strip() if x.gpu_uuid_file else None;d=x.private_dir.resolve();need(d.is_relative_to(ROOT/'.runtime') and not d.exists(),'fresh private directory required');names=('run','verify') if x.kind=='visual' else ('cpu','gpu','verify');paths={n:d/(n.upper()+'_JOB.json') for n in names};result=jobs(x.kind,plan,protocol,PROFILE,uuid,paths);d.mkdir(parents=True,mode=0o700)
 for n,j in result.items():
  paths[n].write_text(json.dumps(j,indent=2)+'\n');os.chmod(paths[n],0o600)
 print(json.dumps({'status':'PRIVATE_SPECS_PREPARED_NOT_AUTHORIZED','job_sha256':{n:sha(p) for n,p in paths.items()}}))
if __name__=='__main__':main()
