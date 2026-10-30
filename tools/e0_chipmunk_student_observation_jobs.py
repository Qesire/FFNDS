"""Prepare private exact supervisor specs after reviewable source freeze; no launch."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_student_observation_owner as owner

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,message):
 if not value:raise ValueError(message)
def jobs(p,protocol,profile,local,gpu,uuid,paths):
 need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True,'frozen protocol required');owner.validate(p,True);need(profile.is_absolute() and sha(profile)==p['execution_private_worker_config_sha256'] and gpu in range(4) and uuid.startswith('GPU-'),'worker or device differs')
 out=owner.HF/p['output_HF_relative'];digest=sha(protocol);aff=','.join(map(str,p['cpu_affinity']));common=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin'];child=['taskset','-c',aff,str(owner.PYTHON),'-I','-B',str(ROOT/owner.SELF)]
 def spec(identifier,argv,worker,seconds):return {'id':identifier,'protocol_path':str(protocol),'protocol_sha256':digest,'cwd':str(ROOT),'argv':argv,'output_dir':str(worker),'hf_root':str(owner.HF),'run_json':str(worker/'results/RUN.json'),'external_timeout_seconds':seconds}
 def tail(job):return ['--protocol',str(protocol),'--hf-root',str(owner.HF),'--local-root',str(local),'--job-spec',str(job)]
 cpu=out.parent.parent/'cpu_preflight/worker';gpuworker=out.parent;verifier=out.parent.parent/'verification/worker'
 first=spec(p['id'],common+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(cpu/'results')]+tail(paths['cpu']),cpu,owner.CPU['outer_seconds'])
 second=spec(p['id'],common+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),str(gpu),uuid]+child+['run']+tail(paths['gpu']),gpuworker,owner.RES['outer_seconds'])
 venv=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(p['cpu_affinity'][0]),str(owner.PYTHON),'-I','-B',str(ROOT/owner.SELF)]
 third=spec(p['id']+'_VERIFY',venv+['verify-closed','--protocol',str(protocol),'--hf-root',str(owner.HF),'--job-spec',str(paths['verify']),'--run',str(out/'RUN.json'),'--exit',str(out.parent/'EXIT_RECEIPT.json'),'--verify-out',str(verifier/'results/RUN.json')],verifier,owner.VERIFY['outer_seconds'])
 return {'cpu':first,'gpu':second,'verify':third}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--worker-profile',type=Path,required=True);ap.add_argument('--local-root',type=Path,required=True);ap.add_argument('--gpu-index',type=int,required=True);ap.add_argument('--uuid-file',type=Path,required=True);ap.add_argument('--private-dir',type=Path,required=True);x=ap.parse_args();d=x.private_dir.resolve();need(d.is_relative_to(ROOT/'.runtime') and not d.exists(),'fresh private directory required');q=x.protocol.resolve();p=json.loads(q.read_text());paths={n:d/(n.upper()+'_JOB.json') for n in ('cpu','gpu','verify')};specs=jobs(p,q,x.worker_profile.resolve(),x.local_root.resolve(),x.gpu_index,x.uuid_file.read_text().strip(),paths);d.mkdir(parents=True,mode=0o700)
 for name,spec in specs.items():paths[name].write_text(json.dumps(spec,indent=2)+'\n');os.chmod(paths[name],0o600)
 print(json.dumps({'status':'PRIVATE_SPECS_PREPARED_NOT_AUTHORIZED','job_sha256':{n:sha(path) for n,path in paths.items()}}))
if __name__=='__main__':main()
