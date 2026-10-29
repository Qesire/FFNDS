"""Private per-part quality launcher verification; no device query at import."""
import os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILE_SHA='5a4e5bf320c525deb8f952e22972fd2f8bc8c35fe8f66dcea1db1e64ef8a90d2'
MONITOR='/usr/bin/nvidia-smi'
HELPERS={'tools/e0_chipmunk_input_video_quality_execution.py','tests/test_e0_chipmunk_input_video_quality_execution.py',
 'tools/durable_worker_job.py','tools/worker_identity.py','scripts/worker_teal_assigned_uuid_exec.sh'}

def need(v,msg):
 if not v:raise ValueError(msg)

def affinity(kind,part):
 need(type(part) is int and part in (0,1,2,3),'Part differs')
 mapping={'quality':([0,1],[2,3],[4,5],[6,7]),'raft':([8],[9],[10],[11]),'visual':([12,13],[14,15],[16,17],[18,19])}
 need(kind in mapping,'Kind differs');return list(mapping[kind][part])

def physical_gpu(kind,part):
 need(type(part) is int and part in (0,1,2,3) and kind in ('quality','raft'),'Kind/part differs')
 return part

def monitor_binding(base):
 p=Path(MONITOR);need(p.is_file(),'Monitor missing')
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':base.sha(p)}

def require_review(plan,scope,base,fingerprint,keys):
 r=base.read_bound(ROOT,scope['review'])
 need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==plan['id']
  and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE'
  and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id']
  and r.get('responses_to_review') and r.get('adversarial_counterexample') and r.get('approval_scope_fingerprint')==fingerprint(plan)
  and r.get('critical_source_hashes')=={b['path']:b['sha256'] for b in plan['repo_sources']}
  and all(r.get(k)==scope.get(k) and isinstance(scope.get(k),str) and len(scope[k])==64 for k in keys)
  and scope.get('session_max_GPUs')==4,'Exact independently reviewed jobs/session scope required')
 for key in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,r[key])
 return r

def verify_job(args,plan,scope,base,kind,phase,source,limits):
 need(args.job_spec is not None,'Private job spec required')
 key='launch_spec_sha256' if kind=='visual' else 'cpu_preflight_spec_sha256' if phase=='preflight' else 'gpu_launch_spec_sha256'
 digest=base.sha(args.protocol);need(base.sha(args.job_spec)==scope[key],'Private spec SHA differs')
 job=base.read(args.job_spec);hf=args.hf_root.resolve();part=plan.get('part',plan.get('shard'));cpus=affinity(kind,part)
 need(sorted(os.sched_getaffinity(0))==cpus,'Actual CPU affinity differs')
 need(job.get('id')==plan['id'] and job.get('protocol_sha256')==digest and Path(job['protocol_path']).resolve()==args.protocol.resolve()
  and Path(job['cwd']).resolve()==ROOT and Path(job['hf_root']).resolve()==hf,'Job protocol/paths differ')
 argv=job['argv'];need(len(argv)>2 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Profile argument missing')
 profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and base.sha(profile)==PROFILE_SHA
  and plan.get('execution_private_worker_config_sha256')==PROFILE_SHA,'Worker profile differs')
 threads=len(cpus);common=['env',argv[1],f'OMP_NUM_THREADS={threads}',f'MKL_NUM_THREADS={threads}',f'OPENBLAS_NUM_THREADS={threads}','PATH=/usr/bin:/bin']
 python=ROOT/'.venv/bin/python' if kind=='visual' else hf/plan['import_environment']['python_HF_relative']
 if kind!='visual':need(python.is_file() and base.sha(python)==plan['import_environment']['python_sha256'],'Original metric interpreter differs')
 flags=['-I','-B'] if kind=='visual' else ['-I','-S','-B']
 if kind=='visual':need(sys.flags.isolated,'visual interpreter isolation required')
 cpu=['taskset','-c',str(cpus[0]) if len(cpus)==1 else f'{cpus[0]}-{cpus[-1]}',str(python),*flags,str(ROOT/source)]
 out=(hf/plan['output_HF_relative']).resolve();tail=['--protocol',str(args.protocol.resolve()),'--hf-root',str(hf),'--job-spec',str(args.job_spec.resolve())]
 if kind=='visual':
  expected=common+['CUDA_VISIBLE_DEVICES=']+cpu+['run']+tail;worker=out.parent;run=out/'RUN.json';outer=limits['outer_seconds']
 elif phase=='preflight':
  worker=out.parent.parent/'cpu_preflight'/'worker';run=worker/'results/RUN.json'
  need(args.out.resolve()==run,'Preflight report path differs');expected=common+['CUDA_VISIBLE_DEVICES=']+cpu+['preflight','--out',str(run)]+tail;outer=limits['outer_seconds']
 else:
  assigned=physical_gpu(kind,part);need(plan['execution_device'].get('physical_gpu_index')==assigned and len(argv)>len(common)+3 and argv[len(common)+2]==str(assigned),'GPU index differs')
  uuid=argv[len(common)+3];need(uuid.startswith('GPU-') and uuid==os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('FFNDS_PHYSICAL_GPU')==str(assigned),'GPU UUID differs')
  monitor=plan['execution_monitor_binary'];path=Path(MONITOR)
  need(shutil.which('nvidia-smi')==MONITOR and monitor==monitor_binding(base) and path.is_file(),'Actual monitor differs')
  expected=common+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),str(assigned),uuid]+cpu+['run']+tail;worker=out.parent;run=out/'RUN.json';outer=limits['outer_seconds']
 need(argv==expected and Path(job['output_dir']).resolve()==worker and Path(job['run_json']).resolve()==run
  and job['external_timeout_seconds']==outer,'Exact job chain/receipt differs')
 return job
