"""Stage-4 custom VBench batch guard. No job submission, no built-in bypass.

Requires one independently reviewed frozen scope and ACTIVE record, exact code,
real generation proof, source/model closure and a real Slurm GPU allocation.
"""
import argparse,hashlib,importlib.util,json,os,re,signal,sys,time,traceback
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def write(path,data):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main(scope_path,scope_sha):
    scope_path=Path(scope_path).resolve();stage=scope_path.parent
    if sha(scope_path)!=scope_sha:raise RuntimeError('scope identity mismatch')
    scope=json.loads(scope_path.read_text(encoding='utf8'))
    if scope['phase']!='FFNDS_STAGE4_QUALITY' or scope['status']!='REVIEW_CANDIDATE':
        raise RuntimeError('incorrect reviewed quality scope')
    review_path=Path(scope['review_path']).resolve();active_path=Path(scope['active_path']).resolve()
    if not review_path.is_relative_to(stage) or not active_path.is_relative_to(stage):
        raise RuntimeError('review/ACTIVE outside frozen stage')
    review=json.loads(review_path.read_text(encoding='utf8'));active=json.loads(active_path.read_text(encoding='utf8'))[scope['active_key']]
    if (review['scope_sha256']!=scope_sha or review['recheck_disposition']!='APPROVE' or
        not review.get('reviewer_agent_id') or review['reviewer_agent_id'] in scope['authors'] or
        active['scope_sha256']!=scope_sha or active['review_sha256']!=sha(review_path) or not active['runnable']):
        raise RuntimeError('independent review or ACTIVE identity mismatch')
    local=Path(__file__).resolve().parent
    required={'quality_guard.py','quality_vbench_batch.py','quality_pair_report.py'}
    code=scope['implementation_sha256']
    if not {str(local/x) for x in required}.issubset(code):raise RuntimeError('quality implementation identities missing')
    for path,digest in code.items():
        if sha(path)!=digest:raise RuntimeError('quality implementation changed: '+path)
    contract_path=Path(scope['paired_contract_path']).resolve()
    if sha(contract_path)!=scope['paired_contract_sha256']:raise RuntimeError('paired contract mismatch')
    contract=json.loads(contract_path.read_text(encoding='utf8'))
    if contract['generation_seed']!=42 or scope['metric_seed']!=contract['metric_seed']:
        raise RuntimeError('fixed seeds mismatch')
    if scope['dimension']=='camera_motion':
        if any(r['source'] not in contract['explicit_camera_instruction_by_source'] for r in scope['records']):
            raise RuntimeError('camera axis not applicable for one or more sources')
    protocol_path=Path(scope['metric_protocol_path']).resolve()
    if sha(protocol_path)!=scope['metric_protocol_sha256']:raise RuntimeError('metric protocol identity mismatch')
    protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    for key in ('dimension','metric_seed','runtime_versions','metric_package_versions','metric_asset_sha256','metric_code_sha256',
                'quality_reference_sha256','ffmpeg_path','ffmpeg_sha256','vbench_working_directory',
                'metric_project_root','metric_cache_root','metric_site_packages','metric_additional_source_roots'):
        if protocol[key]!=scope[key]:raise RuntimeError('metric protocol differs: '+key)
    verifier_path=Path(scope['generation_verifier_path']).resolve()
    if str(verifier_path) not in code or sha(verifier_path)!=scope['generation_verifier_sha256']:
        raise RuntimeError('generation verifier identity mismatch')
    module_spec=importlib.util.spec_from_file_location('reviewed_generation_proof',verifier_path)
    verifier=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(verifier)
    if not callable(getattr(verifier,'verify',None)):raise RuntimeError('generation verifier entry unavailable')
    ref=Path(scope['quality_reference_dir']).resolve()
    for name in ('runner_metrics.py','metric_contract.py','metric_loaders.py','runtime_contract.py'):
        if sha(ref/name)!=scope['quality_reference_sha256'][name]:raise RuntimeError('metric reference changed: '+name)
    guard_path=Path(scope['runtime_guard_path']).resolve()
    if sha(guard_path)!=scope['runtime_guard_sha256']:raise RuntimeError('runtime guard identity mismatch')
    sys.path.insert(0,str(guard_path.parent))
    from runtime_guard import require_compute_allocation,process_peak_rss_bytes
    allocation=require_compute_allocation()
    scheduler=dict(re.findall(r'(\w+)=([^\s]+)',allocation['scheduler_job_record']))
    frozen=scope['scheduler']
    if (scheduler.get('Partition')!=frozen['partition'] or scheduler.get('Account')!=frozen['account']
            or scheduler.get('QOS')!=frozen['qos']):raise RuntimeError('scheduler allocation differs from reviewed scope')
    sys.path.insert(0,str(ref))
    from runtime_contract import validate_allocation
    validate_allocation(scope,allocation)
    if not Path(sys.executable).samefile(scope['python_executable']) or sha(sys.executable)!=scope['python_executable_sha256']:
        raise RuntimeError('metric Python identity mismatch')
    if scope['budget']['gpus_per_job']!=1 or scope['budget']['max_jobs']!=1 or not 0<scope['budget']['soft_seconds']<scope['budget']['wall_seconds']:
        raise RuntimeError('metric budget invalid')
    receipt=Path(scope['receipt_dir']).resolve();out=Path(scope['output']).resolve()
    if not receipt.is_relative_to(stage) or not out.is_relative_to(stage) or receipt==out:
        raise RuntimeError('output/receipt outside reviewed stage or overlap')
    if out.exists():raise FileExistsError(out)
    receipt.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();write(receipt/'RUN.json',{'scope_sha256':scope_sha,'job_id':os.environ['SLURM_JOB_ID'],'allocation':allocation})
    def terminate(signum,frame):raise TimeoutError(f'termination signal {signum}')
    oldsignal=signal.signal(signal.SIGTERM,terminate)
    def check_resources():
        peak=process_peak_rss_bytes()
        if peak is not None and peak>scope['budget']['host_memory_gib']*1024**3:raise MemoryError('host limit')
        if time.monotonic()-started>scope['budget']['soft_seconds']:raise TimeoutError('metric soft deadline')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())>scope['budget']['output_bytes']:
            raise RuntimeError('metric output bytes exceeded')
    result=None;code_exit=1;exception=None
    try:
        from quality_vbench_batch import run
        result=run(scope,scope_sha256=scope_sha,authorize=lambda _:None,
                   verify_generation=verifier.verify,check_resources=check_resources)
        check_resources();code_exit=0
    except BaseException as exc:
        exception=repr(exc);traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,oldsignal)
        write(receipt/'EXIT.json',{'scope_sha256':scope_sha,'job_id':os.environ['SLURM_JOB_ID'],
                                  'exit_code':code_exit,'result':result,'exception':exception,
                                  'elapsed_seconds':time.monotonic()-started})
    if code_exit:raise SystemExit(code_exit)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('scope_sha256');a=p.parse_args()
    main(a.scope,a.scope_sha256)


