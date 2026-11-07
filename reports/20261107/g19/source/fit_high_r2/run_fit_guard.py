"""Finite independently reviewed fit job entry; stdlib before allocation guard."""
import argparse,importlib.metadata,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from runtime_guard import require_compute_allocation,RunReceipt,sha256_file

CODE={'run_fit_guard.py','runtime_guard.py','runner_fit.py','fit_core.py','job_fit.sh'}
def output_bytes(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass  # concurrent atomic rename by the child
    return total

def verify(stage,digest,allocation):
    scope_path=stage/'scope.json'
    if not re.fullmatch('[a-f0-9]{64}',digest) or sha256_file(scope_path)!=digest:raise RuntimeError('scope SHA')
    s=json.loads(scope_path.read_text())
    if s['phase']!='G19_STRENGTHEN_FIT_ALL80' or s['status']!='REVIEW_CANDIDATE':raise RuntimeError('scope phase')
    if s['expert'] not in ('high','low') or s['layers']!=list(range(40)):raise RuntimeError('finite expert/layer scope')
    if s['scope_id']!=f"G19_STRENGTHEN_FIT_{s['expert'].upper()}_R2":raise RuntimeError('scope id')
    if 'old_fit_manifest.json' not in s['frozen_inputs_sha256']:raise RuntimeError('old fit manifest not frozen')
    if set(s['code_sha256'])!=CODE:raise RuntimeError('all executable sources must be bound')
    for name,d in {**s['code_sha256'],**s['frozen_inputs_sha256']}.items():
        p=(stage/name).resolve()
        if not p.is_relative_to(stage) or sha256_file(p)!=d:raise RuntimeError('bundle identity '+name)
    review=json.loads((stage/'INDEPENDENT_REVIEW.json').read_text())
    if (review.get('scope_sha256')!=digest or review.get('disposition')!='APPROVE' or review.get('recheck_disposition')!='APPROVE'
        or not review.get('reviewer_agent_id') or review['reviewer_agent_id'] in s['authors']
        or review.get('root_self_audit_sha256')!=sha256_file(stage/'ROOT_SELF_AUDIT.txt')):raise RuntimeError('exact independent review')
    active=json.loads((stage/'ACTIVE_RESEARCH.json').read_text()); entry=active.get('scopes',{}).get(s['scope_id'],{})
    if (active.get('state')!='SCOPE_FROZEN_IN_ACTIVE' or active.get('session_max_gpus')!=2 or active.get('max_concurrent_jobs')!=2
        or entry.get('state')!='AUTHORIZED_FOR_ONE_JOB' or entry.get('max_jobs')!=1 or entry.get('scope_sha256')!=digest
        or entry.get('review_sha256')!=sha256_file(stage/'INDEPENDENT_REVIEW.json')):raise RuntimeError('ACTIVE identity')
    b=s['budget']
    if tuple(b[k] for k in ('gpus_per_job','max_jobs','cpus','host_memory_gib','wall_seconds','soft_seconds'))!=(1,1,4,192,10800,10680):raise RuntimeError('budget')
    if Path(s['output_dir'])!=stage/'output' or stage.name!=f"fit_{s['expert']}_r2":raise RuntimeError('output identity')
    if allocation['hostname'].split('.')[0]!='anode21':raise RuntimeError('node identity')
    if not re.search(r'(?:^|\s)TimeLimit=0?3:00:00(?:\s|$)',allocation['scheduler_job_record']):raise RuntimeError('time allocation')
    if os.path.realpath(sys.executable)!=os.path.realpath(s['python']):raise RuntimeError('python executable')
    for name,expected in s['package_versions'].items():
        if importlib.metadata.version(name)!=expected:raise RuntimeError('package version '+name)
    import shutil
    if shutil.disk_usage(stage).free<b['min_free_disk_gib']*1024**3:raise RuntimeError('disk budget')
    s['_verified_scope_sha256']=digest
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--scope-sha',required=True);a=p.parse_args()
    stage=Path(a.stage).resolve()
    if Path.cwd().resolve()!=stage:raise RuntimeError('cwd')
    allocation=require_compute_allocation();s=verify(stage,a.scope_sha,allocation)
    with RunReceipt(s['output_dir'],s,allocation) as rec:
        cmd=[sys.executable,str(stage/'runner_fit.py'),'--assets',str(stage/'assets.json'),'--expert',s['expert'],'--output',str(rec.path/'fit'),'--config',str(stage/'fit_config.json'),'--old-fit-manifest',str(stage/'old_fit_manifest.json')]
        child=subprocess.Popen(cmd,cwd=stage,start_new_session=True)
        child_peak_rss=0
        try:
            while child.poll() is None:
                rec.check_deadline()
                used=output_bytes(rec.path)
                if used>s['budget']['total_output_bytes']:raise RuntimeError('output size budget during child run')
                proc_status=Path(f'/proc/{child.pid}/status')
                try:
                    match=re.search(r'^VmHWM:\s+(\d+)\s+kB$',proc_status.read_text(),re.M)
                    if match:child_peak_rss=max(child_peak_rss,int(match[1])*1024)
                except FileNotFoundError:pass
                rec.result['child_peak_rss_bytes_observed_proc']=child_peak_rss
                if child_peak_rss>s['budget']['host_memory_gib']*1024**3:raise MemoryError('FIT_CHILD_HOST_MEMORY_BUDGET')
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,cmd)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        rec.check_deadline()
        summary=rec.path/'fit'/'SUMMARY.json'
        if not summary.is_file():raise RuntimeError('missing complete fit summary')
        result=json.loads(summary.read_text())
        if result.get('expert')!=s['expert'] or result.get('completed_layers')!=list(range(40)):raise RuntimeError('incomplete fit')
        used=output_bytes(rec.path)
        if used>s['budget']['total_output_bytes']:raise RuntimeError('output size budget')
        rec.result.update(status='FIT_ALL_40_COMPLETE_NEEDS_REVIEW',expert=s['expert'],summary_sha256=sha256_file(summary),output_bytes=used)

if __name__=='__main__':main()
