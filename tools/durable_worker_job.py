"""Detached, single-attempt worker supervisor with actual child exit receipt.

Spec and logs are private. No GPU scheduling is performed here; the child owns
its registered UUID lock. start requires a previously registered ACTIVE scope.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from worker_identity import require_worker


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def ticks(pid):
    """Linux /proc starttime, robust to spaces and parentheses in comm."""
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        return int(text[text.rfind(')')+2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, startticks):
    return type(pid) is int and type(startticks) is int and ticks(pid)==startticks


def exclusive_json(path, value):
    # Receipt becomes visible only after its complete bytes are durably written.
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def validate(spec_path, expected_sha, *, active_path=None, require_active=True):
    spec_path=Path(spec_path).resolve()
    if digest(spec_path)!=expected_sha: raise ValueError('Spec SHA differs')
    s=json.loads(spec_path.read_text()); required={'id','protocol_path','protocol_sha256','cwd','argv','output_dir','hf_root','run_json','external_timeout_seconds'}
    if not required <= set(s) or not isinstance(s['id'],str) or not s['id']: raise ValueError('Spec fields missing')
    cwd=Path(s['cwd']).resolve(); hf=Path(s['hf_root']).resolve(); out=Path(s['output_dir']).resolve(); run=Path(s['run_json']).resolve()
    if not cwd.is_dir() or not hf.is_dir() or out==hf or not out.is_relative_to(hf) or not run.is_relative_to(out): raise ValueError('cwd/output/run confinement differs')
    protocol=Path(s['protocol_path']).resolve()
    if digest(protocol)!=s['protocol_sha256']: raise ValueError('Protocol SHA differs')
    if not isinstance(s['argv'],list) or not s['argv'] or not all(isinstance(v,str) and v for v in s['argv']): raise ValueError('Invalid argv')
    if type(s['external_timeout_seconds']) not in (int,float) or not 0<s['external_timeout_seconds']<=86400: raise ValueError('Invalid external timeout')
    if not require_active: return s
    active=Path(active_path) if active_path else cwd/'research/ACTIVE_RESEARCH.json'
    registry=json.loads(active.read_text()); scopes=registry.get('experiment_authority',{}).get('current_user_authorized_scopes',[])
    matches=[x for x in scopes if x.get('id')==s['id'] and x.get('protocol_sha256')==s['protocol_sha256'] and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')]
    auth=registry.get('user_continuation_authorization_20261027',{})
    if len(matches)!=1 or auth.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION': raise ValueError('ACTIVE exact scope not authorized')
    return s


def start(spec_path, expected_sha):
    s=validate(spec_path,expected_sha); out=Path(s['output_dir']).resolve()
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    exclusive_json(out/'START_CLAIM.json',{'id':s['id'],'spec_sha256':expected_sha,'protocol_sha256':s['protocol_sha256']})
    try:
        with (out/'supervisor.stdout.log').open('xb') as stdout, (out/'supervisor.stderr.log').open('xb') as stderr:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise','--spec',str(Path(spec_path).resolve()),'--spec-sha',expected_sha],
                                     cwd=s['cwd'],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
        startticks=ticks(process.pid)
        exclusive_json(out/'START_RECEIPT.json',{'id':s['id'],'spec_sha256':expected_sha,'supervisor_pid':process.pid,'supervisor_startticks':startticks})
        return {'id':s['id'],'supervisor_pid':process.pid,'supervisor_startticks':startticks}
    except BaseException as e:
        exclusive_json(out/'START_FAILURE.json',{'id':s['id'],'spec_sha256':expected_sha,'error_type':type(e).__name__,'automatic_retry':False})
        raise


def supervise(spec_path, expected_sha, *, worker_check=require_worker, active_path=None, kill_grace=30):
    """Run once. Injected worker_check/active_path are for direct CPU unit tests only.

    The CLI never exposes injection or a test-mode bypass.
    """
    s=validate(spec_path,expected_sha,active_path=active_path);out=Path(s['output_dir']).resolve()
    claim=json.loads((out/'START_CLAIM.json').read_text())
    if claim!={'id':s['id'],'spec_sha256':expected_sha,'protocol_sha256':s['protocol_sha256']}: raise ValueError('Start claim differs')
    lock=os.open(out/'SUPERVISOR_CLAIM',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(lock)
    started=time.monotonic(); child=None; returncode=None; exited=False; timed_out=False; error=None; child_ticks=None
    try:
        worker_check()
        # Recheck bindings immediately before child launch.
        validate(spec_path,expected_sha,active_path=active_path)
        with (out/'child.stdout.log').open('xb') as stdout, (out/'child.stderr.log').open('xb') as stderr:
            child=subprocess.Popen(s['argv'],cwd=s['cwd'],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
            child_ticks=ticks(child.pid)
            exclusive_json(out/'CHILD_START.json',{'id':s['id'],'spec_sha256':expected_sha,'child_pid':child.pid,'child_startticks':child_ticks})
            try: returncode=child.wait(timeout=s['external_timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out=True
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: returncode=child.wait(timeout=kill_grace)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    returncode=child.wait()
            exited=True
    except BaseException as e:
        error=type(e).__name__
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: returncode=child.wait(timeout=kill_grace)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                returncode=child.wait()
            exited=True
        elif child is not None:
            returncode=child.wait();exited=True
    finally:
        run=Path(s['run_json']); run_sha=None
        try:
            if run.is_file(): run_sha=digest(run)
        except OSError as e:
            error=error or type(e).__name__
        receipt={'id':s['id'],'spec_sha256':expected_sha,'protocol_sha256':s['protocol_sha256'],
                 'child_pid':child.pid if child else None,'child_startticks':child_ticks,'child_returncode':returncode,
                 'process_exited':exited,'timeout':timed_out,'outer_seconds':time.monotonic()-started,
                 'run_sha256':run_sha,'error_type':error,'automatic_retry':False}
        exclusive_json(out/'EXIT_RECEIPT.json',receipt)
    return receipt


def inspect(spec_path,expected_sha):
    s=validate(spec_path,expected_sha,require_active=False);out=Path(s['output_dir'])
    def read(name):
        p=out/name
        return json.loads(p.read_text()) if p.is_file() else None
    start_record=read('START_RECEIPT.json');end=read('EXIT_RECEIPT.json');child_start=read('CHILD_START.json')
    if child_start and (child_start.get('id')!=s['id'] or child_start.get('spec_sha256')!=expected_sha): raise ValueError('Child receipt differs')
    if start_record and (start_record.get('id')!=s['id'] or start_record.get('spec_sha256')!=expected_sha):raise ValueError('Start receipt differs')
    if end and (end.get('id')!=s['id'] or end.get('spec_sha256')!=expected_sha or end.get('protocol_sha256')!=s['protocol_sha256']):raise ValueError('Exit receipt differs')
    return {'start':start_record,'exit':end,'supervisor_alive':alive(start_record.get('supervisor_pid'),start_record.get('supervisor_startticks')) if start_record else False,
            'child_start':child_start,'child_alive':alive(child_start.get('child_pid'),child_start.get('child_startticks')) if child_start else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('start','supervise','inspect'));ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    if args.action=='start': result=start(args.spec,args.spec_sha)
    elif args.action=='supervise': result=supervise(args.spec,args.spec_sha)
    else: result=inspect(args.spec,args.spec_sha)
    print(json.dumps(result))


if __name__=='__main__':main()
