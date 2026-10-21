"""Wait for all current UB2 GPU work before the separately frozen NF1 pilot."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT


def main():
    r=ROOT/'runs/worker_restart_20261020';out=r/'nonlinear_residual_nf1_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'status':'RUNNING','stage':'WAIT_ENTIRE_UB2_PIPELINE_TERMINAL','source_sha256':digest(Path(__file__)),
        'quality_or_speed_acceptance':False}
    atomic_json(out/'RUN.json',meta)
    try:
        while True:
            p=r/'diverse_confirmation20_controller/RUN.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            status=state.get('status','')
            if status.startswith('FAILED'):raise RuntimeError('Resolve current UB2 pipeline failure before NF1')
            if status.startswith('COMPLETED_DIVERSE_CONFIRMATION20_') or status=='NOT_RUN_OBSERVED_UB2_QUALITY_REJECTED':break
            time.sleep(10)
        freeze=ROOT/'reports/worker_20261020/NF1_before_feature_fit_source_freeze.json'
        for path,sha in json.loads(freeze.read_text())['sources'].items():
            if digest(path)!=sha:raise RuntimeError('Frozen NF1 source changed before its first fit')
        meta.update(stage='FOUR_FIXED_LOCAL_NATIVE_CV_FOLDS',source_freeze_sha256=digest(freeze),
            UB2_terminal_sha256=digest(p))
        atomic_json(out/'RUN.json',meta)
        command=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu','0','--','python',
            'tools/run_nonlinear_residual_pilot.py','--out','/var/tmp/ffnds-xiang/huggingface/ffnds-derived/nonlinear_residual_nf1_20261021',
            '--report',str(r/'nonlinear_residual_nf1/RUN.json')]
        atomic_json(out/'command.json',{'command':command})
        for attempt in range(1,9):
            with (out/f'attempt{attempt}_{time.time_ns()}.log').open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:time.sleep(15);continue
            if result.returncode:raise RuntimeError('Preserve NF1 failure; inspect before any retry or mathematical change')
            break
        else:raise RuntimeError('GPU0 remained occupied')
        resultpath=r/'nonlinear_residual_nf1/RUN.json'
        deadline=time.monotonic()+90
        while not resultpath.exists():
            if time.monotonic()>=deadline:raise RuntimeError('NFS result visibility timeout')
            time.sleep(2)
        result=json.loads(resultpath.read_text())
        if result['status'] not in ['COMPLETED_NF1_NATIVE_GATE_PASSED','COMPLETED_NF1_NATIVE_GATE_REJECTED']:
            raise RuntimeError('All fixed NF1 native folds must finish')
        meta.update(status='COMPLETED_NF1_NATIVE_PILOT',stage='COMPLETE',native_gate_pass=result['native_gate_pass'],
            result_sha256=digest(resultpath))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(out/'RUN.json',meta)


if __name__=='__main__':main()
