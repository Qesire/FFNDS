"""Queue only UB2's registered changed-step fit after complete LR2 rejection."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT


def main():
    r = ROOT/'runs/worker_restart_20261020'
    out = r/'terminal_dense_ub2_fit_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta = {'status':'RUNNING','stage':'WAIT_COMPLETE_LR2_QUALITY_REJECTION',
            'controller_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT/'config/terminal_dense_ub2.json'),
            'quality_or_speed_acceptance':False}
    atomic_json(out/'RUN.json',meta)
    try:
        while True:
            path = r/'late_block_rank_lr2_controller/RUN.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            if value.get('status') == 'COMPLETED_LR2_OBSERVED_COHORT_QUALITY_REJECTED':
                break
            if value.get('status') == 'COMPLETED_LR2_BOTH_OBSERVED_COHORT_QUALITY_PASSED':
                meta.update(status='NOT_RUN_UB2_LR2_REJECTION_CONDITION_FALSE',stage='STOPPED')
                return
            if value.get('status','').startswith('FAILED'):
                raise RuntimeError('Resolve the existing full LR2 quality pipeline first')
            time.sleep(10)
        meta.update(stage='EXACT_OLD_STEP1_CONTROL_THEN_FIT40_QUARTER_POSITIONS',LR2_rejection_sha256=digest(path))
        atomic_json(out/'RUN.json',meta)
        command = [sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu','0','--','python','tools/fit_terminal_dense_ub2.py',
                   '--step','1','--collections',str(r/'calibration24_ffn_operands/shard0/RUN.json'),
                   str(r/'calibration24_ffn_operands/shard1/RUN.json'),
                   '--parent-policy',str(r/'calibration24_residual_fit/C24_RR_policy.json'),
                   '--out','/var/tmp/ffnds-xiang/huggingface/ffnds-derived/terminal_dense_ub2_step1_fit_20261021',
                   '--report',str(r/'terminal_dense_ub2_fit/step1_RUN.json')]
        atomic_json(out/'command.json',{'command':command})
        for attempt in range(1,9):
            with (out/f'attempt{attempt}_{time.time_ns()}.log').open('w') as log:
                result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError('UB2 fit failed; preserve control/report before any correction')
            break
        else:
            raise RuntimeError('GPU0 remained occupied; no fitting settings changed')
        result = json.loads((r/'terminal_dense_ub2_fit/step1_RUN.json').read_text())
        if result['status'] != 'COMPLETED_UB2_STEP1_QUARTER40_FIT' or len(result['positions']) != 40:
            raise RuntimeError('Exact old-control and40 changed-step coefficients required')
        meta.update(status='COMPLETED_UB2_STEP1_FIT_WAITING_NATIVE_CONTROLS',stage='COMPLETE',
                    result_sha256=digest(r/'terminal_dense_ub2_fit/step1_RUN.json'))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(out/'RUN.json',meta)


if __name__ == '__main__':
    main()
