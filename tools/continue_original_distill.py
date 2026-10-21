"""Wait for CPU asset staging and terminal LR2 checks, then run one local OD1 job."""
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
    out = r/'original_distill_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta = {'status':'RUNNING','stage':'WAIT_ASSETS_AND_LR2_TERMINAL','source_sha256':digest(Path(__file__)),
            'quality_or_speed_acceptance':False}
    atomic_json(out/'RUN.json',meta)
    try:
        while True:
            paths = [ROOT/'.runtime/original_o_block20_assets.json',r/'late_block_rank_lr2_acceptance/RUN.json']
            assets,acceptance = [json.loads(p.read_text()) if p.exists() else {} for p in paths]
            testpath = ROOT/'reports/worker_20261020/OD1_preflight_71_tests.json'
            tests = json.loads(testpath.read_text()) if testpath.exists() else {}
            if (assets.get('status','').startswith('FAILED') or acceptance.get('status','').startswith('FAILED')
                    or tests.get('status','').startswith('FAILED')):
                raise RuntimeError('Resolve preserved asset/LR2 failure before OD1; do not overlap isolated timing')
            if (assets.get('status') == 'PASS_ORIGINAL_O_BLOCK20_SHARDS_COMPLETE_FILE_HASHES_AND_HEADERS'
                    and acceptance.get('status','').startswith('COMPLETED_LR2_')
                    and tests.get('status') == 'PASS_COMPLETE_CPU_UNIT_TEST_PREFLIGHT' and tests['tests_run'] == 71):
                break
            time.sleep(10)
        meta.update(stage='NATIVE_DBF16_CONTROLS_THEN_MATCHED_LOCAL_FUNCTIONS',
                    original_asset_sha256=digest(paths[0]),LR2_terminal_sha256=digest(paths[1]))
        atomic_json(out/'RUN.json',meta)
        command = [sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu','1','--','python',
                   'tools/run_original_distill_ffn.py','--out','/var/tmp/ffnds-xiang/huggingface/ffnds-derived/original_distill_od1_20261021',
                   '--report',str(r/'original_distill_od1/RUN.json')]
        atomic_json(out/'command.json',{'command':command})
        for attempt in range(1,9):
            with (out/f'attempt{attempt}_{time.time_ns()}.log').open('w') as log:
                result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError('OD1 failed; preserve native-control evidence before any correction')
            break
        else:
            raise RuntimeError('GPU1 remained occupied; no mathematical settings changed')
        result = json.loads((r/'original_distill_od1/RUN.json').read_text())
        if result['status'] != 'COMPLETED_OD1_EIGHT_NATIVE_CONTROLS_AND_LOCAL_FUNCTION_COMPARISONS':
            raise RuntimeError('All8 native controls and local comparisons required')
        meta.update(status='COMPLETED_ORIGINAL_DISTILL_LOCAL_FUNCTION_DIAGNOSTIC',stage='COMPLETE',
                    result_sha256=digest(r/'original_distill_od1/RUN.json'),
                    scope='partial original-O local functions on D states; no original-O video or sparsity acceptance')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(out/'RUN.json',meta)


if __name__ == '__main__':
    main()
