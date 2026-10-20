"""Wait for the registered OP1 rejection condition, then run exactly the LA1 pilot."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, default=ROOT / 'runs/worker_restart_20261020')
    args = ap.parse_args(); r = args.run_root.resolve(); out = r / 'folded_adapter_la1_controller'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'WAITING_FOR_REGISTERED_OP1_REJECTION', 'source_sha256': digest(Path(__file__)),
        'protocol_sha256': digest(ROOT / 'config/folded_adapter_la1.json'), 'quality_or_speed_acceptance': False}
    def read(path): return json.loads(path.read_text()) if path.exists() else {}
    atomic_json(out / 'RUN.json', meta)
    try:
        while True:
            cv = read(r / 'onpolicy_residual_cv/RUN.json')
            quality = read(r / 'formal20_OP1_comparison/COMPARISON.json')
            if cv.get('status') == 'COMPLETED_OP1_CV_GATE_REJECTED' or (
                    quality.get('status') == 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
                    and not quality['formal20_quality_pass']): break
            primary = read(r / 'onpolicy_pipeline/RUN.json')
            if primary.get('status', '').startswith('FAILED'):
                raise RuntimeError('OP1 failed operationally; this is not registered scientific rejection')
            if quality.get('formal20_quality_pass'):
                meta['status'] = 'SKIPPED_OP1_QUALITY_PASSED'; return
            time.sleep(10)
        meta['status'] = 'RUNNING_LA1_FOUR_FOLD_PILOT'; atomic_json(out / 'RUN.json', meta)
        report = r / 'folded_adapter_la1_cv/RUN.json'
        if read(report).get('status') not in ['COMPLETED_LA1_CV_GATE_PASSED', 'COMPLETED_LA1_CV_GATE_REJECTED']:
            command = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', '1', '--', 'python',
                'tools/run_folded_adapter_cv.py', '--collections',
                *[str(r / f'onpolicy_calibration24/shard{g}/RUN.json') for g in [0, 1]],
                '--student-policy', str(r / 'calibration24_residual_fit/C24_RR_policy.json'), '--run-root', str(r),
                '--out', '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/folded_adapter_la1_cv_20261021',
                '--report', str(report)]
            atomic_json(out / 'command.json', {'command': command})
            for attempt in range(1, 9):
                with (out / f'pilot_attempt{attempt}.log').open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 3: time.sleep(15); continue
                if result.returncode: raise RuntimeError('LA1 pilot failed; inspect saved report/log')
                break
            else: raise RuntimeError('GPU1 remained occupied')
        result = read(report)
        if result.get('status') not in ['COMPLETED_LA1_CV_GATE_PASSED', 'COMPLETED_LA1_CV_GATE_REJECTED']:
            raise RuntimeError('LA1 pilot incomplete')
        meta.update(status='COMPLETED_CONDITIONAL_LA1_PILOT', result=result)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally: atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__': main()
