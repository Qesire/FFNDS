"""Finish the frozen OP1 route, stopping each expensive stage at its registered gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
    args = ap.parse_args(); r = args.run_root.resolve(); out = r / 'onpolicy_pipeline'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'RUNNING', 'stage': None, 'source_sha256': digest(Path(__file__)), 'quality_or_speed_acceptance': False}

    def stage(name):
        meta['stage'] = name; atomic_json(out / 'RUN.json', meta); print(name, flush=True)

    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    def wait(path, status):
        while True:
            value = read(path)
            if value.get('status') == status: return value
            if value.get('status', '').startswith('FAILED'):
                raise RuntimeError(f"Prerequisite failed: {path}: {value.get('error')}")
            time.sleep(10)

    def gpu_job(gpu, command, name):
        argv = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', str(gpu), '--', *map(str, command)]
        atomic_json(out / f'{name}_command.json', {'command': argv})
        for attempt in range(1, 9):
            with (out / f'{name}_attempt{attempt}.log').open('w') as log:
                result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 3: time.sleep(15); continue
            if result.returncode: raise RuntimeError(f'{name} failed; inspect saved log')
            return
        raise RuntimeError(f'{name}: GPU remained occupied')

    def local(command, name):
        with (out / f'{name}.log').open('w') as log:
            subprocess.run([str(ROOT / '.venv/bin/python'), *map(str, command)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    collections = [r / f'onpolicy_calibration24/shard{g}/RUN.json' for g in [0, 1]]
    student = r / 'calibration24_residual_fit/C24_RR_policy.json'
    cv = r / 'onpolicy_residual_cv/RUN.json'; fit = r / 'onpolicy_residual_full/RUN.json'
    policy = r / 'onpolicy_residual_full/OP1_policy.json'
    fit_command = ['python', 'tools/fit_onpolicy_residual.py', '--collections', *collections, '--student-policy', student]
    try:
        stage('WAIT_FOR_BOTH_NATIVE_STUDENT_COLLECTION_SHARDS')
        for path in collections: wait(path, 'PASS_ONPOLICY_CALIBRATION24_SHARD_AND_NOOP')
        stage('FROZEN_TWO_LATE_STEP_FOUR_FOLD_CV_AND_NATIVE_OLD_REPLAY')
        if read(cv).get('status') not in ['COMPLETED_OP1_CV_GATE_PASSED', 'COMPLETED_OP1_CV_GATE_REJECTED']:
            gpu_job(0, [*fit_command, '--mode', 'cv', '--out',
                '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/onpolicy_residual_cv_20261021', '--report', cv], 'CV')
        result = read(cv); meta['CV'] = result
        if result.get('status') == 'COMPLETED_OP1_CV_GATE_REJECTED':
            meta.update(status='COMPLETED_OP1_CV_REJECTED', stage='STOPPED_BEFORE_FULL_FIT_AND_FORMAL'); return
        if result.get('status') != 'COMPLETED_OP1_CV_GATE_PASSED': raise RuntimeError('OP1 CV incomplete')
        stage('SAME_RANK256_LAMBDA1_ALL160_ONPOLICY_REFIT')
        if read(fit).get('status') != 'COMPLETED_RR1_FULL160_CALIBRATION_FIT':
            gpu_job(0, [*fit_command, '--mode', 'full', '--cv-report', cv, '--out',
                '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/onpolicy_residual_full_20261021', '--report', fit], 'full160')
        local(['tools/write_residual_policy.py', '--base-policy', r / 'calibration24_centered_fit/C24_centered_policy.json',
            '--residual-fit', fit, '--experiment', 'OP1_ONPOLICY_C24_STUDENT_FIXED_RANK256_LAMBDA1_20261021',
            '--output', policy], 'write_policy')
        stage('FROZEN_FORMAL20_TWO_DISJOINT_SLICES')
        suites = [r / f'formal20_OP1/gpu{g}/RUN.json' for g in [0, 1]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for gpu in [0, 1]:
                if read(suites[gpu]).get('status') == 'PASS_FORMAL_GENERATION_SLICE_ONLY': continue
                jobs.append(pool.submit(gpu_job, gpu, ['python', 'tools/generate_formal_suite.py', '--indices',
                    *range(10 * gpu, 10 * (gpu + 1)), '--packed-policy', policy, '--out', suites[gpu].parent], f'formal_gpu{gpu}'))
            for job in jobs: job.result()
        merged = r / 'formal20_OP1/FULL_RUN.json'
        local(['tools/merge_formal_suites.py', '--suites', *suites, '--output', merged], 'merge_formal')
        stage('OFFICIAL_THREE_METRICS_ON_REUSED_FORMAL20')
        if read(r / 'formal20_OP1_vbench3/RUN.json').get('status') != 'PASS_THREE_METRIC_EXECUTION_ONLY':
            gpu_job(0, ['bash', 'scripts/worker_eval_env.sh', 'python', 'tools/evaluate_vbench3.py',
                '--suites', merged, '--out', r / 'formal20_OP1_vbench3', '--name', 'OP1_formal20_seed42'], 'official_vbench3')
        comparison = r / 'formal20_OP1_comparison/COMPARISON.json'
        local(['tools/compare_vbench3.py', '--dense', r / 'formal20_dense_vbench3', '--candidate', r / 'formal20_OP1_vbench3',
            '--output', comparison], 'quality_comparison')
        meta['comparison'] = read(comparison)
        if not meta['comparison']['formal20_quality_pass']:
            meta.update(status='COMPLETED_OP1_FORMAL_QUALITY_REJECTED', stage='STOPPED_BEFORE_RESIDENT_ACCEPTANCE'); return
        stage('FULL20_RESIDENT_EQUIVALENCE_THEN_ISOLATED_COMPLETE_FFN_TIMING')
        resident = r / 'formal20_OP1_resident_equivalence/RUN.json'
        if read(resident).get('status') != 'PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT':
            gpu_job(1, ['python', 'tools/verify_resident_formal.py', '--reference-suite', merged, '--policy', policy,
                '--out', resident.parent], 'resident_all20')
        local(['tools/continue_candidate_acceptance.py', '--label', 'OP1', '--policy', policy, '--run-root', r], 'acceptance')
        accepted = read(r / 'OP1_acceptance_controller/RUN.json')
        meta.update(status='COMPLETED_OP1_JOINT_ENGINEERING_CHECKS', acceptance=accepted,
            quality_or_speed_acceptance=bool(accepted.get('joint_engineering_acceptance_pass')))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally: atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__': main()
