"""Finish UB1 identity proof, frozen formal20 quality and resident equivalence."""
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
    args = ap.parse_args(); r = args.run_root.resolve(); out = r / 'ub1_pipeline'; out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'RUNNING', 'stage': None, 'source_sha256': digest(Path(__file__)), 'quality_or_speed_acceptance': False}
    def stage(name):
        meta['stage'] = name; atomic_json(out / 'RUN.json', meta); print(name, flush=True)
    def read(path): return json.loads(path.read_text()) if path.exists() else {}
    def wait(path, status):
        while True:
            d = read(path)
            if d.get('status') == status: return d
            if d.get('status', '').startswith('FAILED'): raise RuntimeError(f"Prerequisite failed: {path}: {d.get('error')}")
            time.sleep(10)
    def local(command, name):
        with (out / f'{name}.log').open('w') as log:
            subprocess.run([str(ROOT / '.venv/bin/python'), *map(str, command)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)
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
    policy = r / 'ub1_step_budget_fit/UB1_policy.json'
    try:
        stage('WAIT_FOR_BOTH_CHANGED_STEP40_FITS_AND_EXISTING_NATIVE_IDENTITY_REQUESTS')
        for step in [0, 3]: wait(r / f'ub1_step_budget_fit/step{step}_RUN.json', 'COMPLETED_UB1_STEP40_FIT')
        for part in ['parent', 'candidate']: wait(r / f'ub1_identity/{part}/run_meta.json', 'PASS')
        stage('VERIFY_ORIGINAL_C24_FULL160_FOUR_LATENTS_AND_VIDEO_BYTES')
        proof = r / 'ub1_identity/verification/RUN.json'
        if read(proof).get('status') != 'PASS_UB1_IDENTITY_FULL160_FOUR_LATENTS_AND_VIDEO_BYTES':
            local(['tools/verify_step_budget_identity.py', '--reference', r / 'ub1_identity/parent/run_meta.json',
                '--candidate', r / 'ub1_identity/candidate/run_meta.json', '--out', proof.parent], 'identity_verify')
        wait(proof, 'PASS_UB1_IDENTITY_FULL160_FOUR_LATENTS_AND_VIDEO_BYTES')
        stage('GENERATE_FROZEN_UB1_FORMAL20_TWO_DISJOINT_SLICES')
        suites = [r / f'formal20_UB1/gpu{g}/RUN.json' for g in [0, 1]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for gpu in [0, 1]:
                if read(suites[gpu]).get('status') == 'PASS_FORMAL_GENERATION_SLICE_ONLY': continue
                jobs.append(pool.submit(gpu_job, gpu, ['python', 'tools/generate_formal_suite.py', '--indices',
                    *range(10 * gpu, 10 * (gpu + 1)), '--packed-policy', policy, '--out', suites[gpu].parent], f'formal_gpu{gpu}'))
            for job in jobs: job.result()
        merged = r / 'formal20_UB1/FULL_RUN.json'
        local(['tools/merge_formal_suites.py', '--suites', *suites, '--output', merged], 'merge_formal')
        stage('OFFICIAL_METRICS_AND_FULL20_RESIDENT_EQUIVALENCE')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            if read(r / 'formal20_UB1_vbench3/RUN.json').get('status') != 'PASS_THREE_METRIC_EXECUTION_ONLY':
                jobs.append(pool.submit(gpu_job, 0, ['bash', 'scripts/worker_eval_env.sh', 'python', 'tools/evaluate_vbench3.py',
                    '--suites', merged, '--out', r / 'formal20_UB1_vbench3', '--name', 'UB1_formal20_seed42'], 'official_vbench3'))
            resident = r / 'formal20_UB1_resident_equivalence/RUN.json'
            if read(resident).get('status') != 'PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT':
                jobs.append(pool.submit(gpu_job, 1, ['python', 'tools/verify_resident_formal.py', '--reference-suite', merged,
                    '--policy', policy, '--out', resident.parent], 'resident_all20'))
            for job in jobs: job.result()
        stage('COMPARE_EACH_METRIC_ON_REUSED_FORMAL20')
        comparison = r / 'formal20_UB1_comparison/COMPARISON.json'
        local(['tools/compare_vbench3.py', '--dense', r / 'formal20_dense_vbench3', '--candidate', r / 'formal20_UB1_vbench3',
            '--output', comparison], 'quality_comparison')
        meta.update(status='COMPLETED_UB1_FORMAL_COMPARISON_AND_RESIDENT_EQUIVALENCE', comparison=read(comparison),
            timing_acceptance_still_required=True)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally: atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__': main()
