"""Run the frozen source-disjoint supplement after primary joint acceptance."""
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
    args = ap.parse_args()
    r = args.run_root.resolve()
    out = r / 'disjoint20_pipeline'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'RUNNING', 'stage': None, 'source_sha256': digest(Path(__file__)),
            'role': 'single frozen source-disjoint supplement; no training or selection from results',
            'changes_primary_acceptance': False}

    def stage(name):
        meta['stage'] = name
        atomic_json(out / 'RUN.json', meta)
        print(name, flush=True)

    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    def gpu(gpu_index, command, name):
        argv = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', str(gpu_index), '--', *map(str, command)]
        atomic_json(out / f'{name}_command.json', {'command': argv})
        for attempt in range(1, 9):
            with (out / f'{name}_attempt{attempt}.log').open('w') as log:
                result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError(f'{name} failed; retain logs and partial checkpoints')
            return
        raise RuntimeError(f'{name}: assigned GPU remained occupied')

    try:
        stage('WAIT_FOR_ORIGINAL_SAME_POLICY_JOINT_ENGINEERING_ACCEPTANCE')
        primary_path = r / 'UB1_acceptance_controller/RUN.json'
        while True:
            primary = read(primary_path)
            if primary.get('status') == 'COMPLETED_JOINT_ENGINEERING_CHECKS':
                break
            if primary.get('status', '').startswith('FAILED'):
                raise RuntimeError('Primary engineering failed; do not consume additional GPUs')
            time.sleep(10)
        if not primary['joint_engineering_acceptance_pass']:
            meta.update(status='NOT_RUN_PRIMARY_ENGINEERING_REJECTED', stage='STOPPED_BEFORE_SUPPLEMENTAL_MODEL_EXECUTION')
            return
        protocol_path = ROOT / 'config/ub1_source_disjoint20.json'
        protocol = read(protocol_path)
        policy = r / 'ub1_step_budget_fit/UB1_policy.json'
        if (digest(policy) != protocol['frozen_candidate_policy_sha256']
                or digest(ROOT / 'tools/step_budget_runtime.py') != protocol['frozen_runtime_source_sha256']
                or digest(ROOT / 'tools/trained_packed_runtime.py') != protocol['frozen_parent_runtime_source_sha256']):
            raise RuntimeError('Frozen supplemental candidate changed')
        meta['primary_acceptance_sha256'] = digest(primary_path)
        meta['protocol_sha256'] = digest(protocol_path)
        stage('GENERATE_PAIRED_SOURCE_DISJOINT20_DQ_AND_UB1_ON_TWO_GPUS')
        suites = {label: r / f'disjoint20_{label}/RUN.json' for label in ['DQ', 'UB1']}
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for gpu_index, label in enumerate(['DQ', 'UB1']):
                if read(suites[label]).get('status') == 'PASS_FORMAL_GENERATION_SLICE_ONLY':
                    continue
                command = ['python', 'tools/generate_disjoint_suite.py', '--indices', *range(20),
                           '--out', suites[label].parent, '--payload-root',
                           f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/source_disjoint20_20261021/{label}']
                if label == 'UB1':
                    command += ['--policy', policy]
                jobs.append(pool.submit(gpu, gpu_index, command, f'generate_{label}'))
            for job in jobs:
                job.result()
        stage('SAME_OFFICIAL_THREE_METRICS_FOR_BOTH_FROZEN_ARMS')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for gpu_index, label in enumerate(['DQ', 'UB1']):
                folder = r / f'disjoint20_{label}_vbench3'
                if read(folder / 'RUN.json').get('status') == 'PASS_THREE_METRIC_EXECUTION_ONLY':
                    continue
                jobs.append(pool.submit(gpu, gpu_index, ['bash', 'scripts/worker_eval_env.sh', 'python',
                    'tools/evaluate_disjoint_vbench3.py', '--suites', suites[label], '--out', folder,
                    '--name', f'{label}_source_disjoint20_seed42'], f'evaluate_{label}'))
            for job in jobs:
                job.result()
        stage('COMPARE_SUPPLEMENTAL_EACH_METRIC_WITHOUT_REFITTING')
        comparison_path = r / 'disjoint20_UB1_comparison/COMPARISON.json'
        with (out / 'compare.log').open('w') as log:
            subprocess.run([sys.executable, 'tools/compare_vbench3.py', '--dense', str(r / 'disjoint20_DQ_vbench3'),
                '--candidate', str(r / 'disjoint20_UB1_vbench3'), '--output', str(comparison_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        meta.update(status='COMPLETED_FROZEN_SOURCE_DISJOINT20_SUPPLEMENT',
                    supplemental_quality_pass=read(comparison_path)['formal20_quality_pass'],
                    comparison=read(comparison_path), refit_or_hyperparameter_selection=False)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
