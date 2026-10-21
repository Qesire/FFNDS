"""Conditional active equivalence and isolated three-arm LR2 engineering checks."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT,SSH


def main():
    r = ROOT/'runs/worker_restart_20261020'
    out = r/'late_block_rank_lr2_acceptance'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy = r/'late_block_rank_lr2_full/LR2_policy.json'
    parent = r/'ub1_step_budget_fit/UB1_policy.json'
    proof = r/'late_block_rank_lr2_identity/RUN.json'
    quality_path = r/'late_block_rank_lr2_controller/RUN.json'
    meta = {'status':'RUNNING','stage':'WAIT_FOR_BOTH_QUALITY_GATES','controller_sha256':digest(Path(__file__)),
            'policy_sha256':digest(policy),'joint_engineering_acceptance_pass':False}

    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    def stage(name):
        meta['stage'] = name
        atomic_json(out/'RUN.json',meta)
        print(name,flush=True)

    def gpu_job(gpu,command,name):
        argv = [sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--',*map(str,command)]
        atomic_json(out/f'{name}_command.json',{'command':argv})
        for attempt in range(1,9):
            path = out/f'{name}_attempt{attempt}_{time.time_ns()}.log'
            with path.open('w') as log:
                result = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect preserved {path} before retry')
            return
        raise RuntimeError(f'{name}: assigned GPU remained occupied')

    try:
        stage('WAIT_FOR_BOTH_QUALITY_GATES')
        while True:
            quality = read(quality_path)
            if quality.get('status') == 'COMPLETED_LR2_OBSERVED_COHORT_QUALITY_REJECTED':
                meta.update(status='COMPLETED_LR2_QUALITY_REJECTED_NO_FURTHER_ACCEPTANCE',stage='STOPPED',
                            quality_report_sha256=digest(quality_path))
                return
            if quality.get('status') == 'COMPLETED_LR2_BOTH_OBSERVED_COHORT_QUALITY_PASSED':
                break
            if quality.get('status','').startswith('FAILED'):
                raise RuntimeError('Finish and audit existing LR2 quality work first')
            time.sleep(10)
        if quality['policy_sha256'] != digest(policy) or not quality['both_observed_cohort_quality_pass']:
            raise RuntimeError('Same frozen active LR2 must pass both quality cohorts')
        meta['quality_report_sha256'] = digest(quality_path)
        stage('ACTIVE_NONRESIDENT_RESIDENT_EQUIVALENCE_BOTH_FULL20_COHORTS')
        def equivalent(cohort,gpu):
            folder = r/f'{cohort}_LR2_nonresident_equivalence'
            value = read(folder/'RUN.json')
            if value.get('status') != 'PASS_FORMAL_GENERATION_SLICE_ONLY':
                if value.get('status','').startswith('FAILED'):
                    raise RuntimeError('Inspect the preserved failed equivalence before resuming')
                gpu_job(gpu,['python','tools/generate_late_block_suite.py','--cohort',cohort,'--policy',policy,
                    '--identity-proof',proof,'--nonresident','--reference-suite',r/f'{cohort}_LR2/RUN.json',
                    '--out',folder,'--payload-root',
                    f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/late_block_rank_lr2_nonresident_20261021/{cohort}'],f'equivalence_{cohort}')
                value = read(folder/'RUN.json')
            if (value['identity']['packed_policy_sha256'] != digest(policy) or len(value['completed']) != 20
                    or not all(v['paired_latents_and_video_equal'] for v in value['completed'])):
                raise RuntimeError('Every active LR2 latent and video must match its paired resident output')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(equivalent,c,g) for c,g in [('formal20',0),('disjoint20',1)]]
            for job in jobs:
                job.result()
        timing = {}
        references = {'DQ':r/'formal20_dense/RUN.json','UB1':r/'formal20_UB1/FULL_RUN.json','LR2':r/'formal20_LR2/RUN.json'}
        for arm in ['DQ','UB1','LR2']:
            stage('ISOLATED_FORMAL00_WARMUP_PLUS_THREE_MEASURED_'+arm)
            source = read(references[arm])
            row = next(v for v in source['completed'] if v['index'] == 0)
            if digest(row['case_report']) != row['case_report_sha256']:
                raise RuntimeError('Fixed formal00 timing reference changed')
            folder = r/f'LR2_timing_formal00_{arm}'
            report = folder/'RUN.json'
            prior = read(report)
            if prior.get('status','').startswith('FAILED'):
                raise RuntimeError('Preserve failed timing and diagnose before any new attempt; never select a best run')
            if prior.get('status') != 'PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE':
                while True:
                    result = subprocess.run(SSH+['nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    if not result.stdout.strip():
                        break
                    time.sleep(10)
                command = ['python','tools/run_late_block_benchmark.py','--arm',arm,'--reference-case',row['case_report'],
                           '--out',folder,'--payload-root',f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/late_block_rank_lr2_timing_20261021/{arm}',
                           '--measured-repeats','3']
                if arm != 'DQ':
                    command += ['--policy',parent if arm == 'UB1' else policy]
                gpu_job(0,command,f'timing_{arm}')
            timing[arm] = report
        stage('COMPARE_ALL_REPEATS_AT_THE_SAME_FIXED_INPUT')
        comparisons = {}
        for arm in ['UB1','LR2']:
            output = out/f'{arm}_FFN_COMPARISON.json'
            command = ['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/compare_late_block_benchmark.py',
                       '--dense',str(timing['DQ']),'--candidate',str(timing[arm]),'--output',str(output)]
            with (out/f'compare_{arm}.log').open('w') as log:
                subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
            comparisons[arm] = read(output)
        if comparisons['LR2']['candidate_policy_sha256'] != digest(policy):
            raise RuntimeError('Timing and quality candidate policies differ')
        meta.update(status='COMPLETED_LR2_JOINT_ENGINEERING_CHECKS',stage='COMPLETE',speed_comparisons=comparisons,
                    joint_engineering_acceptance_pass=bool(comparisons['LR2']['full_FFN_speed_pass']),
                    equivalence={'formal20':20,'disjoint20':20},
                    scope='static final10-layer rank recovery; both quality cohorts observed, no held-out or dynamic novelty claim')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(out/'RUN.json',meta)


if __name__ == '__main__':
    main()
