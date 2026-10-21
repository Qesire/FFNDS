"""Finish UB2's two observed-cohort quality checks after the fixed identity proof."""
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
    out = r/'terminal_dense_ub2_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy = r/'terminal_dense_ub2_fit/UB2_policy.json'
    proof = r/'terminal_dense_ub2_identity/RUN.json'
    flow = r/'terminal_dense_ub2_flow/RUN.json'
    meta = {'status':'RUNNING','stage':'WAIT_FIXED_IDENTITY','controller_sha256':digest(Path(__file__)),
            'policy_sha256':digest(policy),'quality_or_speed_acceptance':False}

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
            logpath = out/f'{name}_attempt{attempt}_{time.time_ns()}.log'
            with logpath.open('w') as log:
                result = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError(f'{name} failed; preserve and inspect {logpath}')
            return
        raise RuntimeError(f'{name}: assigned GPU remained occupied')

    cohorts = [('formal20',0),('disjoint20',1)]
    try:
        stage('WAIT_FIXED_IDENTITY')
        while True:
            value = read(proof)
            if value.get('status') == 'PASS_UB2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES':
                break
            if value.get('status','').startswith('FAILED'):
                raise RuntimeError('Finish the unchanged identity proof before active generation')
            time.sleep(10)
        meta['identity_proof_sha256'] = digest(proof)
        stage('VERIFY_DENSE_TERMINAL_AND_UNCHANGED_NATIVE_FLOW')
        if read(flow).get('status') != 'PASS_UB2_DENSE_TERMINAL_AND_UNCHANGED_STEPS_NATIVE_FLOW':
            gpu_job(1,['python','tools/probe_terminal_dense_flow.py','--policy',policy,
                '--identity-proof',proof,'--out',flow.parent,'--payload-root',
                '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/terminal_dense_ub2_flow_20261021'],'terminal_flow_control')
        control = read(flow)
        if (control['status'] != 'PASS_UB2_DENSE_TERMINAL_AND_UNCHANGED_STEPS_NATIVE_FLOW'
                or control['identity']['policy_sha256'] != digest(policy)
                or control['identity']['identity_control_proof_sha256'] != digest(proof)):
            raise RuntimeError('Same-policy exact terminal native flow controls required')
        meta['terminal_flow_proof_sha256'] = digest(flow)
        stage('GENERATE_BOTH_OBSERVED_COHORTS_SAME_ACTIVE_POLICY')
        def generate(cohort,gpu):
            folder = r/f'{cohort}_UB2'
            prior = read(folder/'RUN.json')
            if prior.get('status') == 'PASS_FORMAL_GENERATION_SLICE_ONLY':
                identity = prior['identity']
                if (identity['packed_policy_sha256'] != digest(policy) or identity['identity_control_proof_sha256'] != digest(proof)
                        or identity['terminal_flow_proof_sha256'] != digest(flow) or identity['cohort'] != cohort or len(prior['completed']) != 20):
                    raise RuntimeError('Completed UB2 generation identity differs')
                for path,sha in {**identity['sources'],**identity['runtime_sources']}.items():
                    if digest(path) != sha:
                        raise RuntimeError('Completed UB2 generation numerical source changed')
                return
            gpu_job(gpu,['python','tools/generate_terminal_dense_suite.py','--cohort',cohort,'--policy',policy,
                '--identity-proof',proof,'--terminal-flow-proof',flow,'--out',folder,'--payload-root',
                f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/terminal_dense_ub2_videos_20261021/{cohort}'],f'generate_{cohort}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(generate,c,g) for c,g in cohorts]
            for job in jobs:
                job.result()
        stage('SAME_OFFICIAL_THREE_METRIC_WORKERS')
        def evaluate(cohort,gpu):
            folder = r/f'{cohort}_UB2_vbench3'
            if read(folder/'RUN.json').get('status') == 'PASS_THREE_METRIC_EXECUTION_ONLY':
                return
            script = 'evaluate_vbench3.py' if cohort == 'formal20' else 'evaluate_disjoint_vbench3.py'
            gpu_job(gpu,['bash','scripts/worker_eval_env.sh','python',f'tools/{script}',
                '--suites',r/f'{cohort}_UB2/RUN.json','--out',folder,'--name',f'UB2_{cohort}'],f'evaluate_{cohort}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(evaluate,c,g) for c,g in cohorts]
            for job in jobs:
                job.result()
        stage('COMPARE_SEPARATE_PAIRED_DQ_REFERENCES')
        comparisons = []
        for cohort,_ in cohorts:
            dense = r/('formal20_dense_vbench3' if cohort == 'formal20' else 'disjoint20_DQ_vbench3')
            output = r/f'{cohort}_UB2_comparison/COMPARISON.json'
            command = ['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/compare_terminal_dense_quality.py',
                       '--dense',str(dense),'--candidate',str(r/f'{cohort}_UB2_vbench3'),'--policy',str(policy),
                       '--identity-proof',str(proof),'--terminal-flow-proof',str(flow),'--cohort',cohort,'--output',str(output)]
            with (out/f'compare_{cohort}.log').open('w') as log:
                subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
            result = read(output)
            if (result['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
                    or result['candidate_policy_sha256'] != meta['policy_sha256']):
                raise RuntimeError('Quality comparison did not preserve the frozen active policy')
            comparisons.append({'cohort':cohort,'path':str(output),'sha256':digest(output),'result':result})
        passed = all(v['result']['formal20_quality_pass'] for v in comparisons)
        meta.update(status='COMPLETED_UB2_BOTH_OBSERVED_COHORT_QUALITY_PASSED' if passed else 'COMPLETED_UB2_OBSERVED_COHORT_QUALITY_REJECTED',
                    stage='COMPLETE',comparisons=comparisons,both_observed_cohort_quality_pass=passed,
                    scope='two observed development cohorts; full FFN speed and active nonresident/resident equivalence still require separate checks')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(out/'RUN.json',meta)


if __name__ == '__main__':
    main()
