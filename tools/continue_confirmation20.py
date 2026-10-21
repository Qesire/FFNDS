"""Run one prospective diverse20 confirmation after UB2's observed gates and timing."""
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
    r=ROOT/'runs/worker_restart_20261020';out=r/'diverse_confirmation20_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=r/'terminal_dense_ub2_fit/UB2_policy.json';proof=r/'terminal_dense_ub2_identity/RUN.json'
    flow=r/'terminal_dense_ub2_flow/RUN.json';inputs=ROOT/'.runtime/inputs/diverse_confirmation20/canonical.json'
    meta={'status':'RUNNING','stage':'WAIT_OBSERVED_QUALITY_AND_ENGINEERING_TERMINAL','policy_sha256':digest(policy),
        'source_sha256':digest(Path(__file__)),'confirmation_quality_pass':False,'full_project_acceptance':False}
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    def stage(name):
        meta['stage']=name;atomic_json(out/'RUN.json',meta);print(name,flush=True)
    def job(gpu,command,name):
        argv=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--',*map(str,command)]
        atomic_json(out/f'{name}_command.json',{'command':argv})
        for attempt in range(1,9):
            logpath=out/f'{name}_attempt{attempt}_{time.time_ns()}.log'
            with logpath.open('w') as log:result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:time.sleep(15);continue
            if result.returncode:raise RuntimeError(f'Preserve and inspect {logpath}')
            return
        raise RuntimeError('Assigned GPU remained occupied')
    try:
        stage('WAIT_OBSERVED_QUALITY_AND_ENGINEERING_TERMINAL')
        while True:
            q=read(r/'terminal_dense_ub2_controller/RUN.json');a=read(r/'terminal_dense_ub2_acceptance/RUN.json')
            if q.get('status')=='COMPLETED_UB2_OBSERVED_COHORT_QUALITY_REJECTED':
                meta.update(status='NOT_RUN_OBSERVED_UB2_QUALITY_REJECTED',stage='STOPPED',
                    reason='Prospective sources remain unused by any confirmation inference; no replacement or candidate retuning')
                return
            if q.get('status','').startswith('FAILED') or a.get('status','').startswith('FAILED'):
                raise RuntimeError('Finish and audit existing UB2 failures before confirmation')
            if (q.get('status')=='COMPLETED_UB2_BOTH_OBSERVED_COHORT_QUALITY_PASSED'
                    and a.get('status')=='COMPLETED_UB2_JOINT_ENGINEERING_CHECKS'):break
            time.sleep(10)
        if q['policy_sha256']!=digest(policy) or not q['both_observed_cohort_quality_pass']:
            raise RuntimeError('Exact frozen observed-cohort candidate required')
        recovery=read(inputs.with_name('recovery_report.json'))
        if recovery.get('status')!='PASS_FROZEN_DIVERSE_PROSPECTIVE_CONFIRMATION20_INPUTS':
            raise RuntimeError('Finish source registration, not replacement, before inference')
        meta.update(observed_quality_sha256=digest(r/'terminal_dense_ub2_controller/RUN.json'),
            engineering_sha256=digest(r/'terminal_dense_ub2_acceptance/RUN.json'),
            input_recovery_sha256=digest(inputs.with_name('recovery_report.json')))
        stage('GENERATE_FROZEN_DQ_AND_UB2_ON_DIVERSE_CONFIRMATION20')
        def generate(arm,gpu):
            folder=r/f'diverse_confirmation20_{arm}'
            if read(folder/'RUN.json').get('status')!='PASS_FORMAL_GENERATION_SLICE_ONLY':
                job(gpu,['python','tools/generate_confirmation_suite.py','--cohort','diverse_confirmation20','--arm',arm,
                    '--policy',policy,'--identity-proof',proof,'--terminal-flow-proof',flow,'--out',folder,
                    '--payload-root',f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/ub2_diverse_confirmation20_20261021/{arm}'],f'generate_{arm}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(generate,'DQ',0),pool.submit(generate,'UB2',1)]:future.result()
        stage('SAME_THREE_OFFICIAL_WORKERS_WITH_EXPLICIT_MULTI_CATEGORY_INDEX')
        def evaluate(arm,gpu):
            folder=r/f'diverse_confirmation20_{arm}_vbench3'
            if read(folder/'RUN.json').get('status')!='PASS_THREE_METRIC_EXECUTION_ONLY':
                job(gpu,['bash','scripts/worker_eval_env.sh','python','tools/evaluate_confirmation_vbench3.py',
                    '--inputs',inputs,'--suites',r/f'diverse_confirmation20_{arm}/RUN.json','--out',folder,
                    '--name',f'diverse_confirmation20_{arm}'],f'evaluate_{arm}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(evaluate,'DQ',0),pool.submit(evaluate,'UB2',1)]:future.result()
        stage('COMPARE_PROSPECTIVELY_FROZEN_POLICY_SEPARATELY')
        output=r/'diverse_confirmation20_UB2_comparison/COMPARISON.json'
        command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/compare_confirmation_quality.py',
            '--dense',str(r/'diverse_confirmation20_DQ_vbench3'),'--candidate',str(r/'diverse_confirmation20_UB2_vbench3'),
            '--policy',str(policy),'--identity-proof',str(proof),'--terminal-flow-proof',str(flow),
            '--cohort','diverse_confirmation20','--output',str(output)]
        with (out/'compare.log').open('w') as log:subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=read(output)
        if result['candidate_policy_sha256']!=digest(policy):raise RuntimeError('Confirmation policy changed')
        meta.update(status='COMPLETED_DIVERSE_CONFIRMATION20_QUALITY_PASSED' if result['formal20_quality_pass'] else 'COMPLETED_DIVERSE_CONFIRMATION20_QUALITY_REJECTED',
            stage='COMPLETE',comparison_sha256=digest(output),comparison=result,confirmation_quality_pass=result['formal20_quality_pass'],
            scope='one frozen prospective multi-category cohort, different composition from development; no dynamic novelty or complete historical blindness claim')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(out/'RUN.json',meta)


if __name__=='__main__':main()
