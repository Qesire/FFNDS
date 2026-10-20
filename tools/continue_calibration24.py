"""Continue registered calibration24 fits and complete the frozen formal comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from run_on_worker import ROOT,SSH


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,default=ROOT/'runs/worker_restart_20261020')
    args=ap.parse_args();r=args.run_root.resolve()
    out=r/'calibration24_pipeline';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'status':'RUNNING','stage':None,'source_sha256':sha(Path(__file__)),'quality_or_speed_acceptance':False}
    local_python=str(ROOT/'.venv/bin/python')

    def stage(name):
        meta['stage']=name;atomic(out/'RUN.json',meta);print(json.dumps({'stage':name}),flush=True)

    def wait(path,status):
        while True:
            if path.exists():
                value=json.loads(path.read_text())
                if value['status']==status:return value
                if value['status']=='FAILED_RESUMABLE':raise RuntimeError(f'Prerequisite failed: {path}: {value.get("error")}')
            time.sleep(10)

    def gpu_job(gpu,command,name):
        argv=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--',*map(str,command)]
        atomic(out/f'{name}_command.json',{'command':argv})
        for attempt in range(1,7):
            with (out/f'{name}_attempt{attempt}.log').open('w') as log:
                result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:
                time.sleep(15);continue
            if result.returncode:raise RuntimeError(f'{name} failed; inspect report/log')
            return
        raise RuntimeError(f'{name}: GPU lock remained busy')

    def local(command,name):
        with (out/f'{name}.log').open('w') as log:
            subprocess.run([local_python,*map(str,command)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

    def cpu_job(command,name):
        argv=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),*map(str,command)]
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(SSH+[shlex.join(argv)],stdout=log,stderr=subprocess.STDOUT,check=True)

    collections=[r/f'calibration24_ffn_operands/shard{g}/RUN.json' for g in [0,1]]
    fits=[r/f'calibration24_centered_fit/{noise}_RUN.json' for noise in ['high','low']]
    residual=r/'calibration24_residual_fit/RUN.json'
    base_policy=r/'calibration24_centered_fit/C24_centered_policy.json'
    policy=r/'calibration24_residual_fit/C24_RR_policy.json'
    try:
        stage('WAITING_FOR_BOTH_24_INPUT_COLLECTION_SHARDS')
        for path in collections:wait(path,'PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP')
        stage('FIT_FROZEN_C_CENTERED_PARAMETERS_BOTH_EXPERTS')
        jobs=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for gpu,noise in enumerate(['high','low']):
                if fits[gpu].exists() and json.loads(fits[gpu].read_text()).get('status')=='COMPLETED_CALIBRATION_PARAMETER_FIT':continue
                command=['python','tools/fit_cstatic_centered.py','--collections',*collections,'--expert',noise+'_noise_model',
                    '--inputs',ROOT/'.runtime/inputs/calibration24/canonical.json','--capture-protocol',ROOT/'config/calibration24_ffn_operands.json',
                    '--fit-protocol',ROOT/'config/cstatic_centered_cal24_fit.json','--out',f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/calibration24_centered_fit_20261021/{noise}',
                    '--report',fits[gpu]]
                jobs.append(pool.submit(gpu_job,gpu,command,f'centered_{noise}'))
            for job in jobs:job.result()
        local(['tools/write_trained_policy.py','--fits',*fits,'--protocol',ROOT/'config/cstatic_centered_cal24_fit.json','--output',base_policy],'write_base_policy')
        stage('FIT_FIXED_RANK256_LAMBDA1_RESIDUAL_ALL160')
        if not residual.exists() or json.loads(residual.read_text()).get('status')!='COMPLETED_RR1_FULL160_CALIBRATION_FIT':
            gpu_job(1,['python','tools/fit_residual_full.py','--calibration24-fixed','--collections',*collections,'--base-fits',*fits,
                '--selection',r/'residual_rr1/RUN.json','--out','/var/tmp/ffnds-xiang/huggingface/ffnds-derived/calibration24_residual_fit_20261021',
                '--report',residual],'residual_full160')
        local(['tools/write_residual_policy.py','--base-policy',base_policy,'--residual-fit',residual,
            '--experiment','RR1_CALIBRATION24_FIXED_RANK256_LAMBDA1_20261021','--output',policy],'write_residual_policy')
        stage('GENERATE_FROZEN_FORMAL20_TWO_DISJOINT_SLICES')
        suites=[r/f'formal20_C24/gpu{gpu}/RUN.json' for gpu in [0,1]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[]
            for gpu in [0,1]:
                if suites[gpu].exists() and json.loads(suites[gpu].read_text()).get('status')=='PASS_FORMAL_GENERATION_SLICE_ONLY':continue
                indices=list(range(gpu*10,(gpu+1)*10))
                jobs.append(pool.submit(gpu_job,gpu,['python','tools/generate_formal_suite.py','--indices',*indices,
                    '--packed-policy',policy,'--out',suites[gpu].parent],f'formal_gpu{gpu}'))
            if not (r/'calibration24_research_diagnostic/run_meta.json').exists() or json.loads((r/'calibration24_research_diagnostic/run_meta.json').read_text()).get('status')!='COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC':
                jobs.append(pool.submit(cpu_job,['python','tools/evaluate_residual_rr1.py','--full-calibration24-fit',
                    '--matrices',r/'dq_matrix/MATRIX_dq_gpu1_tasks0_seeds20261001-20261002.json',r/'dq_matrix/MATRIX_dq_gpu0_tasks6_seeds20261001-20261002.json',
                    '--base-fits',*fits,'--correction',residual,'--parent',r/'ar2_uniform512','--out',r/'calibration24_research_diagnostic'],'research_diagnostic'))
            for job in jobs:job.result()
        merged=r/'formal20_C24/FULL_RUN.json'
        local(['tools/merge_formal_suites.py','--suites',*suites,'--output',merged],'merge_formal')
        stage('OFFICIAL_THREE_METRICS_AND_RESIDENT_ALL20_EQUIVALENCE')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(gpu_job,0,['bash','scripts/worker_eval_env.sh','python','tools/evaluate_vbench3.py','--suites',merged,
                '--out',r/'formal20_C24_vbench3','--name','C24_RR_formal20_seed42'],'vbench3'),
                pool.submit(gpu_job,1,['python','tools/verify_resident_formal.py','--reference-suite',merged,'--policy',policy,
                    '--out',r/'formal20_C24_resident_equivalence'],'resident_equivalence')]
            for job in jobs:job.result()
        stage('COMPARE_EACH_METRIC_ON_REUSED_FORMAL_BENCHMARK')
        comparison=r/'formal20_C24_comparison/COMPARISON.json'
        local(['tools/compare_vbench3.py','--dense',r/'formal20_dense_vbench3','--candidate',r/'formal20_C24_vbench3',
            '--output',comparison],'quality_comparison')
        meta.update(status='COMPLETED_CALIBRATION24_FROZEN_FORMAL_COMPARISON',comparison=json.loads(comparison.read_text()),
            formal20_is_reused_benchmark=True)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic(out/'RUN.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
