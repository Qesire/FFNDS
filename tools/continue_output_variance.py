"""Finish the fixed VM1 control after the primary calibration24 resource window."""
import argparse
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
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,default=ROOT/'runs/worker_restart_20261020')
    args=ap.parse_args();r=args.run_root.resolve()
    out=r/'output_variance_vm1_pipeline';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'status':'RUNNING','stage':None,'source_sha256':digest(Path(__file__)),'quality_or_speed_acceptance':False}

    def stage(name):
        meta['stage']=name;atomic_json(out/'RUN.json',meta);print(name,flush=True)

    def read(path):return json.loads(path.read_text()) if path.exists() else {}

    def wait(path,status):
        while True:
            d=read(path)
            if d.get('status')==status:return d
            if d.get('status','').startswith('FAILED'):raise RuntimeError(f'Required evidence failed: {path}: {d.get("error")}')
            time.sleep(10)

    def local(command,name):
        with (out/f'{name}.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),*map(str,command)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

    def gpu_job(gpu,command,name):
        argv=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--',*map(str,command)]
        atomic_json(out/f'{name}_command.json',{'command':argv})
        for attempt in range(1,9):
            with (out/f'{name}_attempt{attempt}.log').open('w') as log:
                result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:time.sleep(15);continue
            if result.returncode:raise RuntimeError(f'{name} failed; inspect saved log/report')
            return
        raise RuntimeError(f'{name}: GPU lock stayed occupied')

    fit_path=r/'output_variance_vm1/RUN.json';base=r/'residual_rr1_full160/RR1_policy.json'
    policy=r/'output_variance_vm1/VM1_policy.json';control_policy=policy.with_name('VM1_identity_policy.json')
    probe=r/'output_variance_vm1_cuda_probe/RUN.json'
    try:
        stage('WAIT_FOR_MOMENT_FIT_OR_CHECKPOINT_YIELD')
        while True:
            fit=read(fit_path)
            if fit.get('status')=='COMPLETED_VM1_FULL160_MOMENT_FIT':break
            if fit.get('status')=='YIELDED_RESUMABLE_CPU_RESOURCE_WINDOW':
                stage('WAIT_FOR_PRIMARY_CALIBRATION24_CPU_DIAGNOSTIC')
                wait(r/'calibration24_research_diagnostic/run_meta.json','COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC')
                command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/fit_output_variance.py',
                    '--collections',r/'calibration12_ffn_operands_strictseed/shard0/RUN.json',r/'calibration12_ffn_operands_strictseed/shard1/RUN.json',
                    '--base-policy',base,'--out','/home/xiang/.cache/huggingface/ffnds-derived/output_variance_vm1_20261021',
                    '--report',fit_path,'--stage','full','--gate-report',r/'output_variance_vm1_heldout/RUN.json','--max-seconds','1200']
                stage('RESUME_SAME_FROZEN_MOMENT_FIT')
                with (out/'resume_moment_fit.log').open('a') as log:
                    subprocess.run(SSH+[shlex.join(list(map(str,command)))],stdout=log,stderr=subprocess.STDOUT,check=True)
            elif fit.get('status','').startswith('FAILED'):raise RuntimeError(f'Moment fitting failed: {fit.get("error")}')
            else:time.sleep(10)
        stage('WAIT_FOR_PRIMARY_CALIBRATION24_GPU_WORKFLOW')
        primary=wait(r/'calibration24_pipeline/RUN.json','COMPLETED_CALIBRATION24_FROZEN_FORMAL_COMPARISON')
        if primary['comparison']['formal20_quality_pass']:
            meta.update(status='DEFERRED_FOR_CALIBRATION24_QUALITY_PASS',
                stage='PRIMARY_ACCEPTANCE_TIMING_AND_ENGINEERING_TAKE_RESOURCE_PRIORITY',
                reason='Keep frozen VM1 coefficients, prioritize complete acceptance of the candidate that passed formal quality')
            return
        stage('REAL_VALUE_CUDA_AFFINE_BITWISE_AND_IDENTITY_PROBE')
        if not probe.exists():
            gpu_job(0,['python','tools/probe_output_variance_cuda.py','--fit-report',r/'output_variance_vm1/PILOT_RUN.json',
                '--out',probe.parent],'cuda_probe')
        for destination,identity in [(policy,False),(control_policy,True)]:
            command=['tools/write_output_variance_policy.py','--base-policy',base,'--fit-report',fit_path,
                '--kernel-probe',probe,'--output',destination]
            if identity:command.append('--identity-control')
            local(command,'write_'+destination.stem)
        stage('IDENTITY_CONTROL_ALL_FOUR_NATIVE_LATENTS_AND_VIDEO')
        control=r/'output_variance_vm1_identity_task006_seed20261001'
        if not (control/'run_meta.json').exists():
            gpu_job(0,['python','tools/run_dense_hif4.py','--cache-manifest',ROOT/'.runtime/hif4_rebuilt_cache.json',
                '--input-manifest',ROOT/'.runtime/inputs/task_006_seed_20261001.json',
                '--packed-reference-policy',control_policy,'--out',control],'identity_video')
        # A separately scheduled early control may already own GPU0. Presence
        # of its atomic RUNNING report is not evidence that the control passed.
        wait(control/'run_meta.json','PASS')
        local(['tools/verify_output_variance_identity.py','--reference',r/'residual_rr1_task006_seed20261001/run_meta.json',
            '--candidate',control/'run_meta.json','--out',r/'output_variance_vm1_identity_verification'],'verify_identity')
        stage('GENERATE_FROZEN_VM1_FORMAL20_IN_TWO_DISJOINT_SLICES')
        suites=[r/f'formal20_VM1/gpu{g}/RUN.json' for g in [0,1]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[]
            for gpu in [0,1]:
                if read(suites[gpu]).get('status')=='PASS_FORMAL_GENERATION_SLICE_ONLY':continue
                jobs.append(pool.submit(gpu_job,gpu,['python','tools/generate_formal_suite.py','--indices',*range(gpu*10,(gpu+1)*10),
                    '--packed-policy',policy,'--out',suites[gpu].parent],f'formal_gpu{gpu}'))
            for job in jobs:job.result()
        merged=r/'formal20_VM1/FULL_RUN.json'
        local(['tools/merge_formal_suites.py','--suites',*suites,'--output',merged],'merge_formal')
        stage('EVALUATE_ALL_THREE_OFFICIAL_METRICS')
        gpu_job(0,['bash','scripts/worker_eval_env.sh','python','tools/evaluate_vbench3.py',
            '--suites',merged,'--out',r/'formal20_VM1_vbench3','--name','VM1_original12_RR1_formal20_seed42'],'vbench3')
        comparison=r/'formal20_VM1_comparison/COMPARISON.json'
        local(['tools/compare_vbench3.py','--dense',r/'formal20_dense_vbench3','--candidate',r/'formal20_VM1_vbench3',
            '--output',comparison],'quality_comparison')
        meta.update(status='COMPLETED_VM1_FROZEN_FORMAL_QUALITY_COMPARISON',comparison=read(comparison),
            formal20_is_reused_benchmark=True,full_FFN_timing_not_yet_measured=True)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(out/'RUN.json',meta)


if __name__=='__main__':main()
