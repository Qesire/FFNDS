"""Complete isolated engineering checks only after a frozen candidate passes quality."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT,SSH


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--label',required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,default=ROOT/'runs/worker_restart_20261020')
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+',args.label):ap.error('label must be a plain path component')
    r=args.run_root.resolve();label=args.label;out=r/f'{label}_acceptance_controller'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'status':'RUNNING','stage':None,'source_sha256':digest(Path(__file__)),
        'joint_engineering_acceptance_pass':False}

    def stage(name):
        meta['stage']=name;atomic_json(out/'RUN.json',meta);print(name,flush=True)

    def read(path):return json.loads(path.read_text()) if path.exists() else {}

    def wait(path,status):
        while True:
            d=read(path)
            if d.get('status')==status:return d
            if d.get('status','').startswith('FAILED'):raise RuntimeError(f'Required evidence failed: {path}: {d.get("error")}')
            time.sleep(10)

    def gpu_job(gpu,command,name):
        argv=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--',*map(str,command)]
        atomic_json(out/f'{name}_command.json',{'command':argv})
        for attempt in range(1,9):
            with (out/f'{name}_attempt{attempt}.log').open('w') as log:
                result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:time.sleep(15);continue
            if result.returncode:raise RuntimeError(f'{name} failed; inspect saved report/log')
            return
        raise RuntimeError(f'{name}: GPU lock stayed occupied')

    def wait_idle():
        while True:
            result=subprocess.run(SSH+['nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True)
            if not result.stdout.strip():return
            time.sleep(10)

    quality_path=r/f'formal20_{label}_comparison/COMPARISON.json'
    candidate_resident_path=r/f'formal20_{label}_resident_equivalence/RUN.json'
    dense_resident_path=r/'formal20_DQ_resident_equivalence/RUN.json'
    plain=r/f'{label}_task006_seed20261001'
    try:
        stage('WAIT_FOR_EACH_FORMAL_QUALITY_METRIC')
        quality=wait(quality_path,'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON')
        meta['quality']=quality
        if not quality['formal20_quality_pass']:
            meta.update(status='COMPLETED_QUALITY_REJECTED',stage='STOPPED_BEFORE_ADDITIONAL_ACCEPTANCE_WORK');return
        if quality['candidate_policy_sha256']!=digest(args.policy):raise RuntimeError('Quality policy differs')
        for path,sha in quality.get('candidate_runtime_sources',{}).items():
            if digest(path)!=sha:raise RuntimeError('Quality numerical runtime source changed')
        stage('DENSE_FULL20_IN_PARALLEL_WITH_CANDIDATE_RESIDENT_THEN_PAIRED_REFERENCE')
        def candidate_reference():
            resident=wait(candidate_resident_path,'PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT')
            if resident['identity']['packed_policy_sha256']!=quality['candidate_policy_sha256']:
                raise RuntimeError('Candidate resident/quality identities differ')
            if read(plain/'run_meta.json').get('status')!='PASS':
                gpu_job(1,['python','tools/run_dense_hif4.py','--cache-manifest',ROOT/'.runtime/hif4_rebuilt_cache.json',
                    '--input-manifest',ROOT/'.runtime/inputs/task_006_seed_20261001.json','--packed-reference-policy',args.policy,
                    '--out',plain],'candidate_paired_reference')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(candidate_reference)]
            if read(dense_resident_path).get('status')!='PASS_RESIDENT_DQ_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT':
                jobs.append(pool.submit(gpu_job,0,['python','tools/verify_resident_dense_formal.py',
                    '--reference-suite',r/'formal20_dense/RUN.json','--out',dense_resident_path.parent],'dense_resident_full20'))
            for job in jobs:job.result()
        candidate_resident=wait(candidate_resident_path,'PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT')
        dense_resident=wait(dense_resident_path,'PASS_RESIDENT_DQ_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT')
        wait(plain/'run_meta.json','PASS')
        timings={}
        for name,reference,policy in [('DQ',r/'dq_matrix/dq_task006_seed20261001_plain/run_meta.json',None),
                (label,plain/'run_meta.json',args.policy)]:
            stage('ISOLATED_WARMUP_PLUS_THREE_REPEATS_'+name)
            folder=r/f'resident_{name}_for_{label}_isolated_timing';report=folder/'RUN.json'
            if read(report).get('status')!='PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE':
                wait_idle()
                command=['python','tools/run_resident_ffn.py','--input-manifest',ROOT/'.runtime/inputs/task_006_seed_20261001.json',
                    '--reference-meta',reference,'--out',folder,'--measured-repeats','3']
                if policy:command+=['--policy',policy]
                gpu_job(0,command,'timing_'+name)
            wait(report,'PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE');timings[name]=report
        stage('COMPARE_ALL_REPEATS_AND_COMPLETE_COSTS')
        comparison=out/'FFN_COMPARISON.json'
        with (out/'compare_timing.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),'tools/compare_ffn_benchmark.py','--dense',str(timings['DQ']),
                '--candidate',str(timings[label]),'--output',str(comparison)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        speed=read(comparison)
        if speed['candidate_policy_sha256']!=quality['candidate_policy_sha256']:raise RuntimeError('Timing and quality policies differ')
        for path,sha in quality.get('candidate_runtime_sources',{}).items():
            if digest(path)!=sha:raise RuntimeError('Numerical runtime source changed during acceptance')
        meta.update(status='COMPLETED_JOINT_ENGINEERING_CHECKS',stage='COMPLETE',speed=speed,
            joint_engineering_acceptance_pass=bool(speed['full_FFN_speed_pass'] and quality['formal20_quality_pass']),
            resident_equivalence={'candidate_inputs':len(candidate_resident['completed']),'dense_inputs':len(dense_resident['completed'])},
            method_scope='Frozen static whole64-group support plus native BF16 rank256 residual; no input-dynamic or novelty claim')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(out/'RUN.json',meta)


if __name__=='__main__':main()
