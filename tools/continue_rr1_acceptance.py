"""Wait for frozen RR1 evidence, then run isolated complete FFN acceptance checks."""
import argparse
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
    out=r/'RR1_acceptance_controller';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'status':'RUNNING','stage':None,'source_sha256':sha(Path(__file__)),'joint_engineering_acceptance_pass':False}

    def stage(name):
        if meta['stage']!=name:
            meta['stage']=name;atomic(out/'RUN.json',meta);print(json.dumps({'stage':name}),flush=True)

    def wait_report(path,status):
        while True:
            if path.exists():
                value=json.loads(path.read_text())
                if value['status']==status:return value
                if value['status']=='FAILED_RESUMABLE':raise RuntimeError(f'Required evidence failed: {path}: {value.get("error")}')
            time.sleep(10)

    def wait_idle():
        while True:
            result=subprocess.run(SSH+['nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],
                check=True,capture_output=True,text=True)
            if not result.stdout.strip():return
            time.sleep(10)

    def benchmark(arm):
        name='DQ' if arm=='dense' else 'RR1'
        folder=r/f'resident_{name}_isolated_timing'
        path=folder/'RUN.json'
        if path.exists():
            value=json.loads(path.read_text())
            if value['status']=='PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE' and value['identity']['measured_repeats']==3:
                return path
            raise RuntimeError(f'Existing timing requires review before a fresh attempt: {path}')
        reference=r/('dq_matrix/dq_task006_seed20261001_plain/run_meta.json' if arm=='dense'
            else 'residual_rr1_task006_seed20261001/run_meta.json')
        command=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu','0','--','python','tools/run_resident_ffn.py',
            '--input-manifest',str(ROOT/'.runtime/inputs/task_006_seed_20261001.json'),'--reference-meta',str(reference),
            '--out',str(folder),'--measured-repeats','3']
        if arm!='dense':command+=['--policy',str(r/'residual_rr1_full160/RR1_policy.json')]
        for attempt in range(1,7):
            wait_idle()
            atomic(out/f'{name}_command.json',{'command':command})
            with (out/f'{name}_attempt{attempt}.log').open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==3:
                time.sleep(10);continue
            if result.returncode:raise RuntimeError(f'{name} isolated benchmark failed; inspect its report/log')
            return path
        raise RuntimeError('GPU0 remained occupied despite idle probes')

    try:
        stage('WAITING_FOR_THREE_METRIC_COMPARISON')
        quality=wait_report(r/'formal20_RR1_comparison/COMPARISON.json','COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON')
        meta['quality']=quality
        if not quality['formal20_quality_pass']:
            meta.update(status='COMPLETED_RR1_QUALITY_REJECTED',stage='STOPPED_BEFORE_ADDITIONAL_ACCEPTANCE_TIMING')
            atomic(out/'RUN.json',meta);print(json.dumps(meta,indent=2));return
        stage('WAITING_FOR_ALL20_RESIDENT_EQUIVALENCE_BOTH_ARMS')
        resident=wait_report(r/'formal20_RR1_resident_equivalence/RUN.json','PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT')
        dense_resident=wait_report(r/'formal20_DQ_resident_equivalence/RUN.json','PASS_RESIDENT_DQ_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT')
        if resident['identity']['packed_policy_sha256']!=quality['candidate_policy_sha256']:
            raise RuntimeError('Resident and quality candidate identities differ')
        stage('ISOLATED_DENSE_WARMUP_PLUS_THREE_REPEATS')
        dense_path=benchmark('dense')
        stage('ISOLATED_CANDIDATE_WARMUP_PLUS_THREE_REPEATS')
        candidate_path=benchmark('candidate')
        comparison=out/'FFN_COMPARISON.json'
        command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/compare_ffn_benchmark.py',
            '--dense',str(dense_path),'--candidate',str(candidate_path),'--output',str(comparison)]
        with (out/'comparison.log').open('w') as log:
            subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
        # The node wrote an atomic report on NFS; read its exact bytes there.
        result=subprocess.run(SSH+[shlex.join(['cat',str(comparison)])],capture_output=True,text=True,check=True)
        speed=json.loads(result.stdout)
        if speed['candidate_policy_sha256']!=quality['candidate_policy_sha256']:
            raise RuntimeError('Timing and quality candidate identities differ')
        meta.update(status='COMPLETED_JOINT_ENGINEERING_CHECKS',stage='COMPLETE',speed=speed,
            joint_engineering_acceptance_pass=bool(speed['full_FFN_speed_pass'] and quality['formal20_quality_pass']),
            resident_equivalence={'candidate_inputs':len(resident['completed']),'dense_inputs':len(dense_resident['completed'])},
            method_scope='static whole64-group supports with calibrated BF16 low-rank residual; no input-dynamic or novelty claim')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic(out/'RUN.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
