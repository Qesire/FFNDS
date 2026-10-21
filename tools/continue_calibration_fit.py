"""Wait for the two verified calibration shards, fit on both GPUs, then test on CPU."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from run_on_worker import SSH

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/worker_restart_20261020'


def fit(gpu):
    noise='high' if gpu==0 else 'low'
    output=f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/cstatic_centered_fit_20261021/{noise}'
    command=[sys.executable,str(ROOT/'tools/run_on_worker.py'),'--gpu',str(gpu),'--','python','tools/fit_cstatic_centered.py',
        '--collections',str(RUN/'calibration12_ffn_operands_strictseed/shard0/RUN.json'),str(RUN/'calibration12_ffn_operands_strictseed/shard1/RUN.json'),
        '--expert',noise+'_noise_model','--out',output,'--report',str(RUN/f'cstatic_centered_fit/{noise}_RUN.json')]
    for attempt in range(120):
        log=ROOT/f'runs/bootstrap_20261020/cstatic_fit_{noise}_attempt{attempt+1}.log'
        with log.open('wb') as handle:
            result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode==0:
            print(f'COMPLETED {noise} calibration fit',flush=True)
            return
        if result.returncode!=3:
            raise RuntimeError(f'{noise} fit failed, exit={result.returncode}; {log}')
        time.sleep(10)  # Previous owner may still be releasing its CUDA context.
    raise RuntimeError(f'GPU{gpu} did not become available')


def main():
    last=None
    while True:
        states=[]
        for shard in [0,1]:
            path=RUN/f'calibration12_ffn_operands_strictseed/shard{shard}/RUN.json'
            data=json.loads(path.read_text()) if path.exists() else {'status':'WAITING','completed':[]}
            if data['status'].startswith('FAILED'):
                raise RuntimeError(f'Calibration shard{shard} failed: {data.get("error")}')
            states.append((data['status'],len(data['completed'])))
        if states!=last:
            print('CALIBRATION',states,flush=True);last=states
        if all(s=='PASS_SIX_INPUT_FULL160_OPERANDS_AND_NOOP' and n==6 for s,n in states):
            break
        time.sleep(10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fit,[0,1]))
    command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/evaluate_cstatic_centered.py',
        '--matrices',str(RUN/'dq_matrix/MATRIX_dq_gpu1_tasks0_seeds20261001-20261002.json'),str(RUN/'dq_matrix/MATRIX_dq_gpu0_tasks6_seeds20261001-20261002.json'),
        '--fits',str(RUN/'cstatic_centered_fit/high_RUN.json'),str(RUN/'cstatic_centered_fit/low_RUN.json'),
        '--parent',str(RUN/'ar2_uniform512'),'--out',str(RUN/'cstatic_centered_heldout')]
    ssh=SSH+[shlex.join(command)]
    with (ROOT/'runs/bootstrap_20261020/cstatic_centered_heldout.log').open('wb') as handle:
        subprocess.run(ssh,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('COMPLETED fixed-policy held-out evaluation',flush=True)


if __name__=='__main__':
    main()
