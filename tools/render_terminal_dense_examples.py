"""Render the five preregistered examples once their native UB2 videos exist."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT,SSH


def main():
    r=ROOT/'runs/worker_restart_20261020';d=ROOT/'reports/worker_20261020'
    policy=r/'terminal_dense_ub2_fit/UB2_policy.json'
    selections=[('formal20',0),('formal20',10),('disjoint20',0),('disjoint20',10),('disjoint20',5)]
    registration=d/'UB2_visual_selection_before_quality.json'
    if registration.exists(): raise RuntimeError('Selection registration is immutable')
    atomic_json(registration,{'status':'FIXED_BEFORE_ANY_UB2_OFFICIAL_QUALITY_WORKER',
        'selections':[{'cohort':c,'index':i} for c,i in selections],
        'rationale':'same fixed0/10 plus previously known UB1 supplemental5 negative example; no UB2 outcome-based selection',
        'policy_sha256':digest(policy),'source_sha256':digest(Path(__file__)),
        'quality_report_exists_at_registration':{c:(r/f'{c}_UB2_vbench3/RUN.json').exists() for c in ['formal20','disjoint20']}})
    for cohort,index in selections:
        while True:
            path=r/f'{cohort}_UB2/RUN.json'
            suite=json.loads(path.read_text()) if path.exists() else {}
            if suite.get('status','').startswith('FAILED'): raise RuntimeError('Preserve native generation failure before any preview')
            ready=next((v for v in suite.get('completed',[]) if v['index']==index),None)
            if ready: break
            time.sleep(10)
        if suite['identity']['packed_policy_sha256']!=digest(policy): raise RuntimeError('Frozen policy changed')
        names=([r/'formal20_dense/RUN.json',r/'formal20_UB1/FULL_RUN.json'] if cohort=='formal20'
               else [r/'disjoint20_DQ/RUN.json',r/'disjoint20_UB1/RUN.json'])
        names += [r/f'{cohort}_LR2/RUN.json',path]
        cases=[]
        for source in names:
            row=next(v for v in json.loads(source.read_text())['completed'] if v['index']==index)
            if digest(row['case_report'])!=row['case_report_sha256']: raise RuntimeError('Example case changed')
            cases.append(row['case_report'])
        output=d/f'UB2_{cohort}_input{index:02d}_comparison.jpg'
        command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),str(ROOT/'.runtime/envs/vbench_official/bin/python'),
            'tools/render_formal_comparison.py','--cases',*cases,'--labels','D-Q','UB1','LR2','UB2','--output',str(output)]
        with (r/f'terminal_dense_ub2_controller/render_{cohort}_input{index:02d}.log').open('w') as log:
            subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Rendered {cohort} {index}',flush=True)


if __name__=='__main__': main()
