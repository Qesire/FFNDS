"""Describe all40 observed trajectories of frozen UB1/LR2/UB2 without video gates."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from ar2_checkpoint import atomic_json,digest
from run_on_worker import ROOT,SSH


def visible_json(path):
    deadline=time.monotonic()+90
    while True:
        try:return json.loads(path.read_text())
        except FileNotFoundError:
            if time.monotonic()>=deadline:raise
            time.sleep(2)


def main():
    r=ROOT/'runs/worker_restart_20261020';d=ROOT/'reports/worker_20261020';out=r/'observed_trajectory_audit'
    out.mkdir(parents=True,exist_ok=True)
    completed=[]
    for cohort in ['formal20','disjoint20']:
        dense=r/('formal20_dense/RUN.json' if cohort=='formal20' else 'disjoint20_DQ/RUN.json')
        for arm in ['UB1','LR2','UB2']:
            candidate=r/(f'{cohort}_UB1/FULL_RUN.json' if cohort=='formal20' and arm=='UB1' else f'{cohort}_{arm}/RUN.json')
            while True:
                state=json.loads(candidate.read_text()) if candidate.exists() else {}
                if state.get('status')=='PASS_FORMAL_GENERATION_SLICE_ONLY':break
                if state.get('status','').startswith('FAILED'):raise RuntimeError('Inspect native generation failure first')
                time.sleep(10)
            folder=out/f'{cohort}_{arm}';report=folder/'RUN.json'
            if not report.exists():
                command=['bash',str(ROOT/'scripts/worker_cpu_exec.sh'),'python','tools/compare_formal_latents.py',
                    '--dense',str(dense),'--candidate',str(candidate),'--out',str(folder)]
                with (out/f'{cohort}_{arm}.log').open('w') as log:
                    subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
            result=visible_json(report)
            if (result['status']!='COMPLETED_PAIRED_FORMAL20_LATENT_DRIFT_DESCRIPTION'
                    or result['candidate_suite_sha256']!=digest(candidate) or result['dense_suite_sha256']!=digest(dense)):
                raise RuntimeError('Completed trajectory identity differs')
            import shutil
            for name in ['RUN.json','per_source_step.csv']:
                shutil.copyfile(folder/name,d/f'{arm}_{cohort}_trajectory_{name}')
            completed.append({'cohort':cohort,'arm':arm,'report':str(report),'sha256':digest(report),'summary':result['summary']})
            atomic_json(out/'RUN.json',{'status':'RUNNING','completed':completed,'quality_or_speed_acceptance':False})
            print('Described',cohort,arm,flush=True)
    atomic_json(out/'RUN.json',{'status':'COMPLETED_BOTH_COHORTS_THREE_POLICIES_TRAJECTORY_DESCRIPTIONS',
        'source_sha256':digest(Path(__file__)),'completed':completed,'quality_or_speed_acceptance':False,
        'scope':'descriptive propagated state differences only; no terminal attribution or implied video-quality pass'})


if __name__=='__main__':main()
