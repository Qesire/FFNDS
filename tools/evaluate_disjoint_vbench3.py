"""Evaluate the frozen source-disjoint20 supplement with the same official workers.

Derived from evaluate_vbench3; metric/model/worker mechanics are unchanged.
Only the explicit supplemental input manifest and its identity check differ.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DIMS=['aesthetic_quality','imaging_quality','motion_smoothness']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path,value):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suites',type=Path,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--inputs',type=Path,default=ROOT/'.runtime/inputs/disjoint20/canonical.json')
    ap.add_argument('--handoff-root',type=Path,default=Path.home()/'.cache/ffnds/handoff-20261020/Wan22_FFN_HANDOFF_20261020')
    args=ap.parse_args()
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    kit=args.handoff_root/'evidence/context_20261011/reference/vbench3'
    worker=kit/'historical_runner.py'
    if sha(worker)!='abdf211cb14190118a80aeb52a97cc94e85d600300ae445982e8d2d9f781caad':
        raise RuntimeError('Historical worker source changed')
    models_path=ROOT/'.runtime/vbench_models.json'
    models=json.loads(models_path.read_text())
    if models['status']!='PASS_STAGED_VBENCH3_MODELS' or any(sha(r['path'])!=r['sha256'] for r in models['models']):
        raise RuntimeError('Evaluation model identity changed')
    inputs=json.loads(args.inputs.read_text())
    cases,identities,bindings={},{},[]
    for suite_path in args.suites:
        suite=json.loads(suite_path.read_text())
        if suite['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY':
            raise RuntimeError('Formal generation slice incomplete')
        identity={k:v for k,v in suite['identity'].items() if k!='indices'}
        if identity['inputs_sha256']!=sha(args.inputs):
            raise RuntimeError('Supplemental evaluation inputs differ from generation')
        if identities and identity!=next(iter(identities.values())):
            raise RuntimeError('Formal slices use different identities')
        identities[str(suite_path)]=identity
        bindings.append({'path':str(suite_path.resolve()),'sha256':sha(suite_path)})
        for row in suite['completed']:
            if row['index'] in cases or sha(row['case_report'])!=row['case_report_sha256']:
                raise RuntimeError('Duplicate/changed formal input')
            case=json.loads(Path(row['case_report']).read_text())
            if case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or case['input']!=inputs[row['index']] or sha(case['video']['path'])!=case['video']['sha256']:
                raise RuntimeError('Formal video/input identity changed')
            cases[row['index']]=case
    if set(cases)!=set(range(20)):
        raise RuntimeError('Exactly20 independent formal inputs required')
    work=args.out/'workdir'
    crop=work/'vbench2_beta_i2v/data/crop/16-9'
    crop.mkdir(parents=True,exist_ok=True)
    for sample in inputs:
        source=Path(sample['image_path'])
        target=crop/sample['image_name']
        if sha(source)!=sample['input_provenance']['image_sha256']:
            raise RuntimeError('Formal conditioning image changed')
        if target.exists():
            if sha(target)!=sha(source):
                raise RuntimeError('Existing evaluation conditioning image differs')
        else:
            target.symlink_to(source)
    videos=args.out/'videos'
    videos.mkdir(exist_ok=True)
    video_index=[]
    for index,case in sorted(cases.items()):
        for copy in range(5):
            target=videos/f'{case["input"]["prompt"]}-{copy}.mp4'
            if target.exists():
                if sha(target)!=case['video']['sha256']:
                    raise RuntimeError('Existing materialized video differs')
            else:
                try:
                    target.hardlink_to(case['video']['path'])
                except OSError:
                    shutil.copy2(case['video']['path'],target)
        video_index.append({'index':index,'source_group':case['input']['source_group'],'prompt':case['input']['prompt'],
                            'video':case['video'],'identical_filename_copies':5})
    if len(list(videos.glob('*.mp4')))!=100:
        raise RuntimeError('Expected100 names for20 independent sources')
    code=[]
    for name in ['vbench','vbench2_beta_i2v','pyiqa','clip']:
        spec=importlib.util.find_spec(name)
        for p in sorted(Path(spec.origin).parent.rglob('*.py')):
            code.append({'path':str(p),'sha256':sha(p)})
    site=Path(importlib.util.find_spec('vbench').origin).parent.parent
    protocol={'metrics':DIMS,'unique_generation_count':20,'materialized_names':100,'seed':42,'camera_excluded':True,
        'generation_suites':bindings,'generation_identity':next(iter(identities.values())),
        'historical_worker_sha256':sha(worker),'full_info_sha256':sha(kit/'VBENCH_FULL_INFO.json'),
        'code':code,'models_manifest_sha256':sha(models_path),'models':models['models'],
        'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','vbench','pyiqa','openai-clip','decord','setuptools']},
        'python':sys.version,'device':'cuda','local':True,'read_frame':False,'resolution':'16-9',
        'generation_cases':video_index,'source_sha256':sha(Path(__file__))}
    protocol_path=args.out/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise RuntimeError('Evaluation protocol changed; select fresh output')
    atomic(protocol_path,protocol)
    env=dict(os.environ,VBENCH_CACHE_DIR=models['cache_dir'],TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    scores,records={},[]
    meta={'status':'RUNNING','protocol_sha256':sha(protocol_path),'scores':scores,'workers':records,'quality_acceptance':False}
    atomic(args.out/'RUN.json',meta)
    try:
        for dim in DIMS:
            folder=args.out/dim
            folder.mkdir(exist_ok=True)
            report=folder/'worker_report.json'
            if not report.exists():
                command=[sys.executable,str(worker),'--worker-dimension',dim,'--vbench-repo',str(site),'--workdir',str(work),
                    '--full-info',str(kit/'VBENCH_FULL_INFO.json'),'--videos-dir',str(videos),'--output-dir',str(folder/'results'),
                    '--name',args.name,'--device','cuda','--resolution','16-9','--local','--allow-trusted-pickle-checkpoints','--output',str(report)]
                with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
                    subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=7200,check=True)
            result=json.loads(report.read_text())
            if not result['passed'] or result['dimension']!=dim or not math.isfinite(result['score']):
                raise RuntimeError('Invalid metric worker result')
            scores[dim]=result['score']
            records.append({'dimension':dim,'report':str(report),'report_sha256':sha(report),'result_sha256':sha(result['result_path'])})
            atomic(args.out/'RUN.json',meta)
            print(f'METRIC {dim}: {result["score"]}',flush=True)
        meta['status']='PASS_THREE_METRIC_EXECUTION_ONLY'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic(args.out/'RUN.json',meta)


if __name__=='__main__':
    main()
