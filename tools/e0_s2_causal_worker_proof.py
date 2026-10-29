"""Prepare a byte-identical local package and verify existing HF raw masters.

This new finite CPU scope changes no weights and downloads no model. A local
path alias refers to the existing HF cache; the original model paths/numerical
recipe stay identical. Old worker proofs and failed attempts remain immutable.
"""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_s2_causal_private_master',ROOT/'tools/raw_bf16_worker_mirror_proof.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
native=frozen.native
ID='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_20261029'
frozen.ID=ID
TOTAL=frozen.TOTAL
OUTPUT='ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_20261029/attempt1/worker/results'
RESOURCES={'GPU':0,'CPU_threads':4,'hash_threads':2,'wall_seconds':600,'outer_seconds':900,
           'new_HF_bytes':8<<20,'new_local_bytes':64<<20,'automatic_retry':False}
LIMITS={'full_payload_bytes':TOTAL,'models':4,'native_headers':2,'package_files':2302,
        'package_bytes_copied':42053803,'cache_aliases':1,'GPU_calls':0,'model_calls':0}
ALIAS=Path('/var/tmp/ffnds-xiang/huggingface')
LOCAL=Path('/var/tmp/ffnds-xiang/python-packages/b0_20261028_v2')
TEMPLATE='config/e0_s2_new12_dense_baseline_part0_20261029_v2.json'
SOURCES=('tools/e0_s2_causal_worker_proof.py','tests/test_e0_s2_causal_worker_proof.py',
         'tools/raw_bf16_worker_mirror_proof.py','tools/worker_identity.py',TEMPLATE)


def draft():
    source=native.read(ROOT/TEMPLATE);plan=frozen.draft()
    plan.update(output_HF_relative=OUTPUT,resources=RESOURCES,limits=LIMITS,
                local_package=source['local_transformers'],alias=str(ALIAS),local_root=str(LOCAL))
    paths={b['path'] for b in plan['repo_sources']}|set(SOURCES)
    plan['repo_sources']=[native.binding(ROOT/p,ROOT) for p in sorted(paths)]
    return plan


def validate(plan):
    expected=draft();expected['private_worker_config_sha256']=plan.get('private_worker_config_sha256');expected['status']=plan.get('status')
    native.need(plan==expected and plan['id']==ID and plan['status'] in ('DRAFT_NOT_REGISTERED','FROZEN_REGISTERED'),
                'Exact new environment/master proof differs')
    if plan['status']=='FROZEN_REGISTERED':
        value=plan['private_worker_config_sha256'];native.need(isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef'),'Private execution SHA required')


def safe_member(root,name):
    p=Path(name)
    native.need(bool(name) and not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'Unsafe package member')
    target=(root/p).resolve();native.need(target.is_relative_to(root.resolve()),'Package member escaped root');return target


def cache_alias(alias,target):
    native.need(target.is_dir() and not alias.exists() and not alias.is_symlink(),'Fresh cache alias required')
    alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(target,target_is_directory=True)
    native.need(alias.resolve()==target.resolve(),'Cache alias resolution differs')


def run(args):
    started=time.monotonic();plan=native.read(args.protocol);validate(plan);digest=native.sha(args.protocol)
    active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s['id']==ID]
    native.need(len(rows)==1 and rows[0]['protocol_sha256']==digest and rows[0]['resources']==RESOURCES and rows[0]['limits']==LIMITS
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
        and active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules)
        and len(os.sched_getaffinity(0))<=4,'Exact hidden-CUDA CPU proof required')
    frozen.require_worker();native.need(native.sha(frozen.private_config_path())==plan['private_worker_config_sha256'],'Wrong executor')
    for b in plan['repo_sources']:native.bound(ROOT,b)
    out=(args.hf_root/OUTPUT).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh output required');out.mkdir(parents=True)
    env={'id':ID,'status':'STARTED_NODE_LOCAL_ENVIRONMENT','GPU_calls':0,'model_calls':0,
         'copied_files':0,'copied_bytes':0,'model_download_bytes':0,'automatic_retry':False}
    native.write(out/'ENVIRONMENT.json',env)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Environment preparation300s exceeded')));signal.alarm(300)
    try:
        cache_alias(ALIAS,args.hf_root)
        metadata=native.read(ROOT/TEMPLATE);native.model_metadata(metadata)
        info=plan['local_package'];inventory=native.read(native.bound(args.hf_root,info['inventory']))
        files=inventory['files'];native.need(len(files)==2302 and sum(r['bytes'] for r in files)==LIMITS['package_bytes_copied']
            and native.canonical(files)==info['inventory_sha256'] and importlib.metadata.version('transformers')==info['version'],'Frozen package identity differs')
        source=Path(importlib.util.find_spec('transformers').origin).parent
        destination=LOCAL/'transformers';native.need(not destination.exists() and not destination.is_symlink(),'No package overwrite permitted');destination.mkdir(parents=True)
        env.update(predebited_package_files=2302,predebited_package_bytes=LIMITS['package_bytes_copied']);native.write(out/'ENVIRONMENT.json',env)
        for row in files:
            src=safe_member(source,row['path']);dst=safe_member(destination,row['path'])
            native.need(src.is_file() and src.stat().st_size==row['bytes'],'Installed source package differs')
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            native.need(native.sha(dst)==row['sha256'] and dst.stat().st_size==row['bytes'],'Copied package SHA differs')
            env['copied_files']+=1;env['copied_bytes']+=row['bytes']
        env.update(status='PASS_BYTE_IDENTICAL_LOCAL_PACKAGE_AND_HF_ALIAS',wall_seconds=time.monotonic()-started,
            copied_package_SHA=info['inventory_sha256'],local_root_path_sha256=info['local_root_path_sha256'],
            alias_points_to_existing_HF=True,numerical_imports=False)
        native.need(env['wall_seconds']<=300 and not {'numpy','torch'}&set(sys.modules),'Environment wall/import exceeded')
    except BaseException as error:
        env.update(status='FAILED_ENVIRONMENT_NO_RETRY',error=str(error),error_type=type(error).__name__);raise
    finally:
        signal.alarm(0);native.write(out/'ENVIRONMENT.json',env)
    report=frozen.run_proof(plan,out,digest)
    native.need(time.monotonic()-started<=RESOURCES['wall_seconds'],'Total environment+hash wall600s exceeded')
    report.update(environment=native.binding(out/'ENVIRONMENT.json',args.hf_root),total_environment_plus_hash_wall_seconds=time.monotonic()-started)
    native.write(out/'RUN.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','run'));p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));a=p.parse_args()
    if a.action=='draft':native.need(not a.protocol.exists(),'Fresh draft required');native.write(a.protocol,draft())
    else:run(a)


if __name__=='__main__':main()
