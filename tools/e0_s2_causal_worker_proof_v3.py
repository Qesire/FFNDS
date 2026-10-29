"""Validate four retained master links, then add seven missing metadata leaves."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_s2_causal_env_v3',ROOT/'tools/e0_s2_causal_worker_proof_v2.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
core,native=previous.core,previous.native
ID='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V3_20261029'
OUTPUT='ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_v3_20261029/attempt1/worker/results'
RESOURCES=core.RESOURCES;TOTAL=core.TOTAL
LIMITS={**core.LIMITS,'cache_aliases':7,'retained_master_links':4,'metadata_SHA_bytes':2*21454628}
PRIOR='config/e0_s2_causal_idle_a100_master_cpu_20261029_v2.json'
core.ID=core.frozen.ID=ID;core.OUTPUT=OUTPUT;core.LIMITS=LIMITS
core.SOURCES=(*core.SOURCES,'tools/e0_s2_causal_worker_proof_v3.py','tests/test_e0_s2_causal_worker_proof_v3.py',PRIOR)
_draft=previous.draft


def draft():
    plan=_draft();old=native.read(ROOT/PRIOR);hf=Path('/home/xiang/.cache/huggingface');out=hf/old['output_HF_relative']
    plan['failed_environment_v2']={'protocol':native.binding(ROOT/PRIOR,ROOT),
        'exit':native.binding(out.parent/'EXIT_RECEIPT.json',hf),'environment':native.binding(out/'ENVIRONMENT.json',hf)}
    return plan


def validate(plan):
    previous._validate(plan)


core.draft=draft;core.validate=validate


def verify_failed(plan,hf):
    previous.verify_failed(plan,hf)
    proof=plan['failed_environment_v2'];receipt=native.read(native.bound(hf,proof['exit']));env=native.read(native.bound(hf,proof['environment']))
    native.need(receipt['id']=='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V2_20261029'
        and receipt['protocol_sha256']==proof['protocol']['sha256'] and receipt['child_returncode']==1
        and receipt['process_exited'] is True and receipt['timeout'] is False and receipt['run_sha256'] is None
        and env['status']=='FAILED_ENVIRONMENT_NO_RETRY' and env['error']=='Only absent local cache entries may be linked'
        and env['copied_files']==env['copied_bytes']==env['GPU_calls']==env['model_calls']==0,
        'Actual partial-link V2 failure differs')


def link_metadata(view,source_view,metadata):
    pairs=[]
    for row in metadata:
        destination=core.safe_member(view,row['filename']);source=core.safe_member(source_view,row['filename'])
        native.need(source.is_file() and source.stat().st_size==row['bytes'] and native.sha(source)==row['sha256']
            and not destination.exists() and not destination.is_symlink(),'Metadata source identity or absent destination differs')
        pairs.append((destination,source))
    for destination,source in pairs:previous.link_missing(destination,source)
    return [{'filename':row['filename'],'bytes':row['bytes'],'sha256':row['sha256']} for row in metadata]


def cache_entries(alias,hf):
    native.need(alias==core.ALIAS and hf.is_dir(),'Exact cache roots required')
    model=native.read(ROOT/core.TEMPLATE);view=Path(model['model_view_root']);source_view=hf/view.relative_to(alias)
    native.need(view.is_dir() and not view.is_symlink() and source_view.is_dir(),'Retained partial view and cached source required')
    retained=[]
    for item in model['models']:
        path=Path(item['path']);source=hf/path.relative_to(alias)
        native.need(path.is_symlink() and path.resolve()==source.resolve() and source.is_file()
            and source.stat().st_size==item['bytes'],'Actual four retained master links differ')
        retained.append({'kind':item['kind'],'existing_link_reused':True,'payload_SHA':'CHECKED_BY_FOLLOWING_FULL_MASTER_PROOF'})
    native.need(len(retained)==4 and len(model['model_view_metadata'])==7,'Exact eleven-leaf view closure required')
    linked=link_metadata(view,source_view,model['model_view_metadata'])
    native.write(hf/OUTPUT/'CACHE_LINKS.json',{'predebited_metadata_links':7,'completed_metadata_links':len(linked),
        'retained_master_links_verified':retained,'metadata':linked,'model_download_bytes':0,'existing_entries_overwritten':0})


core.cache_alias=cache_entries


def run(args):
    plan=native.read(args.protocol);verify_failed(plan,args.hf_root);core.run(args)
    out=args.hf_root/OUTPUT;report=native.read(out/'RUN.json');report['cache_links']=native.binding(out/'CACHE_LINKS.json',args.hf_root);native.write(out/'RUN.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','run'));p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));a=p.parse_args()
    if a.action=='draft':native.need(not a.protocol.exists(),'Fresh V3 draft required');native.write(a.protocol,draft())
    else:run(a)


if __name__=='__main__':main()
