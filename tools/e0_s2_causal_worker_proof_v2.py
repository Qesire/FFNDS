"""Preserve a failed root-alias attempt; link only five missing cache entries."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_s2_causal_env_v2',ROOT/'tools/e0_s2_causal_worker_proof.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
native=core.native
ID='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V2_20261029'
OUTPUT='ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_v2_20261029/attempt1/worker/results'
RESOURCES=core.RESOURCES;LIMITS={**core.LIMITS,'cache_aliases':5};TOTAL=core.TOTAL
PRIOR='config/e0_s2_causal_idle_a100_master_cpu_20261029_v1.json'
EXTRA=('tools/e0_s2_causal_worker_proof_v2.py','tests/test_e0_s2_causal_worker_proof_v2.py',PRIOR)
core.ID=core.frozen.ID=ID;core.OUTPUT=OUTPUT;core.LIMITS=LIMITS;core.SOURCES=(*core.SOURCES,*EXTRA)
_draft=core.draft;_validate=core.validate


def draft():
    plan=_draft();old=native.read(ROOT/PRIOR);out=Path('/home/xiang/.cache/huggingface')/old['output_HF_relative']
    plan['failed_environment_v1']={'protocol':native.binding(ROOT/PRIOR,ROOT),
        'exit':native.binding(out.parent/'EXIT_RECEIPT.json',out.parents[4]),
        'environment':native.binding(out/'ENVIRONMENT.json',out.parents[4])}
    return plan


def verify_failed(plan,hf):
    proof=plan['failed_environment_v1'];prior=native.read(native.bound(ROOT,proof['protocol']))
    receipt=native.read(native.bound(hf,proof['exit']));env=native.read(native.bound(hf,proof['environment']))
    native.need(prior['id']==receipt['id']=='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_20261029'
        and receipt['protocol_sha256']==proof['protocol']['sha256'] and receipt['child_returncode']==1
        and receipt['process_exited'] is True and receipt['run_sha256'] is None and receipt['timeout'] is False
        and env['status']=='FAILED_ENVIRONMENT_NO_RETRY' and env['error']=='Fresh cache alias required'
        and env['copied_files']==env['copied_bytes']==env['GPU_calls']==env['model_calls']==0
        and not (hf/prior['output_HF_relative']/'RUN.json').exists(),'Actual zero-payload environment failure differs')


def validate(plan):
    _validate(plan)


core.draft=draft;core.validate=validate


def link_missing(destination,target):
    native.need(target.exists() and not destination.exists() and not destination.is_symlink(),
                'Only absent local cache entries may be linked')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(target,target_is_directory=target.is_dir())
    native.need(destination.resolve()==target.resolve(),'Individual cache link differs')


def cache_entries(alias,hf):
    native.need(alias==core.ALIAS and hf.is_dir(),'Exact existing cache roots required')
    metadata=native.read(ROOT/core.TEMPLATE)
    local=[Path(m['path']) for m in metadata['models']]+[Path(metadata['model_view_root'])]
    native.need(len(local)==len(set(local))==5,'Exactly four masters and one view')
    targets=[hf/p.relative_to(alias) for p in local]
    native.need(all(t.exists() and not p.exists() and not p.is_symlink() for p,t in zip(local,targets)),
                'All five fixed destinations must be absent and cached sources present before any link')
    records=[]
    for p,t in zip(local,targets):
        link_missing(p,t);records.append({'local_path':str(p),'HF_relative':str(t.relative_to(hf)),
            'kind':'directory' if t.is_dir() else 'file','existing_cache_entry_overwritten':False})
    native.write(hf/OUTPUT/'CACHE_LINKS.json',{'predebited_links':5,'completed_links':5,'links':records,'model_download_bytes':0})


core.cache_alias=cache_entries


def run(args):
    plan=native.read(args.protocol);verify_failed(plan,args.hf_root);core.run(args)
    out=args.hf_root/OUTPUT;report=native.read(out/'RUN.json')
    report['cache_links']=native.binding(out/'CACHE_LINKS.json',args.hf_root);native.write(out/'RUN.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','run'));p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));a=p.parse_args()
    if a.action=='draft':native.need(not a.protocol.exists(),'Fresh V2 draft required');native.write(a.protocol,draft())
    else:run(a)


if __name__=='__main__':main()
