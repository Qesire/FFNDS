"""Materialize seven owned metadata aliases to satisfy view-root containment."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_s2_causal_env_v4',ROOT/'tools/e0_s2_causal_worker_proof_v3.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
core,native=previous.core,previous.native
ID='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V4_20261029'
OUTPUT='ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_v4_20261029/attempt1/worker/results'
RESOURCES=core.RESOURCES;TOTAL=core.TOTAL
LIMITS={**core.LIMITS,'cache_aliases':0,'metadata_aliases_materialized':7,'metadata_bytes_copied':21454628,
        'metadata_SHA_bytes':3*21454628}
PRIOR='config/e0_s2_causal_idle_a100_master_cpu_20261029_v3.json'
core.ID=core.frozen.ID=ID;core.OUTPUT=OUTPUT;core.LIMITS=LIMITS
core.SOURCES=(*core.SOURCES,'tools/e0_s2_causal_worker_proof_v4.py','tests/test_e0_s2_causal_worker_proof_v4.py',PRIOR)
_draft=previous.draft


def draft():
    plan=_draft();old=native.read(ROOT/PRIOR);hf=Path('/home/xiang/.cache/huggingface');out=hf/old['output_HF_relative']
    plan['failed_environment_v3']={'protocol':native.binding(ROOT/PRIOR,ROOT),'exit':native.binding(out.parent/'EXIT_RECEIPT.json',hf),
        'environment':native.binding(out/'ENVIRONMENT.json',hf),'links':native.binding(out/'CACHE_LINKS.json',hf)}
    return plan


def validate(plan):previous.previous._validate(plan)
core.draft=draft;core.validate=validate


def verify_failed(plan,hf):
    previous.verify_failed(plan,hf);proof=plan['failed_environment_v3']
    receipt=native.read(native.bound(hf,proof['exit']));env=native.read(native.bound(hf,proof['environment']));links=native.read(native.bound(hf,proof['links']))
    native.need(receipt['id']=='E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V3_20261029'
        and receipt['protocol_sha256']==proof['protocol']['sha256'] and receipt['child_returncode']==1
        and receipt['process_exited'] is True and receipt['timeout'] is False and receipt['run_sha256'] is None
        and env['error']=='Bound input bytes/SHA differ: configuration.json'
        and env['copied_files']==env['copied_bytes']==env['GPU_calls']==env['model_calls']==0
        and links['predebited_metadata_links']==links['completed_metadata_links']==7,
        'Actual view-containment failure differs')


def materialize_owned_metadata(view,source_view,rows):
    prepared=[]
    for row in rows:
        name=Path(row['filename']);native.need(not name.is_absolute() and '..' not in name.parts,'Unsafe metadata name')
        dst=view/name;src=source_view/name
        native.need(dst.parent.resolve().is_relative_to(view.resolve()) and dst.is_symlink() and src.is_file()
            and dst.resolve()==src.resolve() and src.stat().st_size==row['bytes'] and native.sha(src)==row['sha256'],
            'Only verified V3 metadata aliases may be materialized')
        tmp=dst.with_name(dst.name+'.ffnds-v4.tmp');native.need(not tmp.exists() and not tmp.is_symlink(),'Fresh atomic copy required')
        st=dst.lstat();prepared.append((dst,src,tmp,(st.st_dev,st.st_ino,st.st_mtime_ns),row))
    for dst,src,tmp,identity,row in prepared:
        shutil.copyfile(src,tmp);native.need(tmp.stat().st_size==row['bytes'] and native.sha(tmp)==row['sha256'],'Metadata copy differs')
        st=dst.lstat();native.need(dst.is_symlink() and (st.st_dev,st.st_ino,st.st_mtime_ns)==identity,'Owned alias changed during copy')
        os.replace(tmp,dst)
        native.bound(view,{'path':row['filename'],'bytes':row['bytes'],'sha256':row['sha256']})
    return len(prepared)


def cache_entries(alias,hf):
    native.need(alias==core.ALIAS and hf.is_dir(),'Exact cache roots required')
    model=native.read(ROOT/core.TEMPLATE);view=Path(model['model_view_root']);source_view=hf/view.relative_to(alias)
    for item in model['models']:
        path=Path(item['path']);source=hf/path.relative_to(alias)
        native.need(path.is_symlink() and path.resolve()==source.resolve() and source.stat().st_size==item['bytes'],'Retained master changed')
    rows=model['model_view_metadata'];native.need(len(rows)==7 and sum(r['bytes'] for r in rows)==LIMITS['metadata_bytes_copied']
        and LIMITS['metadata_bytes_copied']+LIMITS['package_bytes_copied']<=RESOURCES['new_local_bytes'],'Exact metadata/package footprint differs')
    count=materialize_owned_metadata(view,source_view,rows)
    native.write(hf/OUTPUT/'CACHE_LINKS.json',{'retained_master_links':4,'predebited_metadata_regular_copies':7,
        'completed_metadata_regular_copies':count,'metadata_bytes_copied':sum(r['bytes'] for r in rows),
        'replaced_only_this_lineages_V3_metadata_aliases':True,'original_HF_cache_files_overwritten':0,'model_download_bytes':0})


core.cache_alias=cache_entries


def run(args):
    plan=native.read(args.protocol);verify_failed(plan,args.hf_root);core.run(args)
    out=args.hf_root/OUTPUT;report=native.read(out/'RUN.json');report['cache_links']=native.binding(out/'CACHE_LINKS.json',args.hf_root);native.write(out/'RUN.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','run'));p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));a=p.parse_args()
    if a.action=='draft':native.need(not a.protocol.exists(),'Fresh V4 draft required');native.write(a.protocol,draft())
    else:run(a)


if __name__=='__main__':main()
