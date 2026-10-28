"""Export a completed isolated CPU build to a deterministic HF archive."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
ID='E0_SM120_KERNEL_OVERLAY_EXPORT_20261029'
BUILD_PROTOCOL='config/e0_sm120_lightx2v_kernel_build_cpu_20261029_v1.json'
BUILD_RUN='ffnds-derived/e0_sm120_lightx2v_kernel_build_cpu_20261029/attempt1/worker/results/RUN.json'
BUILD_EXIT='ffnds-derived/e0_sm120_lightx2v_kernel_build_cpu_20261029/attempt1/worker/EXIT_RECEIPT.json'
LOCAL=Path('/var/tmp/ffnds-xiang/huggingface/software/lightx2v_kernel_sm120_20261029_v1')
ELF_SHA='474675037c50a97589731cae5225ddf23f9264803205cd538369a75233a3bb88'
NAMES=('__init__.py','gemm.py','kv_cache.py','utils.py','version.py','common_ops.abi3.so')
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':120,'outer_seconds':150,'new_HF_bytes':4*1024*1024}


def need(v,msg):
    if not v:raise ValueError(msg)

def sha(b):return hashlib.sha256(b).hexdigest()
def read_binding(root,b):
    p=Path(b['path']);need(not p.is_absolute() and '..' not in p.parts,'invalid binding path');f=(root/p).resolve();need(f.is_relative_to(root.resolve()),'escaped root');d=f.read_bytes();need(len(d)==b['bytes'] and sha(d)==b['sha256'],'binding differs');return json.loads(d)
def binding(p,root):
    data=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(data),'sha256':sha(data)}

def stable_stat(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def validate_build(plan,hf):
    old=read_binding(ROOT,plan['build']['protocol']);run=read_binding(hf,plan['build']['RUN']);exit=read_binding(hf,plan['build']['exit'])
    need(plan['build']['RUN']['sha256']=='5c9120bcc38cb1491228930a32fc01728dfdb18ca05d2a1089a73382c7166545' and run['id']==exit['id']==old['id']=='E0_SM120_LIGHTX2V_KERNEL_BUILD_CPU_20261029' and run['status']=='COMPLETE_BUILD_NO_GPU_EXECUTION' and run['protocol_sha256']==exit['protocol_sha256']==plan['build']['protocol']['sha256'] and exit['run_sha256']==plan['build']['RUN']['sha256'] and exit['process_exited'] is True and type(exit['child_returncode']) is int and exit['child_returncode']==0 and exit['timeout'] is False and exit['automatic_retry'] is False,'actual completed build differs')
    need(run['artifact']['bytes']==1312256 and run['artifact']['sha256']==ELF_SHA and run['model_calls']==run['GPU_calls']==0,'ELF build identity differs')
    source={Path(x['path']).name:x for x in old['kernel_sources'] if '/python/lightx2v_kernel/' in x['path']}
    expected={n:{'bytes':source[n]['bytes'],'sha256':source[n]['sha256']} for n in NAMES if n!='common_ops.abi3.so'}
    expected['common_ops.abi3.so']={'bytes':1312256,'sha256':ELF_SHA}
    need(plan['files']==expected,'overlay expected identities differ')
    return old,run


def archive_files(folder,expected):
    output=io.BytesIO();records=[]
    with tarfile.open(fileobj=output,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for name in NAMES:
            path=folder/name;st=path.lstat();need(stat.S_ISREG(st.st_mode) and st.st_size==expected[name]['bytes'],'nonregular or wrong size')
            data=path.read_bytes();need(len(data)==st.st_size and sha(data)==expected[name]['sha256'] and stable_stat(path.lstat())==stable_stat(st),'source identity changed')
            if name.endswith('.so'):need(data[:4]==bytes((127,69,76,70)),'not ELF')
            member=tarfile.TarInfo('lightx2v_kernel/'+name);member.size=len(data);member.mode=0o644;member.uid=member.gid=member.mtime=0;member.uname=member.gname=''
            tar.addfile(member,io.BytesIO(data));records.append({'path':member.name,**expected[name]})
    data=output.getvalue();need(len(data)<=RESOURCES['new_HF_bytes'],'archive budget');return data,records


def draft(hf):
    build={'protocol':binding(ROOT/BUILD_PROTOCOL,ROOT),'RUN':binding(hf/BUILD_RUN,hf),'exit':binding(hf/BUILD_EXIT,hf)}
    old=read_binding(ROOT,build['protocol']);run=read_binding(hf,build['RUN']);source={Path(x['path']).name:x for x in old['kernel_sources'] if '/python/lightx2v_kernel/' in x['path']}
    files={n:{'bytes':source[n]['bytes'],'sha256':source[n]['sha256']} for n in NAMES if n!='common_ops.abi3.so'};files['common_ops.abi3.so']={'bytes':run['artifact']['bytes'],'sha256':run['artifact']['sha256']}
    plan={'id':ID,'status':'DRAFT_NOT_AUTHORIZED','build':build,'files':files,'resources':RESOURCES,'repo_sources':[binding(ROOT/'tools/export_sm120_kernel_overlay.py',ROOT),binding(ROOT/'tests/test_export_sm120_kernel_overlay.py',ROOT)],'output_HF_relative':'ffnds-derived/e0_sm120_kernel_overlay_export_20261029/attempt1/worker/results','note':'CPU byte export only; no import or native GPU test.'};validate_build(plan,hf);return plan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('draft','inspect','boundsource','run'));ap.add_argument('--hf-root',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);args=ap.parse_args();hf=args.hf_root.resolve()
    if args.command=='draft':need(not args.protocol.exists(),'draft exists');args.protocol.write_text(json.dumps(draft(hf),indent=2)+'\n');return
    raw=args.protocol.read_bytes();plan=json.loads(raw);need(plan['id']==ID and plan['resources']==RESOURCES,'plan differs');validate_build(plan,hf)
    for b in plan['repo_sources']:
        p=ROOT/b['path'];need(p.stat().st_size==b['bytes'] and sha(p.read_bytes())==b['sha256'],'source binding differs')
    if args.command=='inspect':print(json.dumps({'id':ID,'status':plan['status'],'build_exit':0}));return
    if args.command=='boundsource':
        data,records=archive_files(LOCAL/'overlay/lightx2v_kernel',plan['files']);print(json.dumps({'source_files':records,'archive_bytes':len(data),'archive_sha256':sha(data)}));return
    need(plan['status']=='FROZEN_REGISTERED' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2 and 'torch' not in sys.modules and 'numpy' not in sys.modules,'registered CPU scope required')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());standing=active['user_continuation_authorization_20261027'];scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID];need(standing.get('new_scope_registration_allowed') is True and len(scopes)==1 and scopes[0].get('protocol_sha256')==sha(raw) and scopes[0].get('resources')==RESOURCES and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'),'ACTIVE scope differs')
    out=(hf/plan['output_HF_relative']).resolve();need(out.is_relative_to(hf) and not out.exists(),'fresh output required');out.mkdir(parents=True);start=time.monotonic();report={'id':ID,'status':'STARTED_NO_RETRY','protocol_sha256':sha(raw),'build':plan['build'],'model_calls':0,'GPU_calls':0};(out/'RUN.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        data,records=archive_files(LOCAL/'overlay/lightx2v_kernel',plan['files']);need(time.monotonic()-start<=120,'wall exceeded');(out/'overlay.tar').write_bytes(data);manifest={'schema':'ffnds.sm120.kernel_overlay.v1','build':plan['build'],'files':records,'archive':{'root':'hf','path':str((out/'overlay.tar').relative_to(hf)),'bytes':len(data),'sha256':sha(data)},'native_kernel_executed':False};m=(json.dumps(manifest,indent=2)+'\n').encode();need(len(data)+len(m)<RESOURCES['new_HF_bytes'],'total output budget');(out/'MANIFEST.json').write_bytes(m);report.update(status='COMPLETE_OVERLAY_BYTES_EXPORTED',manifest={'root':'hf','path':str((out/'MANIFEST.json').relative_to(hf)),'bytes':len(m),'sha256':sha(m)},wall_seconds=time.monotonic()-start)
    except BaseException as e:report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:(out/'RUN.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
