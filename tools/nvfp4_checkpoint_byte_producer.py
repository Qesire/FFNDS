"""Download two pinned public checkpoint blobs once, verify SHA and inspect headers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import resource
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ID='E0_NVFP4_CHECKPOINT_BYTES_20261029'
REPO='lightx2v/LightWan2.2-A14B'
REV='f4b91eba1fd7dc2eb038191e2bca7ed33423032d'
SIZE=8379833592
FILES=[{'filename':'Wan2.2-I2V-A14B_NVFP4_Sparse_high.safetensors','bytes':SIZE,'sha256':'c77b92616d678fa8373462973a3d2e3ceefd2fbdb3ccf33454e74f4c79fcf07a'},{'filename':'Wan2.2-I2V-A14B_NVFP4_Sparse_low.safetensors','bytes':SIZE,'sha256':'d76b53286dd9b68f5d956941037c0a6fd5a1fe0fd05eb97f8bf8abf9915da562'}]
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':1800,'outer_seconds':1860,'new_HF_bytes':16777216,'checkpoint_payload_bytes_max':2*SIZE,'minimum_free_disk_bytes':4*SIZE}
LIMITS={'files':2,'checkpoint_payload_bytes':2*SIZE,'header_bytes_max':16777216,'automatic_scope_retry':False}


def need(v,msg):
    if not v:raise ValueError(msg)


def sha(b):return hashlib.sha256(b).hexdigest()


def inspect_header(raw,total):
    data=json.loads(raw);tensors={k:v for k,v in data.items() if k!='__metadata__'}
    need(bool(tensors),'empty checkpoint header')
    end=0
    for name,v in sorted(tensors.items(),key=lambda x:x[1]['data_offsets'][0]):
        need(isinstance(name,str) and isinstance(v.get('dtype'),str) and isinstance(v.get('shape'),list),'bad tensor metadata')
        a,b=v['data_offsets'];need(type(a) is int and type(b) is int and a==end and b>=a and b<=total,'invalid offsets');end=b
    need(end==total,'incomplete payload coverage')
    selected={k:v for k,v in tensors.items() if 'ffn' in k.lower()}
    return {'header_bytes':len(raw),'header_sha256':sha(raw),'all_tensor_count':len(tensors),'ffn_tensors':selected,
            'ffn_tensor_count':len(selected),'interpretation':'Header identities only; key semantics and native kernel execution not validated.'}


def stat_identity(st): return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def check_cache(hf):
    blobs=hf/'hub'/'models--lightx2v--LightWan2.2-A14B'/'blobs';total=0
    for item in FILES:
        final=blobs/item['sha256'];partial=blobs/(item['sha256']+'.incomplete')
        entries=[p for p in (final,partial) if p.exists()]
        need(len(entries)<=1,'final and incomplete duplicate payload')
        for path in entries:
            need(path.is_file() and not path.is_symlink(),'invalid cache blob')
            n=path.stat().st_size;need(n<=SIZE,'oversized cache blob');total+=n
    need(total<=2*SIZE,'checkpoint cache payload budget')
    return total


def file_hash_header(path,expected,read_counter=lambda _:None):
    h=hashlib.sha256();count=0
    with path.open('rb') as f:
        st=os.fstat(f.fileno());need(st.st_size==expected['bytes'],'file size differs')
        prefix=f.read(8);count+=len(prefix);read_counter(len(prefix));h.update(prefix);need(len(prefix)==8,'short header prefix')
        n=struct.unpack('<Q',prefix)[0];need(0<n<=LIMITS['header_bytes_max'] and n+8<=expected['bytes'],'header too large')
        header=f.read(n);count+=len(header);read_counter(len(header));h.update(header);need(len(header)==n,'short header')
        while count<expected['bytes']:
            data=f.read(min(1<<20,expected['bytes']-count));need(bool(data),'short payload');count+=len(data);read_counter(len(data));h.update(data)
        need(stat_identity(os.fstat(f.fileno()))==stat_identity(st) and stat_identity(path.stat())==stat_identity(st),'file identity changed')
    need(h.hexdigest()==expected['sha256'],'LFS SHA mismatch')
    return inspect_header(header,expected['bytes']-8-n)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--hf-root',type=Path,required=True);args=ap.parse_args();raw=args.protocol.read_bytes();p=json.loads(raw)
    need(p['id']==ID and p['status']=='FROZEN_REGISTERED' and p['repo_id']==REPO and p['revision']==REV and p['files']==FILES and p['resources']==RESOURCES and p['limits']==LIMITS,'frozen design differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'numpy' not in sys.modules and len(os.sched_getaffinity(0))==2,'CPU-only isolated worker required')
    for b in p['repo_sources']:
        path=(ROOT/b['path']).resolve();need(path.is_relative_to(ROOT) and path.stat().st_size==b['bytes'] and sha(path.read_bytes())==b['sha256'],'source binding')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());standing=active['user_continuation_authorization_20261027'];scope=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID]
    need(standing.get('status')=='ACTIVE_CURRENT_USER_DIRECTED_SESSION' and standing.get('new_scope_registration_allowed') is True and standing.get('automatic_retry') is False and len(scope)==1 and scope[0].get('protocol_sha256')==sha(raw) and scope[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and scope[0].get('limits')==LIMITS and scope[0].get('resources')==RESOURCES,'ACTIVE scope differs')
    hf=args.hf_root.resolve();out=(hf/p['output_HF_relative']).resolve();need(out.is_relative_to(hf) and not out.exists() and shutil.disk_usage(hf).free>=RESOURCES['minimum_free_disk_bytes'],'output or free disk')
    soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE);need(hard==resource.RLIM_INFINITY or hard>=SIZE,'RLIMIT_FSIZE hard limit too small');resource.setrlimit(resource.RLIMIT_FSIZE,(SIZE,SIZE));need(resource.getrlimit(resource.RLIMIT_FSIZE)==(SIZE,SIZE),'RLIMIT_FSIZE guard failed');check_cache(hf)
    out.mkdir(parents=True);started=time.monotonic();r={'id':ID,'protocol_sha256':sha(raw),'status':'STARTED_NO_AUTOMATIC_RETRY','predebited_files':0,'predebited_download_bytes':0,'completed_files':0,'actual_hash_bytes_read':0,'files':[],'model_calls':0,'GPU_calls':0,'network_transport_retry':'huggingface_hub library transport retry may occur; no scope retry','RLIMIT_FSIZE_bytes':SIZE,'actual_network_bytes':'UNKNOWN_NOT_EXPOSED_BY_CLIENT','cache_budget_identity':'2 files of fixed LFS size, standard cache incomplete file is promoted in place'}
    def save():
        data=(json.dumps(r,indent=2)+'\n').encode();need(len(data)<=RESOURCES['new_HF_bytes'],'output budget');tmp=out/'RUN.tmp';tmp.write_bytes(data);os.replace(tmp,out/'RUN.json')
    def count(n):r['actual_hash_bytes_read']+=n
    save()
    def timeout(*_):raise TimeoutError('download scope wall exceeded; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    try:
        os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        from huggingface_hub import hf_hub_download
        need('torch' not in sys.modules and 'numpy' not in sys.modules,'unexpected numerical import')
        for item in FILES:
            need(time.monotonic()-started<RESOURCES['wall_seconds'],'wall budget')
            r['cache_payload_before_bytes']=check_cache(hf)
            r['predebited_files']+=1;r['predebited_download_bytes']+=item['bytes'];r['active_file']=item['filename'];save()
            path=Path(hf_hub_download(repo_id=REPO,filename=item['filename'],revision=REV,cache_dir=str(hf/'hub'),force_download=False,local_files_only=False)).resolve()
            expected=(hf/'hub'/'models--lightx2v--LightWan2.2-A14B'/'blobs'/item['sha256']).resolve();need(path==expected,'unexpected HF blob path');r['cache_payload_after_bytes']=check_cache(hf)
            header=file_hash_header(path,item,count);r['files'].append({'filename':item['filename'],'cache_path':str(path.relative_to(hf)),'bytes':item['bytes'],'sha256':item['sha256'],'header':header});r['completed_files']+=1;save()
        need(r['completed_files']==r['predebited_files']==2 and r['actual_hash_bytes_read']==2*SIZE,'incomplete hash')
        r.update(status='COMPLETE_TWO_PUBLIC_NVFP4_CHECKPOINT_BYTES',active_file=None,wall_seconds=time.monotonic()-started);save()
    except BaseException as e:
        r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),wall_seconds=time.monotonic()-started);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
