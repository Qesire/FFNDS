"""One registered synthetic Safetensors/shared-HF I/O probe, no tensor framework."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import e0_original_bf16_videos as native
from worker_identity import private_config_path, require_worker

ROOT=Path(__file__).resolve().parents[1]
ID='E0_RAW_BF16_CAPTURE_IO_PROBE_20261029'
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':60,'outer_seconds':90,'local_bytes_max':64<<20,'HF_bytes_max':64<<20,'automatic_retry':False}
LIMITS={'direct_serializations':1,'local_serializations':1,'stream_copies':1,'unbuffered_log_writes':100,'buffered_log_writes':100,'buffered_explicit_flushes':1}
SHAPES={'V1':('BF16',[256,5120]),'h':('BF16',[256,13824]),'Y':('BF16',[256,5120]),'r':('BF16',[256,5120]),'u':('BF16',[256,5120]),'gate':('BF16',[5120]),'shift':('BF16',[5120]),'scale':('BF16',[5120]),'rows':('I64',[256])}
SOURCES=('tools/raw_bf16_capture_io_probe.py','tests/test_raw_bf16_capture_io_probe.py','tools/e0_original_bf16_videos.py','tools/worker_identity.py')


def draft():
    return {'id':ID,'status':'DRAFT_NOT_REGISTERED','private_worker_config_sha256':None,'resources':RESOURCES,'limits':LIMITS,
      'safetensors_version':'0.8.0','shapes':{k:[dtype,shape] for k,(dtype,shape) in SHAPES.items()},
      'local_output':'/tmp/ffnds_raw_bf16_capture_io_probe_20261029/attempt1',
      'output_HF_relative':'ffnds-derived/e0_raw_bf16_capture_io_probe_20261029/attempt1/worker/results',
      'repo_sources':[native.binding(ROOT/p,ROOT) for p in SOURCES],
      'identity':'synthetic zero byte payload only; direct Rust serialize versus local Rust serialize then one stream copy; log writes are 128 zero bytes each'}


def validate(plan):
    expected=draft();expected['status']=plan.get('status');expected['private_worker_config_sha256']=plan.get('private_worker_config_sha256')
    native.need(plan==expected and plan['status'] in ('DRAFT_NOT_REGISTERED','FROZEN_REGISTERED'),'Probe protocol differs')
    sha=plan['private_worker_config_sha256']
    native.need(sha is None or isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'Worker SHA format')
    if plan['status']=='FROZEN_REGISTERED': native.need(sha is not None,'Worker SHA missing')


def authorize(plan,digest,path):
    doc=native.read(ROOT/'research/ACTIVE_RESEARCH.json');a=doc['experiment_authority'];standing=doc['user_continuation_authorization_20261027']
    matches=[x for x in a['current_user_authorized_scopes'] if x.get('id')==ID]
    native.need(doc.get('status')=='E0_OPEN_DISCOVERY_USER_RESUMED_REGISTERED_SCOPES_ONLY' and a.get('all_cpu_gpu_model_experiments')=='USER_RESUMED_EXACT_REGISTERED_SCOPES_ONLY'
      and standing.get('new_scope_registration_allowed') is True and standing.get('automatic_retry') is False and len(matches)==1,'Standing authorization differs')
    scope=matches[0];native.need(scope.get('protocol')==str(path.resolve().relative_to(ROOT)) and scope.get('protocol_sha256')==digest
      and scope.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and scope.get('resources')==RESOURCES and scope.get('limits')==LIMITS,'Registered scope differs')


def make_specs(shapes, st):
    # One live zeroed buffer is shared by specs; Rust reads the pointer only while
    # serialize_file runs. No framework tensors or numerical input are involved.
    def count(shape):
        n=1
        for d in shape:n*=d
        return n
    largest=max(count(shape)*(8 if dtype=='I64' else 2) for dtype,shape in shapes.values())
    backing=bytearray(largest);pointer=ctypes.addressof(ctypes.c_char.from_buffer(backing))
    specs={k:st.TensorSpec(dtype=('bfloat16' if dtype=='BF16' else 'int64'),shape=shape,data_ptr=pointer,data_len=count(shape)*(8 if dtype=='I64' else 2)) for k,(dtype,shape) in shapes.items()}
    return backing,specs


def logs(path,buffering):
    row=bytes(128);start=time.perf_counter()
    with open(path,'xb',buffering=buffering) as stream:
        for _ in range(100): native.need(stream.write(row)==128,'Short log write')
        if buffering: stream.flush()
    return time.perf_counter()-start


def fileinfo(path):
    return {'bytes':path.stat().st_size,'sha256':native.sha(path)}


def run_probe(plan,out,local,digest):
    begin=time.monotonic();report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','automatic_retry':False,'model_calls':0,'GPU_calls':0,'operations':{},'timings':{},'files':{},'safetensors_version':plan['safetensors_version']}
    def persist():native.write(out/'RUN.json',report)
    persist()
    def timeout(*_):raise TimeoutError('Fixed wall60 exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    try:
        local.mkdir(parents=True,exist_ok=True)
        from importlib.metadata import version
        native.need(version('safetensors')==plan['safetensors_version'],'Safetensors runtime differs')
        import safetensors as st
        native.need(not {'torch','numpy'}&set(sys.modules),'Numeric framework imported')
        backing,specs=make_specs(SHAPES,st)
        native.need(len(backing)>0,'Missing live zero backing')
        def operation(key,fn):
            native.need(key not in report['operations'],'Duplicate operation')
            report['operations'][key]='PREDEBITED';persist();start=time.perf_counter();fn();report['timings'][key]=time.perf_counter()-start;report['operations'][key]='OBSERVED';persist()
        direct=out/'direct.safetensors';temp=local/'local.safetensors';copied=out/'copied.safetensors'
        operation('direct_serialization',lambda:st.serialize_file(specs,str(direct)))
        operation('local_serialization',lambda:st.serialize_file(specs,str(temp)))
        def copy():
            with temp.open('rb',buffering=0) as src, copied.open('xb',buffering=0) as dst:
                while True:
                    chunk=src.read(1<<20)
                    if not chunk:break
                    native.need(dst.write(chunk)==len(chunk),'Short copy')
        operation('one_stream_copy',copy)
        unbuffered=out/'unbuffered.log';buffered=out/'buffered.log'
        operation('unbuffered_100_writes',lambda:logs(unbuffered,0))
        operation('buffered_100_writes_one_flush',lambda:logs(buffered,65536))
        report['files']={'direct':fileinfo(direct),'local':fileinfo(temp),'copied':fileinfo(copied),'unbuffered_log':fileinfo(unbuffered),'buffered_log':fileinfo(buffered)}
        native.need(report['files']['direct']==report['files']['local']==report['files']['copied'],'Synthetic serialization/copy bytes differ')
        native.need(report['files']['buffered_log']==report['files']['unbuffered_log'] and report['files']['buffered_log']['bytes']==12800,'Synthetic log bytes differ')
        report['local_filesystem_device']=local.stat().st_dev;report['HF_filesystem_device']=out.stat().st_dev
        report['different_device_ids']=report['local_filesystem_device']!=report['HF_filesystem_device']
        report['local_payload_bytes']=temp.stat().st_size
        report['HF_payload_bytes']=sum(out.joinpath(n).stat().st_size for n in ('direct.safetensors','copied.safetensors','unbuffered.log','buffered.log'))
        native.need(report['local_payload_bytes']<RESOURCES['local_bytes_max'] and report['HF_payload_bytes']+(1<<20)<RESOURCES['HF_bytes_max'],'Byte scope exceeded')
        native.need(time.monotonic()-begin<60 and not {'torch','numpy'}&set(sys.modules),'Wall/import scope exceeded')
        report['status']='COMPLETE_SYNTHETIC_IO_ONLY'
    except BaseException as exc:
        report.update(status='FAILED_SYNTHETIC_IO_PRESERVED_NO_RETRY',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-begin;persist()
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('draft','inspect','run'));ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--hf-root',required=True,type=Path);args=ap.parse_args()
    if args.command=='draft':native.need(not args.protocol.exists(),'Fresh draft required');native.write(args.protocol,draft());return
    plan=native.read(args.protocol);validate(plan)
    if args.command=='inspect':print(json.dumps({'id':ID,'status':plan['status'],'resources':RESOURCES,'limits':LIMITS}));return
    native.need(plan['status']=='FROZEN_REGISTERED','Frozen protocol required');digest=native.sha(args.protocol);authorize(plan,digest,args.protocol)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2 and not {'torch','numpy'}&set(sys.modules),'CPU-only affinity/device/import scope differs')
    require_worker();native.need(native.sha(private_config_path())==plan['private_worker_config_sha256'],'Worker config mismatch')
    hf=args.hf_root.resolve();out=(hf/plan['output_HF_relative']).resolve();local=Path(plan['local_output'])
    native.need(out.is_relative_to(hf) and not out.exists() and local.is_absolute() and not local.exists(),'Fresh HF/local outputs required')
    out.mkdir(parents=True)
    print(json.dumps({'status':run_probe(plan,out,local,digest)['status']}))


if __name__=='__main__':main()
