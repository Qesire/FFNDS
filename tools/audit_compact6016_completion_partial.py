"""Registered CPU-only audit of completed cells in failed completion parts.

Termination observation proves a later read-only process/GPU inspection, not
parent launcher exit. Full export hashing occurs only in registered --execute.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import e0_compact6016_full80_completion as completion

ROOT = Path(__file__).resolve().parents[1]
ID = 'AUDIT_COMPACT6016_COMPLETION_PARTIAL_20261028'
LIMITS = {'parts':4, 'complete_cells':54, 'export_bytes_hashed_max':8 << 30,
          'GPU':0, 'model':0, 'forward':0, 'GEMM':0, 'QDQ':0, 'QR':0, 'fit':0, 'video':0,
          'CPU_threads':2, 'wall_seconds':600, 'outer_seconds':900, 'output_bytes':8 << 20}
SHAPES = {'W1':('BF16',[6016,5120]), 'W2':('BF16',[5120,6016]), 'b1':('BF16',[6016]),
          'b2':('BF16',[5120]), 'smooth1':('F32',[5120]), 'smooth2':('F32',[6016]),
          'groups':('I64',[94]), 'channels':('I64',[6016]), 'training_lora_A1':('F32',[32,5120]),
          'training_lora_B1':('F32',[6016,32]), 'training_lora_A2':('F32',[32,6016]),
          'training_lora_B2':('F32',[5120,32])}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def bind(root, item):
    root=Path(root).resolve(); p=(root/item['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
        raise ValueError('Bound file differs')
    return p


def write(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)


def verify_header(path):
    path=Path(path); size=path.stat().st_size
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('Missing safetensors header')
        n=struct.unpack('<Q',raw)[0]
        if n==0 or n>1<<20 or n+8>size: raise ValueError('Invalid safetensors header length')
        header=json.loads(f.read(n))
    header.pop('__metadata__',None)
    if set(header)!=set(SHAPES): raise ValueError('Tensor names differ')
    spans=[]
    for name,(dtype,shape) in SHAPES.items():
        item=header[name]
        if item['dtype']!=dtype or item['shape']!=shape: raise ValueError('Tensor shape or dtype differs')
        a,b=item['data_offsets']; count=1
        for v in shape: count*=v
        if type(a)!=int or type(b)!=int or a<0 or b-a!=count*({'BF16':2,'F32':4,'I64':8}[dtype]): raise ValueError('Tensor extent differs')
        spans.append((a,b))
    pos=0
    for a,b in sorted(spans):
        if a!=pos: raise ValueError('Tensor gaps or overlap')
        pos=b
    if pos!=size-8-n: raise ValueError('Payload size differs')
    return {'header_bytes':n,'payload_bytes':pos,'tensors':len(SHAPES)}


def check_termination(receipt, protocol_sha):
    if (receipt.get('schema')!='ffnds.completion.termination_observation.v1'
        or receipt.get('original_protocol_sha256')!=protocol_sha
        or receipt.get('parent_returncode')!='UNRECOVERED'
        or not isinstance(receipt.get('observed_utc'),str) or not receipt['observed_utc']
        or receipt.get('scope')!='ALL_FOUR_PARTS_BOTH_WORKERS'
        or receipt.get('same_name_processes')!=0 or receipt.get('assigned_gpu_processes')!=0
        or receipt.get('evidence_type')!='READ_ONLY_OBSERVATION'):
        raise ValueError('Termination observation absent or differs; not an exit receipt')


def parent_records(hf, plan):
    result=[]; protocol=plan['original_protocol']['sha256']
    if [p['part'] for p in plan['parts']]!=list(range(4)): raise ValueError('Four parts required')
    for part in plan['parts']:
        i=part['part']; r=json.loads(bind(hf,part['run']).read_text()); b=json.loads(bind(hf,part['budget']).read_text()); a=json.loads(bind(hf,part['attempt']).read_text())
        for x in (r,b,a):
            if x.get('id')!=completion.ID or x.get('part')!=i or x.get('protocol_sha256')!=protocol: raise ValueError('Parent identity differs')
        if (r.get('status')!='FAILED_NO_RETRY' or r.get('error_type')!='BrokenPipeError'
            or r.get('outer_process_completion')!='REQUIRES_LAUNCHER_PROCESS_EXIT_RECEIPT'
            or a.get('automatic_retry') is not False or len(r['cells'])!=[14,14,13,13][i]
            or any(r.get(k)!=b.get(k) for k in ('predebited','observed'))): raise ValueError('Parent failure/budget differs')
        for key, limit in completion.PART_LIMITS.items():
            pre, obs = b['predebited'].get(key), b['observed'].get(key)
            if type(pre) is not int or type(obs) is not int or not 0 <= obs <= pre <= limit: raise ValueError('Parent budget over limit')
        if (Path(hf)/Path(part['run']['path']).parent/'LAUNCH_EXIT.json').exists(): raise ValueError('Unexpected parent exit receipt')
        allowed=set(completion.run_ids(i)); seen=set()
        fs={(f['expert'],f['block']):f for f in r['functions']}
        if len(fs)!=len(r['functions']) or any(key not in completion.functions_for(i) for key in fs): raise ValueError('Function identity differs')
        for cell in r['cells']:
            if cell['run_id'] not in allowed or cell['run_id'] in seen: raise ValueError('Cell identity differs')
            seen.add(cell['run_id']); function=fs.get((cell['expert'],cell['block']))
            if function is None: raise ValueError('Missing support')
            verified=completion._completed_cell(Path(hf),cell,function,producer_id=completion.ID,producer_protocol=plan['original_protocol'],part=i)
            result.append(verified)
    if len(result)!=54 or len({v['run_id'] for v in result})!=54: raise ValueError('54 unique cells required')
    return result


def authorize(plan, digest):
    if (plan.get('id')!=ID or plan.get('assets_finalized') is not True or plan.get('limits')!=LIMITS
        or plan.get('automatic_retry') is not False or not isinstance(plan.get('termination_observation'),dict)):
        raise ValueError('Audit draft not finalized')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    scopes=active.get('experiment_authority',{}).get('current_user_authorized_scopes',[])
    if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in scopes):
        raise ValueError('Exact audit not registered in ACTIVE')


def worker(protocol, hf):
    plan=json.loads(protocol.read_text()); digest=sha(protocol); authorize(plan,digest)
    hf=hf.resolve(); out=(hf/plan['output_HF_relative']).resolve()
    if out==hf or not out.is_relative_to(hf): raise ValueError('Output outside HF')
    for item in plan['repo_sources']: bind(ROOT,item)
    original=bind(ROOT,plan['original_protocol']); old=bind(hf,plan['original40_manifest'])
    original_plan=json.loads(original.read_text())
    if (original_plan.get('id')!=completion.ID or original_plan.get('assets_finalized') is not True
        or original_plan.get('limits_per_part')!=completion.PART_LIMITS):
        raise ValueError('Original frozen completion design differs')
    prior_cells=completion.validate_partial_reuse(original_plan,hf)
    old_run=json.loads(bind(hf,plan['original40_run']).read_text()); old_exit=json.loads(bind(hf,plan['original40_exit']).read_text())
    if (old_run.get('status')!='COMPLETE_PARTIAL_ASSET_AUDIT' or old_run.get('manifest_sha256')!=plan['original40_manifest']['sha256']
        or old_exit.get('launcher_returncode')!=0 or old_exit.get('process_exited') is not True
        or old_exit.get('RUN_sha256')!=plan['original40_run']['sha256']): raise ValueError('Original 40 audit chain differs')
    termination=bind(hf,plan['termination_observation']); check_termination(json.loads(termination.read_text()),sha(original))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f: json.dump({'id':ID,'protocol_sha256':digest},f)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED','export_bytes_predebited':0,'export_bytes_verified':0};start=time.monotonic()
    def timeout(*_): raise TimeoutError('Audit wall exceeded')
    old_handler=signal.signal(signal.SIGALRM,timeout); signal.alarm(LIMITS['wall_seconds'])
    try:
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
        cells=parent_records(hf,plan); prior=json.loads(old.read_text())
        if (len(prior['cells'])!=40 or len(prior_cells)!=40
            or {(x['expert'],x['block'],x['initialization']) for x in prior['cells']}
               !={(x['expert'],x['block'],x['initialization']) for x in prior_cells}):
            raise ValueError('Original audit chain differs')
        newkeys={(x['expert'],x['block'],x['initialization']) for x in cells}
        oldkeys={(x['expert'],x['block'],x['initialization']) for x in prior['cells']}
        allkeys={(e,b,i) for e,b in completion.FUNCTIONS for i in completion.INITIALIZATIONS}
        if len(oldkeys)!=40 or newkeys & oldkeys or not newkeys|oldkeys <= allkeys: raise ValueError('Reuse overlap or geometry differs')
        remaining=sorted(allkeys-newkeys-oldkeys)
        partial_functions = {(e,b) for e,b,i in newkeys if (e,b,'energy94') in newkeys and (e,b,'random94') not in newkeys}
        if len(remaining)!=66 or len(partial_functions)!=2 or not all((e,b,'random94') in remaining for e,b in partial_functions): raise ValueError('Remaining or unpaired cell count differs')
        for cell in cells:
            ex=cell['export']; n=ex['bytes']
            if type(n)!=int or n<0 or report['export_bytes_predebited']+n>LIMITS['export_bytes_hashed_max']: raise ValueError('Hash budget exceeded')
            report['export_bytes_predebited']+=n; write(out/'RUN.json',report)
            path=(hf/ex['path']).resolve(); verify_header(path)
            if sha(path)!=ex['sha256']: raise ValueError('Export hash differs')
            report['export_bytes_verified']+=n
        manifest={'schema':'ffnds.compact6016.completion_partial_audit.v1','status':'AUDITED_COMPLETE_CELLS_FROM_FAILED_COMPLETION_PARTS',
                  'original40_chain':{'manifest':plan['original40_manifest'],'run':plan['original40_run'],'exit':plan['original40_exit']},
                  'parent_parts':plan['parts'],'parent_exit':'UNRECOVERED','termination_observation':plan['termination_observation'],
                  'completion_cells':cells,'original_cells':40,'completion_cell_count':54,'total_reusable_cells':94,
                  'remaining_cells':[{'expert':e,'block':b,'initialization':i} for e,b,i in remaining]}
        write(out/'MANIFEST.json',manifest)
        if (out/'MANIFEST.json').stat().st_size>LIMITS['output_bytes']: raise ValueError('Output budget exceeded')
        report.update(status='COMPLETE_PARTIAL_ASSET_AUDIT',manifest_sha256=sha(out/'MANIFEST.json'),cells=54,remaining=66,elapsed_seconds=time.monotonic()-start)
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e)); raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler);write(out/'RUN.json',report)


def launch(protocol,hf):
    plan=json.loads(protocol.read_text()); digest=sha(protocol); authorize(plan,digest)
    out=(hf.resolve()/plan['output_HF_relative']).resolve()
    if out==hf.resolve() or not out.is_relative_to(hf.resolve()): raise ValueError('Output outside HF')
    proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--protocol',str(protocol.resolve()),'--hf-root',str(hf.resolve())],timeout=LIMITS['outer_seconds'],check=False)
    run=out/'RUN.json'; out.mkdir(parents=True,exist_ok=True)
    receipt={'id':ID,'protocol_sha256':digest,'launcher_returncode':proc.returncode,'process_exited':True,'automatic_retry':False,'RUN_sha256':sha(run) if run.is_file() else None}
    with (out/'LAUNCH_EXIT.json').open('x') as f: json.dump(receipt,f,indent=2);f.write('\n')
    return proc.returncode


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--hf-root',type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--execute',action='store_true');mode.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    if args.worker: worker(args.protocol,args.hf_root)
    else: sys.exit(launch(args.protocol,args.hf_root))
