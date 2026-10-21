"""Fit exactly the10 LR2 residuals after its complete fixed native-risk gate."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from onpolicy_residual import load_collections, native_sse, replay_rows
from reduced_rank_residual import fit_rrr, native_parameters
from run_ar2_restart import ROOT, tensor_sha
from run_late_rank_pilot import checked_store, ub1_position
from run_residual_rr1 import encode


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args = ap.parse_args()
    r = ROOT/'runs/worker_restart_20261020'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82*1024**3):
        raise RuntimeError('Node HF cache and80GiB reserve plus2GiB allowance required')
    cvpath = r/'late_block_rank_lr2/RUN.json'
    cv,cvreport,cvid = open_verified(cvpath,'COMPLETED_LR2_NATIVE_RISK_GATE_PASSED')
    protocol = ROOT/'config/late_block_rank_lr2.json'
    if (cvid['protocol_sha256'] != digest(protocol) or len(cvreport['CV']) != 4
            or not all(v['gate_pass'] for v in cvreport['CV'])
            or {(v['block'],v['fold']) for v in cvreport['CV']} != {(31,0),(31,1),(39,0),(39,1)}):
        raise RuntimeError('Exact complete preregistered LR2 gate required')
    for path,sha in cvid['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('LR2 pilot numerical source changed')
    records,bindings,capture = load_collections([r/f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]],ROOT)
    parentpath = r/'calibration24_residual_fit/C24_RR_policy.json'
    parent = json.loads(parentpath.read_text())
    ub1path = r/'ub1_step_budget_fit/UB1_policy.json'
    ub1 = json.loads(ub1path.read_text())
    if (bindings != cvid['collection_bindings'] or digest(ub1path) != cvid['UB1_policy_sha256']
            or digest(parentpath) != capture['student_policy_sha256']
            or ub1['parent_policy']['sha256'] != digest(parentpath)):
        raise RuntimeError('Full fit data, main or frozen residual differ from CV')
    old = checked_store(parent['residual_fit'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    changed = checked_store(ub1['changed_fits']['3'],'COMPLETED_UB1_STEP40_FIT')
    cb = ub1['cache_manifest']
    if digest(cb['path']) != cb['sha256']:
        raise RuntimeError('Effective quantized cache binding changed')
    cache = json.loads(Path(cb['path']).read_text())
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    sources = [Path(__file__),ROOT/'tools/run_late_rank_pilot.py',ROOT/'tools/run_folded_adapter_cv.py',
               ROOT/'tools/folded_adapter.py',ROOT/'tools/onpolicy_residual.py',ROOT/'tools/reduced_rank_residual.py']
    args.out.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    store = CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'CV_report_sha256':digest(cvpath),
        'CV_fingerprint':cv.fingerprint,'UB1_policy_sha256':digest(ub1path),'collection_bindings':bindings,
        'rank':512,'lambda':1.,'step':3,'blocks':list(range(30,40)),
        'sources':{str(p):digest(p) for p in sources},'torch':torch.__version__,'TF32':False})
    meta = {'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
            'step':3,'rank':512,'blocks':list(range(30,40)),'positions':[],
            'scope':'static partial last-step recovery; source-exposed C24-student observations, no video or speed acceptance',
            'quality_or_speed_acceptance':False}
    started = time.perf_counter()
    atomic_json(args.report,meta)
    try:
        for block in range(30,40):
            key = f'step3_block{block:02d}'
            part = store.load('positions',key)
            if part is None:
                cases,control = ub1_position(records,cache,parent,old,changed,block,quantizer)
                x = torch.cat([c['V1'] for c in cases])
                y = torch.cat([(c['TeacherY'].float()-c['UB1_main'].float())*c['gate'].float() for c in cases])
                model = fit_rrr(x,y,1.,512)
                parameters = native_parameters(model,cases[0]['gate'],512)
                frozen = sum(native_sse(c['UB1_main']+c['UB1_frozen_R'],c['TeacherY'],c['gate']) for c in cases)
                candidate = sum(native_sse(c['UB1_main']+replay_rows(c['V1'],c['scope'],parameters),c['TeacherY'],c['gate']) for c in cases)
                row = {'step':3,'block':block,'rank':512,'lambda':1.,'training_inputs':list(range(24)),
                       'parameter_sha256':{k:tensor_sha(v) for k,v in parameters.items()},
                       'native_control':control,'training_frozen_R256_SSE':frozen,'training_R512_SSE':candidate,
                       'training_R512_over_frozen_R256':candidate/frozen,'training_score_is_acceptance':False}
                store.save('positions',key,encode(parameters),row)
                part = store.load('positions',key)
                del cases,x,y,model,parameters
            meta['positions'].append(part['metadata'])
            atomic_json(args.report,meta)
            print(f'LR2 fitted {key}',flush=True)
        if len(meta['positions']) != 10 or [v['block'] for v in meta['positions']] != list(range(30,40)):
            raise RuntimeError('Exactly the10 registered last-step positions required')
        meta.update(status='COMPLETED_LR2_TEN_LAST_STEP_RESIDUAL_FIT',elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.report,meta)
        atomic_json(args.out/'run_meta.json',meta)


if __name__ == '__main__':
    main()
