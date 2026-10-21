"""LR1: fixed last-step capacity comparison on verified C24-student states."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from folded_adapter import replay_main_rows
from onpolicy_residual import load_collections, native_sse, replay_rows
from reduced_rank_residual import fit_rrr, native_parameters
from run_ar2_restart import ROOT, tensor_sha
from run_folded_adapter_cv import prepare
from run_residual_rr1 import encode


def checked_store(binding, status):
    if digest(binding['path']) != binding['sha256']:
        raise RuntimeError('Frozen parameter report changed')
    store, _, _ = open_verified(Path(binding['path']), status)
    if store.fingerprint != binding['fingerprint']:
        raise RuntimeError('Frozen parameter identity changed')
    return store


@torch.no_grad()
def ub1_position(records, cache, parent, old, changed, block, quantizer):
    # This independently proves old main/complete output bits against the capture.
    original = prepare(records, cache, parent, old, 3, quantizer, block=block)
    part = changed.load('positions', f'step3_block{block:02d}')
    if part is None:
        raise RuntimeError('Frozen UB1 last-step fit missing')
    arrays, info = part['arrays'], part['metadata']
    groups = arrays['groups']
    if (groups.shape != (162,) or len(np.unique(groups)) != 162
            or not np.array_equal(groups, np.sort(groups)) or groups.min() < 0 or groups.max() >= 216):
        raise RuntimeError('Exact frozen UB1 whole-group budget required')
    ids = torch.tensor((groups[:,None]*64+np.arange(64)).reshape(-1), device='cuda')
    shard = next(s for s in cache['shards'] if s['expert'] == 'low_noise_model' and s['block'] == block)
    if digest(shard['path']) != shard['sha256']:
        raise RuntimeError('Original effective weight shard changed')
    payload = torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)['low_noise_model']
    w0 = payload[f'blocks.{block}.ffn.0']['weight_qdq'].cuda().index_select(0,ids).contiguous()
    w2full = payload[f'blocks.{block}.ffn.2']['weight_qdq']
    if tensor_sha(w2full) != info['original_W2_sha256']:
        raise RuntimeError('Frozen UB1 full W2 changed')
    w2 = w2full.cuda().index_select(1,ids).contiguous()
    smooth = payload[f'blocks.{block}.ffn.2']['smooth_scale'].cuda().index_select(0,ids).contiguous()
    b2 = torch.from_numpy(arrays['b2']).to(device='cuda',dtype=torch.bfloat16)
    residual = {k[3:]:torch.from_numpy(v).to(device='cuda',dtype=torch.bfloat16) for k,v in arrays.items() if k.startswith('rr_')}
    if (tensor_sha(b2) != info['b2_sha256']
            or {k:tensor_sha(v) for k,v in residual.items()} != info['RR_parameter_sha256']):
        raise RuntimeError('Frozen UB1 bias/residual encoding changed')
    view = json.loads((ROOT / '.runtime/node_local_model_view.json').read_text())
    source = next(f for f in view['files'] if '/low_noise_model/' in f['view_path'] and f['filename'].endswith('.safetensors'))
    with safe_open(source['view_path'],framework='pt',device='cpu') as handle:
        b0full = handle.get_tensor(f'blocks.{block}.ffn.0.bias').to(device='cuda',dtype=torch.bfloat16)
    b0 = b0full.index_select(0,ids).contiguous()
    cases = original['cases']
    for case in cases:
        if (tensor_sha(b0full) != case['scope']['original_b0_sha256']
                or info['original_b2_sha256'] != case['scope']['original_b2_sha256']
                or info['gate_sha256'] != case['scope']['gate_sha256']):
            raise RuntimeError('Native raw bias/gate differs')
        case['UB1_main'] = replay_main_rows(case['V1'],case['scope'],w0,b0,w2,b2,smooth,info['alpha'],quantizer)
        case['UB1_frozen_R'] = replay_rows(case['V1'],case['scope'],residual)
    control = {'step':3,'block':block,'original_C24_control':original['control'],
               'UB1_main_recomputed_not_observed':True,'groups':groups.tolist(),'alpha':info['alpha'],
               'UB1_frozen_R_parameter_sha256':info['RR_parameter_sha256'],
               'W0_sha256':tensor_sha(w0),'W2_sha256':tensor_sha(w2),'b0_sha256':tensor_sha(b0),
               'b2_sha256':tensor_sha(b2),'smooth2_sha256':tensor_sha(smooth)}
    return cases,control


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args = ap.parse_args()
    r = ROOT / 'runs/worker_restart_20261020'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    condition_path = r / 'tf2_flow/RUN.json'
    condition = json.loads(condition_path.read_text())
    if condition['status'] != 'COMPLETED_TF2_SAME_DENSE_STATE_DIAGNOSTIC':
        raise RuntimeError('Complete four-source TF2 before LR1')
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82*1024**3):
        raise RuntimeError('HF derived cache and80GiB reserve plus2GiB allowance required')
    protocol = ROOT / 'config/late_rank_lr1.json'
    args.out.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    errors = [v['one_step_post_error']['squared_error'] for v in condition['per_step']]
    if len(errors) != 4 or errors[3] != max(errors):
        atomic_json(args.report,{'status':'NOT_RUN_LR1_TF2_FINAL_STEP_CONDITION_FALSE','condition_sha256':digest(condition_path),
                                'protocol_sha256':digest(protocol),'post_SSE':errors,'quality_or_speed_acceptance':False})
        return
    records,bindings,capture = load_collections([r/f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]],ROOT)
    parentpath = r/'calibration24_residual_fit/C24_RR_policy.json'
    parent = json.loads(parentpath.read_text())
    if digest(parentpath) != capture['student_policy_sha256']:
        raise RuntimeError('Observation student changed')
    old = checked_store(parent['residual_fit'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    ub1path = r/'ub1_step_budget_fit/UB1_policy.json'
    ub1 = json.loads(ub1path.read_text())
    quality = json.loads((r/'disjoint20_UB1_comparison/COMPARISON.json').read_text())
    if digest(ub1path) != quality['candidate_policy_sha256'] or ub1['parent_policy']['sha256'] != digest(parentpath):
        raise RuntimeError('Frozen UB1 baseline changed')
    changed = checked_store(ub1['changed_fits']['3'],'COMPLETED_UB1_STEP40_FIT')
    cb = ub1['cache_manifest']
    if digest(cb['path']) != cb['sha256']:
        raise RuntimeError('Effective cache binding changed')
    cache = json.loads(Path(cb['path']).read_text())
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    sources = [Path(__file__),ROOT/'tools/folded_adapter.py',ROOT/'tools/run_folded_adapter_cv.py',
               ROOT/'tools/onpolicy_residual.py',ROOT/'tools/reduced_rank_residual.py',ROOT/'tools/run_residual_rr1.py']
    store = CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'protocol':json.loads(protocol.read_text()),
        'TF2_condition_sha256':digest(condition_path),'UB1_policy_sha256':digest(ub1path),'collection_bindings':bindings,
        'sources':{str(p):digest(p) for p in sources},'torch':torch.__version__,'TF32':False})
    meta = {'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
            'controls':[],'CV':[],'quality_or_speed_acceptance':False,
            'scope':'static last-step native risk pilot on C24-student states; main source-exposed, not UB1 own states or video acceptance'}
    started = time.perf_counter()
    atomic_json(args.report,meta)
    try:
        for block in [10,30]:
            cases,control = ub1_position(records,cache,parent,old,changed,block,quantizer)
            meta['controls'].append(control)
            for fold in [0,1]:
                key = f'step3_block{block}_fold{fold}'
                part = store.load('CV_models',key)
                if part is None:
                    train = [c for c in cases if c['index']%2 != fold]
                    val = [c for c in cases if c['index']%2 == fold]
                    if len(train) != 12 or len(val) != 12 or sum(c['index']<12 for c in train) != 6:
                        raise RuntimeError('Balanced12/12 source folds required')
                    x = torch.cat([c['V1'] for c in train])
                    y = torch.cat([(c['TeacherY'].float()-c['UB1_main'].float())*c['gate'].float() for c in train])
                    frozen = sum(native_sse(c['UB1_main']+c['UB1_frozen_R'],c['TeacherY'],c['gate']) for c in val)
                    row = {'step':3,'block':block,'fold':fold,'train_inputs':[c['index'] for c in train],
                           'validation_inputs':[c['index'] for c in val],'frozen_UB1_R256_native_SSE':frozen}
                    arrays = {}
                    for rank in [256,512]:
                        model = fit_rrr(x,y,1.,rank)
                        parameters = native_parameters(model,cases[0]['gate'],rank)
                        sse = sum(native_sse(c['UB1_main']+replay_rows(c['V1'],c['scope'],parameters),c['TeacherY'],c['gate']) for c in val)
                        row[f'refit_R{rank}_native_SSE'] = sse
                        row[f'refit_R{rank}_parameter_sha256'] = {k:tensor_sha(v) for k,v in parameters.items()}
                        arrays.update({f'rank{rank}_'+k:v for k,v in encode(parameters).items()})
                        del model,parameters
                    candidate = row['refit_R512_native_SSE']
                    matched = row['refit_R256_native_SSE']
                    if frozen <= 0 or matched <= 0 or not np.isfinite([candidate,matched,frozen]).all():
                        raise RuntimeError('Finite nondegenerate native risks required')
                    row.update(candidate_over_matched_R256=candidate/matched,candidate_over_frozen_UB1_R256=candidate/frozen,
                               gate_pass=candidate/matched <= .95 and candidate/frozen <= .95,lambda_ridge=1.,
                               original_fullM_native_inference=True)
                    store.save('CV_models',key,arrays,row)
                    part = store.load('CV_models',key)
                meta['CV'].append(part['metadata'])
                atomic_json(args.report,meta)
                print('LR1 '+json.dumps(part['metadata']),flush=True)
            del cases
        passed = len(meta['CV'])==4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_LR1_NATIVE_RISK_GATE_PASSED' if passed else 'COMPLETED_LR1_NATIVE_RISK_GATE_REJECTED',
                    native_risk_gate_pass=passed,elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.report,meta)
        atomic_json(args.out/'run_meta.json',meta)


if __name__ == '__main__':
    main()
