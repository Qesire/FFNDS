"""OD1 matched-input local O/D/D-Q functions, after eight native D controls."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open

from anchored_reconstruction import bf16_bits
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from distill_assets import ROOT,verified_assets
from onpolicy_residual import same_bits
from original_distill_functions import raw_native,quantized_native,paired_error,error_alignment,concentration
from run_ar2_restart import tensor_sha


def load_cases(root,expected_checkpoints):
    cases,bindings = [],[]
    for task in [0,6]:
        for seed in [20261001,20261002]:
            name = f'dbf16_task{task:03d}_seed{seed}'
            folder = root/f'{name}_observed'
            metapath = folder/'run_meta.json'
            nooppath = root/f'{name}_observer_noop.json'
            meta,noop = json.loads(metapath.read_text()),json.loads(nooppath.read_text())
            capturepath = folder/'ffn/CAPTURE_MANIFEST.json'
            capture = json.loads(capturepath.read_text())
            if (meta['status'] != 'PASS' or meta['arm'] != 'D-BF16' or meta['seed'] != seed
                    or meta['hif4_runtime_active'] or meta['checkpoint_sha256'] != expected_checkpoints
                    or noop['status'] != 'PASS_OBSERVER_NOOP_THIS_INPUT' or not noop['video_hash_equal']
                    or not all(noop['identity_checks'].values()) or Path(noop['observed']).resolve() != folder.resolve()
                    or meta['versions']['torch'] != torch.__version__
                    or not all(v['bitwise_equal'] for v in noop['checks']) or len(noop['checks']) != 4
                    or capture['status'] != 'PASS_CAPTURE_STRUCTURE_ONLY' or capture['completed_scopes'] != 16
                    or capture['source_id'] != meta['source_id'] or capture['seed'] != seed):
                raise RuntimeError('Exact observed D-BF16 and four-latent/video no-op evidence required')
            if (digest(ROOT/'tools/bf16_observer.py') != meta['observer_source_sha256']
                    or digest(meta['video']['path']) != meta['video']['sha256']
                    or digest(meta['image_path']) != meta['image_sha256']):
                raise RuntimeError('Native observer or captured video changed')
            for path,sha in meta['source_sha256'].items():
                if '/.runtime/code/' in path and digest(path) != sha:
                    raise RuntimeError('Observed native numerical pipeline source changed')
            bindings.append({'task':task,'seed':seed,'source_id':meta['source_id'],'image_sha256':meta['image_sha256'],'meta_sha256':digest(metapath),
                             'noop_sha256':digest(nooppath),'capture_sha256':digest(capturepath)})
            for step in [0,2]:
                scope = next(v for v in capture['scopes'] if v['step'] == step and v['block'] == 20)
                full_rows = int(np.prod(scope['grid']))
                salt = f"FFNDS_RESTART_UNIFORM_V1|{meta['source_id']}|{seed}|{scope['grid']}"
                sampling_seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8],'little')
                ids = np.sort(np.random.default_rng(sampling_seed).choice(full_rows,512,replace=False)).tolist()
                expert = 'high_noise_model' if step == 0 else 'low_noise_model'
                if (scope['row_count'] != 512 or scope['expert'] != expert or scope['selected_rows'] != ids
                        or scope['sampling_seed'] != sampling_seed or not all(scope['checks'].values())
                        or scope['inclusion_probabilities'] != [512/full_rows]*512
                        or scope['design_weights'] != [full_rows/512]*512
                        or digest(scope['tensor_file']) != scope['tensor_sha256']):
                    raise RuntimeError('Actual original uniform scope/coordinate/payload identity changed')
                with safe_open(scope['tensor_file'],framework='pt',device='cpu') as handle:
                    values = {key:handle.get_tensor(key) for key in ['U','Z','H','Y','b2','gate_broadcast','Delta_writeback_operand']}
                if (any(v.dtype != torch.bfloat16 or not torch.isfinite(v).all() for v in values.values())
                        or values['U'].shape != (512,5120) or values['Z'].shape != (512,13824)
                        or values['H'].shape != (512,13824) or values['Y'].shape != (512,5120)):
                    raise RuntimeError('Actual native BF16 captured geometry/precision required')
                cases.append({'task':task,'seed':seed,'step':step,'expert':expert,'scope':scope,'full_rows':full_rows,
                              'source_id':meta['source_id'],**values})
    per_task = [{v['image_sha256'] for v in bindings if v['task'] == task} for task in [0,6]]
    if any(len(v) != 1 for v in per_task) or len(set.union(*per_task)) != 2:
        raise RuntimeError('Exactly two independent images, each repeated across two seeds, required')
    return cases,bindings


def full_input(case):
    ids = torch.tensor(case['scope']['selected_rows'],device='cuda',dtype=torch.long)
    full = torch.zeros((case['full_rows'],5120),device='cuda',dtype=torch.bfloat16)
    full.index_copy_(0,ids,case['U'].cuda())
    return full,ids


def read_parameters(path,block=20):
    result,metadata = {},{}
    with safe_open(path,framework='pt',device='cpu') as handle:
        for slot in [0,2]:
            for kind in ['weight','bias']:
                key = f'blocks.{block}.ffn.{slot}.{kind}'
                raw = handle.get_tensor(key)
                native = raw.to(device='cuda',dtype=torch.bfloat16).contiguous()
                name = f'{slot}.{kind}'
                result[name] = native
                metadata[name] = {'source_dtype':str(raw.dtype),'source_tensor_sha256':tensor_sha(raw),
                                  'native_dtype':str(native.dtype),'native_tensor_sha256':tensor_sha(native),
                                  'evaluation_cast':'explicit source->BF16 before native addmm'}
    return result,metadata


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args = ap.parse_args()
    root = ROOT/'runs/worker_restart_20261020'
    prerequisite = json.loads((root/'late_block_rank_lr2_acceptance/RUN.json').read_text())
    if not prerequisite['status'].startswith('COMPLETED_LR2_'):
        raise RuntimeError('OD1 must wait until the LR2 quality/engineering controller terminates')
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82*1024**3):
        raise RuntimeError('HF derived outputs and80GiB reserve plus2GiB allowance required')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    kwargs,cache,assets = verified_assets()
    view = json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
    d_files = {expert:next(v for v in view['files'] if f'/{expert}/' in v['view_path'] and v['filename'].endswith('.safetensors'))
               for expert in ['high_noise_model','low_noise_model']}
    checkpoints = {v['filename']:v['sha256'] for v in d_files.values()}
    cases,bindings = load_cases(root,checkpoints)
    opath = ROOT/'.runtime/original_o_block20_assets.json'
    original = json.loads(opath.read_text())
    if (original['status'] != 'PASS_ORIGINAL_O_BLOCK20_SHARDS_COMPLETE_FILE_HASHES_AND_HEADERS'
            or original['identity']['revision'] != '206a9ee1b7bfaaf8f7e4d81335650533490646a3'
            or original['identity']['repo'] != 'Wan-AI/Wan2.2-I2V-A14B'
            or original['identity']['protocol_sha256'] != digest(ROOT/'config/original_o_block20_assets.json')
            or len(original['tensor_headers']) != 8 or original['complete_original_model']):
        raise RuntimeError('Verified partial pinned original-O assets required')
    o_files = {}
    for expert in d_files:
        names = original['selected_shards'][expert]
        if len(names) != 1:
            raise RuntimeError('Expected the single complete block20 shard per expert')
        row = next(v for v in original['files'] if v['filename'] == names[0])
        if digest(row['path']) != row['sha256'] or row['sha256'] != row['expected']['LFS_SHA256']:
            raise RuntimeError('Pinned original-O full checkpoint shard changed')
        o_files[expert] = row
    manifestpath = ROOT/'.runtime/hif4_rebuilt_cache.json'
    manifest = json.loads(manifestpath.read_text())
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    protocol = ROOT/'config/original_distill_ffn_od1.json'
    sources = [Path(__file__),ROOT/'tools/original_distill_functions.py',ROOT/'tools/bf16_observer.py',
               ROOT/'tools/onpolicy_residual.py',ROOT/'tools/distill_assets.py']
    args.out.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    store = CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'assets':assets,
        'original_O_asset_manifest_sha256':digest(opath),'original_O_shards':o_files,'D_checkpoints':d_files,
        'observed_DBF16_bindings':bindings,'LR2_terminal_report_sha256':digest(root/'late_block_rank_lr2_acceptance/RUN.json'),
        'sources':{str(p):digest(p) for p in sources},'torch':torch.__version__,'TF32':False})
    meta = {'status':'RUNNING_DBF16_NATIVE_CONTROLS','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
            'controls':[],'weight_precisions':{},'comparisons':[],'independent_source_images':2,'seeds_per_image':2,
            'quality_or_speed_acceptance':False,'scope':'matched-input local functions on D-BF16 U and shared D gate; no original-O trajectory or full-model comparison'}
    atomic_json(args.report,meta)
    started = time.perf_counter()
    try:
        # No O function is evaluated before all eight recorded native D controls.
        for expert in d_files:
            parameters,precision = read_parameters(d_files[expert]['view_path'])
            meta['weight_precisions']['D_'+expert] = precision
            for case in [c for c in cases if c['expert'] == expert]:
                full,ids = full_input(case)
                stride = case['scope']['effective_W2']['runtime_stride']
                z,h,y = raw_native(full,parameters,stride)
                outputs = {'Z':z.index_select(0,ids),'H':h.index_select(0,ids),'Y':y.index_select(0,ids)}
                gate = case['gate_broadcast'].reshape(1,5120).cuda()
                checks = {key:same_bits(value,case[key].cuda()) for key,value in outputs.items()}
                checks.update(b2=same_bits(parameters['2.bias'],case['b2'].cuda()),
                              native_gated_Delta=same_bits(outputs['Y']*gate,case['Delta_writeback_operand'].cuda()))
                row = {'task':case['task'],'seed':case['seed'],'step':case['step'],'expert':expert,'full_rows':case['full_rows'],
                       'tensor_payload_sha256':case['scope']['tensor_sha256'],'checks':checks}
                meta['controls'].append(row)
                atomic_json(args.report,meta)
                if not all(checks.values()):
                    raise RuntimeError('D-BF16 native Z/H/Y/Delta replay failed; no O/D result accepted')
                del full,z,h,y,outputs
            del parameters
        if len(meta['controls']) != 8:
            raise RuntimeError('All8 fixed native D controls required')
        meta['status'] = 'RUNNING_MATCHED_INPUT_O_D_DQ_FUNCTIONS'
        atomic_json(args.report,meta)
        for expert in d_files:
            d,dprec = read_parameters(d_files[expert]['view_path'])
            o,oprec = read_parameters(o_files[expert]['path'])
            meta['weight_precisions']['O_'+expert] = oprec
            shard = next(v for v in manifest['shards'] if v['expert'] == expert and v['block'] == 20)
            if digest(shard['path']) != shard['sha256']:
                raise RuntimeError('Native D-Q effective shard changed')
            payload = torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)[expert]
            records = [{k:v.cuda() if torch.is_tensor(v) else v for k,v in payload[f'blocks.20.ffn.{slot}'].items()} for slot in [0,2]]
            for case in [c for c in cases if c['expert'] == expert]:
                key = f'task{case["task"]:03d}_seed{case["seed"]}_step{case["step"]}'
                previous = store.load('functions',key)
                if previous is None:
                    full,ids = full_input(case)
                    oz,oh,oy = raw_native(full,o,case['scope']['effective_W2']['runtime_stride'])
                    original_h,original_y = oh.index_select(0,ids),oy.index_select(0,ids)
                    del oz,oh,oy
                    q2,qy = quantized_native(full,records,d['0.bias'],d['2.bias'],quantizer)
                    quantized_h,quantized_y = q2.index_select(0,ids),qy.index_select(0,ids)
                    del full,q2,qy
                    distilled_y,distilled_h = case['Y'].cuda(),case['H'].cuda()
                    gate = case['gate_broadcast'].reshape(1,5120).cuda()
                    outputs = {'O_BF16':original_y,'D_BF16':distilled_y,'D_Q_on_DBF16_U':quantized_y}
                    comparisons = {}
                    for name,a,b in [('D_vs_O','D_BF16','O_BF16'),('DQ_vs_D','D_Q_on_DBF16_U','D_BF16'),('DQ_vs_O','D_Q_on_DBF16_U','O_BF16')]:
                        comparisons[name] = {'ungated':paired_error(outputs[a],outputs[b]),
                                             'same_D_gate':paired_error(outputs[a]*gate,outputs[b]*gate)}
                    row = {'task':case['task'],'source_id':case['source_id'],'seed':case['seed'],'step':case['step'],
                           'expert':expert,'block':20,'rows':512,'full_rows':case['full_rows'],'comparison':comparisons,
                           'error_alignment_cosine':{'ungated':error_alignment(distilled_y,original_y,quantized_y),
                               'same_D_gate':error_alignment(distilled_y*gate,original_y*gate,quantized_y*gate)},
                           'C_concentration':{'O_BF16':concentration(original_h,o['2.weight'],gate),
                               'D_BF16':concentration(distilled_h,d['2.weight'],gate),
                               'D_Q_on_DBF16_U':concentration(quantized_h,records[1]['weight_qdq'],gate)}}
                    store.save('functions',key,{'O_Y_bits':bf16_bits(original_y),'DQ_Y_bits':bf16_bits(quantized_y),
                        'O_H_bits':bf16_bits(original_h),'DQ_V2_bits':bf16_bits(quantized_h)},row)
                    previous = store.load('functions',key)
                meta['comparisons'].append(previous['metadata'])
                atomic_json(args.report,meta)
                print(f'OD1 completed {key}',flush=True)
            del o,d,records,payload
        if len(meta['comparisons']) != 8:
            raise RuntimeError('Exactly8 registered matched-input scopes required')
        meta.update(status='COMPLETED_OD1_EIGHT_NATIVE_CONTROLS_AND_LOCAL_FUNCTION_COMPARISONS',elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.report,meta)
        atomic_json(args.out/'run_meta.json',meta)


if __name__ == '__main__':
    main()
