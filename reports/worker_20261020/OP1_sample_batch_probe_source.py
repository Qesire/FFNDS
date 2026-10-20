"""Require bitwise dense FFN teacher replays on four real native calibration scopes."""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json,digest
from dense_ffn_teacher import from_v1
from fit_cstatic_centered import load_training_reports
from run_ar2_restart import ROOT,tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    records,bindings,capture=load_training_reports(args.collections,expected_inputs=24,expected_rows=256)
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture['assets']['cache_manifest_sha256']!=digest(cache_path):raise RuntimeError('Original quantized cache differs')
    shard=next(r for r in cache['shards'] if r['expert']=='low_noise_model' and r['block']==20)
    if digest(shard['path'])!=shard['sha256']:raise RuntimeError('Full teacher shard changed')
    payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)['low_noise_model']
    w0=payload['blocks.20.ffn.0']['weight_qdq'].cuda();w2=payload['blocks.20.ffn.2']['weight_qdq'].cuda()
    smooth2=payload['blocks.20.ffn.2']['smooth_scale'].cuda()
    view=json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
    source=next(r for r in view['files'] if '/low_noise_model/' in r['view_path'] and r['filename'].endswith('.safetensors'))
    with safe_open(source['view_path'],framework='pt',device='cpu') as handle:b0=handle.get_tensor('blocks.20.ffn.0.bias').cuda()
    cases=[];started=time.perf_counter()
    for index in [0,1]:
        for step in [2,3]:
            scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==20)
            if digest(scope['path'])!=scope['sha256']:raise RuntimeError('Frozen dense calibration scope changed')
            with safe_open(scope['path'],framework='pt',device='cpu') as handle:
                x,a,y,b=[handle.get_tensor(k).cuda() for k in ['V1','V2','Y','b2']]
            teacher,v2=from_v1(x,w0,b0,w2,b,smooth2,qdq)
            if (not torch.equal(v2.view(torch.int16),a.view(torch.int16))
                    or not torch.equal(teacher.view(torch.int16),y.view(torch.int16))):
                raise RuntimeError(f'Teacher replay mismatch atinput{index} step{step}; no relaxed tolerance')
            cases.append({'index':index,'step':step,'block':20,'rows':len(x),'grid':scope['grid'],
                'V1_sha256':tensor_sha(x),'V2_sha256':tensor_sha(v2),'Y_sha256':tensor_sha(teacher),
                'bitwise_native_captured_V2_and_Y':True,'scope_sha256':scope['sha256']})
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'RUN.json',{'status':'PASS_DENSE_FFN_TEACHER_FOUR_NATIVE_GPU_REPLAYS',
        'collections':bindings,'protocol_sha256':digest(ROOT/'config/onpolicy_residual_op1.json'),
        'cache_manifest_sha256':digest(cache_path),'raw_checkpoint_sha256':source['sha256'],
        'W0_sha256':tensor_sha(w0),'W2_sha256':tensor_sha(w2),'b0_sha256':tensor_sha(b0),'smooth2_sha256':tensor_sha(smooth2),
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/dense_ffn_teacher.py']},
        'torch':torch.__version__,'GPU':torch.cuda.get_device_name(),'cases':cases,
        'elapsed_seconds':time.perf_counter()-started,'quality_or_speed_acceptance':False})


if __name__=='__main__':main()
