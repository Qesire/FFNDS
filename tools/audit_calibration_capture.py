"""Audit all calibration video geometry, sample definitions and operand headers."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from safetensors import safe_open

from ar2_checkpoint import atomic_json,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--verify-tensor-bytes',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    protocol=json.loads(args.protocol.read_text());inputs=json.loads(args.inputs.read_text())
    count=protocol.get('input_count',12);rows=protocol.get('rows_per_scope',protocol.get('rows',512))
    namespace=protocol.get('sampling_namespace','FFNDS_CALIB12_UNIFORM_V1')
    if len(inputs)!=count or (count,rows) not in [(12,512),(24,256)]:
        raise RuntimeError('Unexpected registered calibration geometry')
    expected_status='PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP' if count==24 else 'PASS_SIX_INPUT_FULL160_OPERANDS_AND_NOOP'
    reports=[];video_records=[];scope_count=0;payload_bytes=0;seen=set();started=time.perf_counter()
    for path in args.collections:
        report=json.loads(path.read_text());identity=report['identity']
        if (report['status']!=expected_status or identity['protocol_sha256']!=digest(args.protocol)
                or identity['calibration_sha256']!=digest(args.inputs)):
            raise RuntimeError('Calibration collection identity/status mismatch')
        reports.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':report['fingerprint']})
        for entry in report['completed']:
            if digest(entry['input_report'])!=entry['input_report_sha256']:raise RuntimeError('Input report changed')
            case=json.loads(Path(entry['input_report']).read_text());index=case['index']
            if index in seen or case['input']!=inputs[index] or case['status']!='PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP':
                raise RuntimeError('Input duplicate or altered no-op status')
            seen.add(index)
            if (case['trajectories']['plain']!=case['trajectories']['observed']
                    or [r['step'] for r in case['trajectories']['plain']]!=list(range(4))
                    or case['videos']['plain']['sha256']!=case['videos']['observed']['sha256']):
                raise RuntimeError('No-op trajectory/video bytes differ')
            for arm,video in case['videos'].items():
                if digest(video['path'])!=video['sha256']:raise RuntimeError('Calibration video changed')
                with imageio.get_reader(video['path']) as reader:
                    frames=reader.count_frames();info=reader.get_meta_data()
                if frames!=81 or abs(float(info['fps'])-16)>.01:raise RuntimeError('Frame count/fps differs')
                shape=case['trajectories'][arm][0]['shape']
                if list(info['size'])!=[shape[3]*8,shape[2]*8]:raise RuntimeError('Video geometry differs from native latent')
                video_records.append({'index':index,'arm':arm,'frames':frames,'fps':info['fps'],'size':info['size'],'sha256':video['sha256']})
            if len(case['scopes'])!=160 or {(s['step'],s['block']) for s in case['scopes']}!={(s,b) for s in range(4) for b in range(40)}:
                raise RuntimeError('Missing or duplicate FFN scopes')
            for scope in case['scopes']:
                n=int(np.prod(scope['grid']));step,block=scope['step'],scope['block']
                salt=f"{namespace}|{index}|{step}|{block}|{scope['grid']}"
                seed=int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8],'little')
                selected=np.sort(np.random.default_rng(seed).choice(n,rows,replace=False)).tolist()
                if (scope['rows']!=rows or scope['selected_rows']!=selected or scope['sampling_seed']!=seed
                        or scope['inclusion_probability']!=rows/n or scope['inverse_inclusion_weight']!=n/rows
                        or scope['expert']!=('high_noise_model' if step<2 else 'low_noise_model')):
                    raise RuntimeError('Sampling coordinates/weights/seed differ')
                p=Path(scope['path'])
                if p.stat().st_size!=scope['bytes']:raise RuntimeError('Operand byte length differs')
                if args.verify_tensor_bytes and digest(p)!=scope['sha256']:raise RuntimeError('Operand payload SHA differs')
                with safe_open(p,framework='numpy') as handle:
                    expected={'V1':[rows,5120],'V2':[rows,13824],'Y':[rows,5120],'gate':[1,5120],'b2':[5120]}
                    if set(handle.keys())!=set(expected):raise RuntimeError('Operand fields differ')
                    for key,shape in expected.items():
                        tensor=handle.get_slice(key)
                        if tensor.get_shape()!=shape or tensor.get_dtype()!='BF16':raise RuntimeError('Operand shape/dtype differs')
                scope_count+=1;payload_bytes+=scope['bytes']
            print(f'AUDITED calibration input{index:02d}: both videos81@16, full160 headers and sampling',flush=True)
    if seen!=set(range(count)):raise RuntimeError('Incomplete distinct calibration inputs')
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'RUN.json',{'status':'PASS_CALIBRATION_VIDEO_GEOMETRY_SAMPLING_AND_HEADERS',
        'collections':reports,'protocol_sha256':digest(args.protocol),'inputs_sha256':digest(args.inputs),
        'source_sha256':digest(Path(__file__)),'input_count':count,'scope_count':scope_count,
        'sampled_rows':scope_count*rows,'tensor_payload_bytes':payload_bytes,
        'all_tensor_payload_SHA_rechecked':args.verify_tensor_bytes,'videos':video_records,
        'same_pipeline_noop_only':True,'elapsed_seconds':time.perf_counter()-started,
        'quality_or_speed_acceptance':False})


if __name__=='__main__':main()
