"""Frozen synthetic cost and correctness comparison of a three-direction FFN.

Loads only random tensors and the original quantizer. No Wan/model weights,
images, fitting or video generation enter this study.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import triton

from ar2_checkpoint import atomic_json, digest
import spatial_contrast_kernels as fused
from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]


def reference_prepare(v1, signs, count, dynamic):
    g, d = len(v1)//4, v1.shape[1]
    a, b, c, e = v1.reshape(g, 4, d).float().unbind(1)
    mean = (((a+b)+(c+e))*.25).bfloat16()
    h = torch.stack(((a+b-c-e)*.5, (a-b+c-e)*.5, (a-b-c+e)*.5), 1).bfloat16().reshape(3*g, d)
    n = g if dynamic else count
    aa, bb, cc, ee, mm = a[:n], b[:n], c[:n], e[:n], mean[:n].float()
    s = signs[:n]
    contrast = ((((aa-mm)*s[:, 0, None]+(bb-mm)*s[:, 1, None])+
                 ((cc-mm)*s[:, 2, None]+(ee-mm)*s[:, 3, None]))*.5).bfloat16()
    score = contrast.float().square().sum(1) if dynamic else None
    return h, mean, contrast, score


def reference_finish(projected, response, x, gate):
    g, d = response.shape
    a, b, c = projected.reshape(g, 3, d).float().unbind(1)
    contrast = torch.stack(((a+b+c)*.5, (a-b-c)*.5,
                            (-a+b-c)*.5, (-a-b+c)*.5), 1).bfloat16()
    y = (contrast+response[:, None]).reshape_as(x)
    return x+y*gate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.output.exists() or torch.cuda.device_count() != 1:
        raise RuntimeError('New output and one standard assigned GPU required')
    protocol_path = ROOT/'research/20261022/R14_three_direction_backend_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    source_paths = ['tools/probe_three_direction_backend.py', 'tools/spatial_contrast_kernels.py']
    source_shas = {p: digest(ROOT/p) for p in source_paths}
    protocol_sha = digest(protocol_path)
    from quant.hif4_backend import hif4_qdq_tensor
    torch.set_num_threads(8)
    torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
    m, d, h, groups, count = [protocol[k] for k in ('M', 'D', 'H', 'stripes', 'two_point_stripes')]
    if (m,d,h,groups,count) != (31668,5120,13824,7917,3958):
        raise RuntimeError('Fixed synthetic dimensions changed')
    rand = lambda shape, scale=1.: torch.randn(shape, device='cuda', dtype=torch.bfloat16)*scale
    x = rand((m,d))
    w1 = hif4_qdq_tensor(rand((h,d), 1/math.sqrt(d)), force_fp32=True)
    w2 = hif4_qdq_tensor(rand((d,h), 1/math.sqrt(h)), force_fp32=True)
    b1, b2, gate = rand((h,), .02), rand((d,), .02), rand((d,), .5)
    scale, shift = rand((d,), .05), rand((d,), .05)
    smooth1, smooth2 = torch.ones(d, device='cuda'), torch.ones(h, device='cuda')
    affine = (.5*w2.float()@w1.float()).bfloat16()
    affine_bias = (b2.float()+.5*w2.float()@b1.float()).bfloat16()
    half_w1, half_w2, half_b1 = w1[:h//2].contiguous(), w2[:,:h//2].contiguous(), b1[:h//2].contiguous()
    first, second = rand((256,d), 1/math.sqrt(d)), rand((d,256), 1/math.sqrt(256))
    first_bias = rand((256,), .02)
    signs = torch.randint(0,2,(groups,4),device='cuda').float()*2-1
    ids = torch.arange(groups,device='cuda')
    fixed_selected, fixed_others = ids[:count], ids[count:]
    n = groups-count
    order_n, order_k = torch.arange(n,device='cuda'), torch.arange(count,device='cuda')+n
    controls, quant_controls, checking = [], [], False
    phase = 'controls'

    def compare(name, a, b, norm=False):
        difference = (a.float()-b.float()).abs()
        error = float(difference.max())
        relative = float((difference/b.float().abs().clamp_min(1.)).max())
        bit_difference = int((a.view(torch.int16)!=b.view(torch.int16)).sum()) if a.dtype==torch.bfloat16 else None
        ok = relative <= 5e-6 if norm else torch.equal(a,b)
        controls.append({'name':name,'shape':list(a.shape),'max_absolute_difference':error,
                         'max_relative_difference_clamped_at1':relative,'BF16_bit_differences':bit_difference,
                         'numeric_exact':torch.equal(a,b),'pass':bool(ok)})
        if not ok:
            raise RuntimeError('Synthetic fusion control failed: '+name)

    def norm_input():
        norm = F.layer_norm(x,(d,),eps=1e-6)
        return ((norm.float()*(1+scale.float())+shift.float())/smooth1).bfloat16()

    def quant(value):
        if value.dtype!=torch.bfloat16:
            raise RuntimeError('Original BF16 quantizer input required')
        output = hif4_qdq_tensor(value,force_fp32=False)
        if checking:
            reference = hif4_qdq_tensor(value,force_fp32=True)
            mismatches = int((output.view(torch.int16)!=reference.view(torch.int16)).sum())
            quant_controls.append({'shape':list(value.shape),'bit_mismatches':mismatches})
            if mismatches:
                raise RuntimeError('Original QDQ control failed')
        return output

    def schedule(score, dynamic, maps):
        if not dynamic:
            return fixed_selected, fixed_others, ids, ids
        selected = score.topk(count,sorted=False).indices.sort().values
        mask = torch.ones(groups,device='cuda',dtype=torch.bool)
        mask[selected] = False
        others = ids[mask]
        if maps:
            slot = torch.empty_like(ids)
            slot.index_copy_(0,others,order_n); slot.index_copy_(0,selected,order_k)
            point_ids = torch.cat((others,selected,selected))
        else:
            point_ids, slot = ids, ids
        return selected, others, point_ids, slot

    def pack_reference(mean, contrast, selected, others, dynamic):
        if dynamic:
            return torch.cat((mean[others],mean[selected]+contrast[selected],mean[selected]-contrast[selected]))
        return torch.cat((mean[count:],mean[:count]+contrast,mean[:count]-contrast))

    def pool_reference(hidden, selected, others, dynamic):
        output = torch.empty((groups,h),device='cuda',dtype=torch.bfloat16)
        paired = ((hidden[n:n+count].float()+hidden[n+count:].float())*.5).bfloat16()
        if dynamic:
            output.index_copy_(0,others,hidden[:n]); output.index_copy_(0,selected,paired)
        else:
            output[count:] = hidden[:n]; output[:count] = paired
        return output

    def forward(arm):
        v1 = quant(norm_input())
        if arm=='dense':
            z = F.linear(v1,w1,b1)
            v2 = quant((F.gelu(z,approximate='tanh').float()/smooth2).bfloat16())
            return x+F.linear(v2,w2,b2)*gate
        if arm=='UB1_half_R256':
            z = F.linear(v1,half_w1,half_b1)
            v2 = quant((F.gelu(z,approximate='tanh').float()/smooth2[:h//2]).bfloat16())
            y = F.linear(v2,half_w2,b2)+F.linear(F.linear(v1,first,first_bias),second)
            return x+y*gate
        if arm=='R13_direct_dynamic':
            linear = F.linear(v1,affine,affine_bias)
            group = v1.reshape(groups,4,d).float()
            mean = group.mean(1).bfloat16()
            c = ((group-mean.float()[:,None])*signs[:,:,None]).sum(1).mul(.5).bfloat16()
            selected,others,_,_ = schedule(c.float().square().sum(1),True,False)
            points = pack_reference(mean,c,selected,others,True)
            hidden = quant((F.gelu(F.linear(points,w1,b1),approximate='tanh').float()/smooth2).bfloat16())
            response = F.linear(pool_reference(hidden,selected,others,True),w2,b2)
            remainder = response-linear.reshape(groups,4,d).float().mean(1).bfloat16()
            y = (linear.reshape(groups,4,d)+remainder[:,None]).reshape(m,d)
            return x+y*gate
        dynamic = arm.endswith('dynamic')
        use_fused = arm.startswith('three_fused')
        hh,mean,c,score = (fused.prepare if use_fused else reference_prepare)(v1,signs,count,dynamic)
        selected,others,point_ids,slot = schedule(score,dynamic,use_fused)
        projected = F.linear(hh,affine)
        points = fused.pack(mean,c,point_ids,count,dynamic) if use_fused else pack_reference(mean,c,selected,others,dynamic)
        hidden = quant((F.gelu(F.linear(points,w1,b1),approximate='tanh').float()/smooth2).bfloat16())
        pooled = fused.pool(hidden,slot,groups,count,dynamic) if use_fused else pool_reference(hidden,selected,others,dynamic)
        response = F.linear(pooled,w2,b2)
        return fused.finish(projected,response,x,gate) if use_fused else reference_finish(projected,response,x,gate)

    try:
        v1 = hif4_qdq_tensor(norm_input(),force_fp32=False)
        for dynamic in (False,True):
            name = 'dynamic' if dynamic else 'fixed'
            reference = reference_prepare(v1,signs,count,dynamic)
            actual = fused.prepare(v1,signs,count,dynamic)
            for label, a, b in zip(('three_contrasts','mean','input_contrast'),actual[:3],reference[:3]):
                compare(name+'/'+label,a[:count] if label=='input_contrast' and not dynamic else a,b)
            if dynamic:
                compare(name+'/score',actual[3],reference[3],norm=True)
            selected,others,point_ids,slot = schedule(actual[3],dynamic,True)
            if dynamic:
                expected_selected = reference[3].topk(count,sorted=False).indices.sort().values
                compare('dynamic/selected_ids',selected,expected_selected)
            compare(name+'/packed_points',fused.pack(actual[1],actual[2],point_ids,count,dynamic),
                    pack_reference(reference[1],reference[2],selected,others,dynamic))
            synthetic_hidden = rand((groups+count,h))
            compare(name+'/pooled_V2',fused.pool(synthetic_hidden,slot,groups,count,dynamic),
                    pool_reference(synthetic_hidden,selected,others,dynamic))
            projected, response = F.linear(reference[0],affine), rand((groups,d))
            compare(name+'/inverse_gate_writeback',fused.finish(projected,response,x,gate),
                    reference_finish(projected,response,x,gate))
            del reference,actual,synthetic_hidden,projected,response
        del v1
        checking = True
        outputs = {a:forward(a) for a in protocol['arms']}
        if len(quant_controls)!=14 or not all(bool(torch.isfinite(t).all()) for t in outputs.values()):
            raise RuntimeError('Incomplete QDQ/finite arm controls')
        for name in ('fixed','dynamic'):
            compare('full_pipeline/'+name,outputs['three_fused_'+name],outputs['three_unfused_'+name])
        direct, rewritten = outputs['R13_direct_dynamic'],outputs['three_fused_dynamic']
        rewrite_relative = float((direct.float()-rewritten.float()).norm()/direct.float().norm())
        del outputs,direct,rewritten
        checking,phase = False,'timing'
        if torch.cuda.max_memory_allocated()>32*2**30:
            raise RuntimeError('Synthetic resident allocation cap exceeded')
        for _ in range(protocol['warmups']):
            for arm in protocol['arms']: forward(arm)
        torch.cuda.synchronize()
        samples,host,orders = {a:[] for a in protocol['arms']},{a:[] for a in protocol['arms']},[]
        for repetition in range(protocol['repetitions']):
            offset = repetition%len(protocol['arms'])
            order = protocol['arms'][offset:]+protocol['arms'][:offset]
            if repetition%2: order = list(reversed(order))
            orders.append(order)
            for arm in order:
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                began = time.monotonic()
                begin.record();output=forward(arm);end.record();end.synchronize()
                host[arm].append(1000*(time.monotonic()-began));samples[arm].append(begin.elapsed_time(end))
                if not bool(torch.isfinite(output).all()): raise RuntimeError('Timed output nonfinite')
                del output
        median = {a:statistics.median(v) for a,v in samples.items()}
        mad = {a:statistics.median(abs(v-median[a]) for v in values) for a,values in samples.items()}
        saving = median['three_unfused_dynamic']-median['three_fused_dynamic']
        margin = max(.5,3*max(mad.values()))
        if digest(protocol_path)!=protocol_sha or any(digest(ROOT/p)!=s for p,s in source_shas.items()):
            raise RuntimeError('Frozen study source changed')
        result={'id':protocol['id'],'status':'COMPLETED_SYNTHETIC_THREE_DIRECTION_BACKEND',
            'evidence_level':'RANDOM_RESIDENT_TENSORS_NOT_WAN_MODEL_OR_VIDEO',
            'protocol_sha256':protocol_sha,'source_sha256':source_shas,
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'triton':triton.__version__,
            'shape':{'M':m,'D':d,'H':h,'stripes':groups,'two_point_stripes':count},
            'fusion_controls':controls,'QDQ_controls':quant_controls,'all_controls_before_timing':True,
            'R13_to_three_direction_random_output_relative_L2':rewrite_relative,
            'samples_ms':samples,'host_samples_ms':host,'median_ms':median,'MAD_ms':mad,'timing_order':orders,
            'speedup_over_same_run_dense':{a:median['dense']/v for a,v in median.items()},
            'same_operator_fusion_saving_ms':saving,'predeclared_noise_margin_ms':margin,
            'backend_fusion_advance':saving>=margin,
            'three_direction_projection_MAC_fraction':.75*d/(2*h)+(2*groups+count)/(2*m),
            'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),
            'models_loaded':0,'new_model_generations':0,'fitted_parameters':0,'new_tensor_files':0,
            'native_model_or_quality_acceptance':False,'limits':protocol['scope']}
        atomic_json(args.output,result)
        print(json.dumps({'median_ms':median,'dense_speedup':result['speedup_over_same_run_dense'],
                          'fusion_saving_ms':saving,'backend_fusion_advance':result['backend_fusion_advance']}),flush=True)
    except BaseException as error:
        failure=ROOT/'runs/r14_three_direction_backend/failure_receipt.json'
        failure.parent.mkdir(parents=True,exist_ok=True)
        if not failure.exists():
            atomic_json(failure,{'status':'SOFTWARE_OR_CONTROL_FAILURE_NO_TIMING_CLAIM','phase':phase,
                'error_type':type(error).__name__,'controls_completed':controls,'QDQ_controls':quant_controls,
                'protocol_sha256':protocol_sha,'source_sha256':source_shas,'models_loaded':0})
        raise


if __name__=='__main__':
    main()
