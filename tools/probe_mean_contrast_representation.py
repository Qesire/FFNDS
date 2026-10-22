"""CPU discriminator for exact-mean / approximate-contrast representations.

Uses small synthetic inputs, the unchanged original Python HiF4 reference,
and four already saved R14N0 payloads with the original frozen affine A.
No original W1/W2, native FFN, model generation or parameter fit is run.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/20261023/R15_mean_contrast_CPU_protocol.json'
TETRAHEDRON = ((1., 1., 1.), (1., -1., -1.),
               (-1., 1., -1.), (-1., -1., 1.))


def gelu(x):
    return .5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + .044715*x**3)))


def mean(values):
    return math.fsum(values) / len(values)


def analytic_checks(scales):
    """A fixed rank-one B differs from W only on centered contrasts."""
    rows = []
    w, b = (1., 1., 1.), (1., 0., 0.)
    z0, low_rank_mean = .7, .2
    for t in scales:
        wd = [t*math.fsum(a*x for a, x in zip(w, d)) for d in TETRAHEDRON]
        bd = [t*math.fsum(a*x for a, x in zip(b, d)) for d in TETRAHEDRON]
        ed = [a-c for a, c in zip(wd, bd)]
        true_quad = mean([(z0+a)**2 for a in wd])
        mean_exact_quad = mean([(z0+a)**2 for a in bd])
        bound_L2 = 2*mean([abs(a*c) for a, c in zip(bd, ed)]) + mean([a*a for a in ed])
        qerror = abs(true_quad - mean_exact_quad)
        if not math.isclose(qerror, 2*t*t, rel_tol=1e-9, abs_tol=1e-14) or qerror > bound_L2+1e-14:
            raise RuntimeError('Centered quadratic cancellation / Lipschitz bound failed')
        true_gelu = mean([gelu(z0+a) for a in wd])
        exact_mean_gelu = mean([gelu(z0+a) for a in bd])
        full_low_rank_gelu = mean([gelu(low_rank_mean+a) for a in bd])
        rows.append({'contrast_scale': t, 'quadratic_mean_error': qerror,
                     'quadratic_L2_bound': bound_L2,
                     'GELU_exact_mean_contrast_error': abs(true_gelu-exact_mean_gelu),
                     'GELU_error_divided_by_contrast_scale2': abs(true_gelu-exact_mean_gelu)/(t*t),
                     'GELU_full_input_low_rank_error': abs(true_gelu-full_low_rank_gelu)})
    return {'fixed_W': list(w), 'fixed_rank1_B': list(b), 'exact_mean_preactivation': z0,
            'full_input_low_rank_mean_preactivation': low_rank_mean,
            'scales': rows,
            'zero_contrast_full_low_rank_GELU_error': abs(gelu(z0)-gelu(low_rank_mean)),
            'zero_contrast_mean_exact_error': 0.}


def mac_ledger(m=31668, d=5120, h=13824, rank=256):
    if m % 4 or not 0 < rank <= min(d, h):
        raise ValueError('Complete four-token stripes and valid positive rank required')
    g, dense = m//4, 2*m*d*h
    terms = {'own_affine_three_directions': 3*g*d*d,
             'exact_mean_W1': g*d*h,
             'contrast_factor_first': 3*g*d*rank,
             'contrast_factor_second': 3*g*rank*h,
             'pooled_W2': g*h*d}
    return {'shape': {'M': m, 'D': d, 'H': h, 'rank': rank, 'stripes': g},
            'MAC_by_term': terms, 'main_MAC_fraction': sum(terms.values())/dense,
            'Q1_rows': m, 'GELU_Q2_rows': m,
            'R14_two_point_GELU_Q2_rows': g+3958,
            'Q2_row_count_over_R14': m/(g+3958),
            'materialized_full_hidden_BF16_bytes': 2*m*h,
            'not_in_MAC_fraction': 'Q1,Q2,GELU,smooth casts,pooling,layout,selector,gate,residual writes and weight traffic. All four hidden nonlinear rows remain. No latency estimate.'}


def quantizer_counterexample(torch, original, qtype, scales):
    # An immutable scale anchor in channel63 fixes the original group64 scale.
    # Solve tanh-GELU(z)=1/8 by bisection, independent of any model/data/loss.
    lo, hi = 0., 1.
    for _ in range(64):
        middle = (lo+hi)*.5
        if gelu(middle) < .125:
            lo = middle
        else:
            hi = middle
    center = (lo+hi)*.5
    wd = torch.tensor([sum(d) for d in TETRAHEDRON], dtype=torch.float64)
    bd = torch.tensor([d[0] for d in TETRAHEDRON], dtype=torch.float64)
    rows = []
    import torch.nn.functional as F
    for t in scales:
        true_z = torch.zeros((4,64), dtype=torch.float64)
        approx_z = torch.zeros_like(true_z)
        true_z[:,0], approx_z[:,0] = center+t*wd, center+t*bd
        true_z[:,63] = approx_z[:,63] = 7.
        if not torch.allclose(true_z.mean(0), approx_z.mean(0), rtol=0., atol=1e-15):
            raise RuntimeError('Counterexample changed the real preactivation mean')
        true_gelu = F.gelu(true_z.bfloat16(), approximate='tanh')
        approx_gelu = F.gelu(approx_z.bfloat16(), approximate='tanh')
        true_q = original(true_gelu.float(), qtype, -1).bfloat16().double()
        approx_q = original(approx_gelu.float(), qtype, -1).bfloat16().double()
        smooth_error = abs(mean([gelu(center+t*float(a)) for a in wd]) -
                           mean([gelu(center+t*float(a)) for a in bd]))
        rows.append({'contrast_scale': t, 'smooth_GELU_mean_absolute_error': smooth_error,
                     'BF16_GELU_mean_absolute_error': float((true_gelu.double().mean(0)-approx_gelu.double().mean(0)).abs().max()),
                     'original_HiF4_mean_absolute_error': float((true_q.mean(0)-approx_q.mean(0)).abs().max()),
                     'true_target_Q2_values': true_q[:,0].tolist(),
                     'approx_target_Q2_values': approx_q[:,0].tolist(),
                     'true_anchor_Q2_values': true_q[:,63].tolist(),
                     'approx_anchor_Q2_values': approx_q[:,63].tolist(),
                     'target_GELU_input_BF16': true_z[:,0].bfloat16().float().tolist(),
                     'approx_target_GELU_input_BF16': approx_z[:,0].bfloat16().float().tolist()})
    if not any(r['original_HiF4_mean_absolute_error'] > .03 for r in rows):
        raise RuntimeError('Fixed original-quantizer counterexample did not discriminate')
    return {'target_tanh_GELU_boundary': .125, 'preactivation_boundary': center,
            'anchor_channel': 63, 'anchor_preactivation': 7., 'rows': rows,
            'interpretation': 'Real preactivation means agree; BF16/GELU/unchanged original group64 HiF4 can have finite different pooled means. These fixed finite-scale cases do not assert a t->0 limit in a finite BF16 lattice or a Wan effect.'}


def saved_data_checks(torch, protocol):
    from safetensors.torch import load_file
    import torch.nn.functional as F
    from r14_stripe_contract import execution_root, read, sha
    from stripe_teacher_capture import writeback
    from analyze_complete_stripe_failure import correlation, ranks
    from onpolicy_residual import same_bits
    native = read(ROOT/'research/20261022/R14N0_native_results.json')
    root = execution_root(read(ROOT/'research/20261022/R14N0_registered_protocol.json'))
    capture = read(root/'CAPTURE.json')
    public_path=ROOT/'research/20261022/R14N0_capture_receipt.json'
    public_capture=read(public_path)
    if (sha(public_path) != protocol['capture_receipt_sha256'] or
            sha(root/'CAPTURE.json') != public_capture['raw_capture_receipt_sha256']):
        raise RuntimeError('Public or original raw saved capture receipt changed')
    cache = read(ROOT/'.runtime/hif4_rebuilt_cache.json')
    record = next(a for a in read(ROOT/'research/20261021/R9A1_native_affine_results.json')['affine_matrices'] if 2 in a['shared_steps'])
    matrix_path = Path(cache['path']).parents[1]/record['HF_derived_relative']
    if sha(matrix_path) != protocol['frozen_affine_sha256']:
        raise RuntimeError('Original frozen affine A changed')
    matrix = load_file(str(matrix_path), device='cpu')['matrix_BF16'].double()
    cases = []
    for index in protocol['indices']:
        entry = next(row for row in capture['completed'] if row['index']==index)
        if sha(entry['report']) != entry['sha256']:
            raise RuntimeError('Saved capture source report changed')
        scope = read(entry['report'])['scope']
        if sha(scope['path']) != scope['sha256']:
            raise RuntimeError('Saved complete-stripe operands changed')
        data = load_file(scope['path'], device='cpu')
        x = data['V1'].reshape(64,4,5120).double()
        mu = x.mean(1,keepdim=True)
        delta = x-mu
        total = x.square().sum((1,2))
        centered = delta.square().sum((1,2))
        mean_energy = 4*mu.square().sum((1,2))
        if not torch.allclose(total,centered+mean_energy,atol=1e-9,rtol=1e-13):
            raise RuntimeError('Orthogonal mean/contrast energy identity failed')
        gram = delta@delta.transpose(1,2)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.)
        fraction = eigenvalues[:,-1]/eigenvalues.sum(1).clamp_min(1e-30)
        infer = getattr(torch,scope['infer_dtype'].split('.')[-1])
        sensitive = getattr(torch,scope['sensitive_dtype'].split('.')[-1])
        teacher_post = writeback(data['Residual'],data['TeacherY'],data['gate'],infer,sensitive)
        student_post = writeback(data['Residual'],data['StudentY'],data['gate'],infer,sensitive)
        if not same_bits(student_post,data['NativePostStudent']):
            raise RuntimeError('Exact saved CPU elementwise writeback differs')
        ub1 = (teacher_post.double()-student_post.double()).reshape(64,4,5120).square().sum((1,2))
        native_case = next(c for c in native['cases'] if c['index']==index)
        if not math.isclose(float(ub1.sum()),math.fsum(native_case['UB1_SSE']),rel_tol=1e-12,abs_tol=1e-8):
            raise RuntimeError('Saved per-group baseline does not reproduce native total')
        own = F.linear(delta.reshape(256,5120),matrix).reshape_as(x)
        post0 = data['Residual'].double().reshape_as(x)+own*data['gate'].double().reshape(1,1,5120)
        error = teacher_post.double().reshape_as(x)-post0
        common = error.mean(1,keepdim=True)
        floor = (error-common).square().sum((1,2))
        headroom = ub1-floor
        ratio = centered/total
        rank1 = eigenvalues[:,-1]/centered.clamp_min(1e-30)
        raw = {'group_ordinal':list(range(64)), 'input_contrast_energy_fraction':ratio.tolist(),
               'input_contrast_top1_energy_fraction':rank1.tolist(),
               'native_UB1_post_SSE':ub1.tolist(),
               'FP64_smooth_additive_common_post_floor_SSE':floor.tolist(),
               'FP64_smooth_reference_headroom':headroom.tolist()}
        cases.append({'index':index,'payload_sha256':scope['sha256'],
                      'scope':{'step':2,'block':20,'groups':64,'tokens_per_group':4,'D':5120},
                      'input_contrast_energy_fraction_aggregate':float(centered.sum()/total.sum()),
                      'input_contrast_top1_energy_fraction_median':float(fraction.median()),
                      'native_UB1_post_SSE':float(ub1.sum()),
                      'FP64_smooth_additive_common_post_floor_SSE':float(floor.sum()),
                      'groups_smooth_floor_above_native_UB1':int((headroom<0).sum()),
                      'groups_smooth_floor_below_native_UB1':int((headroom>0).sum()),
                      'norm_fraction_vs_smooth_headroom_Spearman':correlation(ranks(ratio.tolist()),ranks(headroom.tolist())),
                      'per_group':raw})
        print(f'R15 descriptive savedCPU source{index:02d} complete',flush=True)
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    from worker_identity import require_worker
    from ar2_checkpoint import digest, atomic_json
    require_worker()
    if args.output.exists():
        raise RuntimeError('Preserve first fixed CPU study')
    p=json.loads(PROTOCOL.read_text())
    protocol_sha,source_sha=digest(PROTOCOL),digest(__file__)
    if source_sha != p['source_sha256']:
        raise RuntimeError('Frozen CPU discriminator source changed')
    for name,expected in p['input_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise RuntimeError('Frozen saved/source input changed: '+name)
    import torch
    torch.set_num_threads(8)
    from audit_hif4_group_separability import reference,QROOT
    qtype,original=reference()
    if qtype.blk_size!=64 or torch.cuda.is_initialized():
        raise RuntimeError('Unchanged original group64 CPU reference required')
    with torch.inference_mode():
        analytic=analytic_checks(p['contrast_scales'])
        quant=quantizer_counterexample(torch,original,qtype,p['contrast_scales'])
        saved=saved_data_checks(torch,p)
    if digest(PROTOCOL)!=protocol_sha or digest(__file__)!=source_sha or torch.cuda.is_initialized():
        raise RuntimeError('Frozen source/protocol or CPU-only boundary changed')
    summaries=[]
    for name,indices in [('old_training',(1,11)),('old_development',(0,10)),('all_exposed',(1,11,0,10))]:
        chosen=[s for s in saved if s['index'] in indices]
        base=math.fsum(s['native_UB1_post_SSE'] for s in chosen)
        floor=math.fsum(s['FP64_smooth_additive_common_post_floor_SSE'] for s in chosen)
        summaries.append({'cohort':name,'groups':64*len(chosen),'native_UB1_post_SSE':base,
                          'FP64_smooth_common_post_floor_SSE':floor,'smooth_floor_over_native_UB1':floor/base,
                          'groups_smooth_floor_above_native_UB1':sum(s['groups_smooth_floor_above_native_UB1'] for s in chosen)})
    result={'id':p['id'],'status':'COMPLETED_CPU_MECHANISM_DISCRIMINATOR_NOT_NATIVE_METHOD',
            'protocol_sha256':protocol_sha,'source_sha256':source_sha,'input_sha256':p['input_sha256'],
            'analytic_checks':analytic,'quantizer_counterexample':quant,'MAC_ledger':mac_ledger(),
            'saved_scopes':saved,'saved_summary':summaries,'existing_A_FP64_CPU_projection_calls':4,
            'new_original_W1_W2_calls':0,'new_native_FFN_calls':0,'new_model_generations':0,
            'new_fitted_parameters':0,'synthetic_reference_Q2_calls':2*len(p['contrast_scales']),
            'CUDA_initialized':False,'native_registration':None,'model_execution_eligibility_change':False,
            'elapsed_seconds':time.monotonic()-started,
            'peak_CPU_RSS_bytes_Linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'limitations':p['limitations']}
    atomic_json(args.output,result)
    print(json.dumps({'status':result['status'],'MAC_fraction':result['MAC_ledger']['main_MAC_fraction'],
                      'saved_summary':summaries,'counterexample_mean_gaps':[r['original_HiF4_mean_absolute_error'] for r in quant['rows']]}),flush=True)


if __name__=='__main__':
    main()
