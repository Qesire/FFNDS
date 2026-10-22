"""CPU rounding control for three projections of four centered token rows."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first contrast audit')
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from ar2_checkpoint import digest, atomic_json
    from audit_r12_saved_operands import verified_safetensors
    from run_ar2_restart import tensor_sha
    from ub1_native_data import load_ub1_collections
    from probe_pooled_nonlinear_moments import projection_cost
    torch.set_num_threads(8)
    protocol_path = ROOT/'research/20261022/R13_linear_contrast_protocol.json'
    protocol_sha = digest(protocol_path)
    r9_path = ROOT/'research/20261021/R9A1_native_affine_results.json'
    original = next(a for a in json.loads(r9_path.read_text())['affine_matrices'] if 2 in a['shared_steps'])
    cache_root = Path(json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())['path']).parents[1]
    path = cache_root/original['HF_derived_relative']
    if digest(path) != original['sha256']:
        raise RuntimeError('Original frozen R9 matrix file changed')
    matrix = load_file(str(path), device='cpu')
    old = json.loads((ROOT/'research/20261022/R13_anchored_affine_remainder.json').read_text())
    records, bindings, _ = load_ub1_collections([ROOT/f'reports/worker_20261020/OP2_UB1_calibration24_shard{s}_RUN.json' for s in (0,1)], ROOT)
    signs = torch.tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=torch.float32)
    rows = []
    for index in sorted(records):
        scope = next(s for s in records[index]['scopes'] if (s['step'],s['block']) == (2,20))
        verified_safetensors(scope['path'], scope['sha256'])
        data = load_file(scope['path'],device='cpu')
        linear = F.linear(data['V1'], matrix['matrix_BF16'], matrix['bias_total_BF16'])
        expected = next(s for s in old['scopes'] if s['step']==2 and s['index']==index)
        if tensor_sha(linear) != expected['affine_output_sha256']:
            raise RuntimeError('Original CPU affine output no longer reproduces')
        x = data['V1'].reshape(64,4,5120).float()
        h1 = (x[:,0]+x[:,1]-x[:,2]-x[:,3])*.5
        h2 = (x[:,0]-x[:,1]+x[:,2]-x[:,3])*.5
        h3 = (x[:,0]-x[:,1]-x[:,2]+x[:,3])*.5
        directions = torch.stack((h1,h2,h3),dim=1).bfloat16()
        projected = F.linear(directions.reshape(192,5120),matrix['matrix_BF16']).reshape(64,3,5120)
        centered = (.5*torch.einsum('ij,tjd->tid',signs,projected.float())).bfloat16()
        linear = linear.reshape(64,4,5120)
        response_fixture = data['TeacherY'].reshape(64,4,5120).float().mean(1).bfloat16()
        direct = linear+(response_fixture-linear.float().mean(1).bfloat16())[:,None]
        rewrite = centered+response_fixture[:,None]
        gate = data['gate'].reshape(1,1,5120)
        difference = (rewrite*gate).double()-(direct*gate).double()
        ub1 = (data['TeacherY']*gate.reshape(1,5120)).double()-(data['StudentY']*gate.reshape(1,5120)).double()
        rows.append({'index':index,'original_payload_sha256':scope['sha256'],
            'original_affine_output_sha256':expected['affine_output_sha256'],
            'three_direction_output_sha256':tensor_sha(rewrite),
            'gated_rewrite_discrepancy_SSE':float(difference.square().sum()),
            'original_UB1_gated_SSE':float(ub1.square().sum()),
            'arbitrary_four_row_arithmetic_groups':64,'native_spatial_groups_observed':0})
    discrepancy = math.fsum(r['gated_rewrite_discrepancy_SSE'] for r in rows)
    baseline = math.fsum(r['original_UB1_gated_SSE'] for r in rows)
    if torch.cuda.is_initialized() or digest(protocol_path)!=protocol_sha:
        raise RuntimeError('CPU-only fixed protocol violated')
    result={'id':'R13S4_THREE_DIRECTION_AFFINE_CONTRAST','evidence_level':'SAVED_INPUT_CPU_LINEAR_REWRITE_ROUNDING_ONLY',
        'source_sha256':digest(__file__),'protocol_sha256':protocol_sha,
        'original_R9_result_sha256':digest(r9_path),'original_affine_file_sha256':original['sha256'],
        'prior_affine_result_sha256':digest(ROOT/'research/20261022/R13_anchored_affine_remainder.json'),
        'rows':rows,'gated_rewrite_discrepancy_SSE':discrepancy,'original_UB1_gated_SSE':baseline,
        'rewrite_discrepancy_over_UB1':discrepancy/baseline,'fixed_CPU_precision_screen_pass':discrepancy/baseline<=.01,
        'projection_MAC_fraction':projection_cost()['total']-.25*5120/(2*13824),
        'new_model_generations':0,'new_training_updates':0,'new_tensor_files':0,'CUDA_initialized':False,
        'native_identity_or_speed_pass':False,
        'limits':'Arbitrary saved row quadruples are arithmetic fixtures, not actual spatial groups. The teacher mean is a test fixture only. Native BF16/GEMM transforms and complete-chain speed remain untested; no shared-response quality or selection benefit.'}
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('rewrite_discrepancy_over_UB1','fixed_CPU_precision_screen_pass','projection_MAC_fraction')}))


if __name__=='__main__':
    main()
