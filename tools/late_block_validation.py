"""Validate the actual mixed-rank LR2 trace without changing historical guards."""
from late_block_rank_runtime import expected_rank, validate_policy


def validate_runtime(packed, policy_sha, *, require_timing=True, allow_identity=False):
    policy = packed['policy']
    validate_policy(policy)
    if policy['identity_control'] and not allow_identity:
        raise RuntimeError('An LR2 identity control cannot become candidate acceptance evidence')
    expected = {(s,b) for s in range(4) for b in range(40)}
    calls = {f'step{s}_block{b}_ffn{slot}' for s,b in expected for slot in [0,2]}
    aux = packed['residual_correction']
    if (packed['policy_sha256'] != policy_sha or packed['actual_FFN_linear_calls'] != 320
            or set(packed['coverage']) != calls or not aux['active']
            or aux['actual_positions'] != 160 or aux['pending_predictions'] != 0
            or set(aux['coverage']) != {f'step{s}_block{b}' for s,b in expected}
            or (require_timing and not aux['timing_includes_auxiliary_cost'])):
        raise RuntimeError('Complete same-policy320 main and320 auxiliary GEMMs required')
    numerator = denominator = 0
    for key,row in packed['coverage'].items():
        step = int(key.split('_')[0][4:])
        width = policy['main_groups_by_step'][step]*64
        shape = [width,5120] if key.endswith('ffn0') else [5120,width]
        if (row['actual_weight_shape'] != shape or row['main_GEMM_density'] != width/13824
                or type(row['rows']) is not int or row['rows'] <= 0):
            raise RuntimeError('Actual LR2 main GEMM dimensions/rows differ from UB1')
        numerator += row['rows']*shape[0]*shape[1]
        denominator += row['rows']*5120*13824
    if numerator*2 != denominator or packed['actual_global_main_MAC_density'] != .5:
        raise RuntimeError('Actual row-weighted main MAC density must remain exactly50%')
    rank_rows = rows = 0
    for step,block in expected:
        value = aux['coverage'][f'step{step}_block{block}']
        rank = expected_rank(step,block,policy['identity_control'])
        if (value['auxiliary_GEMM_shapes'] != [[rank,5120],[5120,rank]]
                or value['auxiliary_linear_calls'] != 2
                or value['rows'] != packed['coverage'][f'step{step}_block{block}_ffn0']['rows']
                or value['auxiliary_MACs_over_original_dense_FFN'] != rank/13824
                or (require_timing and value['auxiliary_parameter_H2D_bytes'] != 0)):
            raise RuntimeError('Actual auxiliary matrix shapes/rows/cost differ from LR2 rank schedule')
        rows += value['rows']
        rank_rows += value['rows']*rank
    info = packed['late_block_rank']
    changed = {(v['step'],v['block']) for v in info['changed_positions']}
    if (info['identity_control'] != policy['identity_control'] or len(info['changed_positions']) != 10
            or changed != {(3,b) for b in range(30,40)}
            or info['mean_rank'] != (256 if policy['identity_control'] else 272)
            or info['actual_row_weighted_auxiliary_MAC_density'] != rank_rows/(rows*13824)):
        raise RuntimeError('Complete LR2 schedule/cost metadata differs from the actual trace')
