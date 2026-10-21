"""Validate UB2's real main320 and residual240 GEMMs, including a dense last step."""
from terminal_dense_runtime import effective_groups


def validate_runtime(packed,policy_sha,*,require_timing=True,allow_identity=False):
    p = packed['policy']
    groups = effective_groups(p)
    identity = p['identity_control']
    if identity and not allow_identity:
        raise RuntimeError('Identity controls cannot become active UB2 acceptance evidence')
    positions = {(s,b) for s in range(4) for b in range(40)}
    main_calls = {f'step{s}_block{b}_ffn{slot}' for s,b in positions for slot in [0,2]}
    aux_positions = {(s,b) for s,b in positions if identity or s != 3}
    aux_keys = {f'step{s}_block{b}' for s,b in aux_positions}
    residual = packed['residual_correction']
    if (packed['policy_sha256'] != policy_sha or packed['actual_FFN_linear_calls'] != 320
            or set(packed['coverage']) != main_calls or not residual['active']
            or residual['actual_positions'] != len(aux_positions) or residual['expected_positions'] != len(aux_positions)
            or residual['pending_predictions'] != 0 or set(residual['coverage']) != aux_keys
            or (require_timing and not residual['timing_includes_auxiliary_cost'])):
        raise RuntimeError('Exact complete UB2 main and declared residual coverage required')
    numerator = denominator = 0
    for key,row in packed['coverage'].items():
        step = int(key.split('_')[0][4:])
        width = groups[step]*64
        shape = [width,5120] if key.endswith('ffn0') else [5120,width]
        if (row['actual_weight_shape'] != shape or row['main_GEMM_density'] != width/13824
                or type(row['rows']) is not int or row['rows'] <= 0):
            raise RuntimeError('Actual UB2 GEMM dimensions/rows differ from the frozen schedule')
        numerator += row['rows']*shape[0]*shape[1]
        denominator += row['rows']*5120*13824
    if numerator*2 != denominator or packed['actual_global_main_MAC_density'] != .5:
        raise RuntimeError('Actual complete row-weighted UB2 main MAC must be exactly50%')
    for key,row in residual['coverage'].items():
        if (row['auxiliary_GEMM_shapes'] != [[256,5120],[5120,256]] or row['auxiliary_linear_calls'] != 2
                or row['rows'] != packed['coverage'][key+'_ffn0']['rows']
                or row['auxiliary_MACs_over_original_dense_FFN'] != 256/13824
                or (require_timing and row['auxiliary_parameter_H2D_bytes'] != 0)):
            raise RuntimeError('UB2 actual residual operands/costs differ from the declared120 positions')
    terminal = packed['terminal_dense']
    if (terminal['active'] != (not identity) or terminal['step'] != 3
            or terminal['identity_control'] != identity or terminal['expected_residual_positions'] != len(aux_positions)
            or terminal['expected_auxiliary_GEMMs'] != len(aux_positions)*2
            or terminal['full_main_positions'] != (0 if identity else 40)
            or terminal['nominal_auxiliary_MAC_density'] != len(aux_positions)*256/(160*13824)):
        raise RuntimeError('Dense terminal/residual cost metadata differs from the executed trace')
