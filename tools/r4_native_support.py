"""Native geometry controls and fixed-h readout for the R4 finite support screen."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import digest
from compensated_support import FixedDesign
from folded_adapter import replay_main_rows
from onpolicy_residual import same_bits
from run_ar2_restart import ROOT, tensor_sha


def full_parameters(positions, position):
    expert, block = position['expert'], position['block']
    shard = next(v for v in positions.cache['shards'] if v['expert'] == expert and v['block'] == block)
    if digest(shard['path']) != shard['sha256']:
        raise RuntimeError('Original full native weight shard changed')
    payload = torch.load(shard['path'], map_location='cpu', mmap=True, weights_only=True)[expert]
    first, second = [payload[f'blocks.{block}.ffn.{slot}'] for slot in [0, 2]]
    view = json.loads((ROOT / '.runtime/node_local_model_view.json').read_text())
    raw = next(v for v in view['files'] if f'/{expert}/' in v['view_path'] and v['filename'].endswith('.safetensors'))
    with safe_open(raw['view_path'], framework='pt', device='cpu') as handle:
        b0 = handle.get_tensor(f'blocks.{block}.ffn.0.bias').cuda().bfloat16()
    if tensor_sha(b0) != position['raw_b0_sha256']:
        raise RuntimeError('Original full native FFN0 bias differs')
    return {'w0': first['weight_qdq'].cuda().contiguous(), 'b0': b0,
        'w2': second['weight_qdq'].cuda().contiguous(),
        'smooth2': second['smooth_scale'].cuda().float(), 'shard_sha256': shard['sha256']}


@torch.no_grad()
def reconstruct_teacher_v2(case, full, quantizer):
    scope = case['scope']
    shape, stride = tuple(scope['native_V1_shape']), tuple(scope['native_V1_stride'])
    if shape != (scope['full_rows'], 5120) or stride != (5120, 1):
        raise RuntimeError('Unsupported original teacher GEMM geometry')
    x = torch.zeros(shape, dtype=torch.bfloat16, device='cuda')
    ids = torch.tensor(scope['selected_rows'], device='cuda')
    x.index_copy_(0, ids, case['V1'])
    hidden = F.gelu(F.linear(x, full['w0'], full['b0']), approximate='tanh')
    v2 = quantizer((hidden.float() / full['smooth2']).bfloat16(), force_fp32=True).bfloat16()
    teacher = F.linear(v2, full['w2'], case['b2']).index_select(0, ids)
    if not same_bits(teacher, case['TeacherY']):
        raise RuntimeError('Full native TeacherY does not reproduce the OP2 capture bitwise')
    return v2.index_select(0, ids).contiguous()


def support_parameters(groups, full, position):
    groups = np.asarray(groups, dtype=np.int64)
    if (len(groups) != len(position['groups']) or len(np.unique(groups)) != len(groups)
            or not np.array_equal(groups, np.sort(groups)) or groups.min() < 0 or groups.max() >= 216):
        raise RuntimeError('Sorted same-budget original64-channel support required')
    ids = torch.as_tensor((groups[:, None] * 64 + np.arange(64)).reshape(-1), device='cuda')
    return {'w0': full['w0'].index_select(0, ids).contiguous(),
        'b0': full['b0'].index_select(0, ids).contiguous(),
        'w2': full['w2'].index_select(1, ids).contiguous(),
        'b2': position['b2'], 'smooth2': full['smooth2'].index_select(0, ids).contiguous(),
        'alpha': position['alpha']}


@torch.no_grad()
def native_mains(cases, groups, full, position, quantizer):
    parameters = support_parameters(groups, full, position)
    return {c['index']: replay_main_rows(c['V1'], c['scope'], **parameters, quantizer=quantizer) for c in cases}


def group_contribution(activation, full, group, alpha):
    start = int(group) * 64
    return F.linear(activation[:, start:start + 64].double(),
                    full['w2'][:, start:start + 64].double().contiguous()) * alpha


def action_delta(activation, full, action, alpha):
    result = torch.zeros((len(activation), 5120), device=activation.device, dtype=torch.float64)
    for sign, key in [(-1, 'removed'), (1, 'added')]:
        for group in action[key]:
            result.add_(group_contribution(activation, full, group, alpha), alpha=sign)
    return result


def matched_parameters(design, target, residual):
    coefficients = design.fit(target)
    parameters = {**residual, 'second': coefficients[:-1].T.bfloat16().contiguous(),
                  'bias_second': coefficients[-1].bfloat16().contiguous()}
    if any(not torch.isfinite(v).all() for v in parameters.values()):
        raise RuntimeError('Nonfinite native fixed-h OLS readout')
    return coefficients, parameters
