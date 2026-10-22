"""Actual-shape local operator checks for the fixed, unregistered R14N0 draft.

No pipeline or checkpoint construction here. Callers must pass verified frozen
operands/weights after releasing the complete model process. The output is a
local numeric diagnostic; zero-filled other stripes are never video evidence.
"""
import contextlib
import torch
import torch.nn.functional as F

import spatial_contrast_kernels as fused
from folded_adapter import replay_main_rows
from onpolicy_residual import replay_rows, same_bits
from probe_three_direction_backend import reference_prepare
from stripe_teacher_capture import writeback

M, D, H, G, K = 31668, 5120, 13824, 7917, 3958


@contextlib.contextmanager
def count_linears(expected):
    original = F.linear
    calls = []

    def linear(x, weight, bias=None):
        calls.append({'input': list(x.shape), 'weight': list(weight.shape), 'bias': bias is not None})
        return original(x, weight, bias)

    F.linear = linear
    try:
        yield calls
    finally:
        F.linear = original
    if len(calls) != expected:
        raise RuntimeError('Actual standalone linear count differs from the fixed debit')


class LocalOperator:
    def __init__(self, values, scope, draft, full, affine, position, quantizer):
        self.values = {key: value.to('cuda') for key, value in values.items()}
        self.scope, self.draft = scope, draft
        self.full, self.affine, self.position, self.quantizer = full, affine, position, quantizer
        self.rows = torch.tensor(scope['selected_rows'], device='cuda')
        self.groups = draft['coordinates']['planned_stripe_ids']
        self.group_ids = torch.tensor(self.groups, device='cuda')
        if (scope['selected_rows'] != draft['coordinates']['planned_native_row_ids']
                or scope['native_V1_shape'] != [M, D] or scope['native_V1_stride'] != [D, 1]
                or self.values['V1'].shape != (256, D)):
            raise RuntimeError('Complete real stripe geometry required')
        self.v1 = torch.zeros((M, D), device='cuda', dtype=torch.bfloat16)
        self.v1.index_copy_(0, self.rows, self.values['V1'])
        generator = torch.Generator(device='cuda').manual_seed(20261022)
        signs = torch.randint(0, 2, (G, 4), device='cuda', generator=generator).float() * 2 - 1
        self.h, self.mean, self.c, self.score = fused.prepare(self.v1, signs, K, True)
        reference = reference_prepare(self.v1, signs, K, True)
        if any(not same_bits(a, b) for a, b in zip((self.h, self.mean, self.c), reference[:3])):
            raise RuntimeError('Actual native input fusion differs from its BF16 reference')
        self.score_error = float(((self.score - reference[3]).abs() / reference[3].abs().clamp_min(1.)).max())
        if self.score_error > 5e-6:
            raise RuntimeError('Native contrast norm reduction exceeds frozen tolerance')
        self.norms = self.score.index_select(0, self.group_ids).tolist()
        other_norms = reference[3].index_select(0, self.group_ids).tolist()
        choose = lambda scores: sorted(sorted(range(64), key=lambda i: (-scores[i], i))[:32])
        if choose(self.norms) != choose(other_norms):
            raise RuntimeError('Norm reduction changed local dynamic selection')
        self.dynamic = choose(self.norms)
        self.infer_dtype = getattr(torch, scope['infer_dtype'].split('.')[-1])
        self.sensitive_dtype = getattr(torch, scope['sensitive_dtype'].split('.')[-1])
        self.truth_post = writeback(self.values['Residual'], self.values['TeacherY'], self.values['gate'],
                                    self.infer_dtype, self.sensitive_dtype)
        original_post = writeback(self.values['Residual'], self.values['StudentY'], self.values['gate'],
                                  self.infer_dtype, self.sensitive_dtype)
        if not same_bits(original_post, self.values['NativePostStudent']):
            raise RuntimeError('Captured actual native residual branch no longer reproduces')
        self.quant_controls = []
        self.auxiliary_controls = []
        self.epilogue = ('BF16_FUSED_NATIVE_WRITEBACK' if
                         self.values['Residual'].dtype == self.sensitive_dtype == self.infer_dtype == torch.bfloat16
                         else 'ACTUAL_SOURCE_BRANCH_TORCH_WRITEBACK_NO_SPEED_CLAIM')

    def loss(self, y, post):
        primary = (post.double() - self.truth_post.double()).square().reshape(64, 4, D).sum((1, 2))
        gated = ((y * self.values['gate']).double() -
                 (self.values['TeacherY'] * self.values['gate']).double()).square().reshape(64, 4, D).sum((1, 2))
        return {'SSE': primary.tolist(), 'gated_Y_SSE': gated.tolist(),
                'Y': y.detach().cpu().contiguous(), 'post': post.detach().cpu().contiguous()}

    def teacher(self):
        hidden = F.gelu(F.linear(self.v1, self.full['w0'], self.values['raw_b0']), approximate='tanh')
        v2 = self.quantizer((hidden.float() / self.full['smooth2']).bfloat16(), force_fp32=True).bfloat16()
        teacher = F.linear(v2, self.full['w2'], self.values['b2']).index_select(0, self.rows)
        if not same_bits(teacher, self.values['TeacherY']):
            raise RuntimeError('Actual original-shape teacher replay differs from new capture')
        return {'TeacherY_bitwise_control': True}

    def ub1(self):
        p = self.position
        if (not same_bits(self.values['raw_b0'], self.full['b0'])
                or not same_bits(self.values['b2'], self.full['b2'])):
            raise RuntimeError('Native full biases differ from captured originals')
        base = replay_main_rows(self.values['V1'], self.scope, p['w0'], p['b0'], p['w2'],
                                p['b2'], p['smooth2'], p['alpha'], self.quantizer)
        y = base + replay_rows(self.values['V1'], self.scope, p['residual'])
        if not same_bits(base, self.values['BaseY']) or not same_bits(y, self.values['StudentY']):
            raise RuntimeError('Frozen UB1 replay differs from actual new main/residual output')
        post = writeback(self.values['Residual'], y, self.values['gate'], self.infer_dtype, self.sensitive_dtype)
        return self.loss(y, post)

    def global_schedule(self, selected):
        if sorted(selected) != sorted(set(selected)) or len(selected) != 32 or min(selected) < 0 or max(selected) >= 64:
            raise ValueError('Exactly32 of the64 observed groups required')
        observed = set(self.groups)
        # Constant filler keeps all original GEMM dimensions, not a full-scene policy.
        scaffold = [group for group in range(G) if group not in observed][:K - 32]
        selected_global = sorted(scaffold + [self.groups[i] for i in selected])
        chosen = torch.tensor(selected_global, device='cuda')
        mask = torch.ones(G, device='cuda', dtype=torch.bool)
        mask[chosen] = False
        others = torch.arange(G, device='cuda')[mask]
        slot = torch.empty(G, device='cuda', dtype=torch.long)
        slot.index_copy_(0, others, torch.arange(G - K, device='cuda'))
        slot.index_copy_(0, chosen, torch.arange(K, device='cuda') + G - K)
        return chosen, others, torch.cat((others, chosen, chosen)), slot

    def candidate(self, selected):
        chosen, others, point_ids, slots = self.global_schedule(selected)
        projected = F.linear(self.h, self.affine)
        points = fused.pack(self.mean, self.c, point_ids, K, True)
        reference_points = torch.cat((self.mean[others], self.mean[chosen] + self.c[chosen],
                                      self.mean[chosen] - self.c[chosen]))
        if not same_bits(points, reference_points):
            raise RuntimeError('Native point packing differs from the declared reference')
        del reference_points
        hidden = F.gelu(F.linear(points, self.full['w0'], self.values['raw_b0']), approximate='tanh')
        scaled = (hidden.float() / self.full['smooth2']).bfloat16()
        v2 = self.quantizer(scaled, force_fp32=False).bfloat16()
        original_v2 = self.quantizer(scaled, force_fp32=True).bfloat16()
        mismatches = int((v2.view(torch.int16) != original_v2.view(torch.int16)).sum())
        self.quant_controls.append({'shape': list(v2.shape), 'original_Q2_bit_mismatches': mismatches})
        if mismatches:
            raise RuntimeError('Native BF16 Q2 entry differs from original forceFP32')
        del original_v2, hidden, scaled
        pooled = fused.pool(v2, slots, G, K, True)
        reference_pool = torch.empty_like(pooled)
        reference_pool.index_copy_(0, others, v2[:G - K])
        paired = ((v2[G - K:G].float() + v2[G:].float()) * .5).bfloat16()
        reference_pool.index_copy_(0, chosen, paired)
        if not same_bits(pooled, reference_pool):
            raise RuntimeError('Native postQ2 pooling differs from the declared reference')
        del reference_pool, paired, v2
        response = F.linear(pooled, self.full['w2'], self.values['b2'])
        a, b, c = projected.reshape(G, 3, D).float().unbind(1)
        centered = torch.stack(((a + b + c) * .5, (a - b - c) * .5,
                                (-a + b - c) * .5, (-a - b + c) * .5), 1).bfloat16()
        y = (centered + response[:, None]).reshape(M, D).index_select(0, self.rows)
        post = writeback(self.values['Residual'], y, self.values['gate'], self.infer_dtype, self.sensitive_dtype)
        if self.epilogue == 'BF16_FUSED_NATIVE_WRITEBACK':
            residual = torch.zeros((M, D), device='cuda', dtype=torch.bfloat16)
            residual.index_copy_(0, self.rows, self.values['Residual'])
            fused_post = fused.finish(projected, response, residual, self.values['gate'].reshape(D))
            if not same_bits(fused_post.index_select(0, self.rows), post):
                raise RuntimeError('Fused native residual writeback differs from its source branch')
        self.auxiliary_controls.append({'point_pack_exact': True, 'postQ2_pool_exact': True,
                                        'epilogue': self.epilogue, 'epilogue_control_pass': True})
        return self.loss(y, post)


def compose(complement_a, complement_b, selected):
    """Reference only: final physical masks must be executed and checked."""
    selected = set(selected)
    result = {}
    for key in ('Y', 'post'):
        a, b = [output[key].reshape(64, 4, D) for output in (complement_a, complement_b)]
        result[key] = torch.stack([a[g] if ((g < 32) == (g in selected)) else b[g]
                                   for g in range(64)]).reshape(256, D)
    return result


def check_composed(actual, reference):
    if any(not same_bits(actual[key], reference[key]) for key in ('Y', 'post')):
        raise RuntimeError('Executed native mask does not equal independently measured row-local action values')
    return {'executed_Y_bitwise_matches_local_values': True, 'executed_post_bitwise_matches_local_values': True}
