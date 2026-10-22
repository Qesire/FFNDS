"""Compact native rows into dense and frozen-UB1 FFNs before either W1.

The original Wan code still owns norm/modulation, GELU and gate/writeback.
This module never registers or launches a model. All parameters are the
already fitted UB1 parameters; no per-row teacher or omitted W1 is observed.
"""
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from reduced_rank_residual import predict_y

ARMS = ('dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual')


def row_partition(scores, arm, rows):
    """Exactly floor(M/4) dense rows; deterministic ties use original row ID."""
    ids = torch.arange(rows, device=scores.device)
    count = rows // 4
    if arm == 'UB1':
        dense = ids[:0]
    elif arm == 'fixed_stride':
        dense = ids[::4][:count]
    elif arm == 'fixed_hash':
        # A fixed bijection modulo 2**31; no RNG or source-dependent mask.
        order = torch.argsort((ids * 1103515245 + 12345) % (1 << 31), stable=True)
        dense = order[:count].sort().values
    elif arm == 'dynamic_residual':
        dense = torch.argsort(scores, descending=True, stable=True)[:count].sort().values
    else:
        raise ValueError('Unknown sparse row action')
    keep = torch.ones(rows, dtype=torch.bool, device=scores.device)
    keep[dense] = False
    return dense, ids[keep]


def load_metadata(positions):
    """Read only immutable coefficient arrays, without executing original W1/W2."""
    from run_ar2_restart import tensor_sha
    result = {}
    for step in range(4):
        expert = 'high_noise_model' if step < 2 else 'low_noise_model'
        for block in range(40):
            name = f'step{step}_block{block:02d}'
            if step in positions.changed:
                part = positions.changed[step].load('positions', name)
                arrays, info = part['arrays'], part['metadata']
                groups, bias = arrays['groups'], arrays['b2']
                alpha, bias_sha = info['alpha'], info['b2_sha256']
                rr = {k[3:]: v for k, v in arrays.items() if k.startswith('rr_')}
                rr_sha = info['RR_parameter_sha256']
                w2_sha = info['original_W2_sha256']
            else:
                part = positions.base[expert].load('positions', name)
                arrays, info = part['arrays'], part['metadata']
                groups, bias = arrays['C_groups'], arrays['C_centered_gain_bias_b2']
                parameter = info['policies']['C']['modes']['centered_gain_bias']
                alpha, bias_sha = parameter['alpha'], parameter['b2_sha256']
                residual = positions.residual.load('positions', name)
                rr, rr_sha = residual['arrays'], residual['metadata']['parameter_sha256']
                w2_sha = info['weight_sha256']
            groups = torch.as_tensor(groups.copy(), dtype=torch.long)
            if (len(groups) != [54, 108, 108, 162][step] or
                    len(groups.unique()) != len(groups) or
                    not torch.equal(groups, groups.sort().values) or
                    bool((groups < 0).any()) or bool((groups >= 216).any())):
                raise RuntimeError('Frozen UB1 group support changed')
            b2 = torch.from_numpy(bias.copy()).bfloat16().contiguous()
            residual = {k: torch.from_numpy(v.copy()).bfloat16().contiguous() for k, v in rr.items()}
            if (tensor_sha(b2) != bias_sha or
                    {k: tensor_sha(v) for k, v in residual.items()} != rr_sha or
                    residual['first'].shape != (256, 5120) or
                    residual['second'].shape != (5120, 256)):
                raise RuntimeError('Frozen UB1 parameter encoding changed')
            result[step, block] = {
                'channels': (groups[:, None] * 64 + torch.arange(64)).reshape(-1),
                'b2': b2, 'alpha': float(alpha), 'residual': residual,
                'W2_tensor_sha256': w2_sha,
            }
    return result


def make_metadata_resident(metadata):
    size = 0
    for p in metadata.values():
        for name in ('channels', 'b2'):
            p[name] = p[name].cuda().contiguous()
            size += p[name].numel() * p[name].element_size()
        for name, value in p['residual'].items():
            p['residual'][name] = value.cuda().contiguous()
            size += value.numel() * value.element_size()
    if size > 2 << 30:
        raise RuntimeError('UB1 auxiliary residency exceeds two GiB')
    return size


class RowDispatch:
    def __init__(self, runtime, metadata, arm='dense_DQ', expected_shape=(31668, 5120, 13824)):
        self.runtime, self.metadata, self.arm = runtime, metadata, arm
        self.shape = expected_shape
        self.pending = {}
        self.events = []
        self.gate = None
        self.benchmarking = False
        self.fixed_partitions = {}

    @contextmanager
    def action(self, arm):
        if arm not in ARMS or self.pending:
            raise RuntimeError('Invalid action or unmatched previous FFN')
        before = self.arm
        self.arm = arm
        try:
            yield
        finally:
            self.arm = before
            if self.pending:
                raise RuntimeError('Unmatched compact W1/W2 boundary')

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        self.adapter = adapter
        self.old_apply = self.runtime.apply
        self.infer_class = WanTransformerInfer
        self.old_ffn = WanTransformerInfer.infer_ffn
        owner = self
        # Static masks have no online selector. Precompute them before any
        # timed/native request, giving fixed controls their actual advantage.
        device = next(iter(self.metadata.values()))['channels'].device
        for arm in ('UB1', 'fixed_stride', 'fixed_hash'):
            self.fixed_partitions[arm] = row_partition(torch.empty(self.shape[0], device=device), arm, self.shape[0])

        def ffn(infer, phase, x, attn_out, shift, scale, gate=None):
            before = owner.gate
            owner.gate = gate
            try:
                return owner.old_ffn(infer, phase, x, attn_out, shift, scale, gate)
            finally:
                owner.gate = before

        def apply(runtime, mm, input_tensor, *, expert, step_index, block_index, module_name):
            if owner.arm == 'dense_DQ' or not module_name.endswith(('.ffn.0', '.ffn.2')):
                return owner.old_apply(mm, input_tensor, expert=expert, step_index=step_index,
                                       block_index=block_index, module_name=module_name)
            return owner.apply(mm, input_tensor, expert, step_index, block_index, module_name)

        self.runtime.apply = MethodType(apply, self.runtime)
        WanTransformerInfer.infer_ffn = ffn
        return self

    def __exit__(self, *error):
        self.runtime.apply = self.old_apply
        self.infer_class.infer_ffn = self.old_ffn
        if not any(error) and self.pending:
            raise RuntimeError('Row dispatch ended with an unfinished FFN')

    def apply(self, mm, x, expert, step, block, name):
        if expert != ('high_noise_model' if step < 2 else 'low_noise_model'):
            raise RuntimeError('Native expert/step mismatch')
        key = step, block
        rows, dim, hidden = self.shape
        p = self.metadata[key]
        channels, width = p['channels'], len(p['channels'])
        record = self.runtime.records[expert][name]
        if not record['activation_qdq'] or not record['force_fp32'] or x.dtype != torch.bfloat16:
            raise RuntimeError('Original native HiF4/BF16 regime required')
        weight = self.runtime._weight(expert=expert, block_index=block, module_name=name,
                                      record=record, device=x.device, dtype=x.dtype)
        bias = mm._get_actual_bias()
        if bias is not None:
            bias = bias.to(x.device, dtype=x.dtype)
        qdq = self.adapter.native_hif4_activation_qdq
        if name.endswith('.ffn.0'):
            if key in self.pending or x.shape != (rows, dim) or self.gate is None:
                raise RuntimeError('Unexpected full native row/gate boundary')
            v1 = qdq((x.float() / record['smooth_scale'].float()).bfloat16(), force_fp32=True).bfloat16()
            correction = predict_y(v1, p['residual'])
            # This is a hypothesis about dense exceptions, not an oracle for
            # correction error. It sees only existing post-Q1 input and gate.
            if self.arm == 'dynamic_residual':
                from row_dispatch_kernels import gated_energy
                scores = gated_energy(correction, self.gate.reshape(-1))
            else:
                scores = torch.empty(len(v1), device=x.device)
            if self.arm == 'dynamic_residual' or self.arm not in self.fixed_partitions:
                dense, sparse = row_partition(scores, self.arm, len(v1))
            else:
                dense, sparse = self.fixed_partitions[self.arm]
            sparse_correction = correction if self.arm == 'UB1' else correction.index_select(0, sparse)
            del correction
            w0 = weight.index_select(0, channels).contiguous()
            b0 = None if bias is None else bias.index_select(0, channels)
            zs = F.linear(v1 if self.arm == 'UB1' else v1.index_select(0, sparse), w0, b0)
            if self.arm == 'UB1':
                output = zs
            else:
                zd = F.linear(v1.index_select(0, dense), weight, bias)
                output = torch.cat((zs.reshape(-1), zd.reshape(-1)))
            self.pending[key] = (dense, sparse, sparse_correction)
            self.events.append({'step': step, 'block': block, 'arm': self.arm,
                                'benchmark': self.benchmarking, 'dense_rows': len(dense),
                                'sparse_rows': len(sparse), 'sparse_width': width,
                                'dense_row_ids': dense,
                                'pre_W1_signal': 'gated_frozen_R256_magnitude' if self.arm == 'dynamic_residual' else 'fixed',
                                'W1_physical_MKH': [[len(sparse), dim, width], [len(dense), dim, hidden]],
                                'R256_physical_MKH': [[len(v1), dim, p['residual']['first'].shape[0]],
                                                     [len(v1), p['residual']['first'].shape[0], dim]]})
            physical_qdq = 1
        else:
            if key not in self.pending or x.ndim != (2 if self.arm == 'UB1' else 1):
                raise RuntimeError('Compact GELU buffer lacks its original W1 dispatch')
            dense, sparse, correction = self.pending.pop(key)
            cut = len(sparse) * width
            if x.numel() != cut + len(dense) * hidden:
                raise RuntimeError('Native pointwise GELU changed compact buffer geometry')
            if self.arm == 'UB1':
                hidden_s, hidden_d = x, None
            else:
                hidden_s, hidden_d = x[:cut].reshape(len(sparse), width), x[cut:].reshape(len(dense), hidden)
            smooth = record['smooth_scale'].float()
            vs = qdq((hidden_s.float() / smooth.index_select(0, channels)).bfloat16(), force_fp32=True).bfloat16()
            vs = (vs.float() * p['alpha']).bfloat16()
            w2 = weight.index_select(1, channels).contiguous()
            ys = F.linear(vs, w2, p['b2']) + correction
            if self.arm == 'UB1':
                output = ys
            else:
                output = torch.empty((len(dense) + len(sparse), dim), device=x.device, dtype=x.dtype)
                output.index_copy_(0, sparse, ys)
            if len(dense):
                vd = qdq((hidden_d.float() / smooth).bfloat16(), force_fp32=True).bfloat16()
                output.index_copy_(0, dense, F.linear(vd, weight, bias))
            physical_qdq = 1 + bool(len(dense))
        self.runtime.counters['linear_forward_calls'] += 1
        self.runtime.counters['activation_qdq_calls'] += 1
        self.physical_qdq_calls = getattr(self, 'physical_qdq_calls', 0) + physical_qdq
        return output
