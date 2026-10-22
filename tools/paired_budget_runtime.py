"""R12 D0: real packed cross-layer dispatch with zero pair correction.

The archived UB1 runtime remains unchanged. Outside the selected pair the
arithmetic below is copied in order from its frozen apply method. Only this
separate runtime has a 158-position residual contract.
"""
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from ar2_checkpoint import digest
from r12d0_contract import ACTIONS, ROOT, groups, residual_count
from run_ar2_restart import tensor_sha


def audited_packs(runtime, cache, dense=False):
    audit = json.loads((ROOT / 'research/20261022/R12_E0_operand_audit.json').read_text())
    variants, receipts = {}, []
    for block in (30, 31):
        expected = next(o for o in audit['observations'] if o['family'] == 'ORIGINAL_DQ_WEIGHTS' and o['block'] == block)
        shard = next(s for s in cache['shards'] if s['expert'] == 'low_noise_model' and s['block'] == block)
        if digest(shard['path']) != shard['sha256'] or shard['sha256'] != expected['sha256']:
            raise RuntimeError('Original D-Q pair shard changed')
        payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)['low_noise_model']
        for count in ([216] if dense else [108, 162, 216]):
            frozen = expected['CPU_pack_variants'][str(count)]
            channels = (torch.tensor(frozen['groups'])[:, None] * 64 + torch.arange(64)).reshape(-1)
            if runtime is not None and count == 162:
                position = runtime.positions[3, block]
                if not torch.equal(channels, position['channels']):
                    raise RuntimeError('Frozen UB1 support differs from audited 162 support')
                records = position['records']
            elif runtime is not None and dense:
                records = {s: runtime.records['low_noise_model'][f'blocks.{block}.ffn.{s}'] for s in (0, 2)}
            else:
                records = {s: dict(payload[f'blocks.{block}.ffn.{s}']) for s in (0, 2)}
                records[0]['weight_qdq'] = records[0]['weight_qdq'].index_select(0, channels).contiguous()
                records[2]['weight_qdq'] = records[2]['weight_qdq'].index_select(1, channels).contiguous()
                records[2]['smooth_scale'] = records[2]['smooth_scale'].index_select(0, channels).contiguous()
            for slot, record in records.items():
                frozen_record = frozen['packed_effective_records'][str(slot)]
                if (not record['activation_qdq'] or not record['force_fp32']
                        or record['weight_qdq'].dtype != torch.bfloat16
                        or list(record['weight_qdq'].shape) != frozen_record['weight_shape']
                        or tensor_sha(record['weight_qdq']) != frozen_record['weight_sha256']
                        or tensor_sha(record['smooth_scale']) != frozen_record['smooth_scale_sha256']):
                    raise RuntimeError('Actual packed weight/Smooth differs from E0 CPU audit')
            variants[block, count] = {'records': records, 'channels': channels}
            receipts.append({'block': block, 'groups': count, 'source_shard_sha256': shard['sha256'],
                             'pack_binding': frozen})
    return variants, receipts


def install(runtime, cache, *, dense=False):
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    old_apply, old_report = runtime.apply, runtime.report
    variants, receipts = audited_packs(runtime, cache, dense=dense)
    runtime.r12_variants, runtime.r12_pack_receipts = variants, receipts
    runtime.r12_quantized, runtime.r12_coverage = {}, {}
    runtime.r12_arm = 'DQ' if dense else 'N0'

    def apply(self, mm_weight, input_tensor, *, expert, step_index, block_index, module_name):
        kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
        if not module_name.endswith(('.ffn.0', '.ffn.2')):
            return old_apply(mm_weight, input_tensor, **kw)
        if expert != ('high_noise_model' if step_index < 2 else 'low_noise_model'):
            raise RuntimeError('Native expert/step mismatch')
        slot = 0 if module_name.endswith('.ffn.0') else 2
        count = groups(self.r12_arm, step_index, block_index)
        width = count * 64
        ni, no = (5120, width) if slot == 0 else (width, 5120)
        if input_tensor.shape != (31668, ni) or input_tensor.dtype != torch.bfloat16:
            raise RuntimeError('Native full-M BF16 packed geometry changed')
        raw = step_index == 3 and block_index in (30, 31) and self.r12_arm != 'UB1'
        key = f'step{step_index}_block{block_index}_ffn{slot}'
        if key in self.r12_coverage:
            raise RuntimeError('Duplicate main FFN call')
        if dense and not raw:
            record = self.records[expert][module_name]
            result = old_apply(mm_weight, input_tensor, **kw)
            weight = record['weight_qdq']
        else:
            position = self.r12_variants[block_index, count] if raw else self.positions[step_index, block_index]
            record = position['records'][slot]
            x = (input_tensor.float() / record['smooth_scale'].to(input_tensor.device)).to(input_tensor.dtype)
            x = qdq(x, force_fp32=True)
            if raw:
                x = x.to(input_tensor.dtype)
            self.counters['activation_qdq_calls'] += 1
            if step_index == 3 and block_index in (30, 31):
                self.r12_quantized.setdefault(block_index, {})[f'Q{1 if slot == 0 else 2}_sha256'] = tensor_sha(x)
            pending = (step_index, block_index)
            if not raw and slot == 0:
                if pending in self.pending_residual:
                    raise RuntimeError('Duplicate residual prediction')
                parameters = {k: v.to(x.device) for k, v in self.residual_parameters[pending].items()}
                self.pending_residual[pending] = self.predict_residual(x, parameters)
                self.residual_coverage[f'step{step_index}_block{block_index}'] = {
                    'rows': x.numel() // 5120, 'auxiliary_GEMM_shapes': [[256, 5120], [5120, 256]],
                    'auxiliary_linear_calls': 2, 'auxiliary_parameter_H2D_bytes': 0,
                    'auxiliary_MACs_over_original_dense_FFN': 256 / 13824}
            if slot == 2 and not raw:
                x = (x.float() * position['alpha']).to(x.dtype)
                bias = position['b2'].to(x.device)
            else:
                original = mm_weight._get_actual_bias()
                if original is None:
                    raise RuntimeError('Original raw bias missing')
                if slot == 0:
                    # Exactly the original selected bias rows; no fitted bias.
                    bias = original.to(x.device).index_select(0, position['channels'].to(x.device))
                else:
                    bias = original.to(device=x.device, dtype=x.dtype)
            weight = self._weight(expert=expert, block_index=block_index, module_name=module_name,
                                  record=record, device=x.device, dtype=x.dtype)
            result = F.linear(x, weight, bias)
            if slot == 2 and not raw:
                if pending not in self.pending_residual:
                    raise RuntimeError('Missing matched residual')
                correction = self.pending_residual.pop(pending)
                if correction.shape != result.shape or correction.dtype != result.dtype:
                    raise RuntimeError('Residual shape/dtype mismatch')
                result = result + correction
                if step_index == 3 and block_index == 39:
                    if len(self.residual_coverage) != residual_count(self.r12_arm) or self.pending_residual:
                        raise RuntimeError('Explicit D0 residual coverage mismatch')
            self.counters['linear_forward_calls'] += 1
        if result.shape != (31668, no) or tuple(weight.shape) != (no, ni):
            raise RuntimeError('Actual dispatched GEMM/output shape mismatch')
        self.r12_coverage[key] = {'rows': 31668, 'actual_weight_shape': list(weight.shape),
                                 'groups': count, 'raw_zero_correction': raw}
        return result

    def report():
        result = old_report()
        coverage = runtime.r12_coverage
        density = (sum(r['rows'] * r['actual_weight_shape'][0] * r['actual_weight_shape'][1]
                       for r in coverage.values()) / (2 * 31668 * 5120 * 13824 * 160)
                   if len(coverage) == 320 else None)
        result['R12_D0_dispatch'] = {'arm': runtime.r12_arm,
            'source_sha256': digest(__file__), 'actual_main_linear_calls': len(coverage),
            'main_coverage': coverage, 'actual_global_main_MAC_fraction': density,
            'actual_residual_positions': len(getattr(runtime, 'residual_coverage', {})),
            'expected_residual_positions': residual_count(runtime.r12_arm),
            'pair_groups': [groups(runtime.r12_arm, 3, b) for b in (30, 31)],
            'complete_FFN_timing_or_quality_acceptance': False}
        if 'packed_reference' in result:
            packed = result['packed_reference']
            packed.update(actual_FFN_linear_calls=len(coverage), coverage=coverage,
                          actual_global_main_MAC_density=density,
                          paired_dispatch=result['R12_D0_dispatch'])
            packed['residual_correction']['expected_positions'] = residual_count(runtime.r12_arm)
        return result

    runtime.apply = MethodType(apply, runtime)
    runtime.report = report
    return runtime


def make_pair_resident(runtime):
    # make_resident already moves the aliased UB1162 records and dense216.
    transferred = tensors = 0
    seen = set()
    for position in runtime.r12_variants.values():
        for record in position['records'].values():
            if id(record) in seen:
                continue
            seen.add(id(record))
            for key in ('weight_qdq', 'smooth_scale'):
                value = record[key]
                if value.device.type != 'cuda':
                    transferred += value.numel() * value.element_size()
                    tensors += 1
                    record[key] = value.to('cuda').contiguous()
        channels = position['channels']
        if channels.device.type != 'cuda':
            transferred += channels.numel() * channels.element_size()
            tensors += 1
            position['channels'] = channels.to('cuda')
    torch.cuda.synchronize()
    return {'additional_setup_H2D_bytes': transferred, 'additional_setup_tensors': tensors,
            'steady_weight_H2D_bytes': 0}


def begin(runtime, arm):
    if arm not in ('DQ', 'UB1', 'N0_repeat', *ACTIONS):
        raise ValueError('Unregistered arm')
    if hasattr(runtime, 'begin_request'):
        runtime.begin_request()
    runtime.r12_arm = arm
    runtime.r12_coverage.clear()
    runtime.r12_quantized.clear()


def verify(runtime):
    coverage, arm = runtime.r12_coverage, runtime.r12_arm
    expected = {f'step{s}_block{b}_ffn{slot}' for s in range(4) for b in range(40) for slot in (0, 2)}
    if set(coverage) != expected:
        raise RuntimeError('Not all320 actual main GEMMs were dispatched')
    main = sum(r['rows'] * r['actual_weight_shape'][0] * r['actual_weight_shape'][1] for r in coverage.values())
    dense = 2 * 31668 * 5120 * 13824 * 160
    if main != (dense if arm == 'DQ' else dense // 2):
        raise RuntimeError('Actual full-model main MAC budget mismatch')
    residual = getattr(runtime, 'residual_coverage', {})
    expected_rr = {f'step{s}_block{b}' for s in range(4) for b in range(40)
                   if arm != 'DQ' and not (arm != 'UB1' and s == 3 and b in (30, 31))}
    if set(residual) != expected_rr or getattr(runtime, 'pending_residual', {}):
        raise RuntimeError('Actual residual positions mismatch; no dummy entries allowed')
    return {'main_MAC': main, 'dense_main_MAC': dense, 'actual_global_main_MAC_fraction': main / dense,
            'actual_main_linear_calls': len(coverage), 'main_coverage': coverage,
            'actual_residual_positions': len(residual), 'expected_residual_positions': residual_count(arm),
            'residual_auxiliary_linear_calls': 2 * len(residual),
            'residual_auxiliary_MAC': 2 * 31668 * 5120 * 256 * len(residual),
            'residual_coverage': residual, 'pending_residual_predictions': 0,
            'complete_FFN_timing_or_quality_acceptance': False}
