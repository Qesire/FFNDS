"""Read frozen UB1 coefficients and replay all four native widths, without fitting.

This prepares matched old-state/new-state controls. Existing actual BaseY and
StudentY can be asserted only where that state was collected under this policy.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import digest
from evaluate_residual_rr1 import open_verified
from folded_adapter import replay_main_rows
from onpolicy_residual import position_cases, replay_rows, same_bits
from run_ar2_restart import ROOT, tensor_sha
from run_folded_adapter_cv import prepare


def checked_store(binding, status):
    if digest(binding['path']) != binding['sha256']:
        raise RuntimeError('Frozen UB1 parameter report changed')
    store, report, identity = open_verified(Path(binding['path']), status)
    if store.fingerprint != binding['fingerprint']:
        raise RuntimeError('Frozen UB1 parameter fingerprint differs')
    return store, report, identity


class UB1Positions:
    def __init__(self, policy_path, history, *, expected_policy_sha256):
        self.policy_path = Path(policy_path)
        self.policy_sha256 = digest(policy_path)
        if self.policy_sha256 != expected_policy_sha256:
            raise RuntimeError('UB1 policy differs from the trusted observed baseline')
        self.policy = json.loads(self.policy_path.read_text())
        policy = self.policy
        if (policy['kind'] != 'STEP_BUDGET_PACKED_REFERENCE' or policy['identity_control']
                or policy['groups_by_step'] != [54, 108, 108, 162]):
            raise RuntimeError('Only the original active UB1 policy is supported')
        history.verify(policy['protocol']['path'], policy['protocol']['sha256'])
        binding = policy['parent_policy']
        if digest(binding['path']) != binding['sha256']:
            raise RuntimeError('Frozen UB1 parent policy changed')
        parent = self.parent = json.loads(Path(binding['path']).read_text())
        if (parent['kind'] != 'CALIBRATION_STATIC_PACKED_REFERENCE'
                or parent['support'] != 'C' or parent['mode'] != 'centered_gain_bias'
                or parent['cache_manifest'] != policy['cache_manifest']):
            raise RuntimeError('UB1 requires its original C-centered parent and native cache')
        cb = policy['cache_manifest']
        if digest(cb['path']) != cb['sha256']:
            raise RuntimeError('Frozen UB1 cache manifest changed')
        self.cache = json.loads(Path(cb['path']).read_text())
        self.base = {}
        for expert in ['high_noise_model', 'low_noise_model']:
            store, report, identity = checked_store(parent['fit_reports'][expert], 'COMPLETED_CALIBRATION_PARAMETER_FIT')
            if report['expert'] != expert or len(report['positions']) != 80 or identity['cache_manifest_sha256'] != cb['sha256']:
                raise RuntimeError('Incomplete or mismatched original C24 main coefficients')
            self.base[expert] = store
        self.residual, report, identity = checked_store(parent['residual_fit'], 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
        if (len(report['positions']) != 160 or (report['selected']['rank'] != 256 or report['selected']['lambda'] != 1.0)
                or identity['cache_manifest_sha256'] != cb['sha256']
                or {b['fingerprint'] for b in identity['base_fits']} != {s.fingerprint for s in self.base.values()}):
            raise RuntimeError('Frozen C24 residual does not match its main coefficients')
        self.changed = {}
        if set(policy['changed_fits']) != {'0', '3'}:
            raise RuntimeError('Both original UB1 changed steps are required')
        for step in [0, 3]:
            store, report, identity = checked_store(policy['changed_fits'][str(step)], 'COMPLETED_UB1_STEP40_FIT')
            if (len(report['positions']) != 40 or report['step'] != step
                    or report['groups'] != policy['groups_by_step'][step]
                    or identity['parent_policy_sha256'] != policy['parent_policy']['sha256']
                    or identity['protocol_sha256'] != policy['protocol']['sha256']
                    or identity['cache_manifest_sha256'] != cb['sha256']):
                raise RuntimeError('Changed-step coefficients differ from frozen UB1 bindings')
            self.changed[step] = store

    def position(self, step, block, device='cuda'):
        if step not in range(4) or block not in range(40):
            raise ValueError('Native UB1 position must be inside the four by40 grid')
        expert = 'high_noise_model' if step < 2 else 'low_noise_model'
        name = f'step{step}_block{block:02d}'
        if step in self.changed:
            part = self.changed[step].load('positions', name)
            arrays, info = part['arrays'], part['metadata']
            groups, bias = arrays['groups'], arrays['b2']
            alpha, expected_bias = info['alpha'], info['b2_sha256']
            expected_weight = info['original_W2_sha256']
            rr_arrays = {k[3:]: v for k, v in arrays.items() if k.startswith('rr_')}
            expected_rr = info['RR_parameter_sha256']
        else:
            part = self.base[expert].load('positions', name)
            arrays, info = part['arrays'], part['metadata']
            groups, bias = arrays['C_groups'], arrays['C_centered_gain_bias_b2']
            parameter = info['policies']['C']['modes']['centered_gain_bias']
            alpha, expected_bias = parameter['alpha'], parameter['b2_sha256']
            expected_weight = info['weight_sha256']
            rr = self.residual.load('positions', name)
            rr_arrays, expected_rr = rr['arrays'], rr['metadata']['parameter_sha256']
        keep = self.policy['groups_by_step'][step]
        if (groups.shape != (keep,) or len(np.unique(groups)) != keep
                or not np.array_equal(groups, np.sort(groups)) or groups.min() < 0 or groups.max() >= 216):
            raise RuntimeError('Exact frozen UB1 whole-group support required')
        ids = torch.as_tensor((groups[:, None] * 64 + np.arange(64)).reshape(-1), device=device)
        shard = next(s for s in self.cache['shards'] if s['expert'] == expert and s['block'] == block)
        if digest(shard['path']) != shard['sha256']:
            raise RuntimeError('Original effective teacher shard changed')
        payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)[expert]
        full0, full2 = payload[f'blocks.{block}.ffn.0'], payload[f'blocks.{block}.ffn.2']
        if tensor_sha(full2['weight_qdq']) != expected_weight:
            raise RuntimeError('Frozen UB1 full W2 differs from its native cache')
        w0 = full0['weight_qdq'].to(device).index_select(0, ids).contiguous()
        w2 = full2['weight_qdq'].to(device).index_select(1, ids).contiguous()
        smooth = full2['smooth_scale'].to(device).index_select(0, ids).contiguous()
        b2 = torch.from_numpy(bias).to(device=device, dtype=torch.bfloat16)
        residual = {k: torch.from_numpy(v).to(device=device, dtype=torch.bfloat16) for k, v in rr_arrays.items()}
        if tensor_sha(b2) != expected_bias or {k: tensor_sha(v) for k, v in residual.items()} != expected_rr:
            raise RuntimeError('Frozen UB1 bias or residual parameter encoding changed')
        view = json.loads((ROOT / '.runtime/node_local_model_view.json').read_text())
        raw = next(v for v in view['files'] if f'/{expert}/' in v['view_path'] and v['filename'].endswith('.safetensors'))
        with safe_open(raw['view_path'], framework='pt', device='cpu') as handle:
            raw_bias = handle.get_tensor(f'blocks.{block}.ffn.0.bias').to(device=device, dtype=torch.bfloat16)
        b0 = raw_bias.index_select(0, ids).contiguous()
        # position_cases validates the actual policy's observed folded bias.
        # This explicit expectation is UB1 metadata, not an assertion that the
        # changed positions use the old C24 folded bias.
        expected_capture = {'gate_sha256': info['gate_sha256'], 'original_b2_sha256': info['original_b2_sha256'],
            'policies': {'C': {'modes': {'centered_gain_bias': {'b2_sha256': expected_bias}}}}}
        return {'step': step, 'block': block, 'expert': expert, 'groups': groups.copy(),
            'w0': w0, 'b0': b0, 'w2': w2, 'b2': b2, 'smooth2': smooth, 'alpha': alpha,
            'residual': residual, 'expected_capture': expected_capture, 'raw_b0_sha256': tensor_sha(raw_bias),
            'expected_residual_sha256': expected_rr}


@torch.no_grad()
def replay_position(records, position, quantizer, *, require_captured_UB1):
    if not require_captured_UB1:
        raise ValueError('Old-state recomputation must use a separately explicit observed C24 expectation')
    step, block = position['step'], position['block']
    cases = position_cases(records, step, block, position['expected_capture'])
    for case in cases:
        if case['scope']['original_b0_sha256'] != position['raw_b0_sha256']:
            raise RuntimeError('Original native FFN0 bias differs')
        main = replay_main_rows(case['V1'], case['scope'], position['w0'], position['b0'], position['w2'],
            position['b2'], position['smooth2'], position['alpha'], quantizer)
        residual = replay_rows(case['V1'], case['scope'], position['residual'])
        if not same_bits(main, case['BaseY']) or not same_bits(main + residual, case['StudentY']):
            raise RuntimeError('Complete actual UB1 native main/residual replay differs from its capture')
        case['UB1_main'], case['UB1_frozen_R'] = main, residual
    return cases


@torch.no_grad()
def recompute_on_original_states(records, positions, position, quantizer):
    """Keep the old C24 control explicit before recomputing UB1 on those states."""
    step, block = position['step'], position['block']
    original = prepare(records, positions.cache, positions.parent, positions.residual,
                       step, quantizer, block=block)
    cases = original['cases']
    for case in cases:
        if case['scope']['original_b0_sha256'] != position['raw_b0_sha256']:
            raise RuntimeError('Original-state native FFN0 bias differs')
        main = replay_main_rows(case['V1'], case['scope'], position['w0'], position['b0'], position['w2'],
            position['b2'], position['smooth2'], position['alpha'], quantizer)
        residual = replay_rows(case['V1'], case['scope'], position['residual'])
        if step in [1, 2] and (not same_bits(main, case['BaseY']) or not same_bits(main + residual, case['StudentY'])):
            raise RuntimeError('Unchanged UB1 steps do not match the actual C24 capture')
        case['UB1_main'], case['UB1_frozen_R'] = main, residual
    return cases, {'original_C24_control': original['control'],
                   'UB1_output_observed_under_C24': step in [1, 2],
                   'UB1_changed_step_output_recomputed_not_observed': step in [0, 3]}
