"""Step-budget extension of the frozen C24 runtime, with an exact identity mode."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from ar2_checkpoint import canonical_json, digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import tensor_sha


def validate_budget(policy):
    expected = [108] * 4 if policy['identity_control'] else [54, 108, 108, 162]
    if policy['groups_by_step'] != expected or sum(expected) != 432:
        raise RuntimeError('Only frozen UB1 or its108-group identity control is supported')
    return expected


def create_runtime(cache_path, policy_path):
    from trained_packed_runtime import create_runtime as parent_runtime
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    policy = json.loads(Path(policy_path).read_text())
    if policy['kind'] != 'STEP_BUDGET_PACKED_REFERENCE': raise RuntimeError('Unexpected budget policy')
    counts = validate_budget(policy); protocol = policy['protocol']
    if digest(protocol['path']) != protocol['sha256']: raise RuntimeError('Frozen UB1 protocol changed')
    parent = policy['parent_policy']
    if digest(parent['path']) != parent['sha256']: raise RuntimeError('Frozen C24 parent policy changed')
    runtime = parent_runtime(cache_path, parent['path'])
    if policy['cache_manifest'] != runtime.policy['cache_manifest']:
        raise RuntimeError('Budget and parent quantized cache bindings differ')
    cache = json.loads(Path(runtime.policy['cache_manifest']['path']).read_text())
    old_apply, old_report = runtime.apply, runtime.report
    extra_fits = []
    if not policy['identity_control']:
        if set(policy['changed_fits']) != {'0', '3'}: raise RuntimeError('Both changed-step fits required')
        for step in [0, 3]:
            binding = policy['changed_fits'][str(step)]; path = Path(binding['path'])
            if digest(path) != binding['sha256']: raise RuntimeError('Budget coefficient report changed')
            store, report, identity = open_verified(path, 'COMPLETED_UB1_STEP40_FIT')
            if (store.fingerprint != binding['fingerprint'] or len(report['positions']) != 40
                    or report['step'] != step or report['groups'] != counts[step]
                    or identity['parent_policy_sha256'] != parent['sha256']
                    or identity['protocol_sha256'] != protocol['sha256']
                    or identity['cache_manifest_sha256'] != policy['cache_manifest']['sha256']):
                raise RuntimeError('Budget fit/parent/cache/protocol identity differs')
            expert = 'high_noise_model' if step == 0 else 'low_noise_model'
            extra_fits.append({'step': step, 'fingerprint': store.fingerprint})
            for block in range(40):
                part = store.load('positions', f'step{step}_block{block:02d}')
                if part is None: raise RuntimeError('Missing changed-step position')
                arrays, info = part['arrays'], part['metadata']
                groups = torch.from_numpy(arrays['groups']).long()
                if (groups.shape != (counts[step],) or len(groups.unique()) != counts[step]
                        or not torch.equal(groups, groups.sort().values) or bool((groups < 0).any()) or bool((groups >= 216).any())):
                    raise RuntimeError('Invalid original whole64-group support')
                channels = (groups[:, None] * 64 + torch.arange(64)).reshape(-1)
                shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                if digest(shard['path']) != shard['sha256']: raise RuntimeError('Original effective shard changed')
                payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)[expert]
                r0, r2 = [dict(payload[f'blocks.{block}.ffn.{s}']) for s in [0, 2]]
                if tensor_sha(r2['weight_qdq']) != info['original_W2_sha256']:
                    raise RuntimeError('Budget fit W2 differs from runtime')
                r0['weight_qdq'] = r0['weight_qdq'].index_select(0, channels).contiguous()
                r2['weight_qdq'] = r2['weight_qdq'].index_select(1, channels).contiguous()
                r2['smooth_scale'] = r2['smooth_scale'].index_select(0, channels).contiguous()
                bias = torch.from_numpy(arrays['b2']).bfloat16()
                if tensor_sha(bias) != info['b2_sha256']: raise RuntimeError('Budget folded bias changed')
                rr = {k[3:]: torch.from_numpy(v).bfloat16() for k, v in arrays.items() if k.startswith('rr_')}
                if {k: tensor_sha(v) for k, v in rr.items()} != info['RR_parameter_sha256']:
                    raise RuntimeError('Budget native residual encoding changed')
                if info['original_b2_sha256'] != runtime.positions[step, block]['original_b2_sha256']:
                    raise RuntimeError('Original raw bias changed')
                runtime.positions[step, block] = {'records': {0: r0, 2: r2}, 'channels': channels,
                    'alpha': info['alpha'], 'b2': bias, 'original_b2_sha256': info['original_b2_sha256']}
                runtime.residual_parameters[step, block] = rr
            del payload
    # The resident loader must reference the same first-step tensors as positions,
    # otherwise unused old packed weights would consume extra GPU memory.
    for expert, step in [('high_noise_model', 0), ('low_noise_model', 2)]:
        for block in range(40):
            for slot in [0, 2]: runtime.records[expert][f'blocks.{block}.ffn.{slot}'] = runtime.positions[step, block]['records'][slot]
    runtime.policy, runtime.policy_sha = policy, digest(policy_path)
    runtime.residual_fingerprint = hashlib.sha256(canonical_json({'parent': runtime.residual_fingerprint,
        'changed': extra_fits, 'policy_sha256': runtime.policy_sha}).encode()).hexdigest()
    runtime.channels_cuda.clear()

    def apply(self, mm_weight, input_tensor, *, expert, step_index, block_index, module_name):
        if not module_name.endswith(('.ffn.0', '.ffn.2')):
            return old_apply(mm_weight, input_tensor, expert=expert, step_index=step_index,
                block_index=block_index, module_name=module_name)
        if expert != ('high_noise_model' if step_index < 2 else 'low_noise_model'):
            raise RuntimeError('Budget expert/step mismatch')
        position = self.positions[step_index, block_index]; width = counts[step_index] * 64
        slot = 0 if module_name.endswith('.ffn.0') else 2
        expected_in, expected_out = (5120, width) if slot == 0 else (width, 5120)
        if input_tensor.shape[-1] != expected_in: raise RuntimeError('Budget main input width differs')
        record = position['records'][slot]
        x = (input_tensor.float() / record['smooth_scale'].to(input_tensor.device)).to(input_tensor.dtype)
        x = qdq(x, force_fp32=True); self.counters['activation_qdq_calls'] += 1
        pending = (step_index, block_index)
        if slot == 0:
            if pending in self.pending_residual: raise RuntimeError('Duplicate budget residual prediction')
            parameters = {k: v.to(x.device) for k, v in self.residual_parameters[pending].items()}
            self.pending_residual[pending] = self.predict_residual(x, parameters)
            self.residual_coverage[f'step{step_index}_block{block_index}'] = {
                'rows': x.numel() // 5120, 'auxiliary_GEMM_shapes': [[256, 5120], [5120, 256]],
                'auxiliary_linear_calls': 2, 'auxiliary_parameter_H2D_bytes': sum(v.numel() * v.element_size() for v in parameters.values()),
                'auxiliary_MACs_over_original_dense_FFN': 256 / 13824}
        if slot == 2:
            x = (x.float() * position['alpha']).to(x.dtype); bias = position['b2'].to(x.device)
        else:
            original = mm_weight._get_actual_bias(); key = (step_index, block_index, str(x.device))
            if key not in self.channels_cuda: self.channels_cuda[key] = position['channels'].to(x.device)
            bias = None if original is None else original.to(x.device).index_select(0, self.channels_cuda[key])
        weight = self._weight(expert=expert, block_index=block_index, module_name=module_name,
            record=record, device=x.device, dtype=x.dtype)
        result = F.linear(x, weight, bias)
        if slot == 2:
            if pending not in self.pending_residual: raise RuntimeError('Missing matched budget residual')
            correction = self.pending_residual.pop(pending)
            if correction.shape != result.shape or correction.dtype != result.dtype: raise RuntimeError('Budget residual geometry differs')
            result = result + correction
            if step_index == 3 and block_index == 39 and (len(self.residual_coverage) != 160 or self.pending_residual):
                raise RuntimeError('Budget final position lacks full160 residual coverage')
        key = f'step{step_index}_block{block_index}_ffn{slot}'
        if result.shape[-1] != expected_out or key in self.coverage: raise RuntimeError('Duplicate/invalid budget output')
        self.coverage[key] = {'rows': input_tensor.numel() // expected_in, 'actual_weight_shape': list(weight.shape),
            'main_GEMM_density': width / 13824}
        self.counters['linear_forward_calls'] += 1
        return result

    def report():
        result = old_report(); packed = result['packed_reference']; packed['parent_runtime_source_sha256'] = packed['source_sha256']
        packed.update(source_sha256=digest(Path(__file__)), groups_by_step=counts,
            main_density_by_step=[n / 216 for n in counts], nominal_global_main_MAC_density=.5,
            changed_fit_identities=extra_fits, identity_control=policy['identity_control'])
        if len(runtime.coverage) == 320:
            numerator = sum(r['rows'] * int(np.prod(r['actual_weight_shape'])) for r in runtime.coverage.values())
            denominator = sum(r['rows'] * 5120 * 13824 for r in runtime.coverage.values())
            packed['actual_global_main_MAC_density'] = numerator / denominator
            if numerator * 2 != denominator: raise RuntimeError('Budget actual total mainMAC is not exactly50%')
        return result

    runtime.apply = MethodType(apply, runtime); runtime.report = report
    return runtime
