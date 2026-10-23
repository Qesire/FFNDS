"""Actual complete group probes and next-block native norm readouts.

The main trajectory remains frozen UB1. Counterfactual FFNs receive full
clones of the same native prefix, execute both projections plus correction,
gate/writeback, then the actual next block's norm1. No video-quality claim.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from c1_contract import ARMS, BLOCKS, select_swaps, read, sha
from prew1_norm_statistics import layer_norm_with_statistics
from run_ar2_restart import tensor_sha


def prepare_actions(positions, metadata, runtime):
    """Fixed moment recentering from old calibration; no original FFN call."""
    bindings = read(positions.base['high_noise_model'].root / 'identity.json')['identity']['training_collections']
    cases = {}
    for binding in bindings:
        path = Path(binding['run'])
        if sha(path) != binding['sha256']: raise RuntimeError('Original calibration run changed')
        run = read(path)
        for item in run['completed']:
            p = Path(item['input_report'])
            if sha(p) != item['input_report_sha256']: raise RuntimeError('Original calibration case changed')
            cases[item['index']] = read(p)
    if set(cases) != set(range(24)): raise RuntimeError('All original24calibration cases required')
    actions = {a: dict(metadata) for a in ('UB1', 'swap_A', 'swap_B')}; records = []
    for step in range(4):
        for block in BLOCKS:
            p = metadata[step, block]; groups = (p['channels'].reshape(-1, 64)[:, 0] // 64).tolist()
            pairs = select_swaps(groups)
            needed = sorted(set(g for pair in pairs for g in pair))
            channels = torch.tensor([g * 64 + j for g in needed for j in range(64)])
            total = torch.zeros(len(channels), dtype=torch.float64); count = 0; scopes = []
            for index in range(24):
                s = next(x for x in cases[index]['scopes'] if (x['step'], x['block']) == (step, block))
                path = Path(s['path'])
                if sha(path) != s['sha256']: raise RuntimeError('Calibration V2 payload changed')
                with safe_open(path, framework='pt', device='cpu') as handle:
                    v = handle.get_tensor('V2').index_select(1, channels)
                # Match native alpha BEFORE W2, including BF16 rounding.
                total += (v.float() * p['alpha']).bfloat16().double().sum(0); count += len(v)
                scopes.append({'source': index, 'sha256': s['sha256']})
            means = (total / count).reshape(len(needed), 64)
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'
            w = runtime.records[expert][f'blocks.{block}.ffn.2']['weight_qdq']
            row = {'step': step, 'block': block, 'samples_for_fixed_moment': count,
                   'calibration_payloads': scopes, 'actions': {}}
            for arm, (outgoing, incoming) in zip(('swap_A', 'swap_B'), pairs):
                new_groups = sorted(set(groups) - {outgoing} | {incoming})
                if len(new_groups) != len(groups): raise RuntimeError('Whole-group budget changed')
                ids = torch.tensor([g * 64 + j for g in new_groups for j in range(64)])
                delta = torch.zeros(5120, dtype=torch.float64)
                for sign, group in ((1., outgoing), (-1., incoming)):
                    delta += sign * (w[:, group * 64:(group + 1) * 64].double() @ means[needed.index(group)])
                b2 = (p['b2'].double() + delta).bfloat16().contiguous()
                actions[arm][step, block] = {**p, 'channels': ids, 'b2': b2}
                row['actions'][arm] = {'removed': outgoing, 'added': incoming,
                                      'channels_sha256': tensor_sha(ids), 'folded_bias_sha256': tensor_sha(b2),
                                      'alpha': p['alpha'], 'R256_unchanged': True,
                                      'mean_correction_norm2': float(delta.square().sum())}
            records.append(row)
            print(f'C1 fixed compensated actions prepared step{step}/block{block}', flush=True)
    return actions, records


def row_sse(a, b):
    values = []
    for offset in range(0, len(a), 1024):
        d = a[offset:offset + 1024].float() - b[offset:offset + 1024].float()
        values.append(d.square().double().sum(-1).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all(): raise RuntimeError('Nonfinite actual native loss')
    return result


class Probe:
    def __init__(self, dispatch, actions, budget, root):
        self.dispatch, self.actions, self.budget, self.root = dispatch, actions, budget, root
        self.request, self.mode, self.source = '', '', -1
        self.pending, self.active, self.rows = {}, None, []
        self.extra_counters = {k: 0 for k in dispatch.runtime.counters}

    @contextmanager
    def action(self, name):
        old_arm, old_metadata = self.dispatch.arm, self.dispatch.metadata
        self.dispatch.arm = 'dense_DQ' if name == 'dense_DQ' else 'UB1'
        self.dispatch.metadata = self.actions.get(name, self.actions['UB1'])
        try: yield
        finally: self.dispatch.arm, self.dispatch.metadata = old_arm, old_metadata

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        import ub1_row_dispatch as row_module
        self.cls, self.row_module = WanTransformerInfer, row_module
        self.old_ffn, self.old_post, self.old_block = [getattr(self.cls, x) for x in ('infer_ffn', 'post_process', 'infer_block')]
        self.old_predict = row_module.predict_y
        owner = self

        def predict(x, parameters, *, fp32=False):
            if fp32: return owner.old_predict(x, parameters, fp32=True)
            h = F.linear(x.bfloat16(), parameters['first'], parameters['bias_first'])
            y = F.linear(h, parameters['second'], parameters['bias_second'])
            if owner.active is not None and owner.active.get('capture'):
                ids = owner.active['ids']
                owner.active['V1'] = x.index_select(0, ids).detach().cpu().view(torch.uint16).numpy().copy()
                owner.active['h256'] = h.index_select(0, ids).detach().cpu().view(torch.uint16).numpy().copy()
            return y

        def ffn(infer, phase, x, attention, shift, scale, gate=None):
            step, block = int(infer.scheduler.step_index), int(infer.block_idx)
            if block not in BLOCKS or owner.mode == 'plain':
                return owner.old_ffn(infer, phase, x, attention, shift, scale, gate)
            if owner.active is not None: raise RuntimeError('Unexpected recursive native FFN')
            rng = np.random.default_rng(20261023 + step * 40 + block)
            ids_cpu = np.sort(np.concatenate([t * 1508 + rng.choice(1508, 16, replace=False) for t in range(21)]))
            ids = torch.tensor(ids_cpu, device=x.device)
            before = x.index_select(0, ids).float(); attn = attention.index_select(0, ids).float()
            after = (x.index_select(0, ids) + attention.index_select(0, ids)).float()
            features = torch.stack([before.std(-1, unbiased=False), after.std(-1, unbiased=False),
                attn.square().mean(-1).sqrt(), (before * attn).mean(-1),
                before.square().mean(-1).sqrt(), after.square().mean(-1).sqrt()], dim=1)
            scope = {'step': step, 'block': block, 'source': owner.source, 'ids': ids,
                     'row_ids': ids_cpu, 'pre_features': features.cpu().numpy(), 'capture': False,
                     'states': {}, 'request': owner.request}
            owner.active = scope; old_norm = phase.norm2.apply

            def norm(value):
                y, mean, rstd = layer_norm_with_statistics(value, infer_dtype=infer.infer_dtype,
                    sensitive_layer_dtype=infer.sensitive_layer_dtype, weight=phase.norm2._get_actual_weight(),
                    bias=phase.norm2._get_actual_bias(), eps=phase.norm2.eps)
                if owner.mode == 'identity':
                    with owner.budget.call(f'{owner.request}_norm_identity_{step}_{block}', 'standalone_normalization_calls'):
                        reference = old_norm(value)
                    if not torch.equal(y, reference): raise RuntimeError('Native norm statistics changed actual output')
                if scope['capture']:
                    scope['norm_mean'] = mean.reshape(-1).index_select(0, ids).float().cpu().numpy()
                    scope['norm_rstd'] = rstd.reshape(-1).index_select(0, ids).float().cpu().numpy()
                return y

            phase.norm2.apply = norm
            try:
                if owner.mode == 'diagnostic':
                    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                    for arm in ARMS:
                        xx, aa = x.clone(), attention.clone()
                        old_counters = dict(owner.dispatch.runtime.counters)
                        with owner.budget.call(f'{owner.request}_{step}_{block}_{arm}', 'standalone_FFN_forwards'), owner.action(arm):
                            y = owner.old_ffn(infer, phase, xx, aa, shift, scale, gate)
                            post = owner.old_post(infer, xx, y, gate)
                        scope['states'][arm] = post
                        for k, v in owner.dispatch.runtime.counters.items():
                            owner.extra_counters[k] += v - old_counters[k]
                    if (not torch.equal(cpu_rng, torch.get_rng_state()) or
                            not torch.equal(gpu_rng, torch.cuda.get_rng_state())):
                        raise RuntimeError('Counterfactual FFNs changed native RNG')
                    scope['post_loss'] = {a: row_sse(v, scope['states']['dense_DQ']) for a, v in scope['states'].items()}
                scope['capture'] = owner.mode == 'diagnostic'
                return owner.old_ffn(infer, phase, x, attention, shift, scale, gate)
            except BaseException:
                owner.active = None; raise
            finally:
                phase.norm2.apply = old_norm

        def post(infer, x, y, gate, pre_infer_out=None):
            output = owner.old_post(infer, x, y, gate, pre_infer_out)
            if owner.active is not None:
                scope = owner.active
                if owner.mode == 'diagnostic':
                    if not torch.equal(output, scope['states']['UB1']):
                        raise RuntimeError('Actual full-M UB1 writeback differs from matched-prefix probe')
                    scope['capture'] = False
                    owner.pending[scope['step'], scope['block'] + 1] = scope
                owner.active = None
            return output

        def block(infer, weights, x, pre):
            key = int(infer.scheduler.step_index), int(infer.block_idx)
            if key in owner.pending:
                scope = owner.pending.pop(key); values = {}
                for arm in ARMS:
                    with owner.budget.call(f'{scope["request"]}_{scope["step"]}_{scope["block"]}_nextnorm_{arm}', 'standalone_normalization_calls'):
                        n = weights.compute_phases[0].norm1.apply(scope['states'][arm])
                    if arm == 'dense_DQ': truth = n
                    values[arm] = row_sse(n, truth)
                path = owner.root / f'{scope["request"]}_s{scope["step"]}_b{scope["block"]:02d}.npz'
                arrays = {k: scope[k] for k in ('row_ids', 'pre_features', 'V1', 'h256', 'norm_mean', 'norm_rstd')}
                for arm in ARMS:
                    arrays['post_' + arm] = scope['post_loss'][arm]
                    arrays['nextnorm_' + arm] = values[arm]
                with path.open('xb') as file: np.savez_compressed(file, **arrays)
                owner.rows.append({'source': scope['source'], 'step': scope['step'], 'block': scope['block'],
                    'payload_HF_relative': str(path.relative_to(Path(__import__('os').environ['HF_HOME']))),
                    'payload_sha256': sha(path), 'rows': 31668, 'sampled_feature_rows': 336,
                    'same_prefix_full_input_clones': True, 'actual_UB1_post_bitwise_identity': True,
                    'horizon': 'ACTUAL_NEXT_BLOCK_NORM1_BEFORE_MODULATION_OR_ATTENTION',
                    'whole_site_nextnorm_SSE': {a: float(v.sum()) for a, v in values.items()},
                    'whole_site_post_SSE': {a: float(v.sum()) for a, v in scope['post_loss'].items()}})
                del scope, values, n, truth
                print(f'C1 native complete-action scope finished: {owner.rows[-1]["source"]}/{key}', flush=True)
            return owner.old_block(infer, weights, x, pre)

        row_module.predict_y = predict
        self.cls.infer_ffn, self.cls.post_process, self.cls.infer_block = ffn, post, block
        return self

    def __exit__(self, *error):
        self.row_module.predict_y = self.old_predict
        self.cls.infer_ffn, self.cls.post_process, self.cls.infer_block = self.old_ffn, self.old_post, self.old_block
        if not any(error) and (self.pending or self.active is not None):
            raise RuntimeError('C1 observer ended with unfinished finite-response scope')
