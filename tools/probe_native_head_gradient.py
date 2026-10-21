"""R11G1: one complete native control capture, then fixed head-gradient audit.

This is neither a trained candidate nor a trajectory recovery experiment.
Captured full tensors remain in HF cache; local reports retain raw failures.
"""
import argparse
import contextvars
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from collect_ub1_native_states import policy_alias, verify_runtime
from historical_source_snapshot import HistoricalSources
from native_head_replay import PARAMETERS, normalized_update, patchify_flow, replay, unpatchify_flow
from reduced_rank_residual import predict_y
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from run_cross_layer_transport import references
from run_supplement_teacher_flow import exact_native_latent

METHOD = 'R11G1_NATIVE_HEAD_GRADIENT_READINESS'
PROTOCOL = ROOT / 'research/20261021/R11G1_native_head_protocol.json'
SOURCES = ['probe_native_head_gradient.py', 'native_head_replay.py', 'run_cross_layer_transport.py',
           'run_supplement_teacher_flow.py', 'collect_ub1_native_states.py', 'restored_assets.py',
           'historical_source_snapshot.py', 'resident_ffn_runtime.py', 'step_budget_runtime.py',
           'trained_packed_runtime.py', 'reduced_rank_residual.py']
NATIVE = ['models/networks/wan/infer/transformer_infer.py', 'models/networks/wan/infer/post_infer.py',
          'common/ops/mm/mm_weight.py', 'common/ops/norm/layer_norm_weight.py']


def source_binding():
    result = {f'tools/{n}': digest(ROOT / 'tools' / n) for n in SOURCES}
    result.update({f'.runtime/code/LightX2V/lightx2v/{n}': digest(ROOT / '.runtime/code/LightX2V/lightx2v' / n) for n in NATIVE})
    return result


class FinalBoundary:
    def __init__(self, runtime):
        self.runtime = runtime
        self.context = contextvars.ContextVar('ffnds_head_gradient', default=False)
        self.data = {}

    def put(self, name, value):
        if name in self.data:
            raise RuntimeError('Duplicate complete native operand: ' + name)
        self.data[name] = value.detach().cpu().contiguous().clone()

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
        self.cls = WanTransformerInfer
        self.old_ffn, self.old_post, self.old_head = self.cls.infer_ffn, self.cls.post_process, self.cls.infer_non_blocks
        self.old_apply, self.old_predict = self.runtime.apply, self.runtime.predict_residual
        observer = self

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            selected = (int(infer.scheduler.step_index), int(infer.block_idx)) == (3, 39)
            token = observer.context.set(selected)
            try:
                return observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
            finally:
                observer.context.reset(token)

        def predict(x, parameters):
            result = observer.old_predict(x, parameters)
            if observer.context.get():
                observer.put('u', x)
                observer.put('correction', result)
                for key in PARAMETERS:
                    observer.put('p_' + key, parameters[key])
            return result

        def apply(mm, value, **kw):
            result = observer.old_apply(mm, value, **kw)
            if observer.context.get() and kw['module_name'].endswith('.ffn.2'):
                position = observer.runtime.positions[3, 39]
                record = position['records'][2]
                x = (value.float() / record['smooth_scale']).to(value.dtype)
                x = qdq(x, force_fp32=True)
                x = (x.float() * position['alpha']).to(x.dtype)
                weight = observer.runtime._weight(record=record, device=x.device, dtype=x.dtype, **{k: kw[k] for k in ('expert', 'block_index', 'module_name')})
                main = F.linear(x, weight, position['b2'])
                correction = observer.data['correction'].to(result.device)
                if not torch.equal(main + correction, result):
                    raise RuntimeError('Reconstructed unchanged main/residual sum differs bitwise')
                observer.put('main', main)
                observer.put('native_y', result)
            return result

        def post(infer, x, y, gate, pre):
            if (int(infer.scheduler.step_index), int(infer.block_idx)) == (3, 39):
                observer.put('base', x)
                observer.put('gate', gate.squeeze())
            return observer.old_post(infer, x, y, gate, pre)

        def head(infer, weights, x, embed):
            if int(infer.scheduler.step_index) != 3:
                return observer.old_head(infer, weights, x, embed)
            if infer.sensitive_layer_dtype != infer.infer_dtype or infer.infer_dtype != torch.bfloat16:
                raise RuntimeError('This registered replay only supports the original BF16 sensitive path')
            if embed.dim() != 2 or weights.norm._get_actual_weight() is not None or weights.norm._get_actual_bias() is not None or weights.norm.eps != 1e-6:
                raise RuntimeError('Original parameter-free torch head norm required')
            if weights.head.has_lora_branch or weights.head.has_diff:
                raise RuntimeError('Original head must not have adapters')
            observer.put('native_post', x)
            shift, scale = (weights.head_modulation.tensor + embed.unsqueeze(1)).chunk(2, dim=1)
            observer.put('shift', shift.squeeze())
            observer.put('scale', scale.squeeze())
            observer.put('head_weight', weights.head._get_actual_weight())
            observer.put('head_bias', weights.head._get_actual_bias())
            result = observer.old_head(infer, weights, x, embed)
            observer.put('native_tokens', result)
            return result

        self.cls.infer_ffn, self.cls.post_process, self.cls.infer_non_blocks = ffn, post, head
        self.runtime.apply, self.runtime.predict_residual = apply, predict
        return self

    def __exit__(self, *error):
        self.cls.infer_ffn, self.cls.post_process, self.cls.infer_non_blocks = self.old_ffn, self.old_post, self.old_head
        self.runtime.apply, self.runtime.predict_residual = self.old_apply, self.old_predict


def capture(args, folder, report):
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from step_budget_runtime import create_runtime
    history = HistoricalSources()
    alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    refs, bindings = references(history, assets)
    raw = Path(cache['path']).parents[1] / 'r11g1_native_head' / args.out.name
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 83 << 30:
        raise RuntimeError('Preserve80GiB free reserve plus3GiB full-operand allowance')
    if (raw / 'boundary.safetensors').exists():
        raise RuntimeError('Preserve prior complete/failed capture')
    report.update(original_policy_sha256=policy_sha, assets=assets, reference_bindings=bindings,
                  historical_source_receipts=history.receipts)
    with unused_raw_placeholders(adapter) as audit:
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
    adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
    if len(audit['names']) != 400 or audit['calls'] < 800:
        raise RuntimeError('Incomplete strict placeholder controls')
    runtime = create_runtime(cache['path'], alias)
    report['residency'] = make_resident(runtime)
    adapter.activate_lightx2v_hooks(runtime=runtime)
    scheduler = pipe.runner.scheduler
    old_pre, old_post = scheduler.step_pre, scheduler.step_post
    dq, ub1 = refs['DQ'][0], refs['UB1'][0]
    sample = dq['case']['input']
    if sample['seed'] != 42 or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
        raise RuntimeError('Fixed source identity changed')
    observer = FinalBoundary(runtime)
    steps = []
    scheduler.generator = None
    runtime.begin_request()
    before = dict(runtime.counters)

    def pre(step_index):
        value = old_pre(step_index)
        if not torch.equal(scheduler.latents.detach().float().cpu(), dq['payloads'][step_index]['input']):
            raise RuntimeError('Original DQ trajectory input differs bitwise')
        return value

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        value = old_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        target = ub1['payloads'][step]
        if not torch.equal(flow, target['flow']) or not torch.equal(latent, target['post']):
            raise RuntimeError('Original complete UB1 flow/post control differs bitwise')
        steps.append({'step': step, 'input_bitwise_equal': True, 'flow_bitwise_equal': True,
                      'post_bitwise_equal': True, 'flow_sha256': tensor_sha(flow)})
        if step == 3:
            observer.put('native_flow', flow)
            observer.put('teacher_flow', dq['payloads'][step]['flow'])
        scheduler.latents = exact_native_latent(dq['payloads'][step]['post'], scheduler.latents)
        return value

    scheduler.step_pre, scheduler.step_post = pre, post
    try:
        with torch.inference_mode(), observer:
            pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(raw / 'diagnostic_only.mp4'))
        torch.cuda.synchronize()
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if (delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']
                or audit['fallback_calls'] or [s['step'] for s in steps] != list(range(4))):
            raise RuntimeError('Full native runtime/control coverage changed')
        verify_runtime(runtime, 'UB1')
        if observer.data['u'].shape != (31668, 5120) or observer.data['native_tokens'].shape != (31668, 64):
            raise RuntimeError('Incomplete native geometry')
        path = raw / 'boundary.safetensors'
        save_file(observer.data, str(path))
        report.update(status='PASS_FULL_NATIVE_HEAD_CAPTURE', steps=steps, counter_delta=delta,
                      payload_path=str(path), payload_sha256=digest(path), payload_bytes=path.stat().st_size,
                      tensors={k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_sha(v)} for k, v in observer.data.items()},
                      trajectory='Original DQ post restored only between complete steps; not student free rollout',
                      diagnostic_video_is_DQ_by_construction=True)
    finally:
        scheduler.step_pre, scheduler.step_post = old_pre, old_post
        adapter.deactivate_lightx2v_hooks()


def gradient(args, folder, report):
    control_path = args.out / 'capture/RUN.json'
    control = json.loads(control_path.read_text())
    if control['status'] != 'PASS_FULL_NATIVE_HEAD_CAPTURE' or control['source_sha256'] != report['source_sha256'] or control['protocol_sha256'] != report['protocol_sha256']:
        raise RuntimeError('Complete frozen native capture prerequisite required')
    if digest(control['payload_path']) != control['payload_sha256']:
        raise RuntimeError('Complete operand payload changed')
    data = load_file(control['payload_path'], device='cuda')
    for name, value in data.items():
        if tensor_sha(value) != control['tensors'][name]['sha256']:
            raise RuntimeError('Captured tensor changed')
    parameters = {key: data['p_' + key].float().detach().requires_grad_(True) for key in PARAMETERS}
    grid = (21, 29, 52)
    target = patchify_flow(data['teacher_flow'], grid)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokens = replay(data, parameters)
    post = data['base'] + data['native_y'] * data['gate']
    controls = {'residual_bitwise_equal': torch.equal(predict_y(data['u'], parameters), data['correction']),
                'gate_writeback_bitwise_equal': torch.equal(post, data['native_post']),
                'head_tokens_bitwise_equal': torch.equal(tokens, data['native_tokens']),
                'full_flow_bitwise_equal': torch.equal(unpatchify_flow(tokens, grid).float(), data['native_flow']),
                'flow_permutation_inverse_exact': torch.equal(unpatchify_flow(target, grid), data['teacher_flow'])}
    report['controls'] = controls
    if not all(controls.values()):
        raise RuntimeError('Zero-update complete native forward equivalence failed; stop before gradient')
    loss = (tokens.float() - target).square().mean()
    initial = float(loss.detach())
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = {k: {'connected': v.grad is not None, 'finite': v.grad is not None and bool(torch.isfinite(v.grad).all()),
                     'nonzero_count': int(torch.count_nonzero(v.grad)) if v.grad is not None else 0,
                     'numel': v.numel(), 'L2': float(v.grad.double().norm()) if v.grad is not None else None} for k, v in parameters.items()}
    report['gradients'] = gradients
    if any(not r['finite'] or not r['nonzero_count'] for r in gradients.values()):
        raise RuntimeError('Native gradient not finite/connected for every deployed tensor')
    updated, update = normalized_update(parameters, .001)
    update['BF16_changed_elements'] = {k: int(torch.count_nonzero(updated[k].bfloat16() != parameters[k].detach().bfloat16())) for k in PARAMETERS}
    if sum(update['BF16_changed_elements'].values()) == 0:
        raise RuntimeError('FP32 update does not reach deployed BF16 parameters')
    del tokens, loss, post
    torch.cuda.synchronize()
    after_start = time.perf_counter()
    with torch.no_grad():
        after = replay(data, updated)
        final = float((after.float() - target).square().mean())
        flow_change = float((after.float() - data['native_tokens'].float()).double().square().sum())
    if not bool(torch.isfinite(torch.tensor([initial, final, flow_change])).all()) or flow_change <= 0:
        raise RuntimeError('Fixed update has nonfinite or zero actual native flow change')
    torch.cuda.synchronize()
    report.update(status='PASS_NATIVE_HEAD_GRADIENT_REACHABILITY', capture_sha256=digest(control_path),
                  initial_flow_MSE=initial, one_fixed_update_flow_MSE=final,
                  one_fixed_update_relative_improvement=1 - final / initial,
                  actual_flow_change_SSE=flow_change, update=update,
                  fullM_forward_backward_seconds=elapsed, updated_forward_seconds=time.perf_counter() - after_start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  no_quantizer_STE='Q1 and fixed main are captured native operands; gradients only enter existing R256 and exact BF16 downstream ops',
                  optimization_scope='One fixed .001 relative-global-L2 step on one development-exposed source; not optimization acceptance, validation or recovery',
                  inference_parameter_count=sum(v.numel() for v in parameters.values()), additional_inference_parameters=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['capture', 'gradient'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(ROOT)
    if validate_records(*records) or not method_eligibility(records[0], records[1], METHOD)[0]:
        raise RuntimeError('Fixed readiness diagnostic is not consistently registered')
    args.out = args.out.resolve()
    folder = args.out / args.phase
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve previous completed/failed attempt')
    torch.set_num_threads(8)
    report = {'method_id': METHOD, 'status': 'RUNNING', 'phase': args.phase,
              'protocol_sha256': digest(PROTOCOL), 'source_sha256': source_binding(),
              'quality_or_speed_acceptance': False}
    atomic_json(folder / 'RUN.json', report)
    start = time.monotonic()
    try:
        (capture if args.phase == 'capture' else gradient)(args, folder, report)
        if source_binding() != report['source_sha256'] or digest(PROTOCOL) != report['protocol_sha256']:
            raise RuntimeError('Frozen diagnostic source/protocol changed')
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        atomic_json(folder / 'RUN.json', report)
    print(json.dumps({'phase': args.phase, 'status': report['status']}), flush=True)


if __name__ == '__main__':
    main()
