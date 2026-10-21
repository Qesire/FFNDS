"""R8D1: exact full-token post-Q1 moments on unchanged original UB1 rollouts.

Data-readiness diagnosis only; no quantization, fitting or candidate changes.
"""
import argparse
import contextvars
import fcntl
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from collect_ub1_native_states import policy_alias, verify_runtime
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import same_bits
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from ub1_native_data import load_ub1_collections

METHOD = 'R8D1_FULL_POPULATION_Q1_COVARIANCE'
PROTOCOL = ROOT / 'research/20261021/R8D1_population_covariance_protocol.json'
SOURCES = ['collect_q1_population.py', 'collect_ub1_native_states.py', 'ub1_native_data.py',
           'onpolicy_residual.py', 'historical_source_snapshot.py', 'restored_assets.py',
           'resident_ffn_runtime.py', 'step_budget_runtime.py', 'trained_packed_runtime.py', 'reduced_rank_residual.py']


def source_binding():
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in SOURCES}


class PopulationObserver:
    def __init__(self, runtime, records, train_ids):
        self.runtime, self.records, self.train_ids = runtime, records, set(train_ids)
        self.context = contextvars.ContextVar('ffnds_population_Q1', default=None)
        self.sums, self.counts, self.means, self.scopes = {}, {}, {}, []

    def begin(self, index):
        self.index = index
        self.request_scopes = []

    def __enter__(self):
        self.old_apply, self.old_predict = self.runtime.apply, self.runtime.predict_residual
        observer = self

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            selected = step_index in (2, 3) and block_index == 20 and module_name.endswith('.ffn.0')
            token = observer.context.set((step_index, block_index)) if selected else None
            try:
                return observer.old_apply(mm, value, expert=expert, step_index=step_index,
                                          block_index=block_index, module_name=module_name)
            finally:
                if token is not None: observer.context.reset(token)

        def predict(x, parameters):
            result = observer.old_predict(x, parameters)
            position = observer.context.get()
            if position is not None:
                observer.observe(x, *position)
            return result

        self.runtime.apply, self.runtime.predict_residual = apply, predict
        return self

    def observe(self, x, step, block):
        if any(s['step'] == step for s in self.request_scopes):
            raise RuntimeError('Duplicate full-population position')
        scope = next(s for s in self.records[self.index]['scopes'] if (s['step'], s['block']) == (step, block))
        if list(x.shape) != scope['native_V1_shape'] or list(x.stride()) != scope['native_V1_stride']:
            raise RuntimeError('Original full-M post-Q1 geometry differs')
        if digest(scope['path']) != scope['sha256']:
            raise RuntimeError('Original sampled post-Q1 payload changed')
        saved = load_file(scope['path'], device='cpu')['V1']
        ids = torch.tensor(scope['selected_rows'], device=x.device)
        selected = x.index_select(0, ids).cpu().contiguous()
        if not same_bits(selected, saved):
            raise RuntimeError('Actual population Q1 does not reproduce original256 sampled rows bitwise')
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        values = x.double()
        gram = values.T @ values
        total = values.sum(dim=0)
        del values
        role = 'train' if self.index in self.train_ids else 'validation'
        key = f'{role}_step{step}'
        if key not in self.sums:
            self.sums[key] = torch.zeros_like(gram)
            self.counts[key] = 0
        self.sums[key].add_(gram)
        self.counts[key] += len(x)
        self.means[f'input{self.index:02d}_step{step}_sum'] = total.cpu().contiguous()
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
            raise RuntimeError('Population observer mutated native RNG')
        metadata = {'index': self.index, 'step': step, 'block': block, 'role': role,
                    'population_rows': len(x), 'original_sampled_rows': len(ids),
                    'original_sampled_Q1_bitwise': True, 'original_payload_sha256': scope['sha256'],
                    'full_V1_sha256': tensor_sha(x), 'population_Gram_sha256': tensor_sha(gram),
                    'population_sum_sha256': tensor_sha(total), 'native_RNG_unchanged': True}
        self.scopes.append(metadata); self.request_scopes.append(metadata)

    def __exit__(self, *error):
        self.runtime.apply, self.runtime.predict_residual = self.old_apply, self.old_predict


def collect_one(pipe, runtime, observer, reference, index, folder):
    if folder.exists():
        raise RuntimeError('Preserve every attempted population control')
    folder.mkdir(parents=True)
    sample = reference['input']
    if sample['seed'] != 42 + index or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
        raise RuntimeError('Original OP2 source/seed changed')
    runtime.begin_request(); observer.begin(index)
    scheduler = pipe.runner.scheduler; scheduler.generator = None
    old_post = scheduler.step_post
    trajectory = []
    before = dict(runtime.counters)

    def post():
        result = old_post()
        value = scheduler.latents.detach().float().cpu().contiguous()
        row = {'step': int(scheduler.step_index), 'sha256': tensor_sha(value), 'shape': list(value.shape)}
        if row != reference['trajectories']['plain'][row['step']]:
            raise RuntimeError('Population collection changed original native UB1 trajectory')
        trajectory.append(row)
        return result

    scheduler.step_post = post
    report = {'status': 'RUNNING', 'index': index, 'trajectory': trajectory, 'quality_or_speed_acceptance': False}
    atomic_json(folder / 'INPUT.json', report)
    try:
        with torch.inference_mode():
            pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(folder / 'native_control.mp4'))
        torch.cuda.synchronize()
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if (delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']
                or [s['step'] for s in observer.request_scopes] != [2, 3] or len(trajectory) != 4):
            raise RuntimeError('Incomplete population/native linear coverage')
        verify_runtime(runtime, 'UB1')
        video_sha = digest(folder / 'native_control.mp4')
        if video_sha != reference['videos']['plain']['sha256']:
            raise RuntimeError('Population observer changed original native video bytes')
        report.update(status='PASS_NATIVE_POPULATION_AND_ORIGINAL_VIDEO_BITWISE', scopes=observer.request_scopes,
                      counter_delta=delta, video_sha256=video_sha)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error), scopes=observer.request_scopes)
        raise
    finally:
        scheduler.step_post = old_post
        atomic_json(folder / 'INPUT.json', report)
    print(f'R8D1 input{index:02d} full-Q1 moments and original4-step/video bits PASS', flush=True)
    return {'index': index, 'path': str(folder / 'INPUT.json'), 'sha256': digest(folder / 'INPUT.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    state = load_records(ROOT)
    if validate_records(*state) or not method_eligibility(state[0], state[1], METHOD)[0]:
        raise RuntimeError('Population readiness diagnostic is not registered')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'REGISTERED_UNCHANGED_NATIVE_POPULATION_COLLECTION':
        raise RuntimeError('Unexpected population collection phase')
    args.out = args.out.resolve(); folder = args.out / f'shard{args.shard}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve previous collection attempt')
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    history = HistoricalSources(); alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    runs = ROOT / 'runs/worker_restart_20261020'
    records, bindings, capture = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    if capture['assets'] != assets:
        raise RuntimeError('Population/original OP2 assets differ')
    train = protocol['inputs']['training_indices'][args.shard::2]
    validation = protocol['inputs']['validation_indices'][args.shard::2]
    indices = sorted(train + validation)
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': policy_sha,
                'OP2_collections': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings]}
    raw = Path(cache['path']).parents[1] / 'r8d1_Q1_population_20261021' / args.out.name / f'shard{args.shard}'
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30:
        raise RuntimeError('Preserve80GiB node reserve plus2GiB total collection allowance')
    report = {'status': 'RUNNING', 'scientific_identity': identity, 'shard': args.shard, 'indices': indices,
              'training_indices': train, 'validation_indices': validation, 'completed': [],
              'new_fits_or_encodings': 0, 'quality_or_speed_acceptance': False}
    atomic_json(folder / 'RUN.json', report)
    try:
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Incomplete strict native placeholders')
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        from step_budget_runtime import create_runtime
        runtime = create_runtime(cache['path'], alias); report['residency'] = make_resident(runtime)
        observer = PopulationObserver(runtime, records, protocol['inputs']['training_indices'])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            with observer:
                for index in indices:
                    if shutil.disk_usage(raw).free < 82 << 30:
                        raise RuntimeError('Preserve live80GiB reserve and2GiB pending statistics allowance')
                    if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                        raise RuntimeError('Frozen population observer source changed')
                    report['completed'].append(collect_one(pipe, runtime, observer, records[index], index, raw / f'input{index:02d}'))
                    if audit['fallback_calls']:
                        raise RuntimeError('Forbidden native raw placeholder fallback')
                    atomic_json(folder / 'RUN.json', report)
        finally:
            adapter.deactivate_lightx2v_hooks()
        path = raw / 'Q1_population_moments.safetensors'
        tensors = {key + '_Gram': value.cpu().contiguous() for key, value in observer.sums.items()}
        tensors.update(observer.means)
        save_file(tensors, str(path))
        if path.stat().st_size > 1 << 30:
            raise RuntimeError('Each shard must remain below1GiB saved statistics')
        report.update(status='PASS_UNCHANGED_NATIVE_POPULATION_SHARD', counts=observer.counts,
            scopes=observer.scopes, moments={'HF_derived_relative': str(path.relative_to(Path(cache['path']).parents[1])),
                'sha256': digest(path), 'bytes': path.stat().st_size,
                'tensor_sha256': {k: tensor_sha(v) for k, v in tensors.items()}},
            scope='Exact observed full-Q1 first/second moments only; no encoding, candidate fit or source/row-count sweep')
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)


if __name__ == '__main__':
    main()
