"""Bounded access to accepted native operands for the 2026-10-26 studies."""
import json
from pathlib import Path
import shutil

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
MATRIX_FIELDS = {'q1_post', 'z', 'gelu', 'q2_post', 'ffn_output'}


def prepare(args, expected_id, counters):
    if not args.execute_gpu:
        raise ValueError('Explicit registered saved-GPU execution required')
    plan = json.loads(args.protocol.read_text())
    if plan['id'] != expected_id:
        raise ValueError('Wrong study protocol')
    for name, digest in plan['bound_sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    from worker_identity import require_worker
    require_worker()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < plan['disk_reserve_gib'] * (1 << 30):
        raise RuntimeError('HF output disk reserve reached')
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / 'EXECUTED_SOURCES'
    snapshot.mkdir()
    for name in plan['bound_sources']:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(args.protocol, snapshot / args.protocol.name)
    record = {'status': 'STARTED_NOT_ACCEPTED', 'protocol_sha256': sha256(args.protocol),
              'cases': [], 'counts': dict(counters), 'new_generations': 0,
              'native_FFN_calls': 0, 'native_QDQ_calls': 0, 'optimizer_updates': 0,
              'model_weight_decompositions': 0, 'persisted_fitted_weights': 0,
              'automatic_retry': False, 'quality_or_speed_acceptance': False}
    write_json(args.out / 'RUN.json', record)
    return plan, record


def check_output_budget(directory, plan):
    total = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if total > plan['max_HF_output_gib'] * (1 << 30):
        raise RuntimeError('Frozen HF output budget exceeded')
    return total


class ObservationStore:
    def __init__(self, raw_root, torch):
        self.root, self.torch = raw_root, torch
        accepted = json.loads((ROOT / 'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json').read_text())
        self.runs = {}
        for source in [0, 12]:
            path = raw_root / f'native_source{source:02d}_attempt1/RUN.json'
            binding = next(v for v in accepted['raw_bindings'] if v['source_index'] == source)
            if sha256(path) != binding['sha256']:
                raise ValueError('Accepted native run changed')
            self.runs[source] = (path, json.loads(path.read_text()))

    def load(self, source, step, block, fields):
        path, run = self.runs[source]
        scope = next(s for s in run['scopes'] if (s['step'], s['block']) == (step, block))
        payload = path.parent / 'observed' / scope['path']
        if sha256(payload) != scope['sha256']:
            raise ValueError('Native operand changed')
        row_plan = scope['row_plan']
        center = row_plan['contiguous_tiles'][0]
        uniform = row_plan['uniform_rows']
        ordered = center + uniform
        ids = [row_plan['rows'].index(row) for row in ordered]
        arrays = load_arrays(payload)
        torch = self.torch
        result = {name: torch.as_tensor(arrays[name][ids] if name in MATRIX_FIELDS else arrays[name])
                  .to(device='cuda', dtype=torch.float64) for name in fields}
        result['panels'] = {'contiguous128': list(range(128)), 'uniform32': list(range(128, 160)),
                            'uniform_disjoint': [128 + j for j, row in enumerate(uniform) if row not in set(center)]}
        result['row_ids'] = ordered
        result['seed'] = run['seed']
        return result


def energy(x):
    return float(x.square().sum())


def describe(prediction, target, current, gate, panels):
    result = {}
    for name, ids in panels.items():
        actual = target[ids]
        guessed = prediction[ids]
        error = (guessed - actual) * gate
        actual_gated = actual * gate
        centered_error = ((guessed - guessed.mean(0, keepdim=True)) -
                          (actual - actual.mean(0, keepdim=True))) * gate
        centered_actual = (actual - actual.mean(0, keepdim=True)) * gate
        result[name] = {'rows': len(ids),
                        'relative_target_SSE': energy(error) / max(energy(actual_gated), 1e-30),
                        'relative_current_output_SSE': energy(error) / max(energy(current[ids] * gate), 1e-30),
                        'relative_centered_target_SSE': energy(centered_error) / max(energy(centered_actual), 1e-30)}
    return result
