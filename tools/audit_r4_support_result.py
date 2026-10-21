"""Independent CPU checks of completed R4 finite-screen checkpoints and algebra."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from anchored_reconstruction import from_bf16_bits
from ar2_checkpoint import atomic_json, digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import ROOT, tensor_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or torch.cuda.is_initialized():
        raise RuntimeError('Preserve the first CPU-only audit')
    torch.set_num_threads(8)
    runs = ROOT / 'runs/worker_restart_20261020'
    raw = runs / 'r4e1_compensated_support/RUN.json'
    store, result, identity = open_verified(raw, 'COMPLETED_R4E1_FINITE_NATIVE_SUPPORT_GATE_REJECTED')
    for path, sha in identity['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('Frozen native producer changed after execution')
    baseline = json.loads((runs / 'nf2_reused_hidden_residual/RUN.json').read_text())
    checks, teacher_files, models, identities = [], 0, 0, 0
    for control in result['controls']:
        step, block = control['step'], control['block']
        for index in range(24):
            part = store.load('teacher_V2', f'step{step}_block{block}_input{index}')
            values = from_bf16_bits(part['arrays']['V2_bits'])
            if (values.shape != (256, 13824) or not torch.isfinite(values).all()
                    or tensor_sha(values) != part['metadata']['V2_sha256']
                    or not part['metadata']['native_full_TeacherY_bitwise']):
                raise RuntimeError('Saved full teacher sample failed byte/finite checks')
            teacher_files += 1
    for row in result['CV']:
        step, block, fold = [row[k] for k in ['step', 'block', 'fold']]
        part = store.load('CV', f'step{step}_block{block}_fold{fold}')
        if part['metadata'] != row:
            raise RuntimeError('Run summary and payload-bound fold metadata differ')
        g, linear = part['arrays']['proxy_gram'], part['arrays']['linear_scores']
        pool = row['proposal_pool']
        removed, added = sorted(pool[:8]), sorted(pool[8:])
        if not np.allclose(g, g.T, rtol=1e-13, atol=1e-9):
            raise RuntimeError('Projected Gram is not symmetric')
        eig = np.linalg.eigvalsh(g)
        if eig.min() < -1e-8 * max(1., abs(eig).max()):
            raise RuntimeError('Projected Gram is not numerically PSD')
        lookup = {v: i for i, v in enumerate(pool)}
        enumerated = []
        for count in [0, 1, 2]:
            for out in itertools.combinations(removed, count):
                for into in itertools.combinations(added, count):
                    quadratic = sum(g[lookup[i], lookup[j]] for i in out for j in out)
                    quadratic += sum(g[lookup[i], lookup[j]] for i in into for j in into)
                    quadratic -= 2 * sum(g[lookup[i], lookup[j]] for i in out for j in into)
                    gain = sum(linear[i] for i in into) - sum(linear[i] for i in out) - quadratic
                    enumerated.append((float(gain), list(out), list(into)))
        enumerated.sort(key=lambda v: (-v[0], v[1], v[2]))
        for observed, selected in [(row['proxy_best_including_identity'], enumerated[0]),
                (row['candidate'], next(v for v in enumerated if v[1]))]:
            if (observed['removed'] != selected[1] or observed['added'] != selected[2]
                    or not math.isclose(observed['proxy_advantage'], selected[0], rel_tol=1e-10, abs_tol=1e-8)):
                raise RuntimeError('Independent pairwise quadratic search chose a different action')
        for label in ['K0', 'K1', 'C', 'random']:
            restored = {k: from_bf16_bits(part['arrays'][label + '_' + k + '_bits']) for k in ['second', 'bias_second']}
            if {k: tensor_sha(v) for k, v in restored.items()} != row[label + '_readout_sha256']:
                raise RuntimeError('Saved native readout bits changed')
            models += 1
        historical = next(v for v in baseline['CV'] if (v['step'], v['fold']) == (step, fold))
        if row['native_risks']['K0_frozen'] != historical['frozen_UB1_native_SSE']:
            raise RuntimeError('Frozen baseline native SSE differs from NF2 actual-state reference')
        for label, certificate in row['numeric_discrepancy'].items():
            for suffix, proof in certificate.items():
                exact = row['native_risks']['K0_' + suffix] - row['native_risks'][label + '_' + suffix]
                if (not math.isclose(exact, proof['native_advantage'], rel_tol=1e-10, abs_tol=1e-8)
                        or proof['absolute_advantage_discrepancy'] > proof['bound'] + 1e-8
                        or not proof['inequality_verified']):
                    raise RuntimeError('Native SSE advantage identity or discrepancy bound failed')
                identities += 1
        checks.append({'step': step, 'fold': fold, 'independently_enumerated_candidates': len(enumerated),
            'same_best_and_best_nonzero_actions': True, 'Gram_min_eigenvalue': float(eig.min()),
            'frozen_native_SSE_exactly_equals_historical_NF2': True})
    if (teacher_files != 48 or models != 16 or identities != 24 or torch.cuda.is_initialized()):
        raise RuntimeError('Incomplete CPU checkpoint/algebra audit')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, {'status': 'PASS_R4E1_COMPLETE_NATIVE_CHECKPOINT_AND_INDEPENDENT_QUADRATIC_AUDIT',
        'native_run_sha256': digest(raw), 'source_sha256': digest(Path(__file__)), 'checks': checks,
        'verified_native_teacher_V2_files': teacher_files, 'verified_readout_models': models,
        'native_advantage_and_discrepancy_identities': identities, 'CUDA_initialized': False,
        'quality_or_speed_acceptance': False})
    print('PASS_R4E1_COMPLETE_NATIVE_CHECKPOINT_AND_INDEPENDENT_QUADRATIC_AUDIT', flush=True)


if __name__ == '__main__':
    main()
