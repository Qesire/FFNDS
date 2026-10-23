"""Reanalyze pinned C1/C2 receipts and reproduce PRD mathematics without models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, mean

ROOT = Path(__file__).resolve().parents[1]
OUT = 'research/20261023/PRD_successor_reanalysis.json'
INPUTS = [
    'research/20261023/C1N0_findings.json',
    'research/20261023/C1N0_analysis.json',
    'research/20261023/C1N0_geometry.json',
    'research/20261023/C1N0_dense_value_robustness.json',
    'research/20261023/C2_scale_energy_CPU_results.json',
]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def equivalent(a, b):
    """Allow normal reduction differences across supported Python versions."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    return type(a) is type(b) and a == b


def norm(x, eps=1e-6):
    u = [v - mean(x) for v in x]
    scale = math.sqrt(mean([v * v for v in u]) + eps)
    return [v / scale for v in u]


def sse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def finite_norm_distance(a, b, eps=1e-6):
    """Exact real-arithmetic LayerNorm distance, without affine scale/bias."""
    u, v = [x - mean(a) for x in a], [x - mean(b) for x in b]
    va, vb = mean([x*x for x in u]), mean([x*x for x in v])
    cov = mean([x*y for x, y in zip(u, v)])
    return len(a) * (va/(va+eps) + vb/(vb+eps)
                     - 2*cov/math.sqrt((va+eps)*(vb+eps)))


def norm_jvp(x, d, eps=1e-6):
    u = [v - mean(x) for v in x]
    q = [v - mean(d) for v in d]
    var = mean([v*v for v in u])
    sigma = math.sqrt(var + eps)
    dot = sum(a*b for a, b in zip(u, q))
    return [(a - b*dot/(len(x)*sigma*sigma))/sigma for a, b in zip(q, u)]


def cluster_summary(rows):
    denom = sum(r['total_sampled_nextnorm_loss'] for r in rows)
    old = sum(r['captured']['original_energy_CPU_reference'] for r in rows)
    new = sum(r['captured']['scale_times_energy_CPU_reference'] for r in rows)
    oracle = sum(r['captured']['sample_oracle'] for r in rows)
    return {'scopes': len(rows), 'sampled_loss': denom,
            'original_capture': old/denom, 'scaled_capture': new/denom,
            'oracle_capture': oracle/denom,
            'capture_gain_percentage_points': 100*(new-old)/denom,
            'unselected_sampled_loss_reduction': (new-old)/(denom-old),
            'negative_scopes': sum(r['captured']['scale_times_energy_CPU_reference']
                                   < r['captured']['original_energy_CPU_reference'] for r in rows)}


def split_is_disjoint(manifest):
    """A repeated image or parent scene belongs to only one scientific split."""
    seen = {}
    for row in manifest:
        for field in ('image_sha256', 'scene_family'):
            key = (field, row[field])
            if key in seen and seen[key] != row['split']:
                return False
            seen[key] = row['split']
    return True


def coverage_schedule():
    """12 source slots x 4 steps x 2 seeds, balanced across four depth bands.

    Slots are a coverage design, not actual selected sources or a power result.
    Each source has all steps; each step/depth cell has three sources. Across
    the three source strata every cell uses one block from each of three
    offsets. All 96 contexts are emitted in balanced source-stratum order.
    """
    rows = []
    for seed in range(2):
        for step in range(4):
            for slot in range(4):
                for stratum in range(3):
                    depth = (slot + step) % 4
                    rows.append({'source_slot': 4*stratum + slot, 'stratum': stratum,
                                 'seed_slot': seed, 'step': step, 'depth_band': depth,
                                 'block': 10*depth + (1, 4, 8)[stratum]})
    return rows


def camera_blocks():
    """14 image slots, 3 commands/image; 7 commands, 6 images/command."""
    return [{'image_slot': 7*rep+i, 'commands': sorted({i, (i+1)%7, (i+3)%7})}
            for rep in range(2) for i in range(7)]


def approximate_sources(sd, effect, half_width, alpha=.05, power=.8, primary_count=1):
    """Normal approximation on independent source-level paired differences.

    A planning aid: final protocol uses pilot variance uncertainty, t or
    cluster simulations. No sample size is selected from unknown video data.
    """
    if not (sd > 0 and effect > 0 and half_width > 0 and 0 < alpha < 1
            and .5 < power < 1 and isinstance(primary_count, int) and primary_count >= 1):
        raise ValueError('Positive scales and valid error rates are required')
    z = NormalDist().inv_cdf(1-alpha/(2*primary_count))
    return max(2, math.ceil(((z+NormalDist().inv_cdf(power))*sd/effect)**2),
               math.ceil((z*sd/half_width)**2))


def mathematical_checks():
    rng = random.Random(20261023)
    error, jvp_error = 0., 0.
    for _ in range(40):
        x = [rng.uniform(-2, 2) for _ in range(8)]
        d = [rng.uniform(-.2, .2) for _ in range(8)]
        y = [a+b for a,b in zip(x,d)]
        error = max(error, abs(finite_norm_distance(x,y) - sse(norm(x),norm(y))))
        h = 1e-5
        finite = [(a-b)/(2*h) for a,b in zip(norm([a+h*b for a,b in zip(x,d)]),
                                              norm([a-h*b for a,b in zip(x,d)]))]
        jvp_error = max(jvp_error, max(abs(a-b) for a,b in zip(finite,norm_jvp(x,d))))
    x, radial, tangent = [1.,-1.,1.,-1.], [2.,-2.,2.,-2.], [1.1,-.9,.9,-1.1]
    residual_block = lambda w: [a+b for a,b in zip(w,norm(w))]
    counterexample = {'nextnorm_radial_loss': sse(norm(x),norm(radial)),
                      'nextnorm_tangent_loss': sse(norm(x),norm(tangent)),
                      'residual_block_radial_loss': sse(residual_block(x),residual_block(radial)),
                      'residual_block_tangent_loss': sse(residual_block(x),residual_block(tangent))}
    assert error < 1e-12 and jvp_error < 1e-8
    assert counterexample['nextnorm_radial_loss'] < counterexample['nextnorm_tangent_loss']
    assert counterexample['residual_block_radial_loss'] > counterexample['residual_block_tangent_loss']
    # Additive equal-cost toy: one context reverses the two sites' utilities.
    contexts = [([10,9],[1,.5]), ([1,.5],[10,9])]
    fixed = [mean([sum(a[:q])+sum(b[:2-q]) for a,b in contexts]) for q in range(3)]
    adaptive = mean([max(sum(a[:q])+sum(b[:2-q]) for q in range(3)) for a,b in contexts])
    return {'evidence': 'SYNTHETIC_MATH_ONLY', 'finite_norm_identity_max_error': error,
            'norm_jacobian_max_error': jvp_error, 'residual_bypass_rank_reversal': counterexample,
            'budget_example': {'best_fixed_utility': max(fixed), 'conditional_utility': adaptive,
                               'assumption': 'Equal costs, additive utilities, context known before decision'},
            'dominant_action_information_value': 0,
            'dominance_reason': 'If L(a0,w)<=L(a,w) for every w,a, every policy has >= risk of a0'}


def analyze():
    c1 = read(INPUTS[1]); c2 = read(INPUTS[4]); rows = c2['rows']
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 6
    assert len({(r['source'],r['step'],r['block']) for r in rows}) == 72
    assert c1['results']['oracle_whole_site']['whole_site_action_counts'] == {'UB1':72,'swap_A':0,'swap_B':0}
    groups = {k: {str(v): cluster_summary([r for r in rows if r[k] == v])
                  for v in sorted({r[k] for r in rows})} for k in ('source','step','block')}
    total = cluster_summary(rows)
    macro = mean([s['capture_gain_percentage_points'] for s in groups['source'].values()])
    leave_one = {str(s):cluster_summary([r for r in rows if r['source']!=s])['capture_gain_percentage_points']
                 for s in sorted({r['source'] for r in rows})}
    schedule = coverage_schedule(); cells = Counter((r['step'],r['depth_band']) for r in schedule)
    blocks = camera_blocks(); pairs = Counter()
    for b in blocks:
        for i,a in enumerate(b['commands']):
            for other in b['commands'][i+1:]: pairs[a,other] += 1
    return {'id':'PRD_SUCCESSOR_20261023', 'evidence_base_commit':'7c7019ec66835e740b3573d5572a8da59f0efa69',
            'input_sha256': {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS},
            'saved_native_reanalysis': {'C1_scope_oracle_gain':0, 'C2_pooled':total,
                'C2_grouped':groups, 'C2_equal_source_macro_gain_pp':macro,
                'C2_leave_one_source_out_pooled_gain_pp':leave_one,
                'C2_negative_late_step_scopes':sum(r['step']>=2 and r['captured']['scale_times_energy_CPU_reference']
                    < r['captured']['original_energy_CPU_reference'] for r in rows),
                'inference':'Descriptive reanalysis of six exposed source clusters; no confirmatory CI/p-value'},
            'mathematics':mathematical_checks(),
            'planned_coverage':{'contexts':len(schedule),'source_slots':12,
                'contexts_per_step_depth_cell':sorted(set(cells.values())),
                'camera_image_slots':len(blocks),'camera_command_pair_counts':sorted(set(pairs.values())),
                'not_a_power_calculation':True},
            'new_model_calls':0, 'new_GPU_jobs':0, 'official_quality_scores':0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check',action='store_true')
    a = p.parse_args(); result = analyze()
    if a.check:
        if not equivalent(read(OUT), result): raise SystemExit('Reanalysis differs from saved artifact')
    else:
        (ROOT/OUT).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','C2':result['saved_native_reanalysis']['C2_pooled'],
                      'new_model_calls':0},ensure_ascii=False))


if __name__ == '__main__': main()
