"""Static Wan-shape MAC accounting and a synthetic ICC error-identity witness.

No GPU inference or quality estimate. Sources and limits: NEIGHBOR_MECHANISMS.md.
"""
from pathlib import Path
import json
import numpy as np

N, D, M, K = 31668, 5120, 13824, 2560
dense = 2 * N * D * M
ours = N * D * D + 2 * N * D * K
rows = []
for rank in (128, 192, 256, 512, 1024, 1408):
    hit = 2 * N * (D + M) * rank
    cycle = (dense + hit) / (2 * dense)
    rows.append(dict(rank=rank, hit_macs=hit, hit_ratio=hit / dense,
                     anchor_hit_ratio=cycle, anchor_hit_arithmetic_speedup=1 / cycle))

rng = np.random.default_rng(1414)
xa, xt = rng.normal(size=(2, 11, 8))
w1, w2 = rng.normal(size=(8, 13)), rng.normal(size=(13, 8))
b1, b2 = rng.normal(size=13), rng.normal(size=8)
def lowrank(w, r):
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r]
def gelu(z):
    return .5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z**3)))
wr1, wr2 = lowrank(w1, 3), lowrank(w2, 3)
za = xa @ w1 + b1
ha = gelu(za)
ya = ha @ w2 + b2
ht = gelu(xt @ w1 + b1)
yt = ht @ w2 + b2
zh = za + (xt - xa) @ wr1
hh = gelu(zh)
yh = ya + (hh - ha) @ wr2
rhs = (ht - hh) @ w2 + (hh - ha) @ (w2 - wr2)
identity_error = float(np.max(np.abs((yt - yh) - rhs)))
assert identity_error < 1e-12

alpha = 3968 / N
beta = 64 / N
output = dict(
    status="Static arithmetic at actual Wan dimensions; synthetic FP64 identity witness; no model/GPU experiment",
    shape=dict(N=N, D=D, M=M, selected_K=K),
    excluded="Nonlinearities, softmax, norms, elementwise/casts, solver, routing/grouping, layout/padding, cache traffic/offload, launches, preparation, and all quality effects",
    icc_ffn=rows,
    ours_major_hit_ratio=ours / dense,
    ours_major_anchor_hit_ratio=(dense + ours) / (2 * dense),
    icc_rank_equal_ours_major_hit=ours / (2 * N * (D + M)),
    icc_ffn_four_field_scalars_per_layer=2 * N * (D + M),
    icc_ffn_four_field_BF16_GiB_per_layer=4 * N * (D + M) / 2**30,
    icc_ffn_four_field_BF16_GiB_40_layers=40 * 4 * N * (D + M) / 2**30,
    cache_scope="Direct four-field author-layout extrapolation; not necessary memory lower bound or measured peak",
    mc_core_cycle_model="alpha + 1.5/(H+1): one full QK/AV and one extra full QK per anchor, plus selected core at every anchor and hit",
    selected_alpha=alpha,
    mc_one_hit_per_anchor_core_ratio=alpha + 1.5 / 2,
    mc_fifteen_hits_per_anchor_core_ratio=alpha + 1.5 / 16,
    ours_current_fourcall_core_ratio=(2 + 3 * alpha + 2 * beta) / 4,
    mc_scope="Same density and two-pass selection semantics; ignores grouping, solves, precision and fidelity; not a latency or matched-quality comparison",
    icc_error_identity_max_abs=identity_error,
)
target = Path(__file__).with_name("STRUCTURAL_COST_COMPARISON.json")
target.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
