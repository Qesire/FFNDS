"""Static matched-shape accounting. These are not measured competitor latencies."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
N, d, m, k, p = 31668, 5120, 13824, 2560, 64
alpha_lo, alpha_hi = 3892 / N, 3968 / N
beta = p / N
dense_attention = 4 * N * N * d
dense_ffn = 4 * N * d * m
other = (8 * N * d * d + 4 * N * d * d
         + 4 * 512 * d * d + 4 * N * 512 * d)
total = dense_attention + dense_ffn + other
ffn_major = (1 + (2*N*d*d + 4*N*d*k) / dense_ffn) / 2
ffn_listed = 0.6890326310091037
attention_extra_macs = sum([
    629800960, 38670428160, 10376970240, 5242880, 20971520
])
cases = []
for alpha in [alpha_lo, alpha_hi]:
    ours = (2 + 3*alpha + 2*beta) / 4
    chip_adapted = (2 + 4*alpha) / 4
    mc_short = (3 + 4*alpha) / 4
    full_ratio = (other + dense_ffn*ffn_listed + dense_attention*ours
                  + 2*attention_extra_macs/4) / total
    attention_only_ratio = (other + dense_ffn + dense_attention*ours
                            + 2*attention_extra_macs/4) / total
    cases.append({
        "selected_alpha": alpha,
        "ours_current_fourcall_attention_core_ratio": ours,
        "chip_mechanism_two_anchor_two_hit_core_ratio": chip_adapted,
        "mc_two_pass_two_anchor_two_hit_core_ratio": mc_short,
        "ours_listed_whole_block_ratio": full_ratio,
        "conditional_attention_only_whole_block_ratio": attention_only_ratio,
        "ffn_marginal_dense_whole_block_fraction": attention_only_ratio-full_ratio,
    })
out = {
    "scope": "CPU arithmetic; actual Wan shape, hypothetical competitors with matched lifecycle; no new GPU/model runs",
    "shape": dict(N=N, D=d, FFN_width=m, FFN_selected=k, probes=p),
    "probe_query_fraction": beta,
    "dense_principal_FLOPs_per_layer_step": total,
    "FFN_major_cycle_ratio": ffn_major,
    "FFN_listed_cycle_ratio_from_round13": ffn_listed,
    "cases": cases,
    "conditions": [
        "Chip entry assumes both anchors can immediately construct support and each computes a selected response; default upstream two-dense initialization is not silently removed. Extra work or router adaptation must be charged.",
        "MC entry includes one dense QK/AV, one extra full QK selection scan, and selected attention at each anchor; at each hit only selected attention. It excludes grouping and top-k overhead.",
        "Same selected density is not a matched quality condition; methods can use different grouping/selection.",
        "Ours high uses current absolute P64 reconstruction; low uses centered history plus P64. This asymmetry explains one fewer selected anchor computation.",
        "Attention-only arithmetic is a proposed scope simplification, not an executed model or an assumption that FFN errors are independent.",
        "Useful GEMMs only: not softmax, GELU, solves, padding, launch, precision casts, memory, or wall time.",
        "Whole-block listed ledger matches round13 before the tiny observed rho fallback QK costs."
    ]
}
assert abs(total/1e12 - 39.85196220416) < 1e-10
assert abs(cases[0]["ours_listed_whole_block_ratio"] - 0.7210025490655672) < 1e-12
assert abs(cases[1]["ours_listed_whole_block_ratio"] - 0.721930181092583) < 1e-12
(ROOT/"LIFECYCLE_TRADEOFFS_VERIFIED.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
