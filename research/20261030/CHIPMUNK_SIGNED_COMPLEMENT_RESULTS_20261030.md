# Chipmunk P64 signed complement on true full-M inputs

Two exposed sources and two nested expert pairs per source; each online arm uses the same frozen P64 K5120 mask. Errors below are gate-only relative L2; baseline/signed are 3-repeat means and hindsight is one paid full-current-mean diagnostic.

| Source | Steps | Baseline | Signed P64 | Full-mean hindsight | Baseline DC / total energy | Same-mask reference relative L2 |
|---:|---|---:|---:|---:|---:|---:|
| 898 | 0→1 | 0.426487 | 0.388711 | 0.386715 | 0.177789 | 0 |
| 898 | 2→3 | 0.261199 | 0.260286 | 0.259050 | 0.016434 | 0 |
| 96 | 0→1 | 0.374251 | 0.354046 | 0.352429 | 0.113232 | 0 |
| 96 | 2→3 | 0.256121 | 0.256625 | 0.255117 | 0.007807 | 0 |

| Source | Steps | Dense ms | Baseline online ms | Signed online ms |
|---:|---|---:|---:|---:|
| 898 | 0→1 | 36.324 | 23.661 | 24.772 |
| 898 | 2→3 | 36.181 | 23.807 | 25.108 |
| 96 | 0→1 | 36.124 | 23.780 | 24.712 |
| 96 | 2→3 | 37.171 | 23.750 | 24.910 |

Source-balanced errors (two experts equally averaged within each source): baseline 0.329515, signed 0.314917, hindsight 0.313327.

P64 RMS and signed mean share one paid 64-row current W1/GELU; S selected entries zeroed, remainder cast BF16, full W2 GEMV FP32 output; no current full h before online arms. Full omitted-mean hindsight is separately paid only after current Dense.

Under real arithmetic and fixed S the full omitted mean maps to a constant omitted-output mean; the P64-selected S is sample-dependent, so its sampled signed mean is not claimed unbiased. BF16 rounding makes full-mean hindsight different from a globally optimal gated output constant.

Each baseline/signed/Dense arm has its own cold and three measured calls. Online CUDA and host timing includes selection, sample probe, identity SHA D2H/sync, gather, cache, GEMMs, and gate; persistence is excluded. Old-cache, same-mask frozen-packed direct check, and full-mean hindsight are separate. Guard host intervals are also listed. Local events are not system speed.

The baseline DC/AC values and finite closure, each actual repeat and cold/host interval, direct reference, exact accounting, and source identities are preserved in [whitelist JSON](CHIPMUNK_SIGNED_COMPLEMENT_RESULTS_20261030.json). These are local teacher-state diagnostics, not free-running video or quality evidence.
