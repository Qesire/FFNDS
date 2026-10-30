# Student observation: actual closed results

Two exposed sources at seed 42; 40 sites and 84 groups per site are nested observations.

| Source | Pooled pre rel L2 | Pooled gated rel L2 | Pooled write rel L2 | Gated site mean (range) | Ideal total / diagonal / cross | Actual vs ideal SSE |
|---|---:|---:|---:|---:|---:|---:|
| 898 | 0.473566 | 0.360532 | 0.1696 | 0.366766 (0.0884853–0.510858) | 1.72013e+08 / 9.39851e+07 / 7.80277e+07 | 9979.18 |
| 979 | 0.485746 | 0.342973 | 0.16405 | 0.34933 (0.103575–0.500261) | 1.53163e+08 / 8.58172e+07 / 6.73455e+07 | 9674.44 |

Source 898 selected block gated relative L2: 0: 0.167038, 13: 0.374428, 26: 0.506321, 39: 0.0940643.
Across 84 pooled temporal/quadrant cells, gated relative L2 ranges 0.186398 (τ 0, quadrant 0) to 0.433057 (τ 17, quadrant 0).

Source 979 selected block gated relative L2: 0: 0.132516, 13: 0.349687, 26: 0.458053, 39: 0.12613.
Across 84 pooled temporal/quadrant cells, gated relative L2 ranges 0.15986 (τ 0, quadrant 0) to 0.40746 (τ 18, quadrant 3).

Pooled energies sum over sites; the site mean treats sites as nested descriptions, not independent samples. Write denominator includes residual; do not interpret as motion.

Two exposed sources, one seed each; rows, sites and temporal groups are nested observations, not independent sources.
Write denominator contains residual and is not FFN motion or output energy alone.
Ideal omission diagonal/cross and actual BF16 error are distinct; negative cross indicates cancellation, not causal attribution.
84 additive groups do not measure object coverage or tail quantiles and cannot identify downstream H3 causality.
Byte-identical parent videos imply no new quality measurement or quality recovery claim.
