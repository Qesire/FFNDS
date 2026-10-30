# P64 local and prefix decomposition

L = candidate − Dense shadow on student input; P = student-shadow − free-Dense; T = candidate − free-Dense. All three columns in the main table use the free-Dense energy denominator. The separate local/shadow column uses the same-state reference and must not be confused with L/free-Dense.

| Source | subset | gated L / P / T relative to free-Dense | local L / student-shadow | signed cross / free-Dense energy |
|---|---|---|---:|---:|
| 898 | selector64 | 0.357383 / 0.663002 / 0.675849 | 0.349835 | -0.110523 |
| 898 | eval_weighted_excluding_selector64 | 0.333512 / 0.870673 / 0.869798 | 0.350023 | -0.112754 |
| 898 | full_M_four_snapshots | 0.291017 / 0.493538 / 0.510327 | 0.27315 | -0.0678367 |
| 979 | selector64 | 0.354336 / 0.594583 / 0.596353 | 0.329774 | -0.123446 |
| 979 | eval_weighted_excluding_selector64 | 0.3773 / 0.846447 / 0.867476 | 0.321691 | -0.106312 |
| 979 | full_M_four_snapshots | 0.312905 / 0.495936 / 0.499782 | 0.294689 | -0.0940807 |

Each ratio pools energies before taking a square root; site mean/ranges, pooled84 cells and four full-M gated cell energies are in the compact JSON. Complete per-site cell ratios and numeric residuals remain in the SHA-bound HF RUN.

Two exposed sources and one seed each; rows, blocks and 84 cells are nested, not independent source samples.
Selector64 is a decision sample; weighted evaluation estimates the eligible finite population excluding selector64. Four full-M blocks are separately reported.
L=candidate−student-shadow, P=student-shadow−dense, T=candidate−dense; cross is signed energy interaction, not a causal percentage.
The output decomposition excludes write/residual dynamics and does not identify downstream H3, quality acceptance or free Dense execution.

The retained full-M observations reveal signed cancellation: all252 noninitial temporal/quadrant cells per source have negative cross. At block39, total output discrepancy is0.296151 versus prefix-only0.321224 for898, and0.272916 versus0.306992 for979. Sampled other blocks include positive cross. These are output-level relations, not evidence that replacing an FFN with Dense would worsen a complete video.

Adding prefix differences does not resolve the quality ranking: weighted eval total norms are0.869798/0.867476 and the four-full-M totals0.510327/0.499782, while979 previously had worse video quality. These two-source observations support studying directions and spatial/semantic effects rather than treating a global norm as a quality score. No further experiment was run for that hypothesis.

V1 failed during JSON serialization and has no complete durable site result. A separately reviewed V2 corrected only scalar serialization and completed; both original failures remain in the linked failure record. The coordinator crosschecked all eight full-M same-state ratios against the original native observer (maximum absolute difference below1e-6); this audit is not an additional independent-agent endorsement.
