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

## Reused complete quality evidence

The four observer videos match their historical input-video arms byte for byte. Their Dense-cache control was also neutral to original Dense. The following scores reuse those already-closed original-function single-seed evaluations; no metric or video was rerun. H/X are byte-equivalent videos, not independent samples. MUSIQ is0–100; other axes retain original units.

| Source | axis | Dense | H-P64 | candidate−Dense |
|---|---|---:|---:|---:|
| 898 | imaging_quality | 77.4310628 | 77.4230652 | -0.00799758346 |
| 898 | aesthetic_quality | 0.496315956 | 0.500773013 | +0.00445705652 |
| 898 | motion_smoothness | 0.966414106 | 0.964881786 | -0.00153232029 |
| 898 | dynamic_degree | 1 | 1 | +0 |
| 898 | subject_consistency | 0.900961955 | 0.897711999 | -0.00324995667 |
| 898 | background_consistency | 0.950994873 | 0.947286987 | -0.00370788574 |
| 898 | i2v_subject | 0.956293706 | 0.958781684 | +0.00248797789 |
| 898 | i2v_background | 0.957145845 | 0.954835128 | -0.00231071658 |

Source898 continuous RAFT candidate/Dense=1.01549485. Five fixed8-pair window means (Dense → candidate): 20.1321 → 18.727; 22.2131 → 24.4301; 22.497 → 22.8648; 23.4722 → 24.3157; 25.3134 → 25.051.

| 979 | imaging_quality | 58.4995842 | 52.1175422 | -6.38204203 |
| 979 | aesthetic_quality | 0.667487085 | 0.622025728 | -0.0454613566 |
| 979 | motion_smoothness | 0.991667867 | 0.993474928 | +0.00180706092 |
| 979 | dynamic_degree | 0 | 0 | +0 |
| 979 | subject_consistency | 0.948928324 | 0.95216244 | +0.00323411636 |
| 979 | background_consistency | 0.939831543 | 0.940240479 | +0.000408935547 |
| 979 | i2v_subject | 0.984047607 | 0.988552158 | +0.00450455017 |
| 979 | i2v_background | 0.981561841 | 0.993563023 | +0.012001182 |

Source979 continuous RAFT candidate/Dense=0.787582405. Five fixed8-pair window means (Dense → candidate): 3.33695 → 3.49323; 4.93236 → 4.35383; 4.92716 → 3.6044; 5.73274 → 4.20377; 8.47324 → 5.92645.

camera_motion, temporal_flickering, object/camera motion separation, quantified event alignment and human continuous acceptance: **NOT_MEASURED**. Prior AI81-frame contact sheets observed bird edge/wing and guitar-source late hand/face pose differences; they do not replace continuous human viewing. Original full-frame AI record is linked in the JSON.

The observer added paid full-M Dense shadow W1/GELU/W2 at40 sites per arm; its I/O-heavy runtime is not a deployment benchmark. The linked native/input and full-quality audit preserve actual FFN/system costs and separate the old-full-Y+selected-delta computation from UB1 static groups+R256, S2 low-rank W2, SharQ sparse FP4+dense residual and TEAL masked Dense W2.
