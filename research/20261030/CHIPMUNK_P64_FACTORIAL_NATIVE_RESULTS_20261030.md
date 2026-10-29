# Chipmunk global-P64: high/low expert factorial native runs

Both source-specific producers and their independent artifact verifiers exited 0. For each exposed development source (898 and 96, seed 42), the new NONE, HIGH_ONLY and LOW_ONLY arms generated a complete four-step video. The previously verified Dense and BOTH videos complete the 2×2 action comparison; they are reused observations, not new independent samples. [Machine-readable results and identities](CHIPMUNK_P64_FACTORIAL_NATIVE_RESULTS_20261030.json).

The 26 state byte controls per source passed: NONE reproduced all ten saved states of old Dense; HIGH_ONLY matched old BOTH and LOW_ONLY matched old Dense from initial noise through flow and latent step 2. The new NONE video also matched old Dense byte-for-byte. These hard gates establish the relevant shared prefixes. They do not by themselves measure the downstream causal effect or establish video quality.

Each source executed 480 FFN sites across its three phases: 400 dense FFNs and 80 sparse FFNs. Each single-expert arm performed 40 cache puts, 40 sparse actions and 40 releases; the two arms incurred two expert cleanups and two same-mask pre-gate controls. The recorded relative L2 and maximum absolute difference in each control were zero. Original prefetch remained active: 162 observations per phase. Each independent verifier checked 35 artifacts, including typed headers, whole-file and raw-tensor hashes for the 30 state files. No tensor or video was reread to prepare this report.

| Source | Phase | FFN boundary CUDA seconds | Phase wall seconds including observers | Cumulative peak reserved bytes |
|---|---|---:|---:|---:|
| 898 | NONE | 6.294 | 47.014 | 21,908,946,944 |
| 898 | HIGH_ONLY | 6.052 | 45.485 | 71,051,509,760 |
| 898 | LOW_ONLY | 6.059 | 45.789 | 71,212,990,464 |
| 96 | NONE | 6.587 | 48.086 | 21,908,946,944 |
| 96 | HIGH_ONLY | 6.323 | 47.463 | 71,051,509,760 |
| 96 | LOW_ONLY | 6.309 | 47.214 | 71,051,509,760 |

Per-source total producer walls were 238.103 and 243.634 seconds, with peak host RSS 117,025,091,584 and 117,022,322,688 bytes. Each active expert's logical cache peak was 47,993,487,360 bytes. Phase order, setup, synchronization, byte checks and observers affect these measurements; they are not a fair systems-speed comparison.

This scope generated the six new videos and established their native identities. Quality and causal effect sizes remain to be measured with the frozen same-source factorial comparison. These two exposed images and one seed do not provide a population estimate or human acceptance.
