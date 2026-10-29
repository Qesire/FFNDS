# SharQ Wan all-160 V2: native generation results

**NATIVE / VIDEO generation:** Both registered source runs and their independent output verifications exited 0. Two previously exposed development sources, 1040 and 979, each produced four complete 81-frame videos at seed 42: original BF16, BF16 with cache construction, Dense32 and SharQ. Each generation traversed all four steps and 40 FFN blocks. Quality has not yet been scored or accepted; the two development sources are not a confirmation population.

The two BF16 paths provide a same-device neutral control. For each source, all ten recorded state files and the video were byte-identical between the original and cache-construction phases: 20 state pairs and two video pairs in total. Each candidate follows its own free trajectory; its Q2 operates on its own GELU output. Dense32 and SharQ use the same original BF16 master weights. SharQ executes a 4:8 sparse path plus a full-K dense residual, and the original W1/W2 prefetch remains enabled and paid.

| Source | Mode, in execution order | FFN CUDA events (s) | Phase wall with observers (s) |
|---:|---|---:|---:|
| 1040 | BF16 original | 5.918254 | 45.725396 |
| 1040 | BF16 cache construction | 6.227356 | 40.075170 |
| 1040 | Dense32 | 4.161613 | 36.838721 |
| 1040 | SharQ | 5.778414 | 39.231958 |
| 979 | BF16 original | 5.965078 | 45.557633 |
| 979 | BF16 cache construction | 6.325398 | 39.957321 |
| 979 | Dense32 | 4.159742 | 37.450613 |
| 979 | SharQ | 5.702433 | 38.346506 |

The second BF16 phase builds the packed cache and is not a free baseline. Each part reports 160 static W materializations and quantizations, 22,649,241,600 materialization bytes and 7,102,136,960 cache bytes; later candidate phases reuse that cache. The materialization-byte ledger is a logical weight-view count; `.T.contiguous()` may alias the original buffer, so this number does not establish physical copies or bus traffic. Each phase retains 162 original FFN prefetch observations and no prefetch skips. Fixed phase order, first model loading, caching, observer and output costs confound these timings, so they do not establish a fair end-to-end system speedup. Whole parent RUN wall was 293.856 / 293.479 s for sources 1040 / 979; launcher outer wall was 308.760 / 304.081 s respectively. Setup and phase timings are preserved separately in the JSON.

Per source, the observed ledger records four video generations, 16 full-model flows, 640 FFN calls and 1,280 FFN linear dispatches, 800 stream-bridge calls, 1,600 bridge event records and 1,600 waits. The ledger's predebits and observed totals agree. Both runs recorded zero GPU monitor violations. Each verifier independently rehashed 46 output artifacts: 250,830,813 bytes for source 1040 and 248,185,065 bytes for source 979. The verification reads were already completed; no large states or videos were reread for this report.

The [machine-readable whitelist](SHARQ_WAN_ALL160_VIDEO_V2_NATIVE_RESULTS_20261030.json) contains per-phase video and state identities, timing, operation counts, parent and verification RUN/EXIT hashes and HF-relative paths. It excludes device identifiers and host-specific paths. Recorded output identities and successful execution do not replace visual, motion or official quality evaluation.
