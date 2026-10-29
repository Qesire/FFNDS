# Chipmunk high-expert Y-only reuse: native results

Both source-specific producers, CPU preflights and independent artifact verifiers exited 0. Each source (898 and 96, seed 42) generated NONE, HIGH_Y_CACHE_DENSE and HIGH_Y_REUSE as full four-step, 160-site videos. The latter reuses the same block’s step-0 pre-gate Y at high step 1; the current gate and residual remain on the original path. These are two exposed development sources, not independent held-out confirmation. [Machine-readable results](CHIPMUNK_HIGH_Y_REUSE_NATIVE_RESULTS_20261030.json).

Per source, 24 state byte controls passed: ten each for NONE and cache-dense against old Dense, and four reuse prefix controls. Both NONE and cache-dense videos matched old Dense byte-for-byte. Independent VERIFY read 35 artifacts: 30 typed state files (file and raw tensor hashes), three new videos and two journals. The two reference-video comparisons were producer gates, not verifier artifacts.

Per source, all three phases covered 480 FFN sites and 960 logical linear dispatches. Actual dense GEMMs were 320, 320 and 240; reuse replaced 40 high-step1 FFNs. Original prefetch remained enabled, with 162 recorded observations per phase. These observations do not measure physical bus bytes. The logical 40-block Y cache was 12,971,212,800 bytes (12.08 GiB); a temporary single-block reuse clone adds 324,280,320 bytes before allocator and other live-tensor costs.

| Source | Phase | FFN CUDA event seconds | Phase wall seconds including observers | Cumulative peak reserved bytes |
|---|---|---:|---:|---:|
| 898 | none | 6.307 | 46.701 | 21,908,946,944 |
| 898 | high_y_cache_dense | 6.413 | 46.262 | 33,613,152,256 |
| 898 | high_y_reuse | 4.922 | 44.837 | 33,613,152,256 |
| 96 | none | 6.609 | 48.177 | 21,908,946,944 |
| 96 | high_y_cache_dense | 6.745 | 47.848 | 33,613,152,256 |
| 96 | high_y_reuse | 5.131 | 45.526 | 33,613,152,256 |

The cumulative peak allocated and reserved bytes were 22,848,135,680 and 33,613,152,256 for each source; peaks were not reset between phases. Each caching phase performed a synchronized expert-boundary allocator cleanup; pre/post observations and elapsed time are preserved in the JSON. GPU producer wall seconds were 237.624 and 242.476; independent verifier wall seconds were 1.103 and 1.146. CPU and supervisor identities and durations are also recorded there.

Prior same-recipe HIGH_ONLY P64 FFN event times were 6.052 s (898) and 6.323 s (96), with cumulative peak reserved 71,051,509,760 bytes in that separate scope. Different scope, phase order, observer and setup costs prevent a fair system-speed comparison. These timings include the recorded observer boundaries and do not establish video quality or human acceptance. No tensor or video was reread to create this report.
