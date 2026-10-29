# Chipmunk P4 causal proxy: four local pairs

**NATIVE local diagnostic:** The paid four-row RMS proxy improved gate-weighted relative L2 over the mean proxy in each of four measured source/expert pairs, using the same 5,120 selected channels and real selected W1/GELU and W2/cache-add kernels. These are two previously exposed development images with two nested expert pairs each, at one block and 128 matching nonraster rows—not four independent images or a full-video result.

| Source | Steps | Mean proxy | Four-row RMS proxy | Fixed random | Sample RMS / full-128 RMS relative L2 |
|---:|---|---:|---:|---:|---:|
| 898 | 0→1 | 0.279414 | 0.266417 | 0.321712 | 0.618615 |
| 898 | 2→3 | 0.200512 | 0.181778 | 0.200791 | 0.634657 |
| 96 | 0→1 | 0.263246 | 0.229233 | 0.274488 | 0.636402 |
| 96 | 2→3 | 0.204721 | 0.181994 | 0.204256 | 0.587914 |

Equal averaging of the two expert pairs within each source gives mean/RMS/random values of 0.239963/0.224098/0.261252 for source 898, and 0.233983/0.205613/0.239372 for source 96. Equal averaging of those two sources gives **0.236973/0.214855/0.250312** respectively. These descriptive means carry no population or independent-pair confidence claim.

All masks and selected actions were completed before the full current reference and channel observer. The four-row proxy pays a full-width M4 W1, GELU and RMS of changes against the corresponding previous-step cache; it does not observe full current `h` when choosing channels. The old M128 W1/GELU/W2 cache and its construction cost are paid separately. The 128-row current dense W1/GELU/W2 is the reference, and comparisons apply the actual current gate in BF16. The A100 selected-delta implementation is a controlled port, not the original author's Hopper kernel.

The sample RMS estimate differs substantially from the 128-row reference (relative L2 0.588–0.636; cosine 0.799–0.842). The separate M4-versus-M128 score-shape comparison is much smaller, 0.00207–0.00265. These observations motivate a larger fixed sample and an explicitly charged hindsight reference; they do not, alone, identify sample variance as the exclusive cause of residual output error. Each pair's actual captured Δh energy and M128-versus-saved-h/Y comparisons are in the [whitelist JSON](CHIPMUNK_P4_CAUSAL_PROXY_V2_RESULTS_20261030.json). M128 is not byte-identical full-M replay.

V2 CPU preflight and GPU job actually exited 0, with no monitor violations. The ledger records four previous and four current dense W1/W2 references, four M1 mean W1s, four M4 sample W1s, and 12 each selected W1/GELU and selected W2/cache-add actions, plus explicitly charged observers and I/O. The 77.567 s RUN wall is cold and includes JIT, synchronization, observer and persisted-ledger overhead; it is not a steady-state kernel or system-speed benchmark. V1's entry failure remains separately documented in [its failure report](CHIPMUNK_P4_ENTRY_FAILURE_20261030.md). Exact RUN/EXIT and source identities are in the JSON. No checkpoint or tensor was reread for this report.
