# P64 history encoding: four full-M local cases

Four CPU preflights, four A100 probes and four independent typed verifiers exited 0. The cases are two exposed development sources (898 and 96), each with high and low expert at block 20; they are not four independent sources. [Results and receipt identities](CHIPMUNK_HISTORY_ENCODING_RESULTS_20261030.json).

Each case used the same actual P64 mask in all four arms, with the recorded SHA matching the prior mask. B omitted the old-h clone/update and was **bitwise equal** to A at the gated output. C stored old h in E4M3FN; D used per-column int8. Both kept a BF16 64-row shadow for the selector. No full decoded-h RMS or innovation norm was measured, so these output errors do not identify an error-cancellation mechanism.

| Source | Expert | C relative L2 vs A | D relative L2 vs A | A relative L2 vs actual Dense | Dense vs saved teacher relative L2 / max abs |
|---|---|---:|---:|---:|---:|
| 898 | high | 0.009204 | 0.008320 | 0.426488 | 0.000317 / 0.500 |
| 898 | low | 0.008493 | 0.006705 | 0.261200 | 0.000317 / 0.250 |
| 96 | high | 0.008870 | 0.007949 | 0.374251 | 0.000289 / 0.250 |
| 96 | low | 0.008972 | 0.007019 | 0.256122 | 0.000336 / 0.250 |

**The actual Dense replay was not byte-neutral to stored teacher Y.** Its gated relative L2 was 0.000289–0.000336 and maximum absolute difference was 0.25 or 0.5. Both teacher and same-device Dense comparisons are retained; the observed difference is consistent with a numerical-path difference, without identifying its exact cause.

| Source | Expert | A ms | B ms | C ms | D ms | Dense ms | FP8 encode ms | int8 encode ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 898 | high | 25.194 | 22.566 | 24.013 | 25.058 | 42.592 | 10.177 | 28.031 |
| 898 | low | 25.250 | 22.986 | 24.178 | 24.958 | 43.069 | 9.932 | 27.515 |
| 96 | high | 25.193 | 22.747 | 24.104 | 24.983 | 42.520 | 10.325 | 27.826 |
| 96 | low | 25.236 | 22.805 | 24.099 | 24.957 | 42.695 | 10.354 | 27.558 |

Each number in the action columns is the mean of three CUDA-event measurements after one warmup; a fifth action supplied the output comparison. Encoding is a separately paid one-time operation, not amortized over hypothetical reuse. The JSON also records all individual measurements, shadow construction, four mask hashes, gates and saturation observer. Each action includes the P64 proxy and selection, gathers, W1/GELU/W2, merge, gate and finite observer. Journal persistence is outside the measured intervals.

The following arithmetic adds the CUDA-event history construction to one mean action: A/B use the shared BF16 shadow; C/D use their own encoding interval, which already includes shadow construction. These sums are not measured end-to-end invocations. Each arm’s mask SHA and equality gate, plus the separately measured FP8 saturation observer, are excluded and remain individually recorded in the JSON; full-model creation of old h and Y was not repeated in this scope. No additional reuse is assumed.

| Source | Expert | A construction + action ms | B ms | C ms | D ms |
|---|---|---:|---:|---:|---:|
| 898 | high | 25.710 | 23.083 | 34.190 | 53.089 |
| 898 | low | 25.826 | 23.563 | 34.110 | 52.473 |
| 96 | high | 25.544 | 23.099 | 34.429 | 52.809 |
| 96 | low | 25.705 | 23.274 | 34.453 | 52.515 |

Each encoded payload is 437,778,432 bytes, with a 1,769,472-byte BF16 shadow; int8 additionally stores 55,296 scale bytes. FP8 saturation count was zero in all cases. Cumulative peak reserved memory was 16,085,155,840 bytes (14.98 GiB) per case with BF16 history and both encoded caches co-resident: it is not a standalone compressed-arm peak. Physical disk traffic and uninstrumented metadata I/O are unknown.

The source-equal means are computed within each expert and included in the JSON. This is a local saved-state probe, not a full-video or quality result; no training, video generation or quality scoring was performed.
