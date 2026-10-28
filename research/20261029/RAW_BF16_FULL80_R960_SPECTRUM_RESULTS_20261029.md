# Raw BF16 full80 rank-960 spectrum: actual local diagnostics

**OBS:** The fixed run completed all 80 physical functions, with all 1,280 Dense postwrite byte-neutral checks passing before fits. The durable worker exited 0. RUN SHA-256 `83363b4278ad1e47cfef612e3229e46d309f00f36db0c34df5312fd4809d65af`; exit SHA-256 `c0909ab76731197abe850536a90f00f343b8bc1009fc1483d722fe2d5fb2c3b2`. The [machine result](RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.json) retains per-source metrics and the complete source-equal summary. These are M256 teacher-state operator diagnostics; the independent units are six calibration sources and two previously exposed development sources. The six held-out folds cover eight fixed physical-function anchors only.

| β=1, source-equal mean | Shared80 gated-Y relative SSE | Step160 gated-Y relative SSE | Shared80 normalized-u SSE | Step160 normalized-u SSE |
|---|---:|---:|---:|---:|
| Six calibration sources, train, all 80 | 0.017006 | 0.006659 | 0.004553 | 0.001845 |
| Six source-held folds, eight anchors | 0.044750 | 0.050344 | 0.012858 | 0.014393 |
| Two exposed development sources, all 80 | 0.058298 | 0.066526 | 0.015753 | 0.017850 |

The shared fit has the lower held and development normalized-u proxy for each reported source, despite the step-specific fit's lower training error. This is a finite-source contrast, not evidence that shared factors will yield better generated videos. With shared factors, the variance-fit gain lowers development mean squared standard-deviation ratio error from 0.00002847 to 0.00001734, while normalized-u SSE rises from 0.015753 to 0.015805. The gated least-squares gain changes these aggregate figures only slightly. The per-source and per-family values are in the machine result.

**Candidate construction:** Use the 80 already-exported shared rank-960 BF16 factor pairs at both steps of each physical function, retaining original full W1, GELU, original b2, gate and residual, with β=1. This gives a single all-160-position candidate without another fit. The mechanistic competitor uses the already-exported 160 step-specific factor pairs under the same retained W1 and β=1. The shared and step exports contain 2,910,617,600 and 5,821,235,200 tensor bytes respectively. Per token, the rank-960 W2 factors account for 0.25694 of original W2 MACs; retaining W1 gives 0.62847 of original W1+W2 MACs. Neither system speed nor video quality has been measured for these candidates.

**Rounding:** Captured full-M Y and the M256 original BF16 addmm replay were byte-identical in 0/1,280 cases; their mean relative SSE was 0.000009731. Captured Y versus the FP64 hW2+b2 reference, evaluated with the original gate, had mean relative SSE 0.000002765. The spectral triangle correction is a continuous relaxation, not a native BF16 rank floor. Local proxies and these rounding observations construct the video candidates; full-depth multisource free generation and system measurements remain the deciding evidence.
