# B-H residual-anchored direct65 H128 predictability · 2026-10-28

**A trained 65-statistic student reduces terminal head error on all eight exposed heldout sources for each expert.** The completed GPU M256 pilot predicts the native-captured terminal head target on saved teacher states. Two experts each received 256 Adam updates; 192 initial/final source-step evaluations and two export reload checks completed. The worker exited 0 without timeout. This is local predictability, not a full-model rollout or speed result.

| Expert / source split | Equal-source head relative SSE, initial → final | Equal-source q relative SSE, initial → final | Equal-source variance relative SSE, initial → final |
| --- | ---: | ---: | ---: |
| High / train16 | 0.113140 → 0.004661 | 0.002886 → 0.001137 | 0.192447 → 0.015343 |
| High / heldout8 exposed | 0.093596 → 0.003958 | 0.002739 → 0.001208 | 0.171890 → 0.012401 |
| Low / train16 | 0.065606 → 0.003668 | 0.003626 → 0.001471 | 0.138472 → 0.008785 |
| Low / heldout8 exposed | 0.063256 → 0.003615 | 0.003316 → 0.001265 | 0.134563 → 0.009928 |

Each source mean averages its two expert steps; source means are equally weighted. All 16 training and 8 diagnostic sources were historically exposed, and sampled rows/steps are nested observations. The head error is measured against the native-captured target; it is not a complete-video quality metric.

Teacher-statistic reconstruction is a reference for this parameterization, not an optimal native error lower bound. On heldout8, mean head relative SSE for FP32/BF16 raw65 reconstruction was 4.645e−6/5.211e−6 for high and 3.407e−6/3.989e−6 for low. Four step-specific head geometries had rank 64 and support condition numbers 4.44, 4.50, 4.10 and 4.10. The declared slack floor was 1.526e−11; recorded anchor and teacher floor/negative-repair counts were zero across the 96 source-step cases, as were predicted qmin negative repairs in initial/final evaluations.

Each expert has 672,386 student parameters, 5.66% fewer than A0's 712,704. The 1,016,064 logical MAC/row includes the paid residual projection; the predictor introduces no new Q2. The comparison shares source split and update budget, but **a same-head A0 comparison was not performed**. These are two terminal functions out of 80, and their arithmetic count is not a measured system latency.

[Machine-readable scalar audit](BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.json) records source metrics, teacher references, geometry, export identities and exact budget closure. RESULT and exit receipt SHAs were checked; metric/reconstruction JSON sidecars were freshly hashed in this audit and were not originally bound by RUN. Export file sizes were checked without rereading their tensor payloads.
