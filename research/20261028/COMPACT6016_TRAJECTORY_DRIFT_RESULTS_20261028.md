# Saved six-video trajectory drift · 2026-10-28

The two exposed sources each compare two candidate free trajectories with their matched Dense trajectory at four saved native BF16 post states widened to FP32. Values below are per source and candidate; relative SSE uses the Dense state energy at that step, so its denominator changes. Absolute error norms separately describe accumulated drift.

| Source / candidate | Relative SSE, steps 0–3 | Absolute error norm, steps 0–3 | Norm ratio to previous, steps 1–3 | Increment vs previous error cosine, steps 1–3 |
|---|---|---|---|---|
| 0 / energy94 | 0.0004809, 0.004918, 0.0464, 0.4865 | 29.19, 82.93, 202.3, 615.2 | 2.841, 2.44, 3.041 | 0.7592, 0.8065, 0.8887 |
| 0 / random94 | 0.0004734, 0.004752, 0.04366, 0.4396 | 28.96, 81.52, 196.3, 584.8 | 2.815, 2.407, 2.98 | 0.7629, 0.7624, 0.8894 |
| 12 / energy94 | 0.000688, 0.005484, 0.0474, 0.2729 | 34.85, 87.71, 213.8, 627.6 | 2.517, 2.437, 2.935 | 0.8742, 0.8542, 0.9238 |
| 12 / random94 | 0.0007326, 0.005891, 0.05464, 0.3071 | 35.96, 90.91, 229.5, 665.8 | 2.528, 2.525, 2.9 | 0.8779, 0.8485, 0.9414 |

Each step records all 21 latent-time-slot relative SSE values in the source result. The accompanying JSON gives their minimum, median, maximum, equal-slot mean, maximum/mean tail ratio and extremal slot for every source/candidate/step. The signed identity `||e_t||² = ||e_(t−1)||² + ||Δe_t||² + 2⟨e_(t−1), Δe_t⟩` closes for all 16 pairs; its components and residuals are retained there.

Actual CPU exit 0; 24 state files and their headers, file SHA and raw tensor SHA verified, 16 pairs evaluated. Worker wall 6.769 s and peak RSS 356642816 bytes. Source RESULT SHA256 `bf6634fb8f75cf3f1b51aeb814a48632d8b83ed73b1d89e6fe1cd13705341134`; exit receipt SHA256 `9bf5de1084717b58028e8d260c1dba04bd3b22047f93dfb255809bb6fc85c1b0`; frozen protocol SHA256 `ac435411ce3150d4b18a5f210926c88a6fc7e261767406201eca6bfc6d2c9ab5`. Parent six-video RUN and actual exit bindings are preserved in the JSON.

These are accumulated differences between free trajectories and path-dependent increments. They do not establish same-input local FFN error, flow causation or video quality; two sources are not 16 independent observations.

[Detailed scalar summary](COMPACT6016_TRAJECTORY_DRIFT_RESULTS_20261028.json).
