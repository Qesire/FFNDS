# Chipmunk global-P64 V2: official quality and motion

Two previously exposed development image sources, one seed and one actual Dense/P64 video each. Eight original metric axes were run on all four videos; the two I2V axes are development diagnostics. Human review, five independent official samples, camera motion and temporal flicker remain NOT_MEASURED.

| Axis | 898 Dense | 898 P64 | 96 Dense | 96 P64 | Source-equal paired difference |
|---|---:|---:|---:|---:|---:|
| imaging_quality | 77.431063 | 75.998265 | 72.306570 | 71.622563 | -1.058402 |
| aesthetic_quality | 0.496316 | 0.481741 | 0.694937 | 0.701272 | -0.004120 |
| motion_smoothness | 0.966414 | 0.966775 | 0.988301 | 0.988307 | +0.000184 |
| dynamic_degree | 1.000000 | 1.000000 | 0.000000 | 0.000000 | +0.000000 |
| subject_consistency | 0.900962 | 0.879198 | 0.977322 | 0.971320 | -0.013883 |
| background_consistency | 0.950995 | 0.941910 | 0.962933 | 0.968564 | -0.001727 |
| i2v_subject | 0.956294 | 0.950099 | 0.988172 | 0.978620 | -0.007873 |
| i2v_background | 0.957146 | 0.948899 | 0.991216 | 0.985907 | -0.006778 |

Imaging quality uses raw MUSIQ 0–100; the official normalized aggregate uses 0–1. The binary `dynamic_degree` values are 1/1 for source 898 and 0/0 for source 96, and should not be mistaken for continuous motion magnitude.

| Source | Dense RAFT mean | P64 RAFT mean | P64 / Dense |
|---:|---:|---:|---:|
| 898 | 22.7256 | 21.5953 | 0.9503 |
| 96 | 8.5387 | 7.0504 | 0.8257 |

The mean of the two source-specific ratios is 0.8880. This full-frame top-5% flow statistic uses 40 two-frame intervals and does not isolate the subject; lower motion is not automatically worse quality. The binary axis remains unchanged for both sources while continuous RAFT changes.

An independent AI agent inspected all 8 contact sheets (324 thumbnail frames): source 898 P64 has softer or trailing contours in some wing/beak frames; source 96 preserves the overall scene with small detail differences. See the [actual visual observation](CHIPMUNK_P64_V2_AI_VISUAL_REVIEW_20261030.md). Static sheets cannot establish playback smoothness or exclude subtle flicker.

All six quality, RAFT and visual scopes and their six independently registered verifiers actually exited zero. Per-scope RUN/EXIT and verifier bindings, four metric CPU preflight receipts, operation counts, complete source scores and five RAFT windows are in the [whitelist JSON](CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.json). No new scoring or video decoding was done to prepare this summary.
