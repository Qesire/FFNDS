# High-expert Y reuse: three-arm quality results

Two exposed sources (898, 96), one seed each, were evaluated as NONE, HIGH_P64 and HIGH_Y_REUSE. All six metric/visual scopes and their independent verifiers exited 0. HIGH_P64 reuses an existing video and NONE is byte-equal to the previous Dense video; repeat scoring does not add independent samples. This is a three-arm comparison, not a factorial interaction or human acceptance. [Detailed data and receipt bindings](CHIPMUNK_HIGH_Y_REUSE_QUALITY_RESULTS_20261030.json).

| Source | Arm | Imaging | Aesthetic | Motion smoothness | Dynamic degree | Subject | Background | I2V subject | I2V background |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 898 | none | 77.431063 | 0.496316 | 0.966414 | 1.000000 | 0.900962 | 0.950995 | 0.956294 | 0.957146 |
| 898 | high_p64 | 77.423065 | 0.500773 | 0.964882 | 1.000000 | 0.897712 | 0.947287 | 0.958782 | 0.954835 |
| 898 | high_y_reuse | 77.245527 | 0.501722 | 0.976548 | 1.000000 | 0.883829 | 0.945251 | 0.972660 | 0.960896 |
| 96 | none | 72.306570 | 0.694937 | 0.988301 | 0.000000 | 0.977322 | 0.962933 | 0.988172 | 0.991216 |
| 96 | high_p64 | 74.219019 | 0.699836 | 0.987793 | 0.000000 | 0.978075 | 0.966324 | 0.988987 | 0.991082 |
| 96 | high_y_reuse | 73.115352 | 0.698129 | 0.987748 | 0.000000 | 0.978638 | 0.954163 | 0.988456 | 0.990772 |

| Axis | Source-equal NONE | HIGH_P64 | HIGH_Y_REUSE | Y − Dense | Y − HIGH_P64 |
|---|---:|---:|---:|---:|---:|
| imaging_quality | 74.868816 | 75.821042 | 75.180440 | 0.311624 | -0.640602 |
| aesthetic_quality | 0.595626 | 0.600305 | 0.599926 | 0.004299 | -0.000379 |
| motion_smoothness | 0.977357 | 0.976338 | 0.982148 | 0.004791 | 0.005810 |
| dynamic_degree | 0.500000 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| subject_consistency | 0.939142 | 0.937893 | 0.931234 | -0.007909 | -0.006660 |
| background_consistency | 0.956964 | 0.956805 | 0.949707 | -0.007257 | -0.007098 |
| i2v_subject | 0.972233 | 0.973884 | 0.980558 | 0.008325 | 0.006674 |
| i2v_background | 0.974181 | 0.972958 | 0.975834 | 0.001653 | 0.002875 |

Continuous RAFT below is mean top-5% flow in pixels per two-frame pair, based on 40 stride-2 pairs; five fixed eight-pair windows and all pair values are in the JSON. The official dynamic-degree binary is a separate metric.

| Source | NONE | HIGH_P64 | HIGH_Y_REUSE | Y − Dense | Y − HIGH_P64 | Y / Dense | Y / HIGH_P64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 898 | 22.725567 | 23.077697 | 19.333177 | -3.392390 | -3.744519 | 0.850724 | 0.837743 |
| 96 | 8.538704 | 8.325765 | 7.854691 | -0.684013 | -0.471074 | 0.919893 | 0.943420 |

The JSON contains source-equal averages and paired differences, along with the recorded repeat evaluation of byte-identical Dense. Instrumented SHA read totals and actual CPU, GPU and VERIFY identities are included; other metadata and model-internal I/O remain unknown.

The independent [AI contact-sheet review](CHIPMUNK_HIGH_Y_REUSE_AI_VISUAL_REVIEW_20261030.md) examined all 486 thumbnails. For the bird, Y shows translucent overlapping contours around frames 46–63; its subject-consistency score falls by 0.017133 versus Dense, while I2V subject similarity increases. RAFT motion is lower for both sources. Contact sheets are not continuous playback or human review, and these observations do not establish causal mechanism or quality acceptance.
