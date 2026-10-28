# Six-video official quality observations · 2026-10-28

Eight original metric dimensions completed in the same environment on six already-generated videos. These are two previously exposed source clusters, one seed per source and arm; human full-play remains pending and quality is not accepted. The Dense values are freshly scored in this v4 run.

| Dimension | Source 0 Dense / energy / random | Source 12 Dense / energy / random | Equal-source Dense / energy / random |
|---|---|---|---|
| imaging_quality | 75.05 / 37.459 / 26.513 | 76.986 / 56.428 / 26.814 | 76.018 / 46.944 / 26.664 |
| aesthetic_quality | 0.587 / 0.35997 / 0.32624 | 0.69186 / 0.40864 / 0.29001 | 0.63943 / 0.38431 / 0.30813 |
| motion_smoothness | 0.98469 / 0.99017 / 0.98994 | 0.99224 / 0.9837 / 0.98325 | 0.98846 / 0.98693 / 0.9866 |
| dynamic_degree | false / false / false | false / false / false | 0 / 0 / 0 |
| subject_consistency | 0.97567 / 0.94762 / 0.92242 | 0.98611 / 0.93576 / 0.93537 | 0.98089 / 0.94169 / 0.9289 |
| background_consistency | 0.9765 / 0.95872 / 0.95648 | 0.98882 / 0.96125 / 0.96576 | 0.98266 / 0.95999 / 0.96112 |

The source-specific I2V axes are separate single-source observations: source 0 subject (Dense / energy / random) = 0.98769 / 0.79113 / 0.72182; source 12 background = 0.99249 / 0.84359 / 0.77242.

| Source / candidate | Dense mean | Candidate mean | Candidate / Dense | Five fixed 8-pair window ratios | Official binary Dense / candidate |
|---|---:|---:|---:|---|---|
| 0 / energy94 | 4.5654 | 0.6166 | 0.1351 | 0.112, 0.119, 0.104, 0.136, 0.188 | {'candidate': False, 'Dense': False} |
| 0 / random94 | 4.5654 | 0.6436 | 0.1410 | 0.173, 0.157, 0.158, 0.121, 0.110 | {'candidate': False, 'Dense': False} |
| 12 / energy94 | 2.4779 | 0.3585 | 0.1447 | 0.637, 0.181, 0.103, 0.117, 0.109 | {'candidate': False, 'Dense': False} |
| 12 / random94 | 2.4779 | 0.5769 | 0.2328 | 1.176, 0.294, 0.156, 0.189, 0.159 | {'candidate': False, 'Dense': False} |

RAFT values are 40 two-frame-pair scores, each the original top-5% flow statistic; fixed windows group eight pairs. The official dynamic-degree binary is a separate thresholded output. Historical Dense scores are sanity references only; new Dense supplies every comparison denominator. The JSON retains each window’s Dense and candidate values and historical differences.

Actual exit 0, timeout false: RUN SHA256 `b6ffb121cc6abc59be490d47d3fe9f2bb48bc7d0f4350e262f5b5a371b9bc23e`, exit SHA256 `cd56d8b28c8a646db976a864b4f485b9adfff9d77c93316cbad1c80053592f63`, frozen protocol SHA256 `23f56ab7ae27230e2eb5f4cb813c9eef0bc1ec021bb1d753345b128152e86b50`. Independent byte and imports CPU proofs both actual exit 0. Eight dimension calls, 42 new video-dimension results, 240 RAFT forwards and 4,800 iterations closed exactly; no video generation or training. Worker wall 267.469 s, outer 297.190 s, peak GPU reserved 1480589312 bytes. Previous v1/v2/v3 failures retain their own identities.

[Machine-readable results](COMPACT6016_CONSUMER_QUALITY_RESULTS_20261028.json).
