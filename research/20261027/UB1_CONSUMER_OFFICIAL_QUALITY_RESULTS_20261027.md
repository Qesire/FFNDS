# UB1 fixed-RR consumer: official-function quality on six paired videos

**[VIDEO]** The new consumer restores some continuous motion relative to neutral UB1 on both development sources, but remains below the paired dense videos: mean RAFT pair score is 2.156 versus 1.294 neutral and 3.393 dense for the subject source, and 0.446 versus 0.183 neutral and 1.995 dense for the background source. Candidate/dense ratios are 0.635 and 0.223 respectively. The official binary dynamic degree is **false for all six videos**, so it does not distinguish those continuous changes. Motion smoothness is also a different measurement and should not be read as motion preservation.

The observational units are two image-prompt source clusters, each with seed 42 and three separately generated paired arms: dense, neutral original UB1, and the fixed-RR consumer candidate. Both sources were already development-exposed. These are custom paired subset calls to original VBench-I2V 0.1.5 functions, with one video per source and arm; they are not the official five-independent-samples-per-condition leaderboard protocol. The full per-video values and all 40 RAFT pair scores are in [the machine-readable results](UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.json).

## Per-source, per-arm results

The imaging entries are raw per-video MUSIQ values on the original 0–100 scale. Aesthetic, smoothness, consistency and I2V entries are the original per-video scores. An em dash means the I2V dimension was not applicable to that source.

| Source | Arm | Imaging raw | Aesthetic | Motion smoothness | Subject consistency | Background consistency | I2V subject | I2V background |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Subject 00 | Dense | 75.3573 | 0.61721 | 0.98099 | 0.98357 | 0.97626 | 0.98687 | — |
| Subject 00 | Neutral | 74.5790 | 0.60682 | 0.98268 | 0.99062 | 0.99080 | 0.98409 | — |
| Subject 00 | Candidate | 75.3364 | 0.59225 | 0.97747 | 0.98401 | 0.98610 | 0.98432 | — |
| Background 10 | Dense | 78.1907 | 0.68495 | 0.98093 | 0.98828 | 0.96446 | — | 0.99341 |
| Background 10 | Neutral | 76.4933 | 0.70900 | 0.97421 | 0.99167 | 0.98753 | — | 0.98871 |
| Background 10 | Candidate | 77.4343 | 0.69599 | 0.97191 | 0.98860 | 0.97551 | — | 0.99158 |

| Source | Arm | Mean of 40 RAFT pair scores | Maximum pair score | Official dynamic degree |
| --- | --- | ---: | ---: | --- |
| Subject 00 | Dense | 3.3932 | 7.2359 | false |
| Subject 00 | Neutral | 1.2937 | 3.4269 | false |
| Subject 00 | Candidate | 2.1562 | 4.0267 | false |
| Background 10 | Dense | 1.9950 | 3.0331 | false |
| Background 10 | Neutral | 0.1833 | 0.3769 | false |
| Background 10 | Candidate | 0.4459 | 1.0462 | false |

The original dynamic-degree rule used threshold 10.875 and count threshold 10 for each clip; the continuous score is its original pairwise magnitude summary, recorded before the binary decision. It is not by itself a prompt-following or general quality score. The measured candidate differs across other axes too: subject aesthetic declines versus both comparisons, while background aesthetic rises versus dense and falls versus neutral. This small, mixed pattern motivates paired visual inspection and independent-source confirmation rather than a single quality conclusion.

For the six generic axes, the JSON also reports an equal-weight mean of the two sources per arm. In particular, raw imaging means are 76.7740 dense, 75.5361 neutral and 76.3853 candidate; divide these by 100 for the official imaging aggregate scale. The official function's overall imaging aggregate across all six clips is 0.762318. I2V subject and background each have only one applicable source, so no cross-source average is reported for them.

## Execution and boundaries

Eight official dimensions completed: imaging, aesthetic, motion smoothness, dynamic degree, subject consistency, background consistency, I2V subject and I2V background. **Camera motion was not run**, so this is not a complete nine-dimension I2V result. The additional temporal-flickering dimension was not run because static-video eligibility was not established. Neither population-level acceptance nor human full-play acceptance has been established.

The one scoring attempt exited 0 in 253.617 seconds. The assigned A100 monitor recorded 75 samples, no violations and a sampled peak of 1,946 MiB. RAFT actually recorded 240 forwards and 4,800 iterations. Other per-dimension forward figures in the RUN are **predebited work limits**, not measured counts; opaque internal module calls remain unknown. Each video was checked as 81 decoded frames. These facts are execution evidence for the scored files, not a latency or deployment-speed result.

The original metric scoring functions, frame transforms and aggregation were retained. Checkpoints were resolved offline; the DreamSim compatibility wrapper supplied the same five LoRA configuration fields as the original code after verifying that unsupported extension fields were inactive defaults. Original checkpoint and adapter bytes remained unchanged. The frozen protocol SHA-256 is `6dce8f177eab984f501bcae55de8982de3da2ebbb02a1072146ce202a906df7e`; raw HF RUN SHA-256 is `b19d868babe3033e8e3593d264b8c79d1cd92b7eefdd166e3eda56534be29d85`. The JSON records exact source, video, launch and V8 receipt identities. This review independently checked the recorded outputs and bindings; the reviewer also authored the scoring driver, so this is a source/result audit rather than an independent external peer review.
