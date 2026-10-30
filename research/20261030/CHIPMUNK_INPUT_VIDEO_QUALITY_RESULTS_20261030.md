# Input video: actual quality results

Four exposed sources, one seed each; three paired videos per source, no extra independent N from reused video or neutral controls.

| Source | Arm | Continuous RAFT mean |
|---|---|---:|
| 898 | none | 22.725567 |
| 898 | high_h_readonly_p64 | 23.077697 |
| 898 | high_x_recompute_p64 | 23.077697 |
| 96 | none | 8.538704 |
| 96 | high_h_readonly_p64 | 8.325765 |
| 96 | high_x_recompute_p64 | 8.325765 |
| 1040 | none | 4.627403 |
| 1040 | high_h_readonly_p64 | 5.237096 |
| 1040 | high_x_recompute_p64 | 5.237096 |
| 979 | none | 5.480488 |
| 979 | high_h_readonly_p64 | 4.316336 |
| 979 | high_x_recompute_p64 | 4.316336 |

40 stride-2 pairs per video, 20 iterations each; five eight-pair windows. Continuous top-5% flow is distinct from official binary dynamic degree.

Eight per-source axes, paired differences, five-window observations, source-equal summaries and all twelve actual closure bindings are in the JSON. Independent AI visual review remains separate.

| Source | Candidate/Dense flow | MUSIQ change | Aesthetic change | Subject consistency change |
|---|---:|---:|---:|---:|
| 898 | 1.015495 | -0.007998 | +0.004457 | -0.003250 |
| 96 | 0.975062 | +1.912449 | +0.004900 | +0.000753 |
| 1040 | 1.131757 | -1.118349 | -0.013988 | -0.003874 |
| 979 | 0.787582 | -6.382042 | -0.045461 | +0.003234 |

The source-equal mean of within-source flow ratios is 0.977474; source979 remains a counterexample. Mean MUSIQ change is -1.398985 points.

The two candidates have identical complete output bytes in these four runs; H values equal X except for tiny separately repeated metric variation. This is candidate equivalence, not Dense equivalence or four additional samples.

[Independent AI contact-sheet review](CHIPMUNK_INPUT_VIDEO_AI_VISUAL_REVIEW_20261030.md) covers24 pages and972 thumbnails; continuous human playback and heldout confirmation remain unmeasured.

All eight scoring/RAFT scopes and four visual scopes, plus their12 separate verifiers, actually exited0. Earlier two monitor failures and two taskset failures are retained in the JSON. Quality acceptance is false.
