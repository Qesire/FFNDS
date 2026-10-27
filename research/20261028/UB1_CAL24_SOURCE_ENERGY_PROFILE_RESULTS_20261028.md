# CAL24 whole-group source energy: four teacher positions

**NATIVE / OBS.** The original contribution computation reproduced the stored C_mean vector bit for bit at all four tested positions. Leaving out one of the 24 calibration image clusters yields mean whole64 support Jaccard between 0.980 and 0.998; the lower endpoint of the fixed 200-resample source-cluster bootstrap interval ranges from 0.862 to 0.940. This is evidence that the selected supports at these teacher positions are relatively stable to source removal under the measured statistic.

| Step / block | Retained groups | Leave-one-source Jaccard mean (min) | Bootstrap Jaccard interval |
| --- | ---: | ---: | ---: |
| 0 / 0 | 54 | 0.9985 (0.9636) | [0.8621, 1.0000] |
| 0 / 39 | 54 | 0.9804 (0.9286) | [0.8621, 0.9636] |
| 3 / 0 | 162 | 0.9908 (0.9636) | [0.9401, 0.9756] |
| 3 / 39 | 162 | 0.9847 (0.9636) | [0.9286, 0.9877] |

The calculation uses the legacy exposed CAL24 cohort, 256 original saved rows per source, the original gated W2 group-contribution energy, and teacher input states. It executed 96 source calls and 2,592 observed FP32 batched BMMs. It did not fit or train a model. The launcher exited 0, the recorded computation took 30.03 seconds, and the GPU monitor recorded no violations.

The audit verified the frozen protocol and 20 executed source bindings, all four original reference files, and the identities, original INPUT reports, payload hashes and saved row coordinates for all 96 records. The [machine-readable report](UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.json) binds the raw RUN, launcher, PROFILE and DIAGNOSTIC files and preserves the per-position results. The reproduction comparison has zero maximum absolute difference and bitwise equality in all four cases.

The separate [20-source consumer video evaluation](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md) observed a substantial reduction in continuous RAFT image-flow magnitude relative to fresh Dense. Source-deletion instability of these four measured teacher supports is therefore not the strongest available explanation at these positions. Other positions, sampled-row coverage, student-state changes and consumer behavior remain open alternatives; this measurement alone cannot identify the cause of video motion loss.

These results concern four of 160 positions and the existing 24-source sample. They do not establish that 24×256 is sufficient for fitting or generalization to new sources, and they do not measure signed group cross terms, student states or video quality. Source-cluster bootstrap intervals describe this fixed cohort and diagnostic, not an independent confirmation population.
