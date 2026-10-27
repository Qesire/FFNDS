# Compact6016 recovery: four-position pilot

**NATIVE / OBS.** With the same 128-update rank-32 adapter budget, both energy94 and random94 substantially reduced local held-out-source gated error. Energy94 ended lower at each of the four tested positions. This is a useful recovery result for a 6016-wide native-Q2 student, measured on eight historically exposed diagnostic sources at teacher states.

| Step / block | Initialization | Initial held-out BF16-gated relative SSE | Final held-out BF16-gated relative SSE | Final held-out centered relative SSE | Final held-out mean-error fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 / 0 | energy94 | 24.000% | 2.349% | 4.478% | 3.393% |
| 0 / 0 | random94 | 33.346% | 3.212% | 6.232% | 1.640% |
| 0 / 39 | energy94 | 12.813% | 1.061% | 10.651% | 22.257% |
| 0 / 39 | random94 | 32.975% | 1.463% | 13.426% | 29.026% |
| 3 / 0 | energy94 | 20.813% | 2.546% | 10.662% | 13.396% |
| 3 / 0 | random94 | 30.121% | 3.649% | 14.813% | 15.667% |
| 3 / 39 | energy94 | 18.697% | 2.045% | 13.740% | 16.823% |
| 3 / 39 | random94 | 33.031% | 2.348% | 15.486% | 18.460% |

Entries are equal-source means of per-source ratios; centered error uses the saved-row teacher-state decomposition and its own centered teacher-energy denominator. The [results JSON](COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.json) preserves all 384 initial/final source diagnostics, including each source's values, mean, linearly interpolated p90 and maximum for train and held-out subsets. The four positions are nested within the same eight held-out sources, not 32 independent held-out sources.

The two arms selected 94 complete 64-channel groups, used identical rank, training sources, steps, optimizer and random seed, and began from original selected native weights and fixed biases/scales. Energy selection used only the 16 training sources. Each student's own narrowed Smooth2/HiF4 Q2 path ran, and final diagnostics used serialized and byte-verified reloaded BF16 folded packs. Neither arm used the historical fitted Whole94 readout.

The run completed 1,024 Adam updates and 1,424 native Q2 calls. Predebits and observed counts match the frozen limits, and the native adapter and backend each report 1,424 activation calls. The launcher exited 0 after 135.62 seconds; the run recorded 133.93 seconds, 0.615 GiB peak Torch-reserved GPU memory, 1,168 MiB peak physical GPU memory and no monitor violations. The audit checked 41 executed source snapshots, eight exported files and every exported raw tensor SHA, all zero-adapter/native-STE byte gates, and source roles and diagnostic coverage.

The study covers four fixed step/block sites and an exposed 16/8 source split. It has not evaluated all 80 expert/block functions, both steps per expert, student rollout, video quality or end-to-end speed. Its local teacher-state comparisons should not be mixed with the different state, training history or cost of the earlier consumer bank. A full-depth matched continuation can test whether the local recovery extends across positions, followed by same-noise free-video evaluation.
