# Late dense intervention improves paired pixels unevenly; neutral VAE/media identity closes

Independent `whole_flow_native_review` agent, 2026-10-26. **New empirical object:** rendering the already saved candidate,candidate,candidate,dense endpoints yields complete videos with source-dependent changes. Source12's mean framewise RGB error falls13.58% and is lower at all81 frames; source0's mean falls only0.45% and worsens at30 frames. Both improve the paired temporal-delta proxy at all80 transitions, while source0's Farneback flow error worsens at60/80 transitions. The two dense controls reproduce the historical MP4 bytes exactly.

This closes the standalone VAE/encoding neutrality question for these exact inputs and connects the finite latent intervention to measured video proxies. The competing interpretation is that changing the last whole step can smooth or redistribute errors without restoring necessary motion/conditional information. Latent SSE, paired image errors, estimated motion and perceptual function answer distinct questions. This audit grants no human or official quality acceptance.

Execution boundary: independent CPU reading/decoding/recomputation of existing media only. No VAE/model/GPU calls, rerender, fitting or new video. Earlier source and whole-flow reviews remain unchanged.

## Actual receipts, gates and full-frame recomputation

I independently verified the completed STUDY→RUN→immutable DECODE_RUN/BUNDLE hashes. Both the unchanged core bundle gate and the custom decode inventory returned no errors. I reconstructed both source preparations from actual whole-flow/v3 receipts and confirmed the four media artifacts, source/seed, selected raw latent, decoder/model identities, source snapshots and reference bindings. The authoritative inventory has exactly the four expected MP4s, with no extra/missing clips.

The run records four VAE starts/completions, four encoder starts/completions and zero new denoiser NFE/FFN calls. Dense-control MP4 SHA equals the corresponding old dense reference. Every new report contains81 frames/80 transitions,16fps,464×832×3, official VBench/human NOT_RUN and `quality_accepted=false`.

I then independently used OpenCV/NumPy to decode **all four new clips and both exact prior v3 candidates** against their authoritative dense references. This covered486 candidate frames/480 transitions. I recomputed2418 framewise RGB/luma RMSE, Laplacian energy, paired temporal-delta RMSE and paired Farneback error values, with maximum discrepancy **0** against the reports, and separately verified the aggregate flow-magnitude ratios. Torch was never imported.

For both controls I also checked decoded RGB arrays are identical to their dense references at every frame. Defined paired spatial/temporal/flow errors are exactly zero and energy/flow ratios exactly one. Intrinsic motion/detail values are naturally nonzero for a moving video; neutrality concerns paired differences.

## Latent response versus actual pixel/motion/detail changes

All video quantities below are the same frozen evaluator applied to matching source/seed and the same dense reference. RGB and temporal errors are means of per-frame/per-transition RMSE, not a pooled global RMSE. Motion values are encoded-pixel Farneback proxies, and Laplacian ratios are not perceptual sharpness scores.

| Quantity | Source0 original candidate → late dense | Source12 original candidate → late dense |
| --- | --- | --- |
| Native final latent SSE | 454530.649 →435349.265 (**−4.22%**) | 137447.953 →115041.119 (**−16.30%**) |
| Mean framewise RGB RMSE | .154160 →.153464 (**−.45%**) | .130529 →.112803 (**−13.58%**) |
| Mean paired temporal-delta RGB RMSE | .056418 →.055109 (**−2.32%**) | .065115 →.054826 (**−15.80%**) |
| Mean paired Farneback flow L2 | .854121 →.860995 (**+.80%**) | .178075 →.158782 (**−10.83%**) |
| Mean candidate/reference Laplacian ratio | 1.065137 →.911274 | 1.156966 →.974558 |
| Mean candidate/reference flow-magnitude ratio | .106258 →.108312 | .867577 →.706978 |
| Mean signed encoded-luma drift | .006670 →.007226 | .015196 →.016059 |

The detailed temporal distribution reinforces the source difference:

| Descriptive count | Source0 | Source12 |
| --- | ---: | ---: |
| Frames with lower paired RGB RMSE | 51/81 | 81/81 |
| Transitions with lower paired temporal RMSE | 80/80 | 80/80 |
| Transitions with lower paired flow L2 | 20/80 | 77/80 |
| Frames whose Laplacian ratio is closer to1 | 37/81 | 76/81 |

These correlated within-clip counts are descriptive, not independent statistical trials.

Source12 improves several paired error metrics and brings its mean detail-energy ratio close to1. Its estimated motion-magnitude ratio nevertheless moves farther below1. Source0's mean detail ratio crosses from above1 to below1 and moves farther from1; its estimated motion remains far below the dense reference and paired flow error rises. Both signed luma biases increase even though their luma RMSE decreases. A uniform “motion/detail restored” interpretation is therefore unsupported by these measurements.

Conversely, the recorded improvements are real at this scope: in particular source12's pixel/temporal error reductions are distributed throughout the clip. They identify some terminal function that the final whole dense step can change after three candidate steps. They do not isolate one FFN, numerical error source or lost semantic feature. A lower proxy may arise through smoothing, motion changes or a different pairing to the dense trajectory; actual subject/condition quality requires the independent video-quality evidence still marked NOT_RUN.

## What this adds to the finite-map mechanism

Whole-flow source0/source12 had positive net current candidate effects on native last-step SSE. The saved dense-at-Student endpoint removes that last-step difference while leaving the first three candidate steps fixed. The rendered clips confirm that the resulting latent improvements transfer differently to output domains. The VAE/media bridge is exact for dense controls, so an ambient decoder mismatch is not needed to explain the mixed proxy outcomes in this specific attempt.

The cheapest next scientific use is to combine saved-state error coordinates with these already measured frame/transition changes, distinguishing spatial low-frequency, channel-mean, texture and estimated-motion effects before selecting a repair or FFN representation. A future repaired four-step rollout would be a new policy at new states. Formal quality should be assessed on its full function, rather than promoted from this one macro intervention or from local latent SSE alone.

The four decodes are additional VAE neural work even though no new denoising is performed. Their summed measured render wall is18.8044s with one shared setup63.9212s; the sampled peak is6693MiB. These are render-only costs, not the speed of a complete candidate,candidate,candidate,dense denoising pipeline. The fixed study output totaled8,330,343 bytes at inspection, below0.1GiB. I read the resource receipt but did not independently profile the GPU or verify its later release.

## Exact evidence binding

HF-derived root: `e0_saved_flow_decode_20261026/attempt1/render`.

| Object | SHA256 |
| --- | --- |
| Frozen protocol | `588115b9a0930619045fb42cc1d92803225610ccd66f17b27bb6142130f6baef` |
| Completed STUDY | `58e1183d243c849b83fe57556929a981fe14ddea4ee2452841ac9c9a66b172d8` |
| Final RUN | `cae61ade5219483408c6edf15ed78a958141c64721659e5a723ed9df2180c5d0` |
| Immutable DECODE_RUN | `24943a6febd28e88c3abcabacafff5701b6c718ba9be51565c9dab7d8b769fcf` |
| Core BUNDLE | `70ba9218aa2a67d01ea1641b143c3e2a11cd50f83a71fc1156536210cd7d0013` |
| Source0 dense/control MP4 | `23f5063d3565f672ae82244ddaa7d18591ee84f2484ac8b5a0ab21dea75b1ee4` |
| Source0 late-dense MP4 | `1a132cbf1a01cfe068e325cfa5922d386cbae590ad54c3bcee50d588b9617923` |
| Source12 dense/control MP4 | `6d757a154fc15ca79689b65a5bc95bf9d3338b3d6618d2734645a3414a00919e` |
| Source12 late-dense MP4 | `b5663d93e6e460777a723eaa581159073d3e670bb11669a59a9685e2de6279e9` |
| Source0 late-dense metrics | `035e872b20c02a39774296db3be464a93a1dfdad9c5df3d513554e0b70b3bb3a` |
| Source12 late-dense metrics | `89d7e3e44dffca1e5454ac387ce88bc7da6a5b355c2ebfb8f62a367f34fc4284` |

Comparators are exactly `e0_paired_full_depth_20261026/v3_attempt1/dc_source00/student_metrics.json` SHA `01ef726ac12c143e174a828656e71f0941c8b22a0038fbad80296b640ca8be11` and the source12 counterpart SHA `9946d9438e17ae4d862598ff78ad5c8adbba6108204ef991a46443f27bf3614f`. Their authoritative v3 strict/core bundles and actual videos were rechecked; this comparison does not use the older erroneous reference-metadata reports.

Graph interpretation is **P24→terminal latent→VAE→D09/P25**: actual four-decode observations now exist at the two exposed sources. Independent official/human acceptance remains open. This record uses actual local media and numerical/source checks, with no new web or novelty claim and no manual full-video viewing conclusion.
