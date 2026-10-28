# Fixed-feature B-F rank512: implementation review

2026-10-28. **SUPPORT for the exact bounded pilot below.** Reviewer `/root` selected the experiment scope but did not author the three implementation/test files. `/root/compact6016_six_video_build` authored them. This is implementation and mathematical review, not an executed data result or independent novelty vote.

| File | SHA256 |
| --- | --- |
| `tools/bf_frozen6016_output.py` | `3cf1424473db835db502839d203acf2f7cb0bb736658d8e44821ab5e69d2246c` |
| `tools/e0_bf_frozen6016_terminal.py` | `031ca584a841a4191d11cd1d176835d5d09216b6dfe74049fb9323f8b14873a0` |
| `tests/test_bf_frozen6016_terminal.py` | `b073dda9ef58c692ae4d57001db8cbde1643f0df6f3187d7687d59fe5cf277cf` |
| `config/e0_bf_frozen6016_terminal_20261028_draft.json` | `0cf3009573b4ed03890ec8b476dedae5d40a37ff9a52b164517670a19ce8e9ab` |

The experiment isolates output-space compression after the completed A0 energy6016 nonlinear features. Its own dynamic Q2 and Y must reproduce all96 original diagnostic hashes before training. Inputs remain historical CAL24 teacher states, with train16 and exposed diagnostic8 separated by image source. The feature cache is ordinary detached storage, allowing coefficient backward; it is not a deployed static activation or a teacher13824 hidden cache.

Both expert spaces use their train-only common mean. Ordinary PCA512 competes with the fixed128-column completion of both step-conditioned centered-head/gate row spaces, augmented by384 complement-PCA columns. The fixed completion is explicitly retained if the algebraic head rank is smaller; it is not a rank search. Orthogonality and actual head-span projection are checked in FP64. Preserving the continuous head numerator does not preserve variance or BF16 residual/LN/head instructions, which are measured separately.

The affine factor initialization projects `[W2, b2-mu]` into Phi. Only a rank32 coefficient update is trained:208896 trainable parameters per student,256 Adam updates each,1024 total. The inherited A0 training/calibration/support selection remains paid lineage. The exported static BF16 factors have an explicit post-Q1 → frozen W1/GELU/own-Q2 → Bc → Phi graph, three GEMMs, no intermediate quantizer and no deployed LoRA. This pilot does not time that full deployed graph or generate videos.

There are96 original-u full-M-geometry selected-row neutrals,192 teacher-oracle readouts and384 initial/final student readouts. Other full-M rows are artificial; no complete native state is reconstructed. Actual BF16 head error is decomposed into direct numerator, variance, signed cross and numerical remainder against the same captured head. Source means average the two nested expert steps; rows and steps add no independent sources.

Independent accounting gives5728 BF16 products:192 A0 feature products +4096 training forward/backward products +768 evaluation factor products +672 original heads. FP32 products are1032 merges +2048 adapter backward products =3080. The FP64 geometry/oracle/statistic products sum to1330. The observer includes the original `addmm(out=...)` overload. Failed debits persist and exact budget closure is mandatory.

The coordinator actually ran17 synthetic CPU controls, exit0 in27.588 seconds, including the full1024-update/672-head miniature path and exact dtype matrix counts. Controls also cover heldout isolation, rank/geometry conditions, affine initialization, source aggregation, signed decomposition, deployed own-Q2 timing, immutable output checks and authorization gates. A separate Terra metadata audit matched70/70 source bindings and completed-parent RUN/exit identities without numerical imports. No real saved data, native CUDA, model, fit or video was executed by this review. A frozen protocol and independently exited actual byte preflight must precede the real pilot.
