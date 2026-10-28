# B-F frozen6016/r512 · actual result/source audit

The fixed rank512 output spaces support much smaller teacher-state terminal-head error than the learned coefficient students attain. Both students retain a modest 3–4% held-source improvement over inherited A0 energy; explicitly containing the two-step head×gate span improves the teacher projection substantially but adds little consistent improvement after coefficient training. This separates an available output-space mechanism from the remaining prediction/recovery problem.

Audit by `/root/compact6016_six_video_build`, the implementation author. This is **source-to-raw-result verification**, not an independent review of my own method. Root performs a separate audit. All audit calculations used Python standard library; no Torch/NumPy import, GPU, native quantizer, model, new fit or retry was executed.

| Immutable artifact | SHA256 |
| --- | --- |
| Frozen protocol `config/e0_bf_frozen6016_terminal_20261028_v1.json` | `8a74628b8160d237b42ed4badc20b8aa7602d0d2431e6b36a7c6cc4e270d06c3` |
| HF `ffnds-derived/e0_bf_frozen6016_terminal_20261028/attempt1/worker/results/RUN.json` | `12bc19935b69ab0e5207350ab9241e95fc2718b4268ae7a1a5b3bf2e584d0dc6` |
| Sibling `worker/EXIT_RECEIPT.json` | `97b6967e3ffca55a101c4305ccb66d776d20ff4575b329b4e7de439b3168408a` |
| Result `OPERATIONS.jsonl` | `17c615de9915f8528ac01184c06be758a5e9b565733805417c81472e0e0a5546` |

The actual child exited0, without timeout/retry. Body wall is613.739s, external wall629.444s, peak reserved memory4.1973GiB; the GPU monitor has no recorded violation. The frozen core/driver/tests are respectively `3cf1424473db835db502839d203acf2f7cb0bb736658d8e44821ab5e69d2246c`, `031ca584a841a4191d11cd1d176835d5d09216b6dfe74049fb9323f8b14873a0`, `b073dda9ef58c692ae4d57001db8cbde1643f0df6f3187d7687d59fe5cf277cf`. All70 live bindings and all70 executed-source copies retain their exact registered bytes/SHA.

## Raw evidence and accounting

All77 operation fields agree exactly across frozen limits, predebit, invocation and completion. The94,527 journal events have contiguous serials and preserve predebit≥invoked≥completed at every prefix; their summed counts reproduce every field. This includes1024 Adam updates/backward calls,672 original full-M-geometry head calls,5728 BF16/3080 FP32/1330 FP64 observed aten matrix operations, four eigensolves/four QR operations,694 asset H2D copies and1,258,859,264 payload bytes. Solver-internal BLAS is not represented as separately observed aten operations.

All96 original-u neutrals are bitwise successful. All96 cached inherited A0 own6016 post-Q2 and final-Y hashes equal their original diagnostic hashes. The576 unique records cover96 source×step cases × two bases × oracle/initial/final. Each recorded target energy equals the same captured-head target in immutable matched-head RUN `31baab02…`. All train16 and exposed-held8 source summaries and every parent comparison were independently recomputed from the individual case records, preserving two nested expert steps per source.

I also decoded the serialized BF16 values directly with `struct`, using96 captured `head_rows` payloads and both saved head-prediction packs. All576 raw prediction hashes agree with case/export records; all96 target hashes agree with the CAL24 manifest. Independent FP64 scalar arithmetic over9,437,184 predicted elements reproduces every recorded head SSE and relative SSE: maximum absolute SSE difference3.55e−15, target-energy difference3.64e−12, relative-SSE difference2.17e−19. This verifies oracle, initial and final head metrics against actual saved data rather than only trusting their JSON summaries.

The continuous direct+radial+signed-cross identity and the native continuous+remainder+signed-cross identity close across all576 cases to maximum4.34e−19 in relative units. Gated-Y ratios also reproduce exactly; mean/centered SSE closure differs by at most1.82e−12 in absolute squared-error units. q/variance arrays and the predicted full u states were not saved, so their decomposition components were audited against source and scalar identities, not independently recomputed from missing states. The native-program remainder includes the entire difference from FP64 continuous mathematics, including native normalization/matmul arithmetic; it is not a measurement of one isolated BF16 cast.

All four student whole-file export SHAs match; each is about73.97MB. Actual headers contain BF16 W1[6016,5120], Bc[512,6016], Phi[5120,512], bc[512], mu[5120], plus the registered support/smoothing and FP32 training factors. No full W2 is exported. This verifies the stored three-matrix representation; no full-M FFN latency or video deployment was measured here.

## Same-source held8 results

Numbers are mean head-relative SSE over two steps within each exposed source, then over eight sources. Rows/tokens are not independent N. Each expert uses its own common train16/two-step mean, shared by its two bases.

| Expert / basis | A0 energy | Teacher projection | Initial student | Final student | Final lower than A0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| high / PCA512 | 6.08789e−4 | 1.30555e−5 | 5.98918e−4 | 5.86531e−4 | 7/8 sources |
| high / HEAD128_PCA384 | 6.08789e−4 | 1.02683e−5 | 6.05144e−4 | 5.89806e−4 | 7/8 sources |
| low / PCA512 | 6.60966e−4 | 1.12925e−5 | 6.51456e−4 | 6.38192e−4 | 8/8 sources |
| low / HEAD128_PCA384 | 6.60966e−4 | 6.43884e−6 | 6.58002e−4 | 6.32857e−4 | 8/8 sources |

The final PCA reductions versus A0 are3.66% high and3.45% low; head-containing reductions are3.12% and4.25%. All four students beat saved random-A0 and B-H on all8 held sources. Part of the A0 improvement precedes recovery: the initial PCA projections already improve1.62%/1.44%; its additional256-step recovery improves head error2.07%/2.04% relative to initial. Head-containing recovery improves2.53%/3.82%, on8/8 high and7/8 low sources. This does not establish matching total training cost: each student inherits the trained A0 W1/W2 solution and support calibration, and adds its own256 updates/208896 trainables. Four children add1024 updates to the inherited two-parent512-update lineage.

## What the two bases contribute

The continuous head-span construction worked at the registered object. Its fixed128-column completion and384 covariance directions remain rank512; measured head-span squared leakage is7.21e−31 high and7.11e−31 low. Maximum teacher-oracle pre-cast q leakage is5.77e−36/1.48e−35 in the registered norm-bound-scaled metric. Those are continuous geometry identities, not native-BF16 equality.

For teacher projection, containing head×gate reduces held q-relative SSE by86.43%/93.68% versus ordinary PCA and native head SSE by21.35%/42.98%. Its variance-relative error instead increases9.91%/16.89%. In that head-containing oracle the radial term contributes94.70%/92.07% of continuous SSE before its signed cross correction; native-program remainder SSE contributes36.32%/51.37% of total native head SSE before its signed cross. Thus preserving q alone exposes the remaining normalization/variance and arithmetic mechanisms; it does not preserve the normalized head exactly.

For trained students, the final head-containing basis is0.56% worse than PCA on high and0.84% better on low, winning only2/8 and5/8 sources. The large teacher-oracle head-span benefit is therefore not realized as a stable trained-student advantage in this fixed pilot. The final head-containing direct-numerator term is93.71%/101.08% of native SSE, with radial and signed cross terms accounting for the remainder; the second percentage exceeds100 because the cross is negative. The native-program remainder SSE is only1.21%/1.03% of student native SSE before its cross correction. This is unlike the much smaller teacher-oracle regime.

Final student head errors are44.9×/56.5× the PCA projection controls and57.4×/98.3× the head-containing projection controls. The data therefore show output-space capacity at these sampled teacher states substantially beyond this inherited-feature, rank32 coefficient-recovery solution. They do not distinguish feature information loss from the restricted update family, finite optimization, original gated-Y objective or their combination. The projection control uses teacher Y offline and is an attainable representation witness, not a legal online predictor or a proven global optimum.

## Decision supported by this scope

This pilot is complete as a **HIF4_FAKE_QDQ_DIAGNOSTIC**. Its modest local head improvement and arithmetic MAC ratio0.59255 support a candidate representation and mechanism finding, while full-M FFN execution speed, full80-function free trajectories and video quality remain unmeasured for B-F. Terminal teacher-state head error alone does not establish stability under accumulated errors in all other functions.

The next useful question is how a legal function on original BF16 states or an actual same-checkpoint native-NVFP4 path predicts the head-sensitive coordinates and variance under changed states. Follow the current [BF16_ORIGINAL / NVFP4_NATIVE migration](../../../agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md) for new model work: original BF16 weights and new state/label bindings are required for BF16_ORIGINAL, and native W4A4 needs its real dynamic activation producers/kernels. Old HiF4 decoded weights/cache are retained as historical assets. This result does not motivate another blind width/rank/learning-rate sweep or immediate expansion of this legacy fit to all80 functions. A future discrimination should separate prediction restriction/objective from output-space capacity and check changed-state/full-flow effects, with its own exact registration.
