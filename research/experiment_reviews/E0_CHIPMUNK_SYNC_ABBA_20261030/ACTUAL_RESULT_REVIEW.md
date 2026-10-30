# Independent actual-result review: P64 synchronization ABBA

Reviewer: `/root/sync_abba_actual_review_terra`, independent of the owner and implementation authors. Scope: read-only inspection of actual receipts, metadata and verifier implementation, with metadata arithmetic recomputed. No model, GPU or new artifact-reading verification experiment was run by this reviewer. Decision: **the two recorded executions and their separate 79-artifact verifications are actually closed; the descriptive timing and historical byte-neutrality claims below are supported within this scope.**

## Identity and closure

- Source 898, seed 42: protocol SHA256 `01ae9082c690b226325f875e1136f0ee06a5bfb29876bc85afd92c16407da5ab`; GPU RUN `033a6cd6cae7c7865b08c25593705cc8d73fce76895331b10f8df6fd96267b4e`; GPU EXIT `af08f06695ad988d704ba4e30391c0e485b525f1b43e077a6503faffb00e7952`; verifier RUN `a02f10dc7b382dff78450075ecf4477349cc5e61d5c2f1c44b43139532203614`; verifier EXIT `22ee2568d584cbf7812221372f2fd3c6deb1949f96f2dc0dc478fe2713b0bc1a`.
- Source 979, seed 42: protocol SHA256 `a6e0b24e3d981eb6df5294c42a8fa18a5232f6c949b3c08779a3f50000901164`; GPU RUN `de5006baa40ec75533d5534021785b4247fe6b89cad65761e835a213a56fa09c`; GPU EXIT `25283f73814891b99d016c58743285af1557106a3f43b333b0eba85232da12ae`; verifier RUN `c9265a80004b98e10cd2e95316a60c18995a143995e218b68ca19a8707396477`; verifier EXIT `c66743aea0e259cbdd9f043372b34517cf0b96981540b2bc7f8515f67e77b1a4`.

For each source, CPU preflight, GPU supervisor and verifier supervisor have actual EXIT 0 bound to their RUN and protocol; GPU RUN is complete, verifier RUN is PASS, and completed read bytes equal predebits with no inflight operation. The two verifiers read 442,607,476 and 434,663,881 bytes respectively, each covering 70 typed states, seven complete videos and two journals. The verifier checks output payload identities, replays the operation journal, and binds the GPU receipt. The producer compared each of 70 state files and seven videos byte-for-byte against the corresponding historical reference: NONE against Dense; every warmup and measured candidate against X. The independent verifier checks typed/hash identities and historical metadata; it does not itself reread historical payloads. All 70 + 7 comparisons per source report true. This is exact equality to the historical candidate, not a new quality assessment.

Each seven-phase run has 160 FFN sites per phase; ledger predebits and observations equal the entire frozen nonzero budget, including seven video generations, 240 packed actions, 240 extra old W1 operations, 70 state checks, seven video checks and 120 deferred completion barriers. Per-source 240 cache and mask observations were verified. GPU monitor sampled isolation passed with zero reported violations; output monitor stopped without fault, with tracked peak 475,899,543 / 469,099,904 bytes under 512 MiB. Cumulative peak GPU allocated was 36,967,220,224 bytes for both, reserved 48,219,815,936 / 48,370,810,880 bytes, and peak RSS 117,029,113,856 / 117,025,157,120 bytes; these are cumulative run peaks, not arm-specific peaks. Whole-run wall times were 432.410 / 439.140 s. Recorded precision is PyTorch 2.10.0+cu128 and CUDA 12.8.

## Timing and interpretation

A preserves per-action synchronization; B defers the specified five same-compute-stream actions but synchronizes before cache release, while the mask transfer and cache barriers remain. Both use the same P64 candidate. The measured labels are A1/A2 and B1/B2, excluding NONE and both warmups. Source 898 uses ABBA and source 979 BAAB after mirrored warmups. Recalculation from the bound RUN metadata reproduces the final JSON exactly.

| Source | FFN boundary CUDA A → B (s) | B/A | Phase wall A → B (s) | B/A |
|---|---:|---:|---:|---:|
| 898 | 7.3159 → 6.7927 | 0.9285 | 47.5525 → 47.1724 | 0.9920 |
| 979 | 7.5904 → 7.0458 | 0.9283 | 48.5674 → 48.2992 | 0.9945 |

The equal-source means are FFN 7.4531 → 6.9193 s and phase wall 48.0599 → 47.7358 s. The mean of the two within-source B/A ratios is 0.9284 and 0.9932 respectively. This describes two previously exposed sources, one seed each, with two nested repetitions per arm; it is not four independent sources or a population estimate. Mirror order is confounded with source/device, and phase-level variation is visible, particularly source 898's A1/A2 FFN values 6.9712/7.6605 s. CUDA boundary event sums can include host gaps or prefetch joins; phase wall includes observers and video checks. These are instrumented same-candidate timing observations, not isolated kernel speed, a repeated Dense-relative system speedup, or a deployment acceptance. Physical bus bytes remain unknown. This experiment does not establish source 979 quality recovery.

## Reviewed artifact identities

- Summary generator `tools/e0_chipmunk_sync_abba_results.py`: SHA256 `8f74010f5df0dca5d0a66ce24f55bc1c63f59ca4361e7726eb1addbe971dd75a`.
- Final `research/20261030/CHIPMUNK_SYNC_ABBA_RESULTS_20261030.json`: SHA256 `eb4389e2d5a6af7907cbd58a1d2cb4e5b3de61ee486563ba15f2659f2855b877`.
- Final `research/20261030/CHIPMUNK_SYNC_ABBA_RESULTS_20261030.md`: SHA256 `4553880a6fad50483972876a08172364b2dbb0d49ffa7e4dc6be623bd3f96ff0`.
- Frozen owner `tools/e0_chipmunk_sync_abba_native_owner.py`: SHA256 `3c4ba9b39c50c6d99162e0e6107b43547878f7779f9a1adfb8f93e3c2c3e287c`; driver `tools/chipmunk_p64_sync_abba_driver.py`: `4557f6cff224d7082cd6da2e589e0ed68776cb9b9d8dd72767976c465d14b2ab`; tap `tools/chipmunk_p64_sync_abba_tap.py`: `7d4b8734493ed4a67db9344bdd47cf178c2c4012cdbdf75abc49db3030578687`.
