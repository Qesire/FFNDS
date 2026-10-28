# Independent review: actual eight-axis video20 quality

**OBS / NATIVE:** The actual quality RUN SHA-256 is `8f435d047b2c0f85ca08cefa795246a679e436e7a7dd59710ae5219e8756b6cd`; the durable exit receipt SHA-256 is `29837f9c387dfb4bbf1cad0fc6d7ebeaaa5b371c31f8c78103bd027eba4962b8`. Its child actually exited 0 in 872.72 seconds without timeout or automatic retry, under protocol SHA-256 `6413712794c8d82d3f9a909a6eca48d20f13f6362e4cfcdbe1fc10ec08d091d0`. Predebited, invoked and completed counts all exactly match the frozen budget: 8 official functions, 420 video-axis results, 60 reused CORE reports, 2,400 RAFT forwards and 48,000 iterations. The assigned-GPU report contains no violations. Independent stdlib recomputation of the source summary from the actual 8 result records and 60 RAFT observations equals the recorded summary exactly; no model, metric, saved tensor or GPU was run in this review.

The independent command `python3 tools/recompute_raw_bf16_video20_quality.py --protocol config/e0_raw_bf16_video20_quality_20261029_v1.json --out <fresh temporary file>` also exited 0 and reproduced `research/20261029/RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.json` byte for byte. The replay tool SHA-256 is `bef1ddfbea4588655fb3938d5fd8fd5c5bdc139cc7c7064701c277b1ee7cf585`; the result SHA-256 is `4f8210c5482407410dfdf58835a149819566f265d1938fc12e69bcce050f078c`.

Each of 20 historically exposed development image clusters contributes one seed42 video per arm. The first six axes below use all 20; the I2V subject axis uses 10 subject clusters, and I2V background uses seven background plus three camera-prompt clusters. Values are source-equal means, not a five-sample official VBench aggregate.

| Official axis | Dense | Shared-r960 | Step-r960 |
|---|---:|---:|---:|
| Imaging quality | 73.7205 | 71.5858 | 70.3539 |
| Aesthetic quality | 0.62520 | 0.57478 | 0.55785 |
| Motion smoothness | 0.98323 | 0.97862 | 0.97815 |
| Dynamic degree, binary fraction | 0.35 | 0.10 | 0.15 |
| Subject consistency | 0.96348 | 0.96544 | 0.96723 |
| Background consistency | 0.96746 | 0.96724 | 0.96866 |
| I2V subject | 0.98355 | 0.96502 | 0.95843 |
| I2V background | 0.99170 | 0.97408 | 0.96745 |

The static consistency axes stay close to Dense in these means while image-conditioned subject/background scores and aesthetic scores fall. This pattern motivates testing whether apparent consistency partly accompanies reduced image motion; it does not by itself establish causality or certify prompt, object action or camera behavior. The paired I2V-background difference is negative on all 10 eligible sources in both arms; I2V-subject is negative on 9/10 in both. Aesthetic is negative on 18/20 shared and 20/20 step sources. For dynamic degree, Dense is positive on 7/20, shared on 2/20 and step on 3/20; shared loses 5 Dense-positive cases and step loses 4, with no new positives.

The original RAFT observer's continuous top-5%-flow score, averaged within each video then equally over sources, is 7.05802 pixels per two-frame pair for Dense, 3.45723 for shared and 3.36421 for step. Ratios of these source-equal means are **0.48983** and **0.47665**; means of the 20 defined per-source ratios are **0.52346** and **0.51456**, respectively. Every Dense denominator is nonzero. Shared per-source ratios range 0.06610–1.14018; step ranges 0.09599–1.26203. The direction is heterogeneous: source 312 exceeds 1 in both arms, whereas sources 903 and 256 have very low ratios in both.

Stratified continuous RAFT means (Dense/shared/step) are 10.8996/5.3804/5.1764 for 10 subject sources, 2.3727/1.3471/1.4800 for seven background sources, and 5.1853/1.9702/1.7202 for three camera-prompt sources. These are image-flow statistics; the small camera stratum and prompt text do not prove camera motion, direction or semantic compliance. Dynamic-degree fractions, continuous RAFT and the earlier full-frame Farneback proxy remain distinct observables. Temporal flickering was not scored, and human full-play review, complete quality acceptance and repeat-controlled warmed system speed remain pending.

The next discriminating evidence is source-paired full-play inspection and the independently acquired full-depth 21τ teacher trajectories, with later student-state/causal checks, to separate reduced temporal variation and appearance anchoring from wrong object or camera dynamics. Preserve same-source and all-tail results and compare static versus motion-sensitive axes without treating high consistency or local teacher SSE as a video pass. The currently running new-source acquisition is not an already measured result.
