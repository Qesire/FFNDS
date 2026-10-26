# Signed band effects distinguish error concentration from useful finite correction

Author: independent_e0_audit. Date: 2026-10-27. Status: independent E0 DISCOVERY and completed JSON/artifact audit; no method endorsement or quality acceptance.

**New scientific object.** The six-source record supports a richer signed target than cross sign alone: the finite same-Student-state net effect in each band, \(\tau_b=\|\Pi_bN\|^2+2\langle\Pi_bP,\Pi_bN\rangle\). Source16 has a negative whole-state net effect at step2, while all six trajectories still diverge. Its terminal high-frequency net effect is also negative even though most terminal error energy is spatial-low. This separates the location of error, current candidate injection, and beneficial cancellation at an already displaced state. The strongest competing explanation for video-metric changes remains altered coherent motion/texture and estimator response, together with source/seed/state heterogeneity. The cheapest discriminator is the existing per-source signed vector, followed by the already registered fresh-source crossed-seed observations; no new fitting or polynomial tuning is suggested.

Execution boundary: only source/JSON/scalar arithmetic and artifact byte hashes were used. Zero model/GPU/tensor numerical work, decoder calls, metric reevaluation or new media. Public tool/data and all older reviews are unchanged; this new MD is the sole output.

## DISCOVERY: a local candidate can offset dense finite response

At the fixed native step and actual input pair,

\[
P=T_D(z_S)-T_D(z_D),\quad N=T_S(z_S)-T_D(z_S),\quad \Delta'=P+N.
\]

Thus \(\|\Delta'\|^2-\|P\|^2=\tau=\|N\|^2+2\langle P,N\rangle\). This measures the current candidate map versus the dense map **at this same Student state**, keeping the candidate-generated prefix fixed. It is a finite state-pair observation. It does not allocate the prefix's causal history, identify a universal Jacobian, or value the decoded video.

Source16/step2 gives P energy29,616.670263, N energy2,438.761964 and cross−3,385.644340. Consequently **τ=−946.882376**, and actual next SSE28,669.787887 is below the same-input dense counterfactual. This is the sole negative whole-state τ among the24 step cases. The state remains much farther from the teacher than its input SSE5,839.587532. “The candidate's current injection increases next-state SSE at every step” is therefore contradicted in this cohort; “the current candidate preserves the teacher trajectory” is also contradicted. Both observations fit the exact identity.

The last Source16 step is another discriminating case: P216,234.328998 + N14,224.727817 + cross−13,883.216841 = terminal216,575.839974, with net effect only **+341.510976**. Its terminal band net effects are:

| Native latent band | N energy plus signed cross |
| --- | --- |
| All-axis DC | +3,284.854783 |
| Temporal-DC spatial-low excluding DC | +1,922.775724 |
| Temporal-nonDC spatial-low | −178.514850 |
| Spatial-high remainder | −4,687.604682 |

For an arithmetic endpoint replacement \(\Delta'_B=P+(I-\Pi_B)N\), orthogonality gives

\[
\|\Delta'_B\|^2-\|\Delta'\|^2=-\tau_B.
\]

Therefore replacing Source16's terminal high-band candidate effect with its dense value would **increase this endpoint SSE by4,687.604682**. This is an exact saved-endpoint geometric comparison, not an executed native operator: a projected latent replacement need not be realizable by a packed FFN or commute with native rounding. It is nevertheless a useful counterexample to selecting corrections from N magnitude or negative cross alone. For example, all three Source0 global crosses are negative, yet all three net effects are positive.

A second counterexample separates energy concentration from current injection value. The fixed spatial-low span contains97,104/2,026,752=4.79% of real dimensions and53.47–74.19% of terminal error energy across the six sources. Yet Source12's spatial-high band contributes **15,898.906428 of its22,406.833785 terminal net effect (70.96%)**. Source16's high band instead compensates other components. A fixed “correct the band containing most total error” objective and a signed current-effect objective need not agree. Their eventual video relevance remains an additional empirical question.

## EVIDENCE: source, seed and nested local statistics

The actual six source identities and seeds match canonical24, including input-image byte SHA. Roles mean membership in **this bank's** calibration set, not absence of prior research exposure. All24CAL24 inputs have historical project exposure;0/12 are exposed but were not among the four sources fitting this bank. There are three subject strata, three background strata and zero camera-instruction sources. Each source has one seed, so this cohort does not separate image/content from seed effects.

| Source | Seed | Bank calibration | Raw smaller/equal/larger | Median Student/Teacher operator-SSE ratio | Terminal latent SSE |
| --- | --- | --- | --- | --- | --- |
| 0, subject | 42 | No | 159/1/0 | 0.418601 | 454,530.649087 |
| 4, subject | 46 | Yes | 158/1/1 | 0.479002 | 211,064.900392 |
| 8, subject | 50 | Yes | 156/1/3 | 0.533731 | 251,027.956771 |
| 12, background | 54 | No | 159/1/0 | 0.457672 | 137,447.952941 |
| 16, background | 58 | Yes | 155/1/4 | 0.473232 | 216,575.839974 |
| 20, background | 62 | Yes | 156/1/3 | 0.528594 | 346,171.068199 |

The raw and native-writeback counts both total **943 smaller /6 equal /11 larger**, over960 nested sites. Ten larger sites are block0 and one is Source16/step2/block1. Every equal site is step0/block0. This is a cohort observation with explicit exceptions, not a count of960 independent source samples. The common teacher-energy denominator is retained within each site; absolute SSE comparisons agree with the reported shrink counts.

I independently recomputed all **516** groups of all-site, per-step, per-layer and stratum local counts/medians from raw TRACE JSON. All match. Per-layer medians contain four step observations spanning two experts. The3×3 strata are thirds of the native token-time/vertical raster21×29×52, with horizontal positions pooled; they are not thirds of the81 decoded video frames or semantic subject/background masks. A median of site ratios and a ratio of two medians are distinct quantities; the tool correctly stores the former, plus separately labelled median relative errors.

All18 nonzero state pairs have native dense-map energy gain>1, range3.625849–7.719963, and negative δ–total-flow cosine. Global crosses split **7 positive/11 negative**, whereas dynamic spatial-low crosses split **17 negative/1 positive**. Source4 and Source0 have opposite global signs despite sharing the subject stratum. Background Source16 is globally negative at all three nonzero steps, and Source20 changes sign across steps. A deterministic class→sign account is contradicted. A probabilistic category account remains open but is confounded by image, seed and fit exposure here.

## EVIDENCE: old and new band records are comparable on their shared terms

Old0/12 used differences of nested orthogonal masks at(Tmax,Smax)=(0,0),(0,8),(10,8),full. Since T=21 has maximum absolute Fourier index10, those differences define exactly the same four subspaces as the new direct disjoint masks: dimensions16/4,608/92,480/1,929,648. I reconstructed every old P/N/cross/next band value by subtraction of the original spectrum JSON and checked every new value against its actual direct DISJOINT_BANDS JSON. Band and whole-state identities, gains, flow/TRACE endpoints and saved-state joins agree, with maximum absolute scalar join roundoff5.83e−11.

The old spectral study did not directly report input-band energies. The summary keeps those fields null and records the different provenance; a null is not a measured zero. There is also a cheap **derived** recovery path: for both old sources I verified current flow input native hashes equal TRACE after-step-pre hashes, and at steps1–3 they equal each respective previous after-step-post hash. Differential entry-cast SSE is zero. Consequently input-band energy at step k≥1 equals the preceding next-band energy; step0 has zero input difference. If added later, this should be labelled `DERIVED_FROM_PREVIOUS_NATIVE_OUTPUT_HASH_IDENTITY`, preserving the original missing-direct-measurement provenance. No additional FFT is needed. I have not changed the public table.

## EVIDENCE: the video vector rejects a single scalar ranking

| Source | Mean RGB RMSE | Flow ratio of means | Mean transition flow ratio | Edge ratio of means | Adjacent RGB variation ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.154160 | 0.090937 | 0.106258 | 1.053368 | 0.368354 |
| 4 | 0.108684 | 0.363069 | 0.427176 | 1.451747 | 0.989348 |
| 8 | 0.127557 | 0.701309 | 0.745290 | 1.258801 | 1.089727 |
| 12 | 0.130529 | 0.846493 | 0.867577 | 1.157260 | 1.463009 |
| 16 | 0.107833 | 0.399367 | 0.614131 | 1.178099 | 0.741321 |
| 20 | 0.095078 | 0.259524 | 0.270595 | 1.355580 | 1.108030 |

All ratios are candidate/dense. I recomputed flow ratios and edge ratios from the actual per-frame/transition JSON, and RGB means from all81 paired frame values. The two ratio estimands are correctly distinguished. If dense/candidate transition flow magnitudes are d_i,c_i and r_i=c_i/d_i, then

\[
\frac{\bar c}{\bar d}=\frac{\sum_i d_ir_i}{\sum_i d_i}
=\bar r+\frac{\operatorname{Cov}(d,r)}{\bar d}.
\]

Here covariance uses the empirical denominator80. All actual dense magnitudes exceed the denominator clamp. Source16's0.614131 unweighted mean versus0.399367 dense-motion-weighted ratio is therefore a real weighting effect, not an arithmetic bug. Its covariance is−0.0649234px; all six covariances are negative. Within this estimator's output, high-dense-motion intervals receive lower relative candidate flow on average. This is not identification of true physical motion loss: texture, occlusion and the aperture problem remain competing explanations.

Source20 has the lowest paired RGB RMSE, yet only25.95% of dense estimated flow magnitude, with **increased** adjacent RGB variation and edge energy. Source12 retains84.65% of flow but has higher RGB RMSE and146.30% adjacent variation. All six edge ratios exceed1 while all six flow ratios of means are below1. These observations support retaining reduced coherent displacement, altered texture/noise, photometric variation and estimator response as competing mechanisms. They do not establish which mechanism produced any particular movie. Higher edge energy is neither certified detail nor a general one-kernel blur description. No contact sheet or full video was viewed in this review.

The12 core reports cover972 frame observations and960 transitions, nested within six source pairs; the coincidence with960 FFN sites carries no statistical meaning. Every actual report has official VBench-I2V/VBench2/human NOT_RUN and quality_accepted=false. I personally read the original [VBench dynamic-degree implementation](https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/dynamic_degree.py): it uses RAFT, high-flow pixel statistics and threshold/count rules, rather than this downsampled Farneback mean. The original [subject-consistency implementation](https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/subject_consistency.py) compares normalized DINO features to earlier/first frames. These different computations clarify why the present proxies do not amount to those unexecuted quality measures. No such model was loaded here.

## Audit result and exact provenance

**No substantive arithmetic, role, seed or band-join bug was found in this version.** I read the summarizer, actual six RUN/TRACE/native-flow receipts, old nested spectrum/band reports, new direct band reports, and both current core metric reports per source. All six actual strict inventory/model/media gates pass. The phase1 ledger has four PASS entries. Actual summary-bound source/receipt/metric hashes match. All24 saved-state files were independently stream-hashed as opaque bytes (1,070,142,528bytes including safetensors headers), matching their original receipt SHA; none was parsed as a tensor. This audits identity and scalar consistency, without repeating native or spectral computations.

The exact audited public result is `research/20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.json`, SHA **`84b4177f1eff3e45ddb0fa9c118dbcd114ccd1f28e9630954c22d2b1b3b0a24c`**. It contains each source's full RUN/TRACE/native receipt/core report/bundle SHA bindings and separate flow-run bindings. The summarizer SHA is **`341761d6c6b51fc324f13ff8e327069ba2d8b34a5274de4aa063ef9ee78c9c8a`**. Source inputs checked against those bindings:

| Public source object | SHA256 |
| --- | --- |
| Paired two-source result | `1f1087d2d7fa111933fe2ce308ecbf4a9b7bd19f205d821973411a29115e16b2` |
| Whole-flow two-source result | `d4c1db3a644e7972bdc43e551da2ea59e28508b994730f760396f480a6c99f3a` |
| Old disjoint-band result | `a5390c168d588886d3d6943871ec1a8d88f06d6a573ef0ed2b7939a071ba776c` |
| Old nested spectrum result | `e7ff5a48a1084b1588ea593476c8d88b2dc1e058ac9a984c943894f605414120` |
| Phase1 protocol | `931b8162f8379193697a00899b617f3851f70ef0182ca7126c2ea4511a7f4e77` |
| Completed phase1 STUDY ledger | `701528811ce5d92f848ffab36f103ae91b7110fe62468eb65bea2cff3d0dc9c6` |

Old0/12 video/local traces and later full-flow runs are separate receipts whose full teacher/student trajectories agree; they are not rebranded as new phase1 executions. The summary preserves their spectral provenance and cost scopes. Old native diagnostic cost is latent replay plus extra dense flow/shadows; the new four-source runs also decode two clips. These scopes do not form a matched deployment-timing comparison. The summarizer adds zero metric evaluations, generations, learned-model calls or tensor numerical operations; eight newly added videos and640 newly added sites refer to the completed phase1 experiment, not to running this summary.

Graph anchors: `F04/F06/F08/F09` local function, `F12/F13` writeback, G21 finite state map and D09 decoded function. Local shrink→signed finite response→video function remain separate measured/conditional edges. A band-sensitive functional representation may guide a later FFN objective, but current P/N requires paid dense counterfactuals. Any low-cost deployment proposal must supply obtainable information and actual W1/W2/QDQ savings while preserving independent full-video function. Fresh-source seed replication is the next already specified falsifier; this audit selects no new operator or execution scope.
