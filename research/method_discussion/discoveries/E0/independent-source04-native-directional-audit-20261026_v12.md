# Source04: signed spectral mixtures separate category from finite trajectory response

Author: independent_e0_audit. Date: 2026-10-26. Status: independent E0 DISCOVERY / scalar evidence audit; no method endorsement or quality acceptance.

**New object.** Source04 supplies an exact, measured example of positive whole-state candidate–dense-map interaction coexisting with negative interaction in its dynamic spatial-low subspace at every nonzero step. Its whole-state cross is positive at all three steps, whereas the previously measured subject Source0 has negative whole-state crosses. This changes the useful explanatory unit from a subject/background label to the signed mixture of finite responses in explicitly defined subspaces. A competing explanation is image/seed-specific finite-state geometry, with calibration exposure as another confound. The cheapest present discriminator is already available in the frozen scalars: subtract the DC/static-low contributions and inspect the remaining cross, retaining both signs rather than replacing them with the global sign.

Execution boundary: this review reads source, JSON, hashes and ordinary scalar arithmetic only; it performs no saved-tensor loading, decoding, GPU/model call, fitting, or new generation. Previous v11 source-audit bytes and shared state remain untouched.

## DISCOVERY: what the signed identity explains

For the same step, conditioning and actual native scheduler boundary, define

\[
\Delta=z_S-z_D,\quad P=T_D(z_S)-T_D(z_D),\quad N=T_S(z_S)-T_D(z_S).
\]

Then \(\Delta'=P+N\). For an orthogonal band projector \(\Pi_b\),

\[
E_b=\|\Pi_bP\|^2+\|\Pi_bN\|^2+C_b,
\quad C_b=2\langle\Pi_bP,\Pi_bN\rangle,
\quad C=\sum_b C_b.
\]

The implementation in `tools/e0_disjoint_native_bands.py` constructs four mutually exclusive Fourier masks of the complete 16×21×58×104 state, using an orthonormal transform over T/Y/X, independently per channel. Spatial low means both absolute Fourier indices ≤8. The bands have 16 / 4,608 / 92,480 / 1,929,648 real dimensions, summing to 2,026,752. The conjugate-symmetric masks preserve real subspaces; the dimension count includes both members of each non-DC conjugate pair and therefore gives the real dimension. All-axis DC is one constant per latent channel. Temporal-DC is constant over the 21 latent time positions; these names carry no semantic subject/background or decoded exposure interpretation.

This is an actual BF16-output finite-map decomposition. The related continuous update uses \(R=v_D(z_S)-v_D(z_D)\), \(e=v_S(z_S)-v_D(z_S)\), \(P_{ideal}=\Delta-hR\) and \(N_{ideal}=-he\). The native `N` includes the actual step-post arithmetic and output cast. Keeping these two records separate makes the signed recomposition informative without treating an ideal identity as native bitwise algebra.

### Whole-state response, recomputed from the frozen scalars

| Step | h | Native dense-map energy gain \(\|P\|^2/\|\Delta\|^2\) | \(\|P\|^2\) | \(\|N\|^2\) | Signed cross C | Actual next SSE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0625 | undefined: input Δ=0 | 0 | 660.262339 | 0 | 660.262339 |
| 1 | 0.1041666865 | 6.2477975633 | 4,125.185435 | 729.433956 | +38.665193 | 4,893.284584 |
| 2 | 0.2083333135 | 4.6472070480 | 22,740.106607 | 2,544.506997 | +762.206358 | 26,046.819963 |
| 3 | 0.625 | 7.3154049833 | 190,543.036553 | 17,254.032697 | +3,267.831142 | 211,064.900392 |

The gain is an **energy ratio at the three observed nonzero state pairs**, not a Jacobian norm, global stability constant, or universal amplification factor. Native P/N/cross energy recomposition closes within 5.83e−11 absolute arithmetic difference; the largest recorded Fourier Parseval discrepancy is 2.92e−11. The direct flow recomposition SSE is zero in all four records. Each saved-state SHA recorded by the band result equals its corresponding flow receipt SHA; this review did not reopen those tensors.

At steps1/2/3 the P, N and cross terms account for respectively **84.30/14.91/0.79%, 87.30/9.77/2.93%, 90.28/8.17/1.55%** of the actual next SSE. These are terms of this finite identity. They do not allocate a causal fraction of video damage: P follows states already displaced by the earlier candidate prefix.

The original paired trace also gives δ–flow cosines −0.849762/−0.858487/−0.906036, so the later total flow difference remains anti-aligned with the current latent displacement despite the new cross-sign pattern. Terminal SSE211,064.900392 is less than Source0's454,530.649087, even though Source0's three global crosses are negative (−220.532017/−1,914.643756/−227.407089). Cross sign alone thus does not order terminal divergence: P and N magnitudes also enter the identity.

### Band signs reveal cancellation hidden by the whole-state statistic

| Fixed native latent band | Step1 cross | Step2 cross | Step3 cross | Share of terminal SSE |
| --- | --- | --- | --- | --- |
| all_axis_DC | +278.143019 | +477.471992 | +3,228.289599 | 24.8482% |
| temporal_DC_spatial_low_nonDC | +75.800554 | +282.336063 | +2,770.115315 | 34.8543% |
| temporal_nonDC_spatial_low | −52.480434 | −190.429725 | −935.988983 | 11.4637% |
| remaining_spatial_high | −262.797946 | +192.828028 | −1,794.584789 | 28.8338% |
| Whole | +38.665193 | +762.206358 | +3,267.831142 | 100% |

Subtracting DC alone changes the three whole crosses to **−239.477826 / +284.734366 / +39.541543**. Subtracting both temporal-DC low bands gives **−315.278380 / +2.398303 / −2,730.573771**. The dynamic-low band remains negative in all three steps, while the remaining-high band changes sign. Consequently a positive global cross need not mean that every temporal innovation is reinforced. Conversely, a beneficial interaction inside one band need not dominate its total energy or produce a better decoded video.

These complement scalars are arithmetic descriptions of the observed pair, not interventions that remove latent DC or deploy a spectral correction. A new online selector would also need an available predictor: P and N here were obtained using an extra whole dense flow at each Student state.

## EVIDENCE: local errors and source identity

Source04 is canonical `i2v_subject:4`, original official metadata row876, image type animal, prompt `the otter is standing in the water`, seed **46**. The actual image bytes hash to `64bca1e56a73f87957cefa96ffe7801db9e80506be015da23fcaec1655042f65`, matching RUN/pair/canonical and both metric reports. The prompt-only object digest and full input-prompt digest are different defined objects in the driver; they are not interchangeable hashes. Its role is `EXPOSED_AFFINE54_CALIBRATION`, with bank RUN SHA `fd38f0c148a9284da26adf35bc43c06e720ecb89b00c35500e5b4d375c32dc81`.

All160 layer×step sites exist. Raw FFN absolute Student-prefix operator SSE is smaller at **158**, equal at **1**, larger at **1**; real native writeback has the same count. This independently qualifies the earlier two-source 159/1 pattern rather than copying that count into this source.

| Local quantity | Smaller / equal / larger | Minimum / median / maximum Student-to-Teacher operator-SSE ratio | Ratio of summed SSE |
| --- | --- | --- | --- |
| Raw FFN | 158 / 1 / 1 | 0.189282 / 0.479002 / 1.001553 | 0.521289 |
| Native writeback | 158 / 1 / 1 | 0.191263 / 0.482984 / 1.044047 | 0.475568 |

The equal site is step0/block0, where actual input difference is zero. The larger site is step1/block0: raw teacher/student operator SSE3,529,349.659893/3,534,830.432722; native writeback26,369.204631/27,530.679650. Its post-Q1 input difference SSE10,758.348636 / teacher energy365,155.280159 =2.94624%. Counts by step are39/39/40/40 smaller. Every comparison uses the same teacher-reference energy stored within that site; the shrink observation also holds on absolute SSE and is not a denominator artifact. It does not by itself establish Student-output norm collapse, same-input numerical equivalence, or smaller transported error.

The native-minus-ideal next energies are9.861686/7.728931/4.970595/0.673625 for steps0–3. The corresponding ideal-update defect **vector** energies are9.880397/7.672100/4.824542/4.107889. Their inequality follows from the cross of ideal output error with its defect; defect energy is not the scalar energy difference. Actual step-pre/post dtype observations are BF16; flows are FP32. The initial common FP32→BF16 entry cast has equal teacher/student SSE5.612119 and zero differential cast; later entry casts have zero cast difference.

## EVIDENCE: the motion/detail vector gives a competing visual mechanism

Each immutable report covers81 frames and80 transitions. The table reports stored all-frame/transition means; framewise mean/median/p95/max were independently recomputed for Laplacian energy, encoded mean luma, flow magnitude, paired RGB RMSE and edge ratio. Maximum summary discrepancy was2.23e−16 from ordinary floating arithmetic.

| Diagnostic | Dense teacher | Candidate student | Meaning at this scope |
| --- | --- | --- | --- |
| Downsampled Farneback magnitude, px | 0.268755 | 0.097577 | Ratio of these means0.363069 |
| Adjacent RGB MAE | 0.022685 | 0.022443 | Nearly unchanged temporal pixel variation |
| Absolute Laplacian energy | 0.085073 | 0.123504 | Higher high-frequency proxy, ratio of means1.451747 |
| Encoded BT709 weighted mean | 0.365553 | 0.366543 | Near-equal global encoded brightness |
| Motion-compensated encoded-luma MAE | 0.014548 | 0.013532 | Candidate intrinsic residual slightly lower |
| Forward/backward flow consistent area | 0.983119 | 0.992282 | Consistency is an estimator diagnostic |

The mean of the80 paired flow ratios is **0.427176**; it differs from the **0.363069 ratio of means**. The mean framewise candidate/reference edge ratio is **1.462182**, also distinct from the ratio of edge means. Candidate paired RGB RMSE mean/p95/max=0.108684/0.113594/0.114527; encoded-luma RMSE mean0.106489; signed encoded-luma drift mean+0.000989; adjacent-delta RGB RMSE0.068635; paired flow L2 difference0.317796px; paired compensated-luma residual difference0.021820.

An informative competing explanation is **reduced coherent displacement with retained or increased texture/flicker energy**, rather than a single scalar “blur” process. Similar adjacent RGB variation plus much lower estimated flow and higher Laplacian energy is compatible with stationary noise, altered texture, wrong motion, estimator failure, or a mixture. Higher flow consistency and lower intrinsic compensated residual can occur when a frame sequence moves less; they do not certify better dynamics. An unqualified positive smoothing-kernel explanation is insufficient for the full vector. Conversely, an edge increase alone does not prove noise or sharper true detail. Global encoded-luma agreement also leaves spatial brightness redistribution open.

I read the actual evaluator's uint8 gray flow conversion, Laplacian and encoded-luma definitions, and revisited [OpenCV's primary Farneback API documentation](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html): its input and window/poly settings describe an image-motion estimator, not a semantic or perceptual acceptance metric. I also read [VBench-I2V's original dimension definitions](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md): image-subject/background consistency uses image/video feature comparison, while camera control is adherence to the actual prompt. Thus `i2v_subject` is a benchmark input/evaluation stratum, not a mechanistic statement that only a subject moves. No new literature mechanism or formal novelty claim is made by this focused audit.

Official VBench-I2V, VBench2 and human full-video review are all NOT_RUN, and both reports set quality_accepted=false. I did not inspect a contact sheet or full video in this review. “Motion freezing” and “noise” remain hypotheses, not direct visual findings.

## Counterprediction and cheapest next discriminator

The strong deterministic hypothesis “subject sources have the Source0 global cross sign” is contradicted by Source04 at its actual seed46. A weaker category-conditioned distribution remains OPEN. Source0/Source04 also differ in image, seed42/46 and bank-fit membership; their160 nested sites do not supply160 independent source replications. Source04 therefore cannot identify an effect of calibration, category, or seed separately.

Two predictions now separate useful mechanisms:

1. **Semantic-class sign mechanism:** the sign is stable within a genuinely repeated source across seeds and predominantly separates predeclared classes after image/seed exposure is accounted for.
2. **Finite subspace-mixture mechanism:** the total sign can change with seed or image while individual signed components retain another pattern; the balance of DC/static-low against dynamic/high interaction predicts the global sign directly through the identity.

The cheapest immediate check is scalar-only and completed here: tabulate C, its four summands and C minus the two temporal-DC-low summands for every already-paid source; preserve the outcome as a vector. The registered new-input crossed-seed observations, if executed by the coordinator, can then test within-source sign stability before any category claim. Six source×seed pairs from four images are four source units with nested seed replication, not six independent input contents. Repeated seeds in only two strata do not separately estimate a seed effect in all four strata. Broad video-function transfer still requires the deliberately unexposed source identities and actual quality observations.

For FFN compute, this audit adds no savings measurement. Observation cost includes two complete trajectories, four additional dense whole-network flows and local shadows; an eventual signed-functional objective must predict necessary output directions from paid-or-cheap available information and reduce actual W1/W2/QDQ work. Posthoc P/N/band vectors currently clarify the trajectory problem and define falsifiers; their oracle construction is not an online cost-free correction.

## Provenance and exact frozen identities

Private asset paths and hardware identifiers are intentionally omitted. Artifact names below are relative to HF derived `e0_multisource_directional_20261026/attempt1/dc_source04/`.

| Artifact | SHA256 |
| --- | --- |
| RUN.json | `33ca1e2dcfb6c23cbf4a4cbbdfd1829a17b416540089344ae654164dc9339655` |
| NATIVE_RUN.json | `e3b8de623e7c7c63d18f7b80e3e49e42ba51cf1d1fb7193742c8ef447fdb81ac` |
| TRACE.json | `61a092e4cfed983f1e4201a00a7c471c43f69c5a7721992d981e436febeb0cdd` |
| DISJOINT_BANDS.json | `ce1a05922b57242589d46935718d7677faeef3f5013a358da39d7b02b551221d` |
| BUNDLE.json | `226fcc51141142ca3c199d4d5fe0e8d4fbb9b98191352fd50aa6680128facd94` |
| STRICT_BUNDLE.json | `d6b2a47be61f691467199de52c22adc683711344ee7efbd7f32753934669e6d1` |
| teacher_metrics.json | `a130938b2e0e0a3e4b587d412bf472f27b54c1a693d425fda2cb8ef7f481cebe` |
| student_metrics.json | `a983515a02ef30a48d1e78e021d2c2b05f57085daa457c9ad147197b7a2f6dff` |
| teacher.mp4, bytes hashed only | `8f65f93261e37cb3786cb6bef998f0bdf9792be1131d89496f9434ac2830b1ef` |
| student.mp4, bytes hashed only | `e5a990fd7a40c8762fb9edbd08b7f67d5ba730e4d438e66e3aa16f8026ae4225` |
| flow_s0.json | `ca4abcfadfe9816772a0b6d75574da1975e064421b3a239977103d269ac4198b` |
| flow_s1.json | `836d8564bd68ead3fc8263a5d2b2ef2f6176bc7bfa04a5e25afd1bbb2bb259bb` |
| flow_s2.json | `55ad2fe640d7af10528e0f0cefb4af605e3ddbeff70e7ca5dc915393fbca1688` |
| flow_s3.json | `be7cb181caa702322b283bd134317d81dd30eba2dff7ff6ca50755df9825819f` |
| OBSERVATION.json | `a51689cdf02ac469d464c06b23c01af142bab55fcafadb0fa5362beb61f33524` |

Frozen executed protocol SHA: `931b8162f8379193697a00899b617f3851f70ef0182ca7126c2ea4511a7f4e77`. All26 driver and27 native source snapshot entries match that protocol; the corresponding current source files also match. RUN→NATIVE_RUN, RUN/NATIVE_RUN→TRACE, RUN→bands, strict→core bundle, every core metric/video/reference binding and source/model identity were independently checked. The read-only strict JSON/hash gate returns no errors. Both candidate and reference model digests were rebuilt from native asset/bank identity, not merely copied between reports. The dense reference digest is `c2396ae98e79888dd80c19eaa2a1920b501ef5f67b2e796d1253cc0d52c82329`; candidate `c0c895e1593c130d24b65b8ee0bf2b278789015d4c53e4665b6e98fd8863a101`.

Historical original dense receipt SHA `5a1e9a9523a9be8452925edb62e622d21f80aad5b6300cde7303adc6c15d0bae` matches RUN's frozen reference. All four teacher latent SHA records and the complete teacher MP4 SHA equal that original passive record. Canonical24 SHA `3c24f59ca5960c3c3ca9bc69fcb91ccccaf6c01644ede7361877e8733aa4efcb`; previous Source0 band report `research/20261026/E0_DISJOINT_SIGNED_BANDS.json` SHA `a5390c168d588886d3d6943871ec1a8d88f06d6a573ef0ed2b7939a071ba776c`.

Source implementations personally read: `run_e0_multisource_directional.py` SHA `dd23afd15abd555559b6520090bcb58980f2cd71fca509726b5b47c86749920a`; `e0_whole_flow_counterfactual.py` SHA `0592cfe2a3b92b64b36ecc6e12b99ca653d5939f20305ae02b5ce35d7115306b`; `e0_disjoint_native_bands.py` SHA `1c459aa84082ec4a98ad77333cb072ce2f226780450be3141e31717757c6c28d`; evaluator SHA `0c6ab066674adb2a04097978ced52abcd7cbf1125918b74d59cf976a56ffec66`; strict gate SHA `59d6596cff43cc6197d4b1ab76eb90e9ca92855901b6d3394b7eabc257c9a6d3`.

Observed accounting is160 completed sites, two actual videos, two main trajectories, four extra dense flows, zero fits/training/official-quality calls. Runtime records4,800 linear and4,800 activation-QDQ calls; observer records dense main/shadow GEMMs320/320, candidate main/shadow480/480, native_QDQ1,280 and continuous GEMMs960. These counters have their own scopes and should not be blindly summed. Shared diagnostic generation wall280.031604s, FFN primary-plus-shadow boundary30.911309s, setup206.947269s and sampled physical peak76,847MiB describe this observation scope, not deployment latency or independent per-clip charges. Both metric reports share the same cost-deduplication key.

Graph anchoring: actual Q1/FFN `F04→F06→F08→F09`, writeback `F12/F13`, scheduler `G21`, terminal decode/video `D09`. Signed band→video-function interpretation remains **posthoc/conditional OPEN**, with no new dense-execution edge or method-support label.

Preserved v11 identities checked at completion: source-history MD `41c115e66c6148e65f1b8713be4bfb524bdd2fa1297bdc49497d27882094bfc5`; official-four audit JSON `0575e63303620736ab7b8bca38237d1cdb80491e4eab32249ec9a10865e4bb6b`. Remaining gaps are original tensor-level reduction re-execution, semantic masks, independent seeds/content effects and perceptual/full-video quality; the first was deliberately outside this scalar audit.
