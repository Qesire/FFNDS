# Native FFN secant: full result, descriptive analysis

The completed job reported SUCCESS; the approved postprocessor checked the exact reviewed scope, receipt and ledger hashes, 800 source/expert/layer pairs, 1,600 arm evaluations and 640 weight reads. It verified all 40 layers, 10 sources, both experts and both panels. Outputs are under `secant_full_r2/summary`; `PAIR_ROWS.csv` SHA-256 is `f14e69df82baf3a4f8b8e3884b8a8100c370e824782d652efc3e82004c545639`. Each expert uses its own observed same-expert step pair: high 0→1, low 2→3. Calculations describe a smooth F64 lift and native BF16 diagnostics; numerical errors are not certified.

## Quadrature risk on observed endpoints

Entries below are the **ratio of summed squared errors to summed squared smooth FFN increment energy**, aggregated over 40 layers and 10 sources in each panel. Thus large target energies carry more weight. The grey line in the separate figure instead shows the median of the ten per-source ratios at each layer; those summaries must not be interchanged. “Native discrepancy” is the Dq native-BF16-vs-smooth increment SSE divided by the same smooth target energy, a diagnostic, not a proven lower bound.

| Expert | Panel | Arm | Left | Midpoint | GL2 | Dq native discrepancy |
|---|---|---|---:|---:|---:|---:|
| high | uniform | Dq | 0.927851 | 0.036553 | 0.00071593 | 0.00001235 |
| high | uniform | Oq | 0.928620 | 0.036580 | 0.00071625 | — |
| high | tubelet | Dq | 0.931140 | 0.037382 | 0.00076021 | 0.00001210 |
| high | tubelet | Oq | 0.931930 | 0.037409 | 0.00076065 | — |
| low | uniform | Dq | 0.327542 | 0.004677 | 0.00011184 | 0.00003407 |
| low | uniform | Oq | 0.327717 | 0.004676 | 0.00011195 | — |
| low | tubelet | Dq | 0.326100 | 0.004716 | 0.00009693 | 0.00003358 |
| low | tubelet | Oq | 0.326286 | 0.004714 | 0.00009699 | — |

GL2 has smaller smooth-target SSE than midpoint in all 400 Dq source/layer observations in each of the four panels; the same is true for the fixed midgate-weighted errors. This is descriptive of these observations, not an independence-based statistical confidence statement. Uniform and tubelet selections differ, so they are kept separate. These are squared ratios: for example high/uniform GL2 corresponds to about 2.68% relative L2 error of the aggregate smooth increment, not 0.0716% relative L2.

The full data substantially qualify the preview: in the low expert, GL2 smooth-error SSE is smaller than the native-to-smooth discrepancy SSE in 210/400 uniform and 187/400 tubelet source/layer cases. The native discrepancy is therefore practically material to interpreting the approximant, even though the aggregate GL2 risk exceeds it. The actual GL2-to-native error is not measured directly by those two quantities; only triangle bounds follow without their error-vector cross term. Oq and Dq quadrature ratios are close in this aggregation, which does not establish Oq/Dq function or model equivalence.

Forty-layer Dq ratio-of-sums ranges for GL2 are 0.000136–0.001386 high/uniform, 0.000145–0.001408 high/tubelet, 0.000000295–0.001040 low/uniform, and 0.000000150–0.000665 low/tubelet. Individual source/layer risk can exceed the aggregate: low/uniform reaches 0.004923 (source 08, layer 16). The reference path inside each pair is the virtual straight chord between captured endpoint inputs, not a measured intermediate denoising trajectory.

## Fixed midgate-weighted risk and gate decomposition

For Dq the analogous ratios using the fixed midpoint gate are:

| Expert | Panel | Left | Midpoint | GL2 |
|---|---|---:|---:|---:|
| high | uniform | 0.633759 | 0.029455 | 0.00069879 |
| high | tubelet | 0.611360 | 0.028627 | 0.00069055 |
| low | uniform | 0.431668 | 0.007250 | 0.00071065 |
| low | tubelet | 0.333422 | 0.008008 | 0.00046972 |

The native lifted gate decomposition uses A=Δg⊙(Y₁+Y₀)/2, B=(g₁+g₀)/2⊙ΔY and Δ(g⊙Y)=A+B. The following entries are sums of ||A||², ||B||², and 2⟨A,B⟩ divided by the sum of ||A+B||²; the signed cross term must not be read as an independent positive variance share.

| Expert | Panel | Gate-change A | State-change B | Twice cross |
|---|---|---:|---:|---:|
| high | uniform | 0.01131 | 1.00377 | −0.01508 |
| high | tubelet | 0.01017 | 1.00677 | −0.01694 |
| low | uniform | 0.00676 | 0.99382 | −0.00057 |
| low | tubelet | 0.00615 | 0.98649 | 0.00737 |

These are aggregate energy identities, with small signed floating-point closure residuals; they do not prove gate changes are negligible for every source, token or layer. Products use F64-lifted native outputs and gates rather than native BF16 residual arithmetic.

## Balanced tubelet space–time decomposition

The 21-time × 16-space balanced endpoint decomposition is computed per source and endpoint before energy aggregation. Values are shares of total energy including the mean; the last three terms also sum to one after excluding the mean as shown in parentheses. Spatial sites are source-specific and are not aligned across videos.

| Expert / lifted quantity | Mean | Time | Space | Time × space interaction |
|---|---:|---:|---:|---:|
| high / Y | 0.32366 | 0.10540 (0.15584) | 0.12592 (0.18617) | 0.44503 (0.65799) |
| high / gated Y | 0.48065 | 0.06339 (0.12205) | 0.08404 (0.16183) | 0.37192 (0.71612) |
| low / Y | 0.31406 | 0.11388 (0.16602) | 0.14545 (0.21205) | 0.42661 (0.62193) |
| low / gated Y | 0.68553 | 0.05436 (0.17287) | 0.05383 (0.17119) | 0.20628 (0.65594) |

On these sampled tubelets, interaction accounts for 62–72% of centered energy, depending on expert and gating. This argues against interpreting the observed endpoint output as predominantly a sum of a global temporal component and a source-wide spatial component; it is not a statement about all tokens, rank, or generation quality. Source choice, tubelet selection and energy weighting matter.

## Interpretation and limits

The result identifies a difference between a left tangent and chord-averaged directional behavior on the measured endpoints. It does not establish global low polynomial degree, a reusable low-rank Jacobian, a speedup, or the quality of generated video under replacement. The proposed numerical identities, counterexamples and candidate cost accounting are in `PREVIEW_QUADRATURE_MATHEMATICAL_AUDIT.md`. A full dense midpoint/GL2 computation can require as much as or more than an FFN, and caching has significant memory and bandwidth cost. Source observations share model and sampling conditions and are not independent tokens; no confidence interval or performance claim is implied.
