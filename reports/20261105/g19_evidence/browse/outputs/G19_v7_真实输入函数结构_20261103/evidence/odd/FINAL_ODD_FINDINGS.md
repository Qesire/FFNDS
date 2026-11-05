# Fixed odd-backbone test on native chords

Job 129217 produced 800 paired source/expert/layer records, two arms and two separate panels. The 800 capture SHA pairs, per-panel row/time counts and aggregated direct/midgate target energies match the authenticated secant ledger; this scalar check does not itself re-read token IDs from the NPZ. Results use F64-lifted BF16 weights, gate and native endpoints, on Dq trajectory states. Oq is evaluated on those same Dq states. This is descriptive, with numerical status UNASSESSED.

Each row below pools all 400 source-layer pairs for a Dq expert/panel; energy ratios are ratios of sums, not medians of pair ratios. `R` is the direct squared error of the fixed odd increment divided by the true lifted chord energy; `O`, `E` and `C` are the odd, even and signed twice-cross energies divided by that same chord energy. `cos(O,E)` refers to the two output vectors after concatenating the finite panel, not to individual neurons.

| Expert | Panel | R | O | E | C | cos(O,E) | κ | Global free-gain oracle residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| High | uniform | 2.076 | 3.199 | 2.076 | -4.275 | -0.829 | 3.230 | 0.648 |
| High | tubelet | 2.067 | 3.185 | 2.067 | -4.252 | -0.829 | 3.222 | 0.648 |
| Low | uniform | 1.097 | 1.802 | 1.097 | -1.899 | -0.675 | 2.390 | 0.597 |
| Low | tubelet | 1.082 | 1.794 | 1.082 | -1.875 | -0.673 | 2.379 | 0.591 |

The high/low Dq uniform median of individual pair R is 1.776/0.729; the pooled values in the table are different estimands. Oq on the same Dq states gives pooled uniform R of 2.079/1.096, close to Dq. The negative cross shows opposition of odd and even output components; it does not establish neuron-pair cancellation or explain why hidden structure is attenuated by W2.

The global free-gain oracle is `1−cos(odd,target)^2`, using a single unrestricted real gain for the concatenated group, not a fitted deployable model. Even allowing a separate unrestricted gain for each source-layer pair leaves the following output residual fractions (min / median / max across 400 pairs):

| Expert | Panel | Pair-specific free-gain oracle residual |
|---|---|---|
| High | uniform | 0.492 / 0.611 / 0.892 |
| High | tubelet | 0.501 / 0.616 / 0.892 |
| Low | uniform | 0.201 / 0.499 / 0.863 |
| Low | tubelet | 0.205 / 0.498 / 0.883 |

Thus merely changing one scalar gain cannot completely correct the observed output direction, even when that scalar is an in-sample oracle per pair. This does not reject channel-dependent or input-dependent corrections.

For real native Dq endpoints, the F64-lifted increment prediction `Y0 + odd` evaluated against `Y1`, with endpoint gate `g1`, has pooled error/target energy 1.063 high uniform, 0.995 high tubelet, 0.162 low uniform and 0.172 low tubelet. The corresponding pair medians are 1.135, 1.162, 0.144 and 0.157. It is an endpoint diagnostic, not actual BF16 execution.

The sum-relative condition factor is `κ=(sqrt(Eodd)+sqrt(Eeven))/sqrt(Etarget)`. If each of the odd and even vectors is perturbed by at most ε in its own relative norm, the triangle inequality gives a total relative error bound κ ε. This is a worst-case upper bound, not measured or inevitable amplification. Closure and direct residual are retained separately; no strict floating-point certificate is claimed. Sources are the grouping units; independence of these ten sources is not established. Layers, paired rows and panels are not independent replications. The panels have distinct sampling rules.

Arithmetic source: ODD_DESCRIPTIVE_SUMMARY.json SHA-256 `6adab31cb2605a20457fddcb60659b226418c7460b25216f399bc8e4fe0641c1`, ODD_PAIR_SCALARS.csv SHA-256 `464d23d453fceefc5dd2a791d700d668b52ae8018720b416a09a061d447fa780`. This note awaits independent result review.
