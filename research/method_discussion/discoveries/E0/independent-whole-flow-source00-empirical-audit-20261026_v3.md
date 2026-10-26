# Source0: the native dense step expands the observed displacement while candidate injection partly cancels

Independent `whole_flow_native_review` agent, 2026-10-26. **New measured object:** on source0/seed42, the complete dense model evaluated at each actual Student latent produces a native one-step state separation about **5.97×, 4.98× and 7.72×** the incoming separation energy at steps1–3. Candidate effects have negative signed cross terms with that dense response, yet their net contribution relative to the dense-at-Student update remains positive. This finite decomposition explains coexistence of smaller local Student-prefix FFN operator errors and growing rollout divergence without assigning a universal cause.

The strongest competing interpretation is that the candidate has already selected harmful displaced states through its earlier errors; the observed dense response then transports those particular offsets. The current observations establish the response at these offsets. They do not separate earliest creation from later transport, establish a local/global Jacobian, or test a corrected rollout.

Execution boundary: this review independently reads the **completed source0** native artifacts and recomputes CPU algebra. It runs no model/GPU, fits no policy, creates no video and changes no old receipt. Source12 is still pending in this record; one exposed source/seed is the empirical population here.

## EVIDENCE: complete-state and identity checks

I independently loaded all four safetensors bundles, each containing eight full `(16,21,58,104)` tensors. I checked every bundle SHA against its native RUN binding, its tensor inventory/dtypes/shapes, all three native flow identities, and main before/after identities against both record metadata and TRACE boundaries. Each `flow_s*.json` exactly equaled its corresponding RUN record. The final OBSERVATION and TRACE matched their RUN SHAs. The complete teacher/student trajectory entries were exactly equal to the prior successful v3 RUN, preserving all eight original source0 latent hashes.

I independently reproduced all **12** main/shadow native scheduler updates in CPU FP32 with their original operation order followed by BF16 cast, obtaining exact equality with saved outputs. Step0 had equal dense/Student initial states, equal original dense and dense-at-Student flows, and equal dense native after-states. Full-coordinate recomputation of the flow and native-map scalars matched RUN within `1e-9` relative / `1e-14` absolute tolerance. Both vector recomposition residuals were zero.

The independent projection calculation used per-channel centering and the explicit analytic 2x2 inverse formula, rather than calling the analyzer's reorthogonalization implementation. All four correction families agreed to `1e-10` relative / absolute tolerance. Normalized centered-basis Gram condition numbers at steps0–3 were **12.08, 11.81, 8.33, 4.04**; this source's reported rank2 is well separated from the earlier tiny-direction truncation counterexample. CPU checks printed `torch.cuda.is_initialized()==False`.

## Native finite response and signed candidate effect

For actual native updates `T`, define

```
delta = zS-zD
P = T_D(zS)-T_D(zD)
N = T_S(zS)-T_D(zS)
delta_next_actual = P+N
||delta_next_actual||² = ||P||² + ||N||² + 2<P,N>.
```

| Step | Incoming `||delta||²` | Dense-at-Student `||P||²` | Dense energy gain | Candidate `||N||²` | Signed cross `2<P,N>` | Candidate net `||N||²+2<P,N>` | Actual next energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | undefined | 1731.584 | 0 | 1731.584 | 1731.584 |
| 1 | 1731.584 | 10333.922 | 5.967903 | 1011.935 | −220.532 | 791.403 | 11125.326 |
| 2 | 11125.326 | 55385.827 | 4.978356 | 2921.478 | −1914.644 | 1006.835 | 56392.662 |
| 3 | 56392.662 | 435349.265 | 7.719963 | 19408.791 | −227.407 | 19181.384 | 454530.649 |

The corresponding ideal dense gains are 5.963226, 4.977953 and 7.719771. Actual-minus-ideal next energies are 9.6980, 8.6335, 3.8619 and 13.2748 for steps0–3. Thus the observed expansion exists in both the ideal and actual scheduler representations; this does not remove internal QDQ/BF16 effects from the dense or candidate flow functions.

At this source's fixed observed states, substituting dense for the current whole candidate step would reduce the next squared separation by the reported positive net. Its result would still be displaced and expanded. This is a one-step finite intervention, not an executed dense-recovery sequence. The gains cannot be multiplied to predict such a sequence: a switch changes later states, so later measured gains would no longer refer to its actual inputs.

The negative cross terms show partial cancellation. They do not make the candidate effect beneficial overall here: the self energy is larger and the net remains positive. Conversely, the dense-response norm is not a percentage of the original failure caused by the dense model. The reference states were themselves created by the candidate trajectory, and another intermediate reference would induce a different decomposition.

### A finite gain does not identify endpoint Jacobians

An exact smooth scalar counterexample makes this boundary concrete. Let

```
T_D(z)=0.5*z + 2.5*(3*z²-2*z³).
```

Then `T_D(0)=0`, `T_D(1)=3`, so the finite energy gain between 0 and 1 is9. Yet `T_D'(0)=T_D'(1)=.5`. Both endpoint derivatives are locally contractive even though this finite pair expands. A native quantized map additionally need not be differentiable. The measured Wan result therefore answers a finite state-response question, leaving local sensitivity, global stability and earliest-layer origin separate.

## Cheap correction coordinates retain substantial unexplained ideal state error

For `y=delta/h-[vS(zS)-vD(zD)]`, the ideal error after an additive flow correction `u` is `h*(y-u)`. The largest evaluated family has 16 channel constants and **two global coefficients** multiplying current `vS` and `zS`. It is an exposed-state oracle with teacher information, not a learned/deployed controller.

| Step | Flow scaling only: remaining ideal SSE fraction | Channel bias only | Bias + global flow scale | Bias + global flow/latent scales |
| --- | ---: | ---: | ---: | ---: |
| 0 | .934556 | .760591 | .730767 | .701035 |
| 1 | .936404 | .717321 | .695557 | .667302 |
| 2 | .953856 | .720226 | .707751 | .693299 |
| 3 | .951856 | .737031 | .723630 | .711391 |

Channel constants account for appreciably more of this ideal Euclidean target than one global flow scale. Most target energy remains outside the largest evaluated space. These are FP64 numerical oracle residual estimates, with no certified lower-bound or strict family-exclusion claim. They neither exclude richer channel/spatial directions nor prove that a partial correction cannot help video quality. A real changed-flow BF16 update and its later rollout remain untested.

The inexpensive next discrimination is the already-running independent source12 capture: does the finite dense response again expand and does the same simple direction family leave substantial residual? Replication would support a common descriptive mechanism across the two exposed sources, while transfer to new sources/seeds and corrected full-video function would remain separate. For FFN compute, this identifies useful state-sensitive targets for later approximation or cheap calibration research; the counterfactual observer itself pays extra dense W1/W2 work and is not a speedup.

## Exact provenance and scope

HF-derived root: `e0_whole_flow_counterfactual_20261026/attempt1/dc_source00`.

| Object | SHA256 |
| --- | --- |
| Protocol | `cebee73f6acd646de6b8eddcc03bb7a3982fca652a273cf77f6d5e259ab10b2c` |
| Native RUN | `3e915a01255ddd9a1fc5188a98cc9d1be20b4465da5fb5acb4ce9edb68fb4ae5` |
| TRACE | `791f08648063e91392487f7b77a9f83590bddd80061be4a9aba20282352b0dc8` |
| OBSERVATION | `1b7d9de6406892457393fd29cdada55ae25f0c250831c76e56ce637f4b09e3d7` |
| `flow_s0.safetensors` | `97dc35c29b05e5a40659a34ab45998e693d8eb4dae7c3b9b37fee026958030cf` |
| `flow_s1.safetensors` | `b3c1e87e6b7aaa9f54e56c04c56f43973606c360b5725ac0bf3a8268e45b799a` |
| `flow_s2.safetensors` | `1a72168b7beb7f8e7ff044f494d54de0ae4e279b420f94e046d88e479d6bb0fe` |
| `flow_s3.safetensors` | `707e748a28c73294161f7453c19fcd3a811f8348ca8dd28a31a4488694051196` |
| Coordinator CPU analysis `source00_cpu_analysis.json` | `4d21b9d355bafc045f99b789bdd470f82a9a1658a2f3bd2d35d02e40b429aed8` |

The native source0 RUN records 160 paired sites, 4800 native runtime target linear/QDQ calls each, 640 FFN transfers, zero cache misses, four extra whole dense flows of400 calls each, zero videos/training/fits/official scoring and `quality_accepted=false`. I read these counters and checked the four per-step400-call records; they are instrument accounting, not an independently profiled kernel trace. Observation wall293.936s plus setup381.007s includes paired paths, full dense counterfactuals and local shadows; it is not deployment latency. Physical sampling peaked at75285MiB with no recorded isolation violations. I did not independently query the live GPU or establish post-process release.

Graph evidence is **P23/P24/G02**. This review supports source0's actual finite response and projection observations, preserving **P25** full-video quality as unaccepted. The previous source/math reviews and their primary documentation searches remain immutable; this empirical pass used actual local artifacts and independent CPU arithmetic, with no new literature/novelty claim or new web search.
