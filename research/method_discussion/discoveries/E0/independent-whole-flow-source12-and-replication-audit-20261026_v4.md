# Two-source finite expansion replicates; candidate cross geometry changes sign

Independent `whole_flow_native_review` agent, 2026-10-26. **New measured structure:** source12/seed54 also has expansive native dense responses at all three observed nonzero Student offsets, with one-step energy gains **5.0181, 3.6258 and 6.6580**. This replicates the finite expansion found on source0. At the same time, its native candidate/dense-response cross terms are **positive**, whereas source0's are negative. Shared expansion and source-dependent signed interaction coexist.

The scientific comparison is between whole-network response at the actual candidate-created offset and the current candidate effect at that same state. It measures neither a global Jacobian nor the earliest cause of the offset. The inexpensive empirical replication is now complete for these two exposed source/seed pairs; future changes to function representation or correction objectives can use both cases rather than assuming one universal cross-term sign.

This record adds an independent CPU audit of source12 to the preserved source0 v3 review. It runs no model/GPU, creates no video and fits/deploys no policy. Formal video acceptance remains false.

## EVIDENCE: actual source12 artifacts independently checked

I independently read all four full-state safetensors bundles and their JSON/RUN bindings. The 32 tensors have the declared inventory, dtypes and `(16,21,58,104)` shape. Their file SHAs, all three full-flow identities, before-state identities and TRACE after-state identities match. Every `flow_s*.json` equals the corresponding RUN record. OBSERVATION/TRACE match their RUN hashes. All eight teacher/student trajectory entries equal the original successful v3 source12 record exactly.

All12 main/shadow native scheduler updates were independently recomputed using the original FP32 operations and BF16 output cast, giving exact saved-output equality. Step0's equal inputs, equal dense flows and equal native dense after-states passed. Independent full-coordinate ideal/native geometry matched the native scalars with `1e-9` relative / `1e-14` absolute tolerances, and both signed recomposition residuals were zero.

The original complete-study ledger now contains exactly the two expected successful pair IDs in order. Native per-source counters are4800 target linear and4800 QDQ calls,640 decoded FFN transfers and zero cache misses; each extra complete dense flow accounts for400 linear/QDQ calls and80 FFN transfers. The source12 trace contains160 paired sites. These are checked producer counters, not an independent GPU profiler trace.

I independently computed the projection families through an explicit analytic 2x2 inverse, without invoking the analyzer's projection routine, and compared them with the complete CPU result. The normalized centered-basis Gram condition numbers are **19.07,17.55,12.46,6.19** over steps0–3, comfortably separated from the earlier numerical-rank counterexample. The CPU process printed `torch.cuda.is_initialized()==False`.

## Actual native decomposition and replication

Use `P=T_D(zS)-T_D(zD)`, `N=T_S(zS)-T_D(zS)` and `delta=zS-zD`. The native identity is `delta_next=P+N`.

| Source12 step | Incoming `||delta||²` | Dense `||P||²` | Energy gain | Candidate `||N||²` | Cross `2<P,N>` | Net current candidate effect | Actual next energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | undefined | 474.421 | 0 | 474.421 | 474.421 |
| 1 | 474.421 | 2380.710 | 5.018134 | 782.744 | **119.694** | 902.437 | 3283.147 |
| 2 | 3283.147 | 11904.195 | 3.625849 | 2744.837 | **2629.576** | 5374.413 | 17278.609 |
| 3 | 17278.609 | 115041.119 | 6.658008 | 17467.964 | **4938.870** | 22406.834 | 137447.953 |

The ideal dense gains are5.001499,3.624455,6.657787. Actual-minus-ideal next energies are9.8511,7.8515,4.3489,5.4353 at steps0–3. Expansion is present in both representations. Internal quantization/rounding remains part of each native flow function; these small outer update differences do not isolate its contribution.

The source0/source12 native gain pairs are:

| Nonzero-offset step | Source0 / seed42 | Source12 / seed54 |
| --- | ---: | ---: |
| 1 | 5.967903 | 5.018134 |
| 2 | 4.978356 | 3.625849 |
| 3 | 7.719963 | 6.658008 |

Both sources expand at every sampled nonzero offset, with positive net current candidate effect relative to the whole dense counterfactual. The measured cross signs differ: source0 partially cancels the dense response, source12 reinforces it. A single-source cancellation narrative therefore does not cover both observations.

### Why raw flow cancellation does not fix the state cross sign

Let `R=vD(zS)-vD(zD)`, `e=vS(zS)-vD(zS)` and `P_ideal=delta-hR`. The ideal candidate cross is

```
-2h<P_ideal,e> = -2h<delta,e> + 2h²<R,e>.
```

Source12 has negative `<R,e>` at all later steps, just as source0 does. Nevertheless its ideal cross is positive because the first term dominates:

| Source12 step | `-2h<delta,e>` | `2h²<R,e>` | Sum: ideal cross |
| --- | ---: | ---: | ---: |
| 1 | 464.208 | −336.796 | 127.412 |
| 2 | 2826.087 | −191.323 | 2634.764 |
| 3 | 11914.837 | −6971.822 | 4943.015 |

Flow-error cancellation and next-state cancellation are thus distinct signed geometries. The raw flow cosines between `R,e` are approximately−.1876,−.0289,−.1160 here; their sign alone does not answer the update question.

These decompositions have a clear finite causal reference: substituting dense for the current candidate step at the **observed** Student state. They do not assign fractions of the original video failure to model versus state response. The current Student states were produced by earlier candidate actions. A dense-recovery rollout would visit different later states, so multiplying the measured gains does not predict that rollout. The smooth endpoint-Jacobian counterexample in the source0 v3 report continues to apply.

## Numerical oracle directions: the common limit and the source difference

For the fixed ideal target `y=delta/h-[vS(zS)-vD(zD)]`, the largest family consists of16 channel constants and two global scales multiplying current `vS,zS`. It uses offline teacher information and permits unrestricted real coefficients at each original state.

| Source12 step | Flow scale only: remaining ideal SSE fraction | Channel bias only | Bias + flow scale | Bias + flow/latent scales |
| --- | ---: | ---: | ---: | ---: |
| 0 | .991042 | .638402 | .634856 | .601992 |
| 1 | .992617 | .670399 | .664405 | .605784 |
| 2 | .980936 | .704105 | .693032 | .618754 |
| 3 | .979034 | .733709 | .721232 | .656170 |

Source12's one-global-flow-scale oracle leaves even more of the target than source0's. Its channel/latent family covers somewhat more target energy, while still leaving roughly60–66% over all four states. Across both sources, a substantial majority remains outside this18-direction family. These are numerically stable **oracle residual estimates**, not certified lower bounds, corrected BF16 outcomes or impossibility results for video repair. Richer channelwise, spatial or functional directions are distinct families.

This suggests a constructive distinction: systematic channel means are a real component of the ideal state error, and a scalar output-gain explanation leaves most of it untouched. Subsequent representation research can compare cheap structured coordinates with downstream-valued FFN innovations. Its decisive observable should connect any recovered component to actual four-step/full-video function, since Euclidean pairing alone does not establish semantic or perceptual benefit.

## Exact provenance, accounting and public metadata

HF-derived relative root: `e0_whole_flow_counterfactual_20261026/attempt1/dc_source12`.

| Object | SHA256 |
| --- | --- |
| Frozen protocol | `cebee73f6acd646de6b8eddcc03bb7a3982fca652a273cf77f6d5e259ab10b2c` |
| Completed two-source STUDY | `0ca9a2919e9c7180dd805d3f5cbdb6fb811bb0bd951e3af17637de50e0186124` |
| Source12 native RUN | `dd2e346a9ba1684fd74f7946bf94595a2cb830f0554f277e1dd480f688f2830d` |
| Source12 TRACE | `448b26276bf6cec6c49aa18968503900cfae46910d7d438ac9a4881be6b417e6` |
| Source12 OBSERVATION | `c8734cf3917d5072e9ba93b6f9b4b0793585da87c38656b5830a691948f9625a` |
| `flow_s0.safetensors` | `d64b01b5151e6e251d93f030d0cd3e27031c699169ba8ffbfd5a5448cff2eae2` |
| `flow_s1.safetensors` | `90c78cf513cc63802358212ce15c0bdb0386c883e27fa48dbb8e92709935e1ca` |
| `flow_s2.safetensors` | `5b0dc9e1ba71fe791e2fcc8c3b42b8f82552b3fc8ac691623611192a1f46504a` |
| `flow_s3.safetensors` | `da6addd12dc062293fd6d41692cdf1e6329d8e8539dc5f9a340d7e75d568566f` |
| Complete CPU result snapshot inspected | `d4c1db3a644e7972bdc43e551da2ea59e28508b994730f760396f480a6c99f3a` |

The complete CPU result is `research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.json`, from analyzer SHA `428df2744652226ce5a61646c38ceb4623f09342062437683848ad7a43511ea6`. I checked its two-source COMPLETE status, matching STUDY hash and source12 projection figures; the raw native artifacts above are the primary evidence.

Source12 observation wall244.557s and setup321.993s include research shadows, complete dense counterfactuals and tensor observations. The sampled physical peak is75285MiB with no recorded isolation violations. This is instrument accounting under the existing one-card plan, not deployment latency or independently verified GPU release. The complete fixed study has four latent trajectories/eight extra complete dense flows, preserving16 historical latent hashes and producing zero new videos. Native quality flags remain false.

This public record uses relative evidence locations and hashes. The inspected public CPU result contains no private worker-name strings; no routing/node identity is copied here. These data are two exposed images/seeds and one existing candidate, not an independent video population or quality generalization. Their millions of tensor coordinates do not supply millions of independent examples.

Graph anchors are **P23/P24/G02**, with candidate repair directions still **posthoc/conditional** and **P25** formal quality unaccepted. Prior reviews retain their original scope and bytes. This empirical audit used actual local receipts/tensors and CPU arithmetic, with no new web search or literature novelty claim.
