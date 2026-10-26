# Fixed online directions expose repairable geometry; numerical rank controls the claim

Independent `whole_flow_native_review` agent, 2026-10-26. The scientific object is the portion of the **ideal one-step state discrepancy** spanned by directions already present at inference: the current Student flow, current Student latent and one constant direction per latent channel. This can distinguish a small gain/bias mechanism from missing task-dependent directions before constructing a repair policy.

This is a source/mathematics/CPU synthetic review of the posthoc analyzer. No new model, GPU, fitted parameter, corrected rollout or video is produced by this reviewer. The separate native fixed experiment and the user's continuing-run authorization remain the root coordinator's responsibility. The earlier native-source review remains unchanged.

## DISCOVERY: projection has an exact target, with a numerical-rank caveat

At a frozen observed state pair, write `delta=zS-zD`, `d=vS(zS)-vD(zD)` and `h>0`. Adding a flow correction `u` changes the ideal discrepancy to

```
delta_next(u) = delta-h*(d+u) = h*(y-u),  y=delta/h-d.
```

Consequently the exact-arithmetic oracle optimum over a specified linear space `U` is `h² ||y-Proj_U(y)||²`. The correction sign in the analyzer is correct. This object is conditional on the original state pair; optimizing it separately at each step does not constitute a corrected four-step rollout.

The implemented largest family contains **16 channel biases plus two global scalar coefficients** multiplying `vS` and `zS` in the actual latent shape. It does not contain independently chosen flow/latent scales per channel, nonlinear functions or spatial masks. The target depends on teacher information offline; online availability of the basis does not make the target or optimal coefficients available online.

Centering each channel over all of its spatial/temporal coordinates orthogonally removes the channel-constant space under the unweighted full-latent Euclidean metric. Thus the largest family's projection consists of channel-mean energy plus the projection of centered `y` onto centered `vS,zS`. The smaller families—flow only, channel bias only, and bias plus flow—are likewise computed with the appropriate uncentered or centered flow. This establishes the intended family nesting and correct use of the 2x2 diagnostic Gram.

### Independently reproduced rank counterexample and repair

The initial analyzer discarded Gram eigenvalues below `1e-12` times the largest eigenvalue and labeled the remaining residual a lower bound for the full family. Discarding a nonzero direction instead restricts the family, producing an upper bound on its true optimum.

I executed this counterexample in CPU FP64:

```
u=(1,-1,0,0), v=(1,1,-1,-1), h=1
vS=u, vD=0, zS=1e-7*v, zD=zS-u-v  ->  y=v.
```

The centered Gram is `diag(2,4e-14)`. The initial code reported rank1 and residual4, yet the allowed correction `1e7*zS` leaves only `3.4e-33` squared residual in the explicit arithmetic check. A small basis norm therefore does not make that direction mathematically dispensable when coefficients are unrestricted.

The revised analyzer normalizes each nonzero direction and uses two-pass reorthogonalization. The decisive small direction is retained. All projection outputs are now explicitly **FP64 oracle residual estimates**, with `certified_lower_bound=false` and `strict_family_exclusion_allowed=false`. This is the appropriate claim scope: ill-conditioned numerical rank does not support a mathematical impossibility statement. Three further independently executed exactly dependent examples retained numerical rank1 and the expected residual. The existing diagnostic Gram and normalized residual norms make degeneracy visible.

A high estimated projection fraction would motivate testing transferable cheap coefficients, while a substantial residual would motivate richer directions. Both statements concern this fixed family at the observed state; neither is a native BF16 repair result or a full-video quality judgment. The original uncorrected scheduler defect cannot be reused for a hypothetical changed flow.

## EVIDENCE: receipt binding and CPU recomputation

The initial receipt reader checked counts and PASS strings without enforcing distinct ordered source pairs or all state identity bindings. I requested the following checks and verified their presence in the final source:

- Ledger attempts must be the unique ordered prefix of the frozen pair list; a completed scope requires both successful pairs.
- RUN status, study ID, protocol SHA, full pair object, source ID and seed must match the protocol. TRACE source and seed must match that same pair.
- Four counterfactual records must cover unique steps0–3. RUN, TRACE and every safetensors file are SHA-bound, and loaded tensor inventory, shape and dtype match the stored binding.
- All three saved flow identities are verified. Saved before-state identities match both the counterfactual record and TRACE post-`step_pre` boundaries; saved main after-states match TRACE post-`step_post` boundaries.
- The frozen geometry, paired identity and trace-contract consumers are checked against protocol source hashes and included in output provenance.

The CPU path recomputes full-coordinate ideal flow geometry and native-map geometry, comparing their numeric fields to the native records with explicit tolerances. It separately reproduces all three original FP32 scheduler update sequences followed by the BF16 cast and requires exact saved-output equality. This checks the actual scheduler arithmetic at these tensors, beyond a recomposed squared-error identity. A cross-device mismatch will stop analysis rather than silently substitute an ideal update.

Native response `P_native=T_D(zS)-T_D(zD)` and candidate effect `N_native=T_S(zS)-T_D(zS)` are distinguished from their ideal analogues. `||N_native||²+2<P_native,N_native>` is correctly reported as the candidate's signed net effect relative to the dense-at-Student update at this finite state pair. A negative value can be compensating at that step; it does not identify the earliest error source or a beneficial long-term policy.

Graph anchors remain **P23/P24** for measured local operators versus whole-network response, with proposed correction directions a **posthoc/conditional** interpretation. **P25** still consumes independent full-video function evidence. The projection observer itself eliminates no W1/W2 work; a later successful cheap correction could help retain savings from a separately validated FFN approximation.

## Verification and final snapshot

I independently reran `PYTHONPATH=tools .venv/bin/python -m unittest discover -s tests -p 'test_e0_flow_projection_bounds.py' -v`: **4 tests passed**. These cover a target outside the complete span, exact dependent directions, channel-only modes, and the tiny decisive direction. Counterexample/extra dependent-case checks printed `torch.cuda.is_initialized()==False`.

| Reviewed file | SHA256 |
| --- | --- |
| `tools/analyze_e0_whole_flow_results.py` | `428df2744652226ce5a61646c38ceb4623f09342062437683848ad7a43511ea6` |
| `tests/test_e0_flow_projection_bounds.py` | `552707399343148f580cb915e7c09db4913a75292783c58419bd8a67c11ff7f5` |
| `config/e0_whole_flow_counterfactual_20261026_v1.json` | `cebee73f6acd646de6b8eddcc03bb7a3982fca652a273cf77f6d5e259ab10b2c` |

No native whole-flow receipt was analyzed by this reviewer at closure: the first-source ledger was still STARTED when inspected. Actual CPU/GPU recomputation, actual numerical rank and measured projection fractions are outcomes to be checked when completed artifacts become available. At the above snapshots I found no remaining source/math blocker for that scoped posthoc analysis.

## Actual primary-source search

On 2026-10-26 I searched the official PyTorch documentation for `matrix_rank tolerance singular values` and `numerical_accuracy ill conditioned matrices`. The [numerical-rank definition](https://docs.pytorch.org/docs/main/generated/torch.linalg.matrix_rank.html), [numerical accuracy discussion](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html), and [least-squares rank/driver documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) confirm the distinction between numerical truncation and the unrestricted algebraic span. They informed the rank caveat, not a novelty claim. The decisive evidence here is the explicit counterexample and corrected source; no broader new-method literature claim is made.
