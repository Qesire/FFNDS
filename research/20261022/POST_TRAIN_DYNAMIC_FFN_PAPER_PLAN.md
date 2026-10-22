# R12 — post-training dynamic FFN sparsity: mechanism-first paper plan
Date: 2026-10-22. **Exploration, not an approved GPU protocol or reported Wan result.**

## Central scientific question
Can a frozen four-step distilled Wan2.2 I2V D-Q model use low-cost information available **before W1/FFN0** to choose an input-conditioned FFN computational action that beats the **best equal-budget fixed action**, retains new-source free-student visual quality and produces measurable full-FFN acceleration? Prefer no full distillation and at most modest offline calibration/selector/recovery training.

AutoResearcher v2.3: generate hypotheses widely, reconstruct true numerical operators, derive conditional constraints, perform the cheapest discriminating check, then commit only mature arms to a frozen model protocol. A failed registered recipe is a scope-bound observation, not a ban on a distinct mathematical action.

## Existing findings motivating, not proving, a new mechanism
- UB1 is fixed static step widths 54/108/108/162 out of 216 groups; nominal dense main MAC 50%, historical FFN 1.546×, original formal20 quality inside bounds but supplementary20 imaging fell 3.166%. UB1 is the strongest existing static engineering comparator.
- R4E1 step2/3 block20 supports q<=2 substitutions in a 16-group shortlist with matched residual readout. All four fixed gates failed; small matched step2 gains and step3 preferred identity. This **does not** test cross-layer allocation.
- R7D1 block30/31 full teacher restoring changed the native downstream response, and 3/4 inputs reversed local/downstream ranking. But its full teacher arms violate the paired equal-MAC action and cannot be reused as R12 values.
- R7D2 secant model had poor native directional fidelity. R11T1 fixed step3/block39 native-head flow training beat local training on exposed development flow SSE, but failed its frozen gate. Neither tested input-conditioned routing.
- OD1 original-vs-distilled two-image block20 contribution concentration nearly matched; it does not establish how distillation changes conditional action regret, cross-layer signed error, or original-O free trajectory.

## Reconstruct and preserve native compute
Pin actual `tools/step_budget_runtime.py`, `tools/trained_packed_runtime.py`, `tools/native_head_replay.py`, `docs/08_上游源码校对_20261020.md` and `docs/65_研究目标层级与R11解释修正_20261022.md`. The source-backed path includes modulation, Smooth, native HiF4 activation QDQ, W1, GELU(tanh), Q2, W2, casts, b2, gate/residual and downstream flow. Dimensions: d=5120, h=13824=216×64, 40 blocks, 4 steps [1000,750,500,250], high/high/low/low. The script `step_budget_runtime.py` intentionally accepts frozen UB1/identity only: new packed actions require their *own* native runner and byte controls, not changes presented as already supported.

## Proposed R12 action and exact rationale
First discriminate in step3/block30+31, where UB1 selects 162 groups per block. Use three equal **pair main-MAC** arms: `UB1_162_162`, `A_216_108`, `B_108_216`. Each totals 324 groups. Other FFNs unchanged means nominal all-positions main MAC remains 50%. The same group total does **not** imply equal real latency or storage; account for dense/sparse packed weights, full native QDQ, selector, correction and launches. Match arm-specific correction capacity and fitting budget; also include no-correction ablation. Choose the pair action before block30 W1 from only currently available upstream inputs/conditioning; block31 must receive block30's actual altered state.

For identical incoming **native step** state x, D-Q flow F_D, UB1 flow F_0 and action flow F_a, let e=F_0−F_D and delta_a=F_a−F_0. The exact signed squared-error gain is

```text
G_a(x)=||e||^2-||e+delta_a||^2
      =-2<e,delta_a>-||delta_a||^2 .
```

This reveals that local FFN error magnitude alone need not order downstream value. Actual native F_a absorbs nonlinear propagation and quantization; this equation does not substitute a synthetic linear Jacobian for native execution. Source-specific L_{i,a} from the *same* frozen objective and equal action set then defines `best_fixed=min_a mean_i L_{i,a}`, `oracle=mean_i min_a L_{i,a}`; conditional headroom is their difference. An oracle uses post-action labels **offline** and is not a deployable pre-W1 predictor. Compare gains versus both best fixed and UB1, not just the incumbent. Normalize if target dimensions differ.

## Execution stages (agent may act on E0; E1+ require separate registration)
**E0 SOURCE / MATH / CPU NOW**: read current decision, triage and objective, run `python tools/check_research_routes.py`; inspect saved R4/R11 records and [E0 recheck](R12_E0_SAVED_EVIDENCE_AUDIT.json); run the synthetic tool unit tests and fixture; map which OP2/TF2/R7 full M operands exist for block30/31 and which native flow targets are missing. Do not reshape unrelated sampled rows into full M or pretend R7 full-teacher recovery is equal MAC. Produce a source-locator coverage ledger. Derive two contradictory predictions: some input-dependent action rank reversals versus a universally better fixed action.

**E1 NATIVE COUNTERFACTUAL VALUE**: separately freeze and register source identities and old exposure, pack widths/actions, corrections and matched fitting, all numeric controls, actual device resources and precommitted stop criterion. Complete identical incoming step-input UB1/A/B native propagation including modified block31 input, compare per-source losses, compute best-fixed/oracle gap, signed direction and controlled noise. If no useful conditional headroom (relative to uncertainty and chosen task impact), close this action family and expand to a genuinely different representation; do not post-hoc widen old q/widths.

**E2 PRE-W1 PREDICTOR**: only after E1 has meaningful headroom, fit very small linear and MLP predictors from already-available modulated input sketches/gate/step/context, compare random and step-only/fixed controls, and report true action regret, generalization and end-to-end feature/selector cost. The old HP1/HP2 C-score target is a different label. Do not use complete W1 hidden values as free decision inputs.

**E3 ACTUAL DEPLOYMENT / ROLLOUT**: prepacked native action variants, real GPU complete FFN benchmark with all QDQ, layouts, predictor and correction charged, then free-running four-step student on independent frozen source images. Compare dense D-Q, UB1, best fixed new action, learned dynamic selector, an oracle diagnostic, near-prior baselines, ablations, quality and per-frame background illumination/motion stability. Frozen joint gates: global main MAC≤50%, complete FFN≥1.5×, independent official aesthetic/imaging/motion each no worse than 2% relative D-Q. New independent images have not been used for selection.

## Divergent exploratory seeds and failure-to-new-mechanism branches
1. **Executable signed functional units**: different group basis based on *signed* W2/gate/downstream output effects rather than C energy. Must change actual computational action geometry and packing, not only rename/reorder R4.
2. **Distillation changes conditional geometry**: compare original O vs D-BF16 under matched input/precision and only then investigate D-Q. Top-k energy alone is not a sufficient diagnosis.
3. **Temporal risk**: equal local FFN SSE but distinct error direction and time-frequency structure may explain background light flicker. Intervene on signed directions before reflexive EMA/smoothing.
4. **Earlier state information**: if oracle headroom exists but current pre-W1 sketches cannot predict it, seek information already emitted by earlier operations rather than growing the old C-score head.

These are seeds, not passed screens or mandatory GPU jobs. Use signed native loss, operator differences and lowest-cost falsifiers to prioritize.

## Conference-paper comparison and writing contract
Claims require a demonstrated **mechanism**, a new post-training **dynamic FFN action/estimator**, and a credible deployment-quality result. Contrast dynamic width/time-step methods, recent diffusion routing, 2:4 kernels, post-training sparsification and few-step methods using the R10 primary-literature ledger, revisiting latest official paper/code before claiming originality. Do not claim novelty for a router, dynamic width, teacher flow SSE, or generic sparse GEMM alone. Keep a table of strongest equal-budget **fixed** actions, dense D-Q, UB1, signed-target ablations, oracle/predictor gap, full operator costs, free-rollout metrics and another regime for transfer if available.

This plan makes no new raw-model result and leaves `CURRENT_DECISION.eligible_model_method_ids=[]`.
