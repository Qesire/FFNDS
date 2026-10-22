# FFNDS · scientific research library and open exploration navigator

**Current as of 2026-10-22. Readable archive and research compass, not a mandatory experiment sequence.**

## What is fixed

The goal is to discover a **new, deployable, post-training input/state-conditioned FFN dynamic-sparsity method** on the frozen Wan2.2 I2V A14B four-step D-Q student, improving real complete FFN/whole-video inference with acceptable *free-running video quality*. The existing joint project criteria remain: main FFN GEMM MAC ≤50% dense D-Q; complete FFN speed ≥1.5× including all auxiliary costs; independent new-source official aesthetic/imaging/motion each ≤2% relative drop. The scientific paper also needs incremental improvement over the strongest matched static or prior-art control. These criteria evaluate final registered candidates, not the imagination stage.

Live status: [CURRENT_DECISION](../../CURRENT_DECISION.json); completed latest [R12 E0](../../../docs/66_R12等预算动作的操作数与条件粒度审计.md) and its [handoff](../../20261022/HANDOFF_R12_E0.md). No equal-budget native R12, native spatial-tile policy, new router, or new video/GPU result was obtained by importing these documents. At the import snapshot: "Model method eligibility remains **empty**." A separate later user approval registered only the fixed [D0 diagnostic](../../20261022/R12_D0_registered_protocol.json); its current status is in CURRENT_DECISION, and no native D0 result has yet been measured. The library itself grants no execution authority.

## Read the imported studies as distinct evidence types

| Document | What it contributes | Status / required correction |
|---|---|---|
| [V1 full historical mechanism atlas](history_full_mechanism_atlas_V1.md) | Research families; measured positive, negative and untested mechanisms, original experiment locators | Historical synthesis, fixed to source evidence before docs/66; old “next experiments” not binding |
| [Literature/consistency comparison](prior_art_and_consistency_review.md) | PARE, E-DiT, FIS and other direct precedents; novelty boundaries and AdaLN source timing | Audit and comparison, not final originality proof; check up-to-date primary work |
| [V3 conditional-value derivation](conditional_value_derivation_V3.md) | Best fixed vs post-hoc oracle vs pre-W1 information decomposition, compact test logic | Algebra valid in its explicit two-action matched-loss assumptions; R12 pair ordering superseded |
| [V4 whole-trajectory seeds](full_trajectory_dynamic_ffn_seeds_V4.md) | Global layer/step compute allocation and actual tile×hidden FFN operator proposals, end-to-end post-training experiments | Proposed, unexecuted algorithms. No privilege or requirement to develop GBA/STB; docs/66 terminal-output splicing is NOT a native FFN action |
| [This scientific synthesis](../../../docs/67_后训练动态FFN开放探索与质量速度目标_20261022.md) | Current priorities, agent behavior, provenance and open route selection | Orientation, not experiment registration |

**Not promoted as current document:** the original V2 broad mechanisms draft, because it was superseded by the academic/source correction review and subsequent V3/V4; see the reviewed V1 plus corrected V2 prior-art comparison instead. Past full history remains in original FFNDS `docs/01–66`.

## Evidence model

- **Measured:** source-linked UB1 1.546x full-FFN speed and formal20 performance; supplemental20 imaging 3.166% drop; R7D1 fixed-input recovery/flow; R11T1 matched-flow objective; R12 E0 saved native flow/operands recheck.
- **Derived:** signed downstream counterfactual identity, finite-action conditional oracle under same sample/action losses; group-weight MAC algebra.
- **Exploration:** real tile execution, multi-layer dynamic budget, functional-group alternatives, available information in earlier modules, causal full-rollout action values, fused native QDQ/packing. These are neither accepted implementations nor a closed menu.

R12 E0 on four **already exposed** TF2 sources measured 0.171022% *source-level post-hoc full-teacher A/B* gain, 0.314848% *frame output splice* and 5.089427% *output-token splice*. Only the first is a whole-source selection between two already calculated R7 full restorations. Splicing already computed final output with D-Q labels is neither a feasible pre-W1 policy nor an upper bound on feasible new native tile actions. The R7 A/B restorations are also **not equal-budget R12 actions**. The practical value of dynamic spatial FFN computing remains an empirical question.

At a fixed step and layer, AdaLN modulation/gate is governed by timestep/block parameters in the checked paths; do not assert AdaLN-only content adaptation. Incoming post-attention residual and modulated FFN input are genuine content-aware candidates, but their *predictive value for a new action* must still be demonstrated. Distinguish configured scheduler coordinates from observed native model timesteps ([docs/66](../../../docs/66_R12等预算动作的操作数与条件粒度审计.md)).

## Research selection: no algorithm whitelist

The default *scientific question* is: **What inexpensive observation and executable intervention permits a nontrivial dynamic improvement in full video quality at genuine lower cost, beyond the strongest static/prior-art alternative?** Possible intervention scopes include individual FFNs, multiple blocks, whole four-step trajectory, function-space regrouping, regular spatial tiles, cross-operator fusion and wholly different proposals discovered from new trace/quality evidence. There is no mandate to begin with block30/31 or to implement V4 A/B.

When an observation exposes a new causal degree of freedom, formulate several explanations that make **different predictions**, perform the cheapest discriminating derivation/source/saved-data check, then focus compute on the small number whose expected research impact justifies it. An untested seed is not “rejected” for lacking terminal video scores; a paper claim of quality/speed is not accepted without them.

Avoid substituting C-score or forced single-step flow for final video, and avoid passing static step schedules off as input-conditioned novelty. Do not simply repeat rejected R4 q/pool, R9 greedy coefficients or R11 fixed training lengths. New operators, representations and information channels are allowed to be devised and tested.

## Agent execution output (one compact scientific record)

Every substantive research iteration should leave:
1. The observed finding and its source/provenance (or a plainly labeled new conjecture);
2. A precise new computation/information/action mechanism and comparison with the nearest strong method;
3. One minimal discriminator including observable, math and plausible opposite result;
4. An actual saved-data/source/CPU outcome when accessible, else a prepared executable native protocol with missing operands named;
5. Status of evidence and a decision about next work **without** silently registering model jobs.

Read [AutoResearcher v2.3](../../agent_protocol/README.md) as the discovery protocol. Source and data evidence outrank any archived priority order. Keep an unrestricted low-cost exploration backlog beside a small registered experiment portfolio.

## Literature touchstones — not originality guarantees

- [PARE](https://arxiv.org/abs/2605.27336): Wan2.1 content/time block routing and progressive training.
- [E-DiT (ICML 2026)](https://proceedings.mlr.press/v306/wang26dd.html): sample-dependent block skip and MLP width.
- [FIS-DiT](https://arxiv.org/abs/2605.11869): Wan2.2 frame-interleaved compute.
- [Dynamic-in-Few-Step](https://arxiv.org/abs/2607.06631): distillation-time step-structured computation.

Nearest methods/codes must be rechecked at novel-algorithm claim time. Recombining known router, static width and standard fusion is not automatically novel; neither is a new named loss identity alone.
