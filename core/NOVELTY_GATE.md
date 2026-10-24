# Mechanism-first Novelty Gate (MNG)

A frequently seen failure is to select internet-famous solutions (smoothing, EMA, regularization, distillation, classic sparsity) immediately after observing an anomaly. This bypasses the computation-specific mechanism that could yield a genuinely new method.

**Generate first, audit before novelty claims.** Early exploratory ideas have no novelty-permission gate. For a candidate being promoted toward a research contribution, use this ordering:
1. Sketch a testable mechanism from the observation; reconstruct and verify the relevant true computational structure before claiming source-grounded behavior.
2. Extract algebraic constraints, invariants, sensitivity or dependency patterns.
3. **Independently construct and diversify** candidate interventions from the mechanism; idea generation can precede the final proof of a hypothesis.
4. Predict when each intervention must work, fail or differ from a simpler alternative.
5. Search nearest prior art and negative results, inspect official code where needed, and set matched known-method baselines.
6. Find whether the proposed difference lies in the operator, causal pathway, regime, or falsifiable behavior, beyond a renamed combination.
7. Require ablation, null-mechanism control and equal-budget comparison before claiming a useful new contribution.

## Mechanism-signature comparison (PRD)

After generating an idea, compare its *operator, information timing, representation, state/reuse, action granularity, causal prediction and true cost* against nearest prior art; see [mechanism taxonomy](MECHANISM_TAXONOMY.md). Separate exact algebraic equivalence, shared mechanism with changed regime, and a new falsifiable causal prediction. A renamed formula alone is not a contribution; a known method can nevertheless be a component of a nontrivial verified interaction.

For cross-problem reductions, compare the encoder / borrowed solver / output readout and target objective, not only neural architecture. For a hybrid, predict a distinct interaction against matched components. This refines claims without rationing exploratory invention.

Statuses: DIAGNOSTIC, BASELINE, CANDIDATE, DISTINCT_PREDICTION, AUDITED, VALIDATED. Neither a high metric nor a self-awarded novelty score alone establishes scientific contribution. If the prior-art search is incomplete, use CANDIDATE rather than invent originality.

Existing techniques remain legitimate as controls, building blocks and sometimes the best deployment solution. They simply must not replace mechanism discovery by reflex.

A surprising hypothesis may be worth pursuing even if nearest prior work later shows an earlier conceptual analogue; that literature then becomes a mechanism comparison and opportunity to refine the idea. Reserve AUDITED/VALIDATED for the appropriate evidence. Do not make the phrase 'novelty gate' into a gate on curiosity.
