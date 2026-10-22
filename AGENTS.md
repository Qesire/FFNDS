# FFNDS research continuity

For research review, method proposals, or experiment work, use the available `mechanism-research` skill and read in this order:

1. `research/CURRENT_DECISION.json` (current decision and evidence pointers).
2. `research/20261021/method_triage.json` (current route dispositions).
3. `research/OBJECTIVE_CONTRACT.json` and `docs/65_研究目标层级与R11解释修正_20261022.md` (primary scientific goal versus auxiliary losses).
4. The current decision document and the referenced registered protocol/results.

Scientific objective guard: optimize **deployable input-/state-conditioned FFN dynamic sparsity** in four-step distilled Wan, preserving quality and honest hardware speed. Static step-budget UB1 is the matched engineering control, NOT an input-conditioned mechanism. Quantization is the frozen execution regime, not an interchangeable target model. Full-teacher/local FFN error, fixed-step native flow SSE and free-student-rollout official quality are separate evidence tiers. R11T1's 1.49121% held-out (development-exposed) flow SSE improvement is neither official video quality nor dynamic routing. Do not silently pivot the research goal to static compression, large model distillation, or post-hoc teacher-state restoration.

The original three acceptance constraints remain joint: main-GEMM MAC at most 50% of dense, complete FFN speedup at least 1.5x **including selectors/QDQ/auxiliary work**, and aesthetic/imaging/motion each <=2% relative deterioration on genuinely new, audited frozen inputs on the free-running student. Conditional gain must beat the strongest matched fixed support/schedule; pre-W1 availability and inference cost of the signal are mandatory. These objective rules do not register an experiment or re-open a closed method.

Historical R1-R4 `CANDIDATE`, `PLANNED`, primary-route, next-action, pause, and resume statements are snapshots. They do not override later completed evidence or create standing authorization. Preserve original experimental protocols, metrics, and raw records.

Before recommending or running a model route, compare its actual operator, information, objective, action set, compensation, budget, and data with the nearest rejected recipe. A new name, seed, bigger shortlist/q/rank/router, different source split, relaxed threshold, or unchanged proposal on more positions is not by itself a new mechanism. Use `research/NEW_BRANCH_TEMPLATE.md` to explain a substantive difference, a falsifiable prediction, and why the old obstruction no longer applies. This is a research record, not a substitute for user authorization or a claim of success.

Run `python tools/check_research_routes.py` after editing current research decisions. Before selecting a registered method, also run `python tools/check_research_routes.py --method-id METHOD_ID`. Unknown, closed, deferred, untriggered, and unreviewed routes fail closed; never silently fall back to an archived plan. The checker only enforces recorded consistency and eligibility; it does not understand arbitrary renamed algorithms or automatically establish scientific validity. Do not mark a route eligible merely to bypass it.

The registered R4E1 16-group q<=2 screen at step2/3 block20 is rejected. Joint swaps and matched fixed-h readouts were already included. Its R4E2/R4E3 downstream stages were not triggered. No model job is currently eligible from that protocol. Do not infer a global impossibility theorem, but do not use that logical opening as permission to widen the failed search after observing validation results.

When a new result arrives, update the current decision, triage, README entry, and current handoff together; keep older states as explicitly historical. Do not reintroduce local workstation paths, private worker names, addresses, credentials, or old unredacted snapshot contents. Start edits from current remote files, not a stale local export. Existing public-metadata checks remain applicable.

Preserve concurrent work: read the current branch before publishing and use a non-forced expected-head update. If it moved, inspect the intervening work and reconcile; do not overwrite it.
