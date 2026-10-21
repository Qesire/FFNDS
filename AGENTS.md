# FFNDS research continuity

For research review, method proposals, or experiment work, use the available `mechanism-research` skill and read in this order:

1. `research/CURRENT_DECISION.json` (current decision and evidence pointers).
2. `research/20261021/method_triage.json` (current route dispositions).
3. The current decision document and the referenced registered protocol/results.

Historical R1-R4 `CANDIDATE`, `PLANNED`, primary-route, next-action, pause, and resume statements are snapshots. They do not override later completed evidence or create standing authorization. Preserve original experimental protocols, metrics, and raw records.

Before recommending or running a model route, compare its actual operator, information, objective, action set, compensation, budget, and data with the nearest rejected recipe. A new name, seed, bigger shortlist/q/rank/router, different source split, relaxed threshold, or unchanged proposal on more positions is not by itself a new mechanism. Use `research/NEW_BRANCH_TEMPLATE.md` to explain a substantive difference, a falsifiable prediction, and why the old obstruction no longer applies. This is a research record, not a substitute for user authorization or a claim of success.

Run `python tools/check_research_routes.py` after editing current research decisions. Before selecting a registered method, also run `python tools/check_research_routes.py --method-id METHOD_ID`. Unknown, closed, deferred, untriggered, and unreviewed routes fail closed; never silently fall back to an archived plan. The checker only enforces recorded consistency and eligibility; it does not understand arbitrary renamed algorithms or automatically establish scientific validity. Do not mark a route eligible merely to bypass it.

The registered R4E1 16-group q<=2 screen at step2/3 block20 is rejected. Joint swaps and matched fixed-h readouts were already included. Its R4E2/R4E3 downstream stages were not triggered. No model job is currently eligible from that protocol. Do not infer a global impossibility theorem, but do not use that logical opening as permission to widen the failed search after observing validation results.

When a new result arrives, update the current decision, triage, README entry, and current handoff together; keep older states as explicitly historical. Do not reintroduce local workstation paths, private worker names, addresses, credentials, or old unredacted snapshot contents. Start edits from current remote files, not a stale local export. Existing public-metadata checks remain applicable.

Preserve concurrent work: read the current branch before publishing and use a non-forced expected-head update. If it moved, inspect the intervening work and reconcile; do not overwrite it.
