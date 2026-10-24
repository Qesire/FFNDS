# FFNDS | research agent entrypoint

**AutoResearcher is the default operating method.** At every substantive scientific decision, automatically apply `Qesire/AutoResearcher@5bd7ec53`: mechanism-first discovery, source reconstruction, PRD representation changes, informative discrimination and concise handoff. No separate user trigger or repeated "use skill" request is required. The linked version is authoritative for research behavior; dated locally copied v2.3 docs are history, not a second prompt authority.

## Current research: three questions, in order
Read [active research state](research/ACTIVE_RESEARCH.json), then only the relevant evidence for the current fork in [Q1/Q2/Q3 research](docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md).
1. **Information** — Before the FFN/block, what useful signal about a computation or its downstream necessity already exists? Distinguish deterministic local hidden prediction from residual/attention-dependent downstream action value.
2. **Decision** — Can that signal change an actionable choice over the best same-cost static policy? Calculate conditional decision regret/NetVOI including the information-read, selector, memory and physical operator costs.
3. **Compute** — Which native FFN operations can actually be avoided while retaining free four-step video quality, real complete FFN latency and total generation-time value?

The objective is better measured **video-quality × actual compute Pareto**, not dense-restoration accuracy. Algorithms, pruning, transformations, compensation, R256, dense completion, static/conditional schedules and post-training are candidate mechanisms—not premises. A novel research question can precede an algorithm. Old R4/R8/R9/R12/C1/C2 candidates constrain only their tested instances.

## Default work loop
- **Discover:** lead with the strongest new causal/mathematical mechanism, not a survey of limitations. Reframe the information, representation, operation or objective when evidence breaks the current model.
- **Ground:** inspect source/config, real tensor/time/QDQ semantics and saved evidence sufficient for that specific mechanism; write a compact equation and check any decisive hidden assumption.
- **Discriminate:** state two meaningfully different predictions; select a source inspection, algebraic proof or smallest worthwhile measurement. Keep early creativity open; novelty search is constructive and closest papers are useful controls.
- **Act and learn:** produce a concrete result, correction, artifact or next decisive discriminator within the available permissions. Record the exact failure scope instead of extrapolating it to an entire method family. Do not repeat already-run controls by renaming them.
- **Communicate:** result → derivation/source → discriminating prediction → next action. Keep precision/unknowns attached to claims. Avoid preemptive refusal lists, rigid quotas, unproductive gatekeeping and verbose protective qualifiers.

On demand read **one relevant** local skill: [mechanism-research](.agents/skills/mechanism-research/SKILL.md), [research-expansion](.agents/skills/research-expansion/SKILL.md) or [problem-reformulation](.agents/skills/problem-reformulation/SKILL.md). Consult upstream [compact harness](https://github.com/Qesire/AutoResearcher/blob/5bd7ec537ae63f2b6a6dd579f0e886aff58a177c/core/AGENT_HARNESS.md) for mode routing and [prompt evaluation](https://github.com/Qesire/AutoResearcher/blob/5bd7ec537ae63f2b6a6dd579f0e886aff58a177c/core/PROMPT_EVALUATION.md) when revising prompts. Do not preload the historical task ledger or all archived skills for every request.

## Execution & evidence
**All FFNDS CPU/GPU/model experiments, queues, retries and automated experimental follow-ups remain paused** by explicit user request: [signed state](research/20261024/EXPERIMENT_PAUSE.json). Source reading, literature, symbolic reasoning and administrative editing are permitted. Only an explicit subsequent user instruction can resume a resource-consuming experiment; historical authorizations do not. No eligible model method is active.

Retain every completed experiment's original protocol, hashes, results, raw logs and exact negative scope. When an authorized method advances to a performance claim, compare matched native controls and real full four-step free-running I2V; a synthetic counterexample, proxy score and theoretical MAC are not video quality or wall-clock speed. Frozen registration gates apply to those historic registrations, not to all new hypotheses.

**Single source of live tasks:** `research/ACTIVE_RESEARCH.json`. `research/CURRENT_DECISION.json`, `research/20261021/method_triage.json`, dated handoffs and proposals are historical evidence/eligibility ledgers, not priority queues. An older line saying ACTIVE, running, retry or mandatory has no execution authority.
