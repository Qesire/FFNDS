# FFNDS method discussion agents — AutoResearcher by default

## E0 active brief for all parallel discussants

Read [the single live stage](../ACTIVE_RESEARCH.json) and [E0 agent anchor](../agent_protocol/E0_STAGE_ANCHOR.md) **before treating an old proposal, review or stale handoff as the current task**. The current discussion is independent scientific discovery of real phenomena, competing mechanisms and one decisive next observation. Prefer an individually owned file under [E0 discoveries](discoveries/E0/README.md) for this stage; an unexpected operator can still be recorded as an unpromoted idea, but **is not an instruction to build it or to change the live research priority**. Reviewers challenge the mechanism with a competing source/math explanation rather than demanding an arbitrary implementation. Only the user can authorize a stage change; serialized coordination avoids shared-file races.


> **Applies to every agent and every role in this directory**: independent proposer, external reviewer, challenger, re-formulator, and discussion/consensus integrator. It also applies to agents entering directly through a `proposals/`, `reviews/` or `endorsements/` file. **Do not wait to be asked to use AutoResearcher.** Inherit [root AGENTS.md](../../AGENTS.md), then this local rule.

## Source of method

Use the [currently pinned AutoResearcher source manifest](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json), [research harness](../../core/AGENT_HARNESS.md), [mechanism-research](../../.agents/skills/mechanism-research/SKILL.md), [research-expansion](../../.agents/skills/research-expansion/SKILL.md), [problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md), and [FFNDS native domain skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md).

The active AutoResearcher revision is recorded in [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json), currently `5bd7ec537ae63f2b6a6dd579f0e886aff58a177c`. Use the **locally pinned** source files, not a mixture of old handoffs or an unverified new upstream commit. You may apply several relevant skills, but do **not** force every skill, every Q1/Q2/Q3 lens, or every reasoning step into a bureaucratic checklist. Choose the most informative route for the scientific decision.

## Default scientific loop — shared by all roles

1. **Discover / open:** Find a structural anomaly, conceptual tension, source-level freedom, surprising old result or productive external paper. A mathematically explicit speculative construction can be registered immediately as a conjecture.
2. **Reconstruct:** For an assertion about actual Wan execution, ground the decision-critical source/config/shape and native Q1→W1→GELU→Q2→W2, AdaLN/gate, residual consumer and quantization order. Do not reconstruct irrelevant blocks just to lengthen the discussion.
3. **Reformulate / derive:** Find an informative equation or nontrivial representation (exact/conditional algebra, mechanism taxonomy, alternative reduction, operator change), not a superficial rebranding of a score, mask or baseline.
4. **Discriminate:** State a closest competing mechanism and a different prediction, exact proof, counterexample, archived observation or feasible **future** discriminator. Seek the strongest defensible interpretation of the idea, not reasons to stall it.
5. **Advance:** Leave a distinct construction, sharper testable proposition, scoped rejection, or concrete improvement. Research value and peer agreement are judged on the **substance of that reasoning**, not the existence of a filled checklist.

For FFNDS always keep the real objective in view: lower actual FFN / total-generation cost while maintaining four-step video quality. Q1 (available information), Q2 (net decision value), Q3 (physically eliminable compute) are **coupled explanatory viewpoints, not sequential approvals**. Dense restoration, R256, masks and a particular representation are options/comparators, not defaults.

## Role-specific duties under the same method

**Proposer:** originate an operator, decision principle or structural hypothesis and explain the freedom it exploits; give a distinguishing prediction and most instructive failure case. Relate to original native source, prior results and nearest work at the scope required by the claim. **Do not gate novel ideas on already having a full implementation or measured video**.

**Independent reviewer:** reconstruct the claimed mechanism **yourself** before adopting the proposer’s conclusion. Inspect the original model/evidence where it changes the judgment; attempt a stronger alternative interpretation, algebraic check and discriminating counterexample. Support a clearly bounded claim when the construction survives; `CHALLENGE` identifies the minimal broken assumption and a constructive repair. Do not respond with generic safety/uncertainty boilerplate or expand a failed *old action* into disproof of an entire direction.

**Integrator / synthesizer:** only combine **real recorded** independent reviews, preserve competing mechanisms and unresolved challenges, and state precisely which proposition is supported, contested or superseded. Do not manufacture extra agent identities or shift a `PEER_SUPPORTED` claim to `MEASURED`. A synthesis should reveal what the cross-agent reasoning discovered, not tally votes without arguments.

## Work products and collaboration

See [discussion rules](README.md), [queue](QUEUE.md) and [record templates](TEMPLATES.md). Write proposals and reviews in separate, immutable versioned files; another agent's approval must come from another agent's distinct file and independent argument. In review records, identify the **chosen AutoResearcher route(s)** and the decisive scientific reasoning *in content*, not via a hollow compliance certificate. Keep the upstream commit provenance visible through this local rule; the proposal/review file may cite particular skills when material.

**Discussion only:** No agent in this workflow implements, performs CPU/GPU/model experiments, spawns jobs or edits the experiment authorization. The [explicit experiment pause](../20261024/EXPERIMENT_PAUSE.json) remains active. Reading sources, papers, and already generated experimental data is permitted. Even unanimous agreement cannot restart experiments.

If an external agent system does not read repository instructions automatically, its launcher must supply **both root and local AGENTS.md**. Git rules cannot force an unrelated agent to read instructions; participation in this discussion process assumes it has been given these rules.
