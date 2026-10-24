# FFNDS — AutoResearcher default research agent

## Current stage: E0_OPEN_DISCOVERY — first understand the phenomenon

For **every new FFNDS scientific turn or continuation**, first read the single live [ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json) `current_stage`, then its [E0 task](research/20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md) and the concise [stage anchor](research/agent_protocol/E0_STAGE_ANCHOR.md). **The live phase is phenomenon/competing-mechanism discovery, not a selected G6D/G6E/G6F operator program.** Start by identifying a source-grounded observation or explicit structural hypothesis, an alternative explanation and a discriminating next observation. Q1/Q2/Q3 still inspire each other and creative operator conjectures are welcome; keep such ideas **unpromoted seeds** until an explicit user-directed phase change. Do not silently turn a candidate into the default research agenda. In method discussion, only discuss; no implementation or CPU/GPU/model tasks.


**AutoResearcher is the default operating method** for every scientific FFNDS task: discover a mechanism, reconstruct the decisive real computation, derive an informative representation, distinguish competing predictions, and carry the result into the next useful action. Use the pinned [research harness](core/AGENT_HARNESS.md) and [current study](research/ACTIVE_RESEARCH.json) automatically, including when the user simply says “继续探索” or “继续推进”. The local [evidence packet](research/agent_protocol/CURRENT_CONTEXT_PACKET.md) is a retrieval index, not a substitute for original evidence.

## Scientific aim and three complementary lenses

Improve the **real FFN / end-to-end compute versus full four-step I2V quality** frontier in native Wan2.2 D-Q/HiF4. The problem is to discover what information, mathematical structure and execution unit actually makes computation dispensable.

- **Information (Q1):** What state is already available and what does it tell us? Distinguish cheap prediction of local FFN computation from prediction of the downstream consequence of changing that computation.
- **Decision (Q2):** When is an input/state-dependent choice worth its observation, routing, memory and physical packing cost? Compare its *net decision value* with a strong matched fixed alternative.
- **Compute (Q3):** What operations can be eliminated, shared, transformed or amortized, and what happens through quantization, nonlinearities, residuals, attention, later blocks and the free four-step video?

**These are interacting questions, not sequential gates.** An information discrepancy may suggest a new operator; an algebraic simplification or physical execution discovery may reveal what information a selector should collect; a full-video artifact can force a new representation of both. Develop useful operator-first and information-first ideas in parallel when they make distinct predictions. Dense completion, R256, SVD, static sparsity, conditional operators and post-training remain available as *hypotheses or comparators* according to mechanism and actual costs, not as required recipes.

## Default scientific moves

**EXPLORE / research-expansion:** Turn a meaningful anomaly or mathematical tension into distinct mechanisms. Seek freedoms in representation, timing, information, causality, spatial/temporal structure and physical execution. Produce a concrete equation or construction, including bold exploratory conjectures whose decisive assumptions are explicit.

**REFORMULATE / problem-reformulation (PRD):** Try exact or conditional rewrites (E), a new mechanism taxonomy (T) or reduction to a better-understood task (R). Change the unit of explanation when repetitive scores/thresholds fail to explain outcomes.

**GROUND / mechanism-research:** Follow the actual Wan source/config, Q1→W1→GELU→Q2→W2, AdaLN/gate, residual and downstream path for the *decision-critical* question. A creative seed can precede a complete source read; confirm semantics before asserting implementation equivalence or measured performance.

**DISCRIMINATE:** Derive predictions that separate candidate mechanisms. Choose an algebraic proof, saved trace, paper/code check or—when separately permitted—small native study with actual action/compute accounting. Borrow prior science as a source of mechanisms and strong controls; distinguish complete-method equivalence from valuable use of an existing primitive.

**ADVANCE:** Deliver a new structural insight, valid refutation, source-grounded correction, useful implementation or decisive next move. Tie evidence scope to the specific claim instead of expanding local negative results into universal exclusions. Start handoffs with the new result and the alternative it opens.

Use the [FFNDS domain skill](.agents/skills/ffnds-paper-discovery/SKILL.md) for model-specific application. Pull [mechanism-research](.agents/skills/mechanism-research/SKILL.md), [research-expansion](.agents/skills/research-expansion/SKILL.md) and [problem-reformulation](.agents/skills/problem-reformulation/SKILL.md) on demand; combine their methods when a real scientific question spans them. The source-pinned [AutoResearcher research harness](core/AGENT_HARNESS.md), [prompt evaluation](core/PROMPT_EVALUATION.md) and [behavior cases](research/agent_protocol/DEFAULT_BEHAVIOR_CASES.json) guide further improvement. Their existence alone does not establish research quality.

## Evidence and execution

Retrieve empirical claims through the [ranked experimental evidence index](research/EXPERIMENT_EVIDENCE_INDEX.md), whose file priority and maturity labels never grant execution authority. Historical runs and closed branches remain source evidence at their original scope: [method/authorization ledger](research/CURRENT_DECISION.json), [triage and archived methods](research/20261021/method_triage.json), [research navigation](research/RESEARCH_NAVIGATION.md). Research priority comes from **one live task**: `research/ACTIVE_RESEARCH.json`. The [dated handoff archive](archive/20261024/historical_handoffs/README.md) retains old agent instructions and run-status narratives as source material. Dated “ACTIVE”, “NEXT”, or “RUN” in old reports describes the historical snapshot.

**Explicit experiment pause remains active:** all FFNDS CPU/GPU/model experiments, queued work, retries and automatic experimental follow-ups are paused in [EXPERIMENT_PAUSE.json](research/20261024/EXPERIMENT_PAUSE.json). Literature/source reading, mathematical derivation and repository documentation/administrative work proceed. Only a new explicit user instruction may resume resource-consuming experiments. Original logs, protocol hashes and results are preserved.

For a later authorized empirical contribution, compare the original dense D-Q reference, strong matched competitors, native FFN plus total wall time, and independent full free-running four-step videos. Mathematical identities, synthetic counterexamples and local error metrics already contribute scientific insight at their actual scope.

## Multi-agent method discussion — default repository behavior

**Every participating agent**—proposer, external reviewer, challenger, re-formulator and queue integrator—**must use the pinned AutoResearcher methodology by default, not merely cite its name**. Read the [directory-specific AGENTS.md](research/method_discussion/AGENTS.md) in addition to this root file when working anywhere in the method discussion queue. The methodological behaviors to demonstrate in reasoning are: scientific mechanism discovery, decision-critical native source reconstruction, useful mathematical/problem reformulation, prediction-driven discrimination against alternatives, and constructive advance. Select relevant upstream skills adaptively; a creative seed need not satisfy a bureaucratic checklist before discussion. Reviewer endorsements require an independent mechanism-level argument, not a formulaic approval.



**For multiple agents working on FFNDS method ideas, use the [method discussion queue](research/method_discussion/QUEUE.md) and its [protocol](research/method_discussion/README.md).** The agents in this workflow **only participate in method discussion**: independent formulation, mathematical/source/historical-evidence reasoning, competing explanations, constructive criticism and independent peer agreement. They are not implementers, evaluators or experiment runners. Existing user experiment pause remains unchanged.

- **PROPOSE:** Each agent owns its own `research/method_discussion/proposals/<method-id>/vN.json`. Derive an operator/mechanism and falsifiable predictions, not just a score tweak. Declare whether the idea is independent or imported; look for competing ideas without suppressing an alternative merely because an older recipe failed. An early conjecture is welcome; no terminal video/latency proof is necessary to submit a proposal.
- **CROSS-REVIEW:** A *different* agent owns an independent `reviews/<method-id>/vN/<review-id>.json` with a real derivation, strongest counterexample or discriminating prediction, and `SUPPORT`, `CHALLENGE`, `REVISE` or `ABSTAIN`. The proposer never votes on its own method; no assistant may invent other agents' approval. Prefer forming your own argument before reading other reviews and record whether reviews were seen.
- **RECORD AGREEMENT:** At least one genuinely independent `SUPPORT` (without unresolved material challenge) permits a `PEER_SUPPORTED` discussion snapshot; two different outside reviewers permit `CROSS_CONFIRMED`. Save a source-linked `endorsements/<method-id>/vN.json` only *after* those separate reviews exist. Substantive dissent remains `CONTESTED`, even if other reviews support the idea. This is **opinion-level agreement on a bounded mechanism claim**, not empirical validation, implementation authorization, a paper novelty certificate or a change of research priority.
- **KEEP HISTORY AND SEPARATE WRITES:** Version proposal revisions; preserve previous reviewer positions, exact SHA and provenance. Multiple agents create separate files rather than overwriting the same shared registry. Only a serialized integrator refreshes `QUEUE.md` and endorsement snapshots. Never overwrite historical experiments, elevate a peer discussion vote into `CURRENT_DECISION.json`, or silently change `ACTIVE_RESEARCH.json` and `EXPERIMENT_PAUSE.json`.

Use the [record templates](research/method_discussion/TEMPLATES.md) when creating or assessing a method. **Do not allocate implementation/experiments as agent discussion tasks.** The three Q1/Q2/Q3 viewpoints remain coupled openings, not approval gates.
