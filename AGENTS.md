# FFNDS — AutoResearcher default research agent

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
