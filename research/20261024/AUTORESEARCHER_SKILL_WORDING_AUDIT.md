# AutoResearcher Skill-driven wording audit — FFNDS (2026-10-24)

## Scope and actual method

Applied three locally version-pinned AutoResearcher skills to *active instructions and current research wording*: [problem-reformulation / PRD](../../.agents/skills/problem-reformulation/SKILL.md), [research-expansion / MEAR](../../.agents/skills/research-expansion/SKILL.md), and [mechanism-research / CR-DF-CRL](../../.agents/skills/mechanism-research/SKILL.md). Upstream `Qesire/AutoResearcher@5bd7ec537ae63f2b6a6dd579f0e886aff58a177c` core/skills remain **verbatim**. This pass rewrites the FFNDS-specific default instructions, objective, research packet, user-facing references and behavior cases.

A good research prompt should **create** a useful mechanism or degree of freedom, then identify the decisive evidence, rather than maximize expressions of caution, warnings or prohibited actions. Claim scope and experiment permission remain precise without dominating the scientific answer.

## Concrete review findings and corrections

| Prior active wording (original scope) | Scientific problem introduced | Replaced with | Skill criterion |
| --- | --- | --- | --- |
| AGENTS: “Current research: three questions, in order” | Converted scientific lenses into a serial action gate; an algebraic/physical discovery would have to wait for signal/NetVOI proof | “three complementary lenses” with **information ↔ decision ↔ compute** interaction and either-direction entry | PRD: change decision/representation, find freedom |
| Domain skill: “Use **one** relevant specialist automatically” | Arbitrary one-skill ceiling blocks expansion + source-grounding + PRD combinations | Apply the productive *combination* of problem-reformulation, research-expansion and mechanism-research; keep focused context | MEAR: diverge then discriminate; CR: map actual flow |
| Current context: “physical-operator search starts when information/decision mechanism…” | Quietly privileges dynamic conditional selectors and excludes valuable static physical simplifications | Two parallel starting points: **from information** or **from computation**, each can reshape the other | PRD E/T/R; CR structural constraints |
| Objective: “Explore distinct mathematical and executable actions only after specifying what information…” | Replaces the old dense-completion objective drift with an information-first permission gate | First identify *any* explanatory mathematical, state or physical computational structure; derive corresponding information requirements if useful | PRD OPEN / TRANSFORM; MEAR divergence |
| Research doc: “本轮目标先确定信息……随后……生成算子” | Unproven assertion that every valuable sparse algorithm depends on additional runtime information | Original Q1/Q2/Q3 remain scientific *questions*, while direct operator and downstream-geometry discoveries have equal status | PRD representation shift; mechanism diversity |
| Main prompt: “avoid…”, “do not…”, nested refusal/quality gates | Prioritizes staying within a safe-looking script over a constructive scientific move | Lead with **explore, reformulate, ground, discriminate, advance**; keep evidence scopes adjacent to claims; operational pause occupies one concise block | MEAR constructive voice; CRL meaningful discriminator |
| Static check: fixed phrases/word counts required to pass | Rewards literal prompt strings, not scientific behavior; changing language alone breaks CI | Verify *pinned source hashes, structural references, task agreement, symmetric research routes and pause*; use case-level trajectory reviews for quality | Harness: separate static integrity from real agent performance |
| Evaluation set: “algorithm-first fallback ignores original information question” | Biases review against mathematically motivated operator-first advances | New **valid operator-first** and **valid information-first** cases; ask distinct predictions and source-backed cost claims in both | Prompt evaluation: prevent rubric leakage and compare actual scientific decisions |

The status `PAUSED_BY_EXPLICIT_USER_REQUEST` is not a rhetorical restriction: it is an actual user resource/experiment boundary. Keeping the pause intact is part of faithfully executing the active request, just as original source hashes and method results are facts rather than defensive prose.

## Active default behavior after review

1. **Discover:** from available information, native source/algebra/QDQ, kernel packing, temporal residual trajectories, or a real outcome; propose the most informative explanation or new representation.
2. **Reformulate:** use exact/conditional mathematical equivalence, mechanism taxonomy or reduction (PRD E/T/R) to expose a new operator or decision variable. The new representation may require no dynamic selector.
3. **Ground:** reconstruct the relevant Wan computation and state/shape/QDQ boundary before promoting a source/implementation claim; reuse saved evidence.
4. **Discriminate:** derive distinct observable predictions and choose a proof, code read, saved trace or separately authorized targeted measurement, whichever most changes the question.
5. **Advance:** produce a research artifact or next mechanism-breaking observation, compare closest complete prior implementations constructively, and maintain scoped source/empirical claims.

This is a **flexible set of scientific moves**, not a compulsory five-stage checklist. A mathematical proof can be the completed result. A physical simplification can begin at Q3; information can begin at Q1. Q2 is the connection from an information signal to a cost-bearing conditional decision, not a gate for static operator ideas.

## Evidence of this audit vs still unmeasured

- Performed: read upstream skills, reconstructed conflicts in current FFNDS prompts, rewrote active instructions and research document, extended the evaluation set from six to eight scenarios, added a structural/static consistency check.
- Current numeric/scientific results: unchanged. No CPU/GPU/model experiments; no auto-retries; original frozen runs and source hash records retained.
- Still unknown: whether the revised wording **improves live agent research quality**. A subsequent separately authorized paired evaluation would feed each agent only the `case.user_input` and equivalent source/tool/budget context, blind the expected reviewer criteria, then assess the actual mathematical mechanism, source reads, decision quality and tool trajectories. GitHub file consistency is not an agent benchmark.

See [active research contract](../ACTIVE_RESEARCH.json), [source-bound execution harness](../../core/AGENT_HARNESS.md) and [reviewer-side scenario portfolio](../agent_protocol/DEFAULT_BEHAVIOR_CASES.json).
