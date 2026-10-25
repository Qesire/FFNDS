# FFNDS · Wan2.2 four-step video FFN research

## Current E0 science — intrinsic FFN computation first

**Research direction reset, 2026-10-25:** [intrinsic FFN mechanism program](research/20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) and [current sole discovery task](research/20261025/E0_FFN_FIRST_TASK_20261025.md). For specific missing source fields, mathematical contrasts and cheapest discriminators, read [knowledge gaps K1–K5](research/20261025/E0_FFN_KNOWLEDGE_GAPS_AND_DISCRIMINATORS_20261025.md). Study W1–GELU–W2 signed contributions, conditional token/step/expert structure, downstream functional effect, and actual avoidable GEMM work. The old [E0 gap dossier](research/20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md), radial pairing, four-arm and HiF4/NVFP4 comparison now serve as historical/scoped evidence rather than default science agenda. Default [deep Web Search](research/agent_protocol/DEEP_WEB_RESEARCH.md) and AutoResearcher v2.4 remain in force.

**Goal:** Understand and exploit real FFN computational redundancy in the distilled Wan I2V model, reduce *actual* FFN and end-to-end inference time, and preserve complete free-running video quality. No default dense restoration, R256 compensation, mask family or representation.

## Experiment findings and source-linked computation graph

The [complete source-anchored execution graph](docs/98_端到端计算图与证据注解_20261025.md) traces four-step scheduling, self/cross attention, norm2/AdaLN, HiF4 QDQ, W1–GELU–W2, gated residual writeback and propagation into flow/video. Every graph boundary links to a relevant historical or E0 result with an explicit evidence scope. The companion [experiment findings atlas](research/EXPERIMENT_FINDINGS_ATLAS.md) classifies early mask/quantization, group support, R256, temporal reuse, native finite actions, information-value, system timing and current intrinsic-FFN discoveries. They synthesize existing evidence; no paused experiments were resumed.

## Current scientific questions

The three questions are **complementary, not a compulsory sequence**. Discovery may begin with either upstream information, native FFN algebra/physical execution, or downstream video behavior, and revise the others.

1. **Information:** Which already-available signals before FFN/block execution reveal the necessity or downstream value of different computations?
2. **Decision:** Which signals have positive *net* value after observation, routing and physical execution costs over strong static same-cost alternatives?
3. **Compute:** Which FFN computations are avoidable or reparameterizable while maintaining four-step I2V quality, including residual/step propagation and eventual native hardware deployment?

Read [ranked experimental evidence](research/EXPERIMENT_EVIDENCE_INDEX.md), [current live research](research/ACTIVE_RESEARCH.json), [short evidence packet](research/agent_protocol/CURRENT_CONTEXT_PACKET.md), [research navigation](research/RESEARCH_NAVIGATION.md) and [mechanism & information study](docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md). The source-informed C0 residual-scale example is a synthetic construction, not a measured Wan speed/quality win.

## Default scientific workflow

Every research agent uses [AutoResearcher](https://github.com/Qesire/AutoResearcher/tree/8b88a52d894568dc303db727decfbf13b5fabf3b) automatically through [AGENTS.md](AGENTS.md): discover a mechanism → reconstruct relevant source → derive and discriminate → compare closest published work → implement/measure when authorized → record what was learned. The minimal [FFNDS paper-discovery skill](.agents/skills/ffnds-paper-discovery/SKILL.md) provides domain context. Suggestions and exploratory math are not gated by whether an exact experiment already exists.

## Project state

**Saved radial mechanism followthrough, 2026-10-25:** [pairing, common-mode and source-transfer controls](research/20261025/E0_RADIAL_PAIRING.md) show that Block0 raw radial alignment survives wrong token pairing (19.79% paired,20.54% deranged). Position-centered directions retain a pairing relation (8.84% versus3.59%); neither observation is a deletion or speed claim. A common residual direction from the other five sources transfers at cosine0.99645, within this six-source seed42 cohort. All1440 historical radial values reproduce; zero new model/GPU calls. This separates shared geometry from token dependence before investing in conditional routing.

**Saved temporal followthrough, 2026-10-25:** [source-held-out calibration and mechanism comparison](research/20261025/E0_TEMPORAL_CALIBRATION.md) finds transferable common-component drift: Block39 relative SSE falls from37.34% with reuse to8.32% with fixed scale and4.35% with fixed affine correction. Position-centered variation still has39.25% relative SSE under the affine reference. Current-input RMS adds some expert-dependent information, while strong fixed controls remain essential. This uses existing dense-prefix observations: zero new model/GPU calls, no executed cache, and no speed or video-quality acceptance.

Following the independently authored mechanism record synchronized at `0bd9bbc`, [full four-arm distance and subspace checks](research/20261025/E0_SHARED_REFERENCE_MOMENTS.md) quantify the shared-reference decorrelation alternative. The best nonnegative moment fit has2.24%–20.01% discrepancy on final-flow squared distances; whole-output subadditivity can coexist with superadditive low-energy components. This is a saved-tensor diagnostic, not proof of independent noise or beneficial functional cancellation.

**E0 native finite-action discovery, 2026-10-25:** [current scope](research/ACTIVE_RESEARCH.json), [results and competing explanations](research/20261025/E0_FOLLOWTHROUGH.md), and [six-source numerical results](research/20261025/E0_FINITE_GROUP_DIAGNOSTIC.json). Thirty complete four-step generations establish that BF16 residual writeback is a measurable boundary between continuous group predictions and native actions; local and final-flow rankings differ in three of six exposed sources. Ten [cross-worker technical generations](research/20261025/E0_CROSS_WORKER_REPRODUCTION.json) also completed, with all210 tensor/video hash comparisons matching. The user ceiling is four simultaneous idle GPUs; actual peak was two, using four distinct cards in batches. No jobs remain. This remains phenomenon discovery, with full dense GEMMs and no speed or video-quality acceptance.

The preceding [passive six-source study](research/20261025/E0_DISCOVERY.md) completed twelve dense generations with historical latent/video identity. Its geometry, quantization and temporal observations remain evidence at their original scope. The [earlier pause receipt](research/20261024/EXPERIMENT_PAUSE.json) and historical closures are preserved.

**Experimental file importance:** [28 curated, source-hash-bound priority records](research/experiment_evidence/PRIORITY_RANKING.json) and a [searchable inventory of 1,348 historical experiment-related files](research/experiment_evidence/FILE_INVENTORY.json). Priority means usefulness to current scientific questions, not experiment authorization or a quality verdict.

**Frozen evidence is preserved** in [historical decisions](research/CURRENT_DECISION.json), [method outcomes](research/20261021/method_triage.json), `research/`, `reports/`, `docs/` and `upstream_audit/`. These are archives of measured or proposed *past* work, not active tasks. Historical registration rules and results remain unchanged. See [archival lifecycle and redirect index](ARCHIVE.md) and [metric/quality protocol](docs/77_FFNDS学术指标族与评测协议_20261023.md) for eventual native full-video evaluation.

### Key reproducibility references

- [Native FFN structure and conditional computation evidence](docs/86_归一化前信息与净动作耦合的离线检验_20261023.md)
- [Historical objective drift / optional completion analysis](docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md)
- [Full four-step video quality gate](docs/75_备选算法必须完整生成与防止视频质量崩溃_20261022.md)
- [Q1/Q2/Q3 competing mechanism record](research/20261024/three_question_information_first_exploration.json)

The 2026-10-20 release/hardware restoration details remain in [ARCHIVE.md](ARCHIVE.md). Old plans that say `ACTIVE` or `mandatory` remain historical snapshots and do not supersede the active file.

## Multi-agent method discussion

**Default research method:** AutoResearcher applies automatically to **every** participating discussion agent (proposer, independent reviewer, challenger or integrator), according to [the nested agent instructions](research/method_discussion/AGENTS.md) and pinned [source manifest](research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json). Apply its substantive discover → source-ground → mathematical reformulation → discriminating prediction → constructive advance principles without making them a creativity gate.

Agents collaborate **only at the scientific method-discussion level**: each independently proposes an operator/mechanism, other agents review and challenge its mathematics and source evidence, and independent agreement is stored with traceable review records. Start at the [method discussion queue](research/method_discussion/QUEUE.md), [rules](research/method_discussion/README.md), and [record templates](research/method_discussion/TEMPLATES.md). A method supported by another agent is not an executed or validated method; reviews cannot resume paused experiments or authorize implementation.
