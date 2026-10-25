# FFNDS · Wan2.2 four-step video FFN research

**Goal:** Understand and exploit real FFN computational redundancy in the distilled Wan I2V model, reduce *actual* FFN and end-to-end inference time, and preserve complete free-running video quality. No default dense restoration, R256 compensation, mask family or representation.

## Current scientific questions

The three questions are **complementary, not a compulsory sequence**. Discovery may begin with either upstream information, native FFN algebra/physical execution, or downstream video behavior, and revise the others.

1. **Information:** Which already-available signals before FFN/block execution reveal the necessity or downstream value of different computations?
2. **Decision:** Which signals have positive *net* value after observation, routing and physical execution costs over strong static same-cost alternatives?
3. **Compute:** Which operations are truly removable without degrading four-step I2V quality, after native HiF4/BF16, residual and cross-step propagation?

Read [ranked experimental evidence](research/EXPERIMENT_EVIDENCE_INDEX.md), [current live research](research/ACTIVE_RESEARCH.json), [short evidence packet](research/agent_protocol/CURRENT_CONTEXT_PACKET.md), [research navigation](research/RESEARCH_NAVIGATION.md) and [mechanism & information study](docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md). The source-informed C0 residual-scale example is a synthetic construction, not a measured Wan speed/quality win.

## Default scientific workflow

Every research agent uses [AutoResearcher](https://github.com/Qesire/AutoResearcher/tree/5bd7ec537ae63f2b6a6dd579f0e886aff58a177c) automatically through [AGENTS.md](AGENTS.md): discover a mechanism → reconstruct relevant source → derive and discriminate → compare closest published work → implement/measure when authorized → record what was learned. The minimal [FFNDS paper-discovery skill](.agents/skills/ffnds-paper-discovery/SKILL.md) provides domain context. Suggestions and exploratory math are not gated by whether an exact experiment already exists.

## Project state

**E0 six-source observations completed on 2026-10-25 under the user's renewed execution instruction:** [current scope and results](research/ACTIVE_RESEARCH.json), [fixed protocol](config/e0_observation_replication_20261025.json), and [discoveries and figures](research/20261025/E0_DISCOVERY.md). Twelve full dense D-Q generations passed paired four-step latent/video identity; at most two idle A100s were used. Saved-operand analysis found residual-radial alignment and separated quantization/temporal mean effects. These are structural observations; candidate selection, video-quality and acceleration claims remain open. The [earlier pause receipt](research/20261024/EXPERIMENT_PAUSE.json) and historical closures are preserved.

**Experimental file importance:** [28 curated, source-hash-bound priority records](research/experiment_evidence/PRIORITY_RANKING.json) and a [searchable inventory of 1,348 historical experiment-related files](research/experiment_evidence/FILE_INVENTORY.json). Priority means usefulness to current scientific questions, not experiment authorization or a quality verdict.\n\n**Frozen evidence is preserved** in [historical decisions](research/CURRENT_DECISION.json), [method outcomes](research/20261021/method_triage.json), `research/`, `reports/`, `docs/` and `upstream_audit/`. These are archives of measured or proposed *past* work, not active tasks. Historical registration rules and results remain unchanged. See [archival lifecycle and redirect index](ARCHIVE.md) and [metric/quality protocol](docs/77_FFNDS学术指标族与评测协议_20261023.md) for eventual native full-video evaluation.

### Key reproducibility references

- [Native FFN structure and conditional computation evidence](docs/86_归一化前信息与净动作耦合的离线检验_20261023.md)
- [Historical objective drift / optional completion analysis](docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md)
- [Full four-step video quality gate](docs/75_备选算法必须完整生成与防止视频质量崩溃_20261022.md)
- [Q1/Q2/Q3 competing mechanism record](research/20261024/three_question_information_first_exploration.json)

The 2026-10-20 release/hardware restoration details remain in [ARCHIVE.md](ARCHIVE.md). Old plans that say `ACTIVE` or `mandatory` remain historical snapshots and do not supersede the active file.

## Multi-agent method discussion

**Default research method:** AutoResearcher applies automatically to **every** participating discussion agent (proposer, independent reviewer, challenger or integrator), according to [the nested agent instructions](research/method_discussion/AGENTS.md) and pinned [source manifest](research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json). Apply its substantive discover → source-ground → mathematical reformulation → discriminating prediction → constructive advance principles without making them a creativity gate.

Agents collaborate **only at the scientific method-discussion level**: each independently proposes an operator/mechanism, other agents review and challenge its mathematics and source evidence, and independent agreement is stored with traceable review records. Start at the [method discussion queue](research/method_discussion/QUEUE.md), [rules](research/method_discussion/README.md), and [record templates](research/method_discussion/TEMPLATES.md). A method supported by another agent is not an executed or validated method; reviews cannot resume paused experiments or authorize implementation.
