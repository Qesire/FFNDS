# FFNDS · Wan2.2 four-step video FFN research

**Goal:** Understand and exploit real FFN computational redundancy in the distilled Wan I2V model, reduce *actual* FFN and end-to-end inference time, and preserve complete free-running video quality. No default dense restoration, R256 compensation, mask family or representation.

## Current scientific questions

1. **Information:** Which already-available signals before FFN/block execution reveal the necessity or downstream value of different computations?
2. **Decision:** Which signals have positive *net* value after observation, routing and physical execution costs over strong static same-cost alternatives?
3. **Compute:** Which operations are truly removable without degrading four-step I2V quality, after native HiF4/BF16, residual and cross-step propagation?

Read [current live research](research/ACTIVE_RESEARCH.json), [short evidence packet](research/agent_protocol/CURRENT_CONTEXT_PACKET.md), [research navigation](research/RESEARCH_NAVIGATION.md) and [mechanism & information study](docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md). The source-informed C0 residual-scale example is a synthetic construction, not a measured Wan speed/quality win.

## Default scientific workflow

Every research agent uses [AutoResearcher](https://github.com/Qesire/AutoResearcher/tree/5bd7ec537ae63f2b6a6dd579f0e886aff58a177c) automatically through [AGENTS.md](AGENTS.md): discover a mechanism → reconstruct relevant source → derive and discriminate → compare closest published work → implement/measure when authorized → record what was learned. The minimal [FFNDS paper-discovery skill](.agents/skills/ffnds-paper-discovery/SKILL.md) provides domain context. Suggestions and exploratory math are not gated by whether an exact experiment already exists.

## Project state

**All experimental work remains paused:** [pause receipt](research/20261024/EXPERIMENT_PAUSE.json). CPU/GPU/model experiments, retries and automatic follow-ups are not enabled. Current exploration is source, mathematics and literature only.

**Frozen evidence is preserved** in [historical decisions](research/CURRENT_DECISION.json), [method outcomes](research/20261021/method_triage.json), `research/`, `reports/`, `docs/` and `upstream_audit/`. These are archives of measured or proposed *past* work, not active tasks. Historical registration rules and results remain unchanged. See [archive guide](ARCHIVE.md) and [metric/quality protocol](docs/77_FFNDS学术指标族与评测协议_20261023.md) for eventual native full-video evaluation.

### Key reproducibility references

- [Native FFN structure and conditional computation evidence](docs/86_归一化前信息与净动作耦合的离线检验_20261023.md)
- [Historical objective drift / optional completion analysis](docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md)
- [Full four-step video quality gate](docs/75_备选算法必须完整生成与防止视频质量崩溃_20261022.md)
- [Q1/Q2/Q3 competing mechanism record](research/20261024/three_question_information_first_exploration.json)

The 2026-10-20 release/hardware restoration details remain in [ARCHIVE.md](ARCHIVE.md). Old plans that say `ACTIVE` or `mandatory` remain historical snapshots and do not supersede the active file.
