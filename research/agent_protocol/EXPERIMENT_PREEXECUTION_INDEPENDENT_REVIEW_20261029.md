# FFNDS · 每次新实验执行前的独立科学与实现复审（强制）

**2026-10-29 用户指令｜适用新的 CPU / GPU / 原生 capture / replay / 校准 / 全生成 / system benchmark 实验范围。** 本规范与 AGENTS.md、.agents/skills/ffnds-experiment-evidence/SKILL.md 共同约束；不是单纯的建议或以方法通过代替实验通过。科研研究自由探索，不将各门槛当作禁止提出新假说的理由。

## 执行许可状态机

\`IDEA -> DESIGN_DRAFT -> AUTHOR_SELF_AUDIT -> INDEPENDENT_REVIEW_REQUESTED -> INDEPENDENT_REVIEW_RETURNED -> REVISED_DESIGN -> REVIEWER_RECHECK_APPROVED -> SCOPE_FROZEN_IN_ACTIVE -> PRELAUNCH_IDENTITY_RECHECK -> ACTUAL_RUN -> ACTUAL_EXIT_AND_EVIDENCE\`

- **AUTHOR_SELF_AUDIT**：主 Agent 对真实研究问题和 *本次实验* 负责：其相反预测、原论文/代码核对、正常计算图、执行成本、数据/风险及能在结果后做什么决策。与问题无关的额外实验要明确其独立价值。
- **INDEPENDENT_REVIEW**：确有不同 agent_id / context / author 的研究员从零审核实验可辨识度、技术可信性与强基线，至少提出一个可能推翻主 Agent 机制的具体反例，并审核论文/源码差异；其文本不是预设支持票。
- **REVISED_DESIGN**：主 Agent 对独立意见逐项给出 ACCEPT/REVISE/TEST/REJECT_WITH_REASON，修改同一次计划；实质变更需 reviewer re-check。评审人可以给 APPROVE / REVISE / REJECT / CONDITIONAL_APPROVE（需满足明确条件并 recheck），只有 APPROVE 或条件全部核实的批准才进入新运行范围。
- **SCOPE_FROZEN_IN_ACTIVE**：review 已通过不意味着 GPU 授权。另按 ACTIVE_RESEARCH.json 新建本次有限范围、资源/数据/代码哈希/预算，获得实际执行许可。旧 RUN、失败与两卡 session 限制继续有效。
- **PRELAUNCH_IDENTITY_RECHECK**：在实际 CPU/GPU 运行之前，检查当前 git HEAD/关键源码/模型/输入 manifest/数值 backend/分层/指标/核预算/协议 SHA；与审定版本不同时，回到 REVISED_DESIGN/REVIEWER_RECHECK，不能拿旧审批覆盖新实验。
- **ACTUAL_RUN**：只有真实 job、真实 RUN/EXIT 和输出 hash 才代表执行。无可用独立 Agent、无资源权限或未满足审查意见则保留 \`INDEPENDENT_REVIEW_REQUESTED\`/ \`NEEDS_REVISION\`，允许继续静态阅读/代数设计但不假称已有实验。
- **每个新 experiment scope 都重新审查**：原方法旧 review / 同方法不同源/seed/精度/层/稀疏率/算子融合/部署后端 不自动获得新许可；可以引用历史证明减少重复，但必须具体重新审查本次设计及其强对照。显然无物理实验的单元测试/语法检查/阅读可按静态工程验证，不得包装成研究实验结果。

## 审查内容：实验本身必须有可判别性

| 审查轴 | 主 Agent 与 reviewer 要回答的具体问题 |
| --- | --- |
| Scientific question | 实验会区分哪两个竞争机制？两种假说对 **来源、层、step、动作方向或成本** 的预期为何不同？哪些实际结果会改变研究结论？ |
| Novelty & fidelity | 最近论文作者原算法、开源 commit、模型/训练/精度/硬件分别是什么？是忠实复现、受控移植还是独立新方法？公式同构/实现不同的地方是什么？ |
| Computational graph | 真实 \`W1→GELU→W2→gate/e5→residual→next consumer\`、两个专家与4步调度、40层/site、latent τ 与去噪 step、FP4 Q1/Q2 数据生产截止均已核定吗？ |
| Nontrivial controls | 同checkpoint/initial noise/seed/scheduler的真正 Dense 与 neutral、真实随机/静态强基线、同预算/同信息比较；oracle 已付费信号与在线合法信号分开了吗？ |
| Data/statistics | 统计独立单位是否为 source/image-cluster？已有开发数据与新的 blind confirmation 隔离？分层覆盖主体动作、背景、相机、低运动及强运动尾部？完整视频指标有来源级向量吗？ |
| Physical validity | sparse mask 真正令哪个 GEMM/物理 tile 少算？选择、转换、分组、packing、cache、预取和写回总成本是多少？是否只是高 sparsity ratio 的 BF16 dense GEMM？ |
| Numerical identity | 原 BF16/HiF4 fake-QDQ/native packed FP4 不混同；原始参数/偏置/写回顺序与模型行为 neutral 一致；原生/kernel/完整视频闭环清楚？ |
| Resource/failure | 预算、时间/显存上界、当前可用 GPU 与适用 session_max_GPUs、失败与下次重试状态、真实已存在文件的 SHA 是什么？ |
| Decision value | 通过/失败/矛盾输出分别淘汰或推进什么？最小先验实验是否可用已存数据或证明替代？是否需要新的 source/producer，而非适应旧数据降低科学问题？ |

## 每次复审的机器可读证据

为每个新实验保存与固定源码/数据身份绑定的 \`research/experiment_reviews/<scope_id>/vN.json\`，至少含：

\`\`\`json
{
  "review_schema": "FFNDS_PREEXEC_REVIEW_V1",
  "scope_id": "EXACT_NEW_SCOPE",
  "purpose": "faithful_reproduction | wan_transfer | new_method | mechanism | system",
  "research_question": "...",
  "hypotheses_with_opposing_predictions": [
    {"id": "H1", "prediction": "...", "discriminator": "..."},
    {"id": "H2", "prediction": "...", "discriminator": "..."}
  ],
  "author_agent_id": "ACTUAL_AGENT",
  "reviewer_agent_id": "DIFFERENT_REAL_AGENT",
  "author_review_path": "...",
  "independent_review_path": "...",
  "reviewer_disposition": "APPROVE",
  "adversarial_counterexample": "...",
  "responses_to_review": ["..."],
  "recheck_disposition": "APPROVE",
  "code_commit": "FULL_GIT_SHA",
  "critical_source_hashes": {"path": "sha256"},
  "checkpoint_numeric_recipe": "...",
  "input_cohort_manifest_sha256": "...",
  "control_arms": ["neutral", "dense", "candidate", "matched_strong"],
  "unit_of_independence": "image-cluster",
  "full_depth_or_claim_scope": "40x4 / explicit bounded diagnostic",
  "predeclared_metrics": ["subject", "camera", "motion", "temporal", "semantic", "imaging", "FFN_wall", "system_wall"],
  "expected_physical_savings": {"W1": "...", "W2": "...", "observer_and_cache": "..."},
  "applicable_session_gpu_cap": 2,
  "budget_and_failure_plan": "...",
  "approval_scope_fingerprint": "sha256(frozen methods/config/ckpt/inputs/metrics/resource, canonical representation)",
  "state": "REVIEWER_RECHECK_APPROVED"
}
\`\`\`

这个 JSON 是**结构示例，不是现存正式审批**。严禁用模板中的 \`APPROVE\` 或 \`2\` 作为任何新实验的实际评议结果或资源额度。无可用 reviewer 的设计记录要写 \`REVIEW_REQUESTED\`，绝不合成 agent_id、反例、报告或签名。实际记录可使用单独 P0 review 对象加独立 GPU ACTIVE scope；历史实验保持原版。

建议在原生 runner 的 scope-registration / launch 前核对 review.path、fingerprint、真实 reviewer 结果和 ACTIVE 授权，以实现程序阻断。尚未改造到的现有 runner 仍需人工执行本协议；仅修改本文件不会让旧 runner 自动获得强制检查能力。每次正式实验审查与运行需保留可追踪 SHA、链接到论文原码和当前计算图。

## 首批适用对象

- \`B1_CHIPMUNK_MLP_ONLY_AUTHOR_REPRO\` / \`B1_CHIPMUNK_WAN22_FOUR_STEP_TRANSFER\`：两个不同 scope，分别审原作忠实度/当前四步复用合法性；wan2.1的50步收益不可转写为wan2.2四步已测收益。
- \`B2_TEAL_POSTGELU_W2\` / \`B2_TEAL_PREW1_INPUT\`：两个不同物理动作，独立读写、kernel、数值和质量。
- \`S2_REAL_NORMAL_FORWARD_CAUSAL\`：先审 frozen action 对比是否真正区分层敏感与后缀传播；只有早/晚对照不足时补充交错/局部因果及全深度轨迹，而不再基于单末层选方法。
- S1、S3、S4 或其他新增方法按本合同各审各 scope，彼此独立，不做统一的唯一科研漏斗。

完整评议与基线研究案：[20261029 独立研究评议与免训练正向基线](../20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)。
