# FFNDS · AutoResearcher 默认科研引导

## 当前阶段与研究任务

活跃阶段从 [ACTIVE_RESEARCH.json](../research/ACTIVE_RESEARCH.json) 读取：`E0_OPEN_DISCOVERY`。结合 [E0 第一任务](../research/20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md)、[当前科学问题](../research/20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md) 和 [E0 Agent 锚点](../research/agent_protocol/E0_STAGE_ANCHOR.md) 选择本次最值得研究的现象。

**当前最有价值的入口：** 独立解释 Block 0 真实保存组输出与 residual 的径向对齐，检查理想 LN 局部几何、真实残差旁路和后续传播之间的差别；同时保持量化层级、FFN 内部几何、蒸馏时空和异常 token 作为竞争路线。

**本轮应交付：** 新的结构认识或精确理论命题；源码/数学/已保存证据；最强不同解释与一项决定性观察。任何有启发的新算子均可作为后续探索种子。

## 默认进行广泛深入的 Web Search

每项实质性科学工作主动执行 [DEEP_WEB_RESEARCH.md](../research/agent_protocol/DEEP_WEB_RESEARCH.md)：从经典到最新论文、相邻学科、不同命名、前后向引用及官方代码/硬件规格扩展；具体阅读论文方程、方法、消融和失败条件。独立 reviewer 寻找最接近的同构机制及最强反例。成果写明检索日期、覆盖范围、原始链接和哪些外部结论改变了研究解释；遇到无网络环境则用真实可达的来源继续数学/源码研究并注明搜索缺口。

## 每位 Agent 自动使用 AutoResearcher

继承 [根目录 AGENTS.md](../AGENTS.md)、[FFNDS domain skill](../.agents/skills/ffnds-paper-discovery/SKILL.md) 和固定 [AutoResearcher 来源](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json)。按问题组合 mechanism-research、problem-reformulation 与 research-expansion：

**发现张力 → 复建关键计算 → 推导新表征/机制 → 区分竞争解释 → 交付最有信息量的下一步。**

Q1 信息、Q2 净决策价值、Q3 真实计算是相互作用的科学视角。探索允许从任一角度开始；E0 首先把现象解释清楚，再由机制引出可能的算法自由度。

## 多 Agent 独立讨论与审查

阅读 [方法讨论 AGENTS](../research/method_discussion/AGENTS.md) 和 [独立 E0 发现目录](../research/method_discussion/discoveries/E0/README.md)。提出者在自己的文件提交新解释；挑战者独立核对数学、真实源码或反例；整理者凭真实审核记录总结争议。旧 G6 提案与 PR 的认可只对应它们各自精确的原始命题。共享 [QUEUE](../research/method_discussion/QUEUE.md) 由串行整理者在相应指令下维护。

## 实验记录与权限

[ACTIVE_RESEARCH.json](../research/ACTIVE_RESEARCH.json) 记录 2026-10-25 获用户授权且**已经完成**的六来源 E0 被动采集；早期 [暂停回执](../research/20261024/EXPERIMENT_PAUSE.json) 保留历史状态。当前方法讨论可进行公开论文/源码、数学和获准的已保存资产分析。新 CPU/GPU/model 实验需要用户另行明确指令；Agent 自身的研究论证和支持票仅代表讨论结果。
