# FFNDS · AutoResearcher 默认科研引导

**AutoResearcher v2.4 (pinned):** Load [web-research-discovery Skill](../.agents/skills/web-research-discovery/SKILL.md), [generic deep web protocol](../core/WEB_RESEARCH_DISCOVERY.md) and [FFNDS-specific search](../research/agent_protocol/DEEP_WEB_RESEARCH.md) for substantial research. The exact upstream revision is recorded in [active manifest](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json); this shared default applies to independent research agents and reviewers while E0 remains open discovery.


## 当前阶段与研究任务

唯一活跃阶段是 [ACTIVE_RESEARCH.json](../research/ACTIVE_RESEARCH.json) 中的 `E0_OPEN_DISCOVERY`；立即阅读 [本征 FFN 重置纲领](../research/20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[新的第一任务](../research/20261025/E0_FFN_FIRST_TASK_20261025.md) 、[最新机制分歧审查](../research/20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md) 与 [Agent 锚点](../research/agent_protocol/E0_STAGE_ANCHOR.md)。

**默认从 FFN 而不是量化信息出发**：W1/GELU/W2、有符号 hidden 贡献、token/step/expert 动态结构、下游功能敏感性及可能重排或减少的实际计算。旧 HiF4 QDQ、MXFP4/NVFP4 元信息仅作为次级幅值/数值实现线索。局部激活小、W2 净贡献小、视频功能价值小、物理 tile 可跳过，是四个不同的科学对象。旧径向/四臂报告继续作为与具体假说相关的证据，而非第一任务。

**2026-10-25 新的研究节点**：已完成全13824 hidden、两来源连续M128、同专家创新/共享列和HiF4组闭包分析。后验逐token的25%通道创新SSE为2.44%–7.07%，但中间层25%共享列后验SSE约39%–42%。从这一落差打开**有符号函数坐标、时空组织、下游视频作用、非线性表示、物理执行和蒸馏层/步特异性**，并允许超出这些切面的新想法。已完成K1字段地图/原布局支持检查无需再作为第一任务。micro16所谓“pre-W1”需额外全宽W1/GELU，其信息时点与开销分开记账；原HiF4的尺度闭包不自动推出其它数值实现的计算下界。

**每次研究交付**：一个值得深入的新函数/结构解释，真实源码或数学上的支撑，与最相近机制相反的预测，最廉价的判别观察，以及它打开的计算自由度。探索种子可以先于算子。

## 默认进行广泛深入的 Web Search

每项实质性科学工作主动执行 [DEEP_WEB_RESEARCH.md](../research/agent_protocol/DEEP_WEB_RESEARCH.md)：从 FFN key-value、superposition、扩散时空动态、视频闪烁评价、Tensor Core 稀疏 tiling 五族经典/最新论文扩展，补充替代命名、前后向引用、官方代码与硬件规格；具体阅读论文方程、方法、消融和失败条件。独立 reviewer 寻找最接近的同构机制及最强反例。成果写明检索日期、覆盖范围、原始链接和哪些外部结论改变了研究解释；遇到无网络环境则用真实可达的来源继续数学/源码研究并注明搜索缺口。

## 每位 Agent 自动使用 AutoResearcher

继承 [根目录 AGENTS.md](../AGENTS.md)、[FFNDS domain skill](../.agents/skills/ffnds-paper-discovery/SKILL.md) 和固定 [AutoResearcher 来源](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json)。按问题组合 mechanism-research、problem-reformulation 与 research-expansion：

**发现张力 → 复建关键计算 → 推导新表征/机制 → 区分竞争解释 → 交付最有信息量的下一步。**

Q1 信息、Q2 净决策价值、Q3 真实计算是相互作用的科学视角。探索允许从任一角度开始；E0 首先把现象解释清楚，再由机制引出可能的算法自由度。

## 多 Agent 独立讨论与审查

阅读 [方法讨论 AGENTS](../research/method_discussion/AGENTS.md) 和 [独立 E0 发现目录](../research/method_discussion/discoveries/E0/README.md)。提出者在自己的文件提交新解释；挑战者独立核对数学、真实源码或反例；整理者凭真实审核记录总结争议。旧 G6 提案与 PR 的认可只对应它们各自精确的原始命题。共享 [QUEUE](../research/method_discussion/QUEUE.md) 由串行整理者在相应指令下维护。

## 实验记录与权限

[ACTIVE_RESEARCH.json](../research/ACTIVE_RESEARCH.json) 记录 2026-10-25 获用户授权且**已经完成**的六来源 E0 被动采集；早期 [暂停回执](../research/20261024/EXPERIMENT_PAUSE.json) 保留历史状态。当前方法讨论可进行公开论文/源码、数学和获准的已保存资产分析。新 CPU/GPU/model 实验需要用户另行明确指令；Agent 自身的研究论证和支持票仅代表讨论结果。
