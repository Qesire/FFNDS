# FFNDS · S2 原生时空因果采集：主Agent/独立理论子Agent分工

2026-10-29 · USER DIRECTED · 对 [S2下一轮真实实验](../20261029/S2_NEXT_NATIVE_TEACHER_STUDENT_SPATIOTEMPORAL_CAUSAL_EXPERIMENT_20261029.md) **生效**。本合同赋予主Agent分派与整合的职责，不冒充已真实启动子Agent或已运行Slurm。S2是现有研究中的统计/科学解释维度，不独立立项为S5。

## 1 · 主Agent先进行真实分派

当主Agent在有子Agent功能的执行环境接手本任务时，**第一步立即启动一个单独上下文的「S2 Theory & Experimental Design」子Agent**（可用独立推理模型/子Agent接口，不限定厂商和名称；选择有能力做源码与数学校验的模型）。独立子Agent读本合同和下面冻结输入，但**不要先读主Agent即将给出的推断或方法偏好**，避免相互确认偏差。该子Agent专门承担推导、解释与设计；主Agent同步负责数据/代码/授权/物理执行。若执行环境根本没有创建子Agent的接口，明确写出 BLOCKED_NO_SUBAGENT_TOOL 并保留本合同待可委派Agent接手；不能把主Agent自己写的论文评议署名成独立子Agent产出。此次GitHub connector无法直接起集群或子Agent，任务状态仍是 REQUESTED_NOT_DISPATCHED。

### 提供给子Agent的原始输入包（按顺序）

1. 仓库根 AGENTS.md、.agents/skills/ffnds-paper-discovery/SKILL.md、mechanism-research、problem-reformulation、research-expansion与web-research-discovery；
2. 原始BF16计算图与真实LightX2V Wan正常forward（W1/GELU/W2+bias/gate/BF16 residual, Attention消费者），原始tensor维度/状态读出；
3. [21τ实际CPU统计](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)，[同Dense h原生factor重放](../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)，[新10 fitting](../20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)，[新40视频八轴质量](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)，以及[已完成高运动ROI采集坐标审计](../method_discussion/discoveries/E0/s2-motion-roi-coordinator-review-20261029.md)；
4. 冻结当前任务边界：2-source pilot是验证链路，真正S2结论需要扩大的跨source统计；两卡会话限制优先于动态GPU默认；不重复上轮value10/dcac10拟合权重搜索；
5. [下一轮实验规格](../20261029/S2_NEXT_NATIVE_TEACHER_STUDENT_SPATIOTEMPORAL_CAUSAL_EXPERIMENT_20261029.md)仅作待审对象，不是必须赞同的理论结论；同时找原论文先例并核对真实作者实现，不用抽象标题代替算法同构性。

## 2 · 子Agent具体必须推导的内容（一次完整独立审查）

**A. 原生张量与等价计算**：重建每次4step×40层真实V1→W1→GELU→W2→gate→BF16 residual→Attention/后缀，写明形状、条件调制、bias位置、额外状态以及高/低expert切换；说明同τ跨模型与跨step对齐合法条件。

**B. 竞争解释**：独立提出至少三类可证伪机制（例如source分布失配/频段容量、student prefix变迁与下游放大、运动对齐后条件低维、gate/residual数值关系），为每个指出**符号相反或量级可分离**的预测、最可能混淆因素、最低有效对照与边界。解释为什么AC损失从同Dense-h到自由视频没有提升，明确区分真实观察、连续算术、后验oracle和未测假说。

**C. 数学分解与数值时点**：独立推导 L/P/I恒等式，检查同bias/同g、不同gate/residual与BF16 rounding的修正；再提出优于简单SSE的有符号交叉/时间谱/相位/时空lag/因果局部响应指标。若 H3 不成立，给出更强替代解释而不是强行保留motion compensation。

**D. 数据与来源方案**：重新审查ord898/96从真实Dense flow选ROI是否混淆背景运动，固定central5×5为何必要，是否应再采对象ROI/遮挡/负对照；真实空间邻域、完整21τ、不同source/student状态的producer、内存IO/算力预算（估算与实际冻结分离），独立未暴露source扩大规则。校验机制代表性、统计依赖、τ0与高噪声步、motion camera/foreground相混问题。

**E. 真实干预与实验设计**：给出最小但是决定性的pilot、离线same-input原W2 reader反事实、student free-state追踪、正常模型下游因果干预和完整video有效性之间的执行顺序。对每个实验写清原生输入、观测、预期相反结果、源码路径、数值recipe、来源层/步/时空坐标、运行/输出成本及明确退出条件。考虑是否可以先用更轻的原始变换获得同等辨别力。

**F. 前沿和新颖性边界**：结合已审论文图谱与外部一手论文/源码（TokenFlow、FlowVid、AdaCache、SVD-Cache、OmniCache以及必要的新近工作），区分视频frame运动匹配与denoise-step缓存、普通低秩拟合与W2时空误差运输，避免把现有方法重新命名为创新。引用精确source/代码/公式，不把论文观点伪造为实验结论。

## 3 · 子Agent独立产物与文件协议

子Agent使用自己的agent标识生成独立研究文档：

research/method_discussion/discoveries/E0/s2-theory-subagent-<agent-id>-20261029.md

内容顺序：Original graph/ALG → OBS facts → competing predictions table → math and numerical corrections → experiment schema (parents/sources/states/rows/costs/acceptance) → closest original papers → recommended decisive pilot / alternative pilot → open uncertainties。对潜在代码/设计问题优先列具体可运行反例或数学推导。另提供与主Agent设计不一致的条目（即使结论是完全同意，也明确独立推导过程）。

**状态规则**：子Agent文件实际创建前，机器状态为 ASSIGNMENT_REQUESTED，不能预创建伪装作者文件，也不标独立审阅已完成。此委派用于研究设计，不等同方法队列正式PEER_SUPPORTED投票；确实独立讨论的方法版本需遵守research/method_discussion/AGENTS.md的另一次版本级评议。

## 4 · 主Agent不被理论流程阻断的实操责任

- 主Agent立即检查真实仓库HEAD、ACTIVE/实验两卡有效限制、源ROI coords、native probe hook、已存video及正确可重用状态、CPU/GPU/磁盘可用性；在子Agent推导期间可先完成静态数据身份preflight、合成测试、接口/资源预算的准备，**不等待子Agent后才允许无副作用工作**。
- 在冻结第一个**新native capture scope**前整合独立设计意见，尤其是Student的实际normal-flow数据身份、各步头/门控/下游变量、是否足以区分H1–H4；接受或给出一条有根据的拒绝理由，记录主Agent vs 子Agent差异。
- 主Agent实现和实际运行新producer，真实RUN/EXIT、全部source/arm×160site的原生中性与BF16 typed校验；不为获得一次“成功状态”而将采集缩为只首/末Block。
- pilot退出后启动下一批源分层采集以及另行登记的same-input反事实与必要干预；是否扩充来源取决于统计精度、层/step/动作尾部，不用只做两source作为结论。
- 更新research/ACTIVE_RESEARCH.json、ACQUISITION_QUEUE、CURRENT_TASK_LEDGER、Graph agent bindings以及当前context/handoff；**状态只能按真实证据从REQUESTED→DISPATCHED→REVIEWED→REGISTERED→RUNNING→EXIT_VERIFIED变更**，不得跳级或写“后台运行”。

## 5 · 协作任务/验收状态

| 工作ID | 负责方 | 现在状态 | 正式完成证据 |
| --- | --- | --- | --- |
| S2-THEORY-EXPLAIN-DESIGN-20261029 | 独立理论子Agent | ASSIGNMENT_REQUESTED_NOT_SPAWNED | 新agent-id的独立doc blob、版本及主Agent差异回应 |
| S2-ROI-PAIR-PILOT-20261029 | 主Agent执行 | REQUESTED_NOT_REGISTERED | 真实native code/protocol SHA、二卡内有效RUN/EXIT、source×arm×160、byte-bound FFN/latent状态 |
| S2-CROSS-SOURCE-EXPAND-20261029 | 主Agent执行 | DEPENDENT_ON_PILOT | 预声明独立来源分层/闭合source bank，多批实际EXIT |
| S2-LPI-SAME-INPUT-20261029 | 主Agent+理论解释 | DEPENDENT_ON_NATIVE_TENSORS | L/P/I分解与真实gate/r/BF16对照，source-level错误尾部及计算耗时 |
| S2-CAUSAL-INTERVENTION-20261029 | 主Agent+理论解释 | CONDITIONAL_AFTER_LPI | 正常完整模型后缀与多来源配对视频因果响应及真实SYS |

子Agent推理/原生工作可并发，但 **GPU不是子Agent默认配额**：子Agent优先论文、数学和保存资产，主Agent统一资源登记和调度，整个适用会话严格遵守真实session_max_GPUs=2。
