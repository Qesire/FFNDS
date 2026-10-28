# FFNDS · Copilot科学研究默认引导（E0_OPEN_DISCOVERY）

统一科学问题：**视频生成模型如何组织信息、计算与生成行为，怎样在保持完整视频能力时降低真实执行成本？** 可从I（信息与表示）、F（函数与动力学）、C（计算与执行）任意入口研究，G13是F线专题。见[统一目标](../research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

**新数值/实验主线**：[原始BF16机制 + RTX5090 SM120成熟NVFP4 W4A4原生部署](../research/agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md)；本会话用户最新确认没有RTX5090、使用A100且最多同时2张空闲GPU；当前继续原始BF16，NVFP4等待将来实际可用的SM120硬件。旧HiF4 fake-QDQ保留科学数值诊断和历史结果，不再充当原生低位加速证据。当前方法M-B/A0及QCFB分析保持并行，静态文档不触发GPU运行。

**方法统一五操作分类**：[O-A计算选择、O-B表示、O-C历史复用、O-D算子代数、O-E学习/验证](../research/20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)。与I/F/C科学入口正交，按G14一手论文source+decision_time+runtime action建立组合指纹。原三候选明确为M-A/M-B/M-C，下一Agent先研究**M-B敏感坐标**，避免与操作O-B同名混用；现有方法讨论批准及运行范围不变。

**当前方法交接**：[现行M-A/M-B/M-C整合方案](../research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)与[下一任务技术合同](../research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。B0原生捕获、CAL24 96flow标签与第一轮B1固定候选池、B-H Direct65 H128真实训练pilot已完成（[B1](../research/20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)、[B-H](../research/20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)），同head的A0/B-H比较、B-F固定输出pilot和A0六视频已实际完成，**下一Agent依据唯一台账推进原始BF16及视频/成本验证**。新来源/GPU/模型作业按ACTIVE精确scope登记；讨论队列只登记真实peer review。

**阅读优先级**：先按[P0–P4导航](../research/RESEARCH_NAVIGATION.md)的P0恢复上下文/方法/当前任务/ACTIVE；需要具体数学、原文、G13或历史失配时再选择性阅读P1–P4。根[AGENTS](../AGENTS.md)及[领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)是行为规则，不要求每次独立Agent重读全部G0–G14报告和历史handoff。

**量化operator/时间强类型**：[源码和成熟FP4审计](../research/20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)区别静态W1/W2权重码/scale与每次调用在线Q1/A1、Q2/A2激活码/scale；原生W4A4具两侧scale，局部activation尺度动态，global可能提前校准。看见scale不等于能及时/免费读。W4A16无activation FP4 scale；任何QCFB路由同时声明可得时点/producer和访问代价。

## 数据、校准与评测的实验思维

遵循[新实验数据协议](../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)与[专用Skill](../.agents/skills/ffnds-experiment-evidence/SKILL.md)。恢复真实来源与source×seed独立性、样本行坐标和状态来源、候选/ Dense完全配对、指标覆盖与视频原始证据；用VBench-I2V官方五次真实生成和CUSTOM_PAIRED单次自定义评价的严格区分建立科学结论。每个问题正向给出来源级对照和最便宜的可判别实验；官方未测维度保留NOT_RUN。历史记录维持原有身份。

## 默认科研动作

1. **任选真实入口**：从有趣的视频行为/有符号差异、已可得的信息与表示、数学算子恒等式、硬件执行异常或原论文机制提出新构造及竞争解释。
2. **追溯关键计算**：按主张定位真实Wan2.2四步D-Q的norm2/AdaLN→Q1→W1→GELU→HiF4 Q2→W2→e5 BF16 gate/residual→flow/scheduler；仅对功能因果主张使用G13 O/D/Q/P完整定位方法。
3. **推导不同预测**：信息时点、计算粒度、功能迁移、因果路径和部署代价都可成为独立新规律；按AutoResearcher MEAR/PRD分析，结合最强相近方法与反例。
4. **主动深读原文与源码**：使用[WARD](../.agents/skills/web-research-discovery/SKILL.md)查原始数学、消融、作者代码、负例及引用链；保留实际模型、训练、数值与运行环境。
5. **评价恰当主张**：机制论文先验证数学/源码/因果范围；执行加速论文另外计量真实FFN+系统成本与独立完整视频质量，严格区分静态图验证、原生运行与外部论文证据。
6. **独立协作与真实图谱**：经[Graph-Agent协议](../research/agent_protocol/GRAPH_AGENT_COLLABORATION.md)与[讨论队列](../research/method_discussion/QUEUE.md)回写I/F/C任一路线的源节点、ALG/LIT/NATIVE/OPEN关系和独立评议。

**事实与权限**：当前G14为**98主＋10补充论文、16个算子焦点**，原生研究图217节点/451关系；[机器先验索引](../research/agent_protocol/METHOD_OPERATION_CROSSWALK.json)额外登记12条typed方法约束。旧10source/12条轨迹/1920站点属于早期机制研究，不是后续consumer20/P30来源集合。[动态GPU调度规则](../research/agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)的通用默认不预设全局卡数；本会话当前用户指定同时最多2张GPU，优先于通用默认。[ACTIVE_RESEARCH](../research/ACTIVE_RESEARCH.json)逐项核验实验身份；新模型与视频按真实来源、冻结代码和执行scope登记，实时/流式实验仍未恢复。

**HiF4源码事实**：[格式审计](../research/20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)确认元素**signed S1P2 自带符号位**，尺度无符号；当前Wan CUDA fake-QDQ写出带符号BF16，不直接导出packed code/scale。符号信息可研究，提取与在线决策时点的成本要实际计算。

**AutoResearcher方法来源**：核心科学Skills来自[固定manifest](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，采用[mechanism-research](../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../.agents/skills/problem-reformulation/SKILL.md)及WARD。新回复首先给出**科学发现或构造性解释**，再写方程与真实证据、相反预测和下一条可区分的观察。源级/数学/实验的证据等级紧贴具体命题，而非用长串限定句铺在发现前面。
