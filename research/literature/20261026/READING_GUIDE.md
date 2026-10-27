# FFNDS · 文献Agent的一手核读、证据等级与计算图回写契约

> 现行版本：2026-10-27。全部研究记录索引：[79篇唯一论文](PRIMARY_LITERATURE_INDEX.json)；综合机制解释：[现行研究报告](../../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)；机器关系：[Crosswalk](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)。原2026-10-26首批33/51篇是历史检索阶段，而非当前总数。文献研究持续，原生CPU/GPU/模型/视频执行暂停。

## 首要文献辨识：功能意义与承担功能的计算结构

在探索何种FP4-code符号、mask、路由、Cache值得部署前，先从真实视频/状态差和原W1/GELU/Q2/W2发现**函数是什么、在哪里发生、哪个上下文成立**，再确定哪些**正常前向计算路径**承担此功能。新增[L80–L85 功能–因果–时空定位理论](../20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)，以及[G13研究综合](../../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)。语义concept/SAE可做函数候选观测；necessity/sufficiency仍需要合法原生干预；能否物理省算是另一层实验。当前目录85项，机器图171节点/354关系，14机制家族。

## 一个paper ID应回答什么

原始论文/论文版本与正式会议；确切读过的摘要、原文§/Eq/Table/Fig、公开源码文件与blob SHA；作者所**实际采用的架构、训练/蒸馏、quant格式和步数**；真实当次输入/历史/量化code在**哪个计算节点之前可用**；额外router、proxy、量化scale、KV/cache、metadata、tensor gather、weight读与GPU调度价格；缩减的W1/W2 GEMM、frame、block或whole-step调用；作者报告质量/速度的硬件、source、step和任务；最强同构先例、替代机制和可区分预测。

分别给出四级认知身份：**摘要/作者描述**、**原文方程+消融**、**作者代码实际分支**、**本Wan原生数值/视频RUN**。只有后者才支持本模型已实现和达到质量。实现存在不等于其全部默认配置开启；速度提升要区分模型算子/单核/完整pipeline以及多GPU/单流/TTFF/p99。

## 按机制研究，而非按发表日期排序

| 科学交集 | 首选一手材料 | 迁移至原四步Wan的条件 |
|---|---|---|
| FFN hidden条件决策/代价 | [Prox/Spark/SpQt/SQ-DM/CodeGEMM原论文核查](../20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ算子代码与T2V](../20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) | 在线数据真实可用、group64 Q2闭包、native packed dense强基线和完整视频功能 |
| FP4有符号直方图与编码域 | [G11原理](../20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12单行紧证书和LUT](../20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | weight/input scale同步分段、signed多重集合获取成本、shared-input多W1行joint约束、真实可跳tile |
| 当前GEMM N:M、训练MoE和部分历史缓存 | [G8四篇审计](../20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md) | W1前or W2前信息时点、same expert历史、router/cost；RT-Lynx原文PDF仍待深入 |
| 逐step/逐frame/训练压缩 | [DIFS/FIS v1原始公式](PRIMARY_METHOD_EQUATION_AUDIT_20261027.md)、[G9历史/树结构](../20261027/G9_CONDITIONAL_EXECUTION_TRAJECTORY_AND_FUNCTION_REALLOCATION.md) | 训练后新函数与冻权FFN区分，I2V vs T2V任务及time/expert区别 |
| 实时causal视频与FFN SLO | [14项模型方法及源码](../20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)、[source ledger](../20261027/REALTIME_VIDEO_SOURCE_LEDGER.json) | causal新模型与原全片双向DiT分开，chunk真实M、KV/decoder总成本和p99/TTFF |
| 数学/理论/解释与质量 | [机制交叉索引](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)、[当前研究综合](../../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[原生证据谱系](../../EXPERIMENT_FINDINGS_ATLAS.md) | 预先定义终端功能而非后验SSE；同源seed、流状态与完整视频指标 |

阅读升级顺序：从强相关原论文**精准对象/公式**入手，再读其最强先例/失败消融及实际源码分支，最后把数学表示连接到**原Wan对应SRC或native证据节点**。独立Agent可以同时提出不同科学机制；研究的排序由新信息价值决定而非固定TopK、dense恢复或现成proxy。

## 实际回写顺序

1. 在[PRIMARY_LITERATURE_INDEX.json](PRIMARY_LITERATURE_INDEX.json)保持一个稳定LID，记录论文/版本/一手阅读级别与最强竞争解释，原方法详细内容放独立审计与source ledger。
2. 在[Graph Bindings](../../agent_protocol/GRAPH_AGENT_BINDINGS.json)定位原生F/G节点，必要时增加新的**方法级LIT/ALG/OPEN节点**及有类型比较边；SOURCE/LIT边不可画成原生已经运行的实线。合成关系更新[Crosswalk](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)与[可读图G0–G12](../../../docs/98_端到端计算图与证据注解_20261025.md)。
3. **仅当新原文改变机制分类、证据资格、在线可用性、真实成本或可区分结论时**，更新[统一综合报告](../../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)，避免在所有导航文件再写一遍进度。
4. 用轻量静态检查确保论文ID唯一、边端点存在、相互链接及权限不被覆盖；数学/数值/视频必须分别有相应来源，不把CI“通过”解释为科学成立。

对于摘要无法获得的方法，明确记录摘要可支持的结论和进一步要核的公式；正式PDF读取和作者源码核对缺一项时只提升已实际检查的层级。所有原本历史审查文件作为那个时点的真实工作记录保留，现行科学解释由本页引用的综合报告给出。
