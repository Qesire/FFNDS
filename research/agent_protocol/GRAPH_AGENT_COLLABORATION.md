# FFNDS · 科研Agent与真实计算图的双向协作

**科学对象**：并行研究[I信息、F函数动力学、C计算执行](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)与[O/D/Q/P](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)是F线专题；[G0–G13源图](../../docs/98_端到端计算图与证据注解_20261025.md)、[机器节点关系](GRAPH_AGENT_BINDINGS.json)和[Crosswalk](MECHANISM_GRAPH_CROSSWALK.json)连接所有路线。目前图谱89项文献、177节点、371关系、14机制家族。

研究者可从数学信息界、代码结构、执行成本或自然视频作用任一方面生成新的图节点与有类型关系；新I/C机制可直接登记，不需先被解释成既有FUNCTION/STRUCTURE标签。

## 1. 功能定位专题与跨线路回写

**图→Agent**：沿G18→F03/F04→F06→F08R→F09→F12→F13/G21重建当前输入条件、共享q上的多W1行、GELU、HiF4 Q2整组尺度、门控输出及下游真实路径；结合自然视频和跨step/seed的已保存状态变化。

**Agent→函数/结构地图**：提出一个行为现象或无标签函数候选，登记实际条件、观察出处、有符号/时空响应以及可能的计算承担者（联合W1行、W2有符号组合、跨层attention补偿、expert/step职责或时空共享）。从中推导新结构、竞争解释及有不同结果的判别。

**独立评议→图谱**：另一个Agent核对源码时序、公式和实验身份，对函数的正常计算路径提供支持、具体反例或替代解释；整合者将成熟命题标为SRC/ALG/OBS/NATIVE、原作者LIT或OPEN条件边。函数候选、物理执行单元和后验oracle信息分别登记，便于未来完整因果验证。

**信息与效率（I/C）**：独立比较G7–G12中的FP4 code/sign、proxy、码本查表、合法group稀疏、cache、step策略或小训练模型；G13的已定位功能也可为这些方法提供额外约束。解释信息获取点、物理观察成本以及实际变化的W1/Q2/W2和视频行为。

## 2. Graph Evidence Packet：足够支持新认识的最小信息

以[模板](../method_discussion/GRAPH_CLAIM_TEMPLATE.json)记录：

| 项目 | 有价值的研究内容 |
|---|---|
| 身份与版本 | claim_id、agent_id、source_revision及真实模型/代码范围 |
| 行为/函数/结构 | 哪种视频行为、FFN输出及其可能的原计算路径 |
| 条件 | source、seed、step、expert、block、token时空位置 |
| 图锚点 | 原生上游、正常计算、下游消费者及其证据身份 |
| 数学与机制 | 一条新的方程、交互、共享结构或可证伪预测 |
| 证据与竞争机制 | 精确源码/保存数据/论文、最强不同解释和决定性的下一观察 |
| 信息/物理价格 | 在讨论执行时记录observer、量化、访存、核时间和可省计算 |
| 独立评议与回写 | 对应版本的支持/反例与需要补充的图边 |

探索性FUNCTIONAL-OBS或数学新结构可以先形成有价值的科研成果；正式部署加速需要进一步核对normal-forward因果身份、整个HiF4 Q2分组、终端视频及实际物理成本。

## 3. 三种互补的探索方式

- **正常计算中的联合函数**：研究真实W1/GELU/HiF4-Q2/W2及其它数学基下的有符号响应、补偿与条件职责；与旧Affine或SVD/TopK先例比较新的可区分预测。
- **真实视频状态中的功能职责**：以source425跨seed的signed P/N翻转、频带内外cross异号及背景/光照/运动失配为入口，追踪原生计算消费者维护的时空关系；保留旧状态传播与当前注入两个不同解释。
- **前沿论文启发的结构**：读原论文及实际源码，核对训练、在线决策、量化/核、任务和硬件身份；抽取科学自由度，并据此构造对当前Wan有辨识力的新功能假说。

多Agent经[QUEUE](../method_discussion/QUEUE.md)、[独立评议规则](../method_discussion/AGENTS.md)和[图注表面](../method_discussion/graph_annotations/E0/README.md)交换版本与可复核证据。

## 4. 只读图谱检索

~~~bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py mechanisms
python tools/graph_agent_bridge.py mechanism M13
python tools/graph_agent_bridge.py inspect F08R
python tools/graph_agent_bridge.py trace G18 P12
~~~

validate检查JSON结构、证据文件、节点、89项论文ID、14机制家族和84条跨机制关系；mechanism按家族返回函数/结构及先例；inspect/trace沿登记边检索。结果是科学图谱/条件关系的索引，真实数值、因果作用与性能另由对应证据支持。

## 5. 科研状态与记录维护

**当前范围**：原论文/源码、数学、已有证据、函数/结构索引与独立讨论。**新的CPU/GPU/模型/视频执行暂停**；用户未来按[ACTIVE](../ACTIVE_RESEARCH.json)单独授权。历史RUN、质量、SHA与对应失败结论保留原始身份。

当F线研究发现改变函数或承担结构的认识，按对应命题更新[Function–Structure](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[主图](../../docs/98_端到端计算图与证据注解_20261025.md)与[科学综合](../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)；I/C线的新表示、算子和成本结构可直接更新Graph Bindings与Crosswalk。原论文详细方法与阅读等级保留在各自source ledger。好的交接留下**新的科学认识、真实证据、竞争预测和下一条有信息价值的图边**。


## G14 · 逐论文计算作用注释

原文事实/源码阅读层级继续在 `PRIMARY_LITERATURE_INDEX.json`及各原文ledger维护；G14 `PAPER_METHOD_FOCUS_GRAPH.json`对每篇文献绑定稳定T01–T16 focus、原生Wan锚点、决策时点、具体行动、适用性判别。图G14将所有L01–L89和校准C01–C10逐项落入可读Mermaid图，用`graph_agent_bridge.py paper L34`/`focus T06`进行双向访问。映射边保留LIT，不自动升级成本仓库NATIVE/VIDEO/SYS的执行结果。数据/校准/评测型论文优先进入T01/T02/T16及视频作用边，训练结构方法保留其训练前提与物理算子。
