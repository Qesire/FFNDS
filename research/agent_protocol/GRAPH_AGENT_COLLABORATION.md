# FFNDS · 科研 Agent × 源码计算图双向协作协议

**当前任务**：[E0 FFN本征机制发现](../20261025/E0_FFN_FIRST_TASK_20261025.md)；**权威可读计算图**：[G0–G4](../../docs/98_端到端计算图与证据注解_20261025.md)；**结构化索引**：[GRAPH_AGENT_BINDINGS.json](GRAPH_AGENT_BINDINGS.json)；**研究范例与竞争解释**：[E0机制再审](../20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md)。

## 协作定义

**图是研究问题的可定位计算对象，Agent 是提出/验证新解释的主体。** G0–G4 是源码绑定的计算流与证据注解；`GRAPH_AGENT_BINDINGS.json` 提供图中稳定ID、语义、信息时点、证据文件、未解分歧和必要的跨层边。它不是运行时动态图，也不意味着所有备选融合/硬件分支已被执行。

双向反馈：
1. **图→Agent**：从科研问题选图节点（例如 `F06→F08R→F08→F09`），沿执行边追溯上游输入与合法状态（G0/G1），沿消费者追踪功能后果（F12/F13/G21/D09），将数值证据、真实源码和未知边组成小型上下文。
2. **Agent→图**：提出可能的函数关系、可执行变换或竞争解释；创建**与节点/边绑定**的讨论注解，附公式、源证据、反向预言和最小判别。新想法仍为`HYPOTHESIS`；已存推导和测量的命题以 `ALG`、`OBS`、`NATIVE-OBS`、`NATIVE-ACT`、`VIDEO`、`SYS` 等标注。
3. **独立复核→图谱**：第二个 Agent 沿图重新检查实际时序、数学及证据，以独立文件提出支持/异议/修订；整合者串行地将**证实且范围明确**的新认知回写到计算图文字、[实验发现图谱](../EXPERIMENT_FINDINGS_ATLAS.md)或索引。尚未证实的研究空间留在图的“OPEN边”和注解，不改变真实Dense执行实线。

**此流程是 Agent 必须履行的研究交接步骤，不要求所有研究按同一机制、同一节点序列或同一稀疏算子展开。** 允许理论先发现表示，再找到对应图边；也允许从视频现象逆向寻找FFN/残差传播路径。AutoResearcher v2.4 `mechanism-research`、`research-expansion`、`problem-reformulation` 与 `web-research-discovery` 始终默认生效。

## Graph Evidence Packet：每项研究留下什么

最小记录可用 [模板](../method_discussion/GRAPH_CLAIM_TEMPLATE.json)：

| 字段 | 内容 |
|---|---|
| `claim_id`, `agent_id`, `source_revision` | 可追踪的发现身份/代码版本 |
| `anchor_nodes`, `upstream_nodes`, `consumer_nodes` | 从 G/F/D/P 注册表中选择**相关**节点，跨G0–G4追踪；不要求每个方向都有完整闭环 |
| `mechanism`, `math_or_trace` | 一条正向新认识、推导或可证伪想法，明确 GELU/QDQ/BF16 等边界 |
| `availability`, `real_cost` | 观测在 W1 前还是后？哪些W1/W2/量化/缓存/索引开销已付？属于后验 oracle 还是可在线观测？ |
| `evidence`, `countermechanism`, `discriminator` | 原始文件及精确来源/step/block/token 范围，最强不同解释与决定性预测 |
| `graph_change_suggestion` | 哪条边/节点可能新增条件注释、改变覆盖范围或等待证据；此字段只是建议 |
| `review_status` | `PROPOSED`、`INDEPENDENT_REVIEWED` 或 `CONTESTED`，由实际 reviewer 文件支持 |

单个有价值的数学问题不必补齐全链路未来质量与时延；只有声称实际可部署加速时，才把到 `D09/P12` 的费用和视频证据闭合。理论/源码、后验保存数据、原生有限动作和自由四步视频是不同来源。

## 三条不同的工作路径

**A. 从原生计算路径提出新数学表示。** 例如 `G18→F04→F06→F08R→F08→F09`。区分真实后Q2系数与 pre-Q2 GELU，探索有符号value basis、GELU特殊结构、条件子空间，以及何处能改变W1/W2的可计算形式。比较 `F15` 的后验有符号证据与 `D08` oracle，不让观察数据自动升级为在线路由。

**B. 从时空现象反推需要的共同/独有计算。** 例如从 `G02→P06→P08` 开始：逐token新变化高度集中但原raster共享支持恶化。允许构造完全不同的时空聚类/条件基、共同更新+位置特有修正、专家专属预测；计算图负责明确跨专家合法时间历史，Agent负责产生反预测，报告真实信息与开销。

**C. 从画质或物理成本反向找机制。** 从 `D09/F13/G21` 的光照/时间质量变化回溯 `F12/F09` 或 `P12/P10/P11` 的执行单位。功能结论不能用局部FFN SSE替代。原HiF4 `P09` 只限定相同QDQ的原始求值闭包；不同格式/函数表示允许另一条条件边，不用假装图的原生实线已经执行新算子。

三个入口互补；新解释可来自 WARD 查到的相邻理论、其它计算图节点或者独立先例，不以此列举限制创意。

## 具体操作（不启动模型）

从仓库根目录使用无第三方依赖的索引工具：

```bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py nodes
python tools/graph_agent_bridge.py inspect F08
python tools/graph_agent_bridge.py trace G18 P12
```

- `validate` 检查 registry JSON结构、边端点、文档中存在的稳定节点ID、节点的本地证据链接；它不验证 GPU算子实际实现、数学结论或任意 Agent推理。
- `inspect` 从注册表返回当前节点的上下文与其关联边。
- `trace` 在本索引的**因果/候选依赖边**中寻找最短路径；它是快速检索提示，不宣称一条科学机制已经证实。
- Agent 以 `research/method_discussion/graph_annotations/E0/<unique-claim>.json` 保存**新的**图锚点发现；提案与独立评议继续沿旧 [讨论规则](../method_discussion/AGENTS.md) 运行。不覆盖他人记录。
- 整理者将证实的命题更新到 [图](../../docs/98_端到端计算图与证据注解_20261025.md)及[图谱](../EXPERIMENT_FINDINGS_ATLAS.md)，并根据需要扩展新的节点ID及本registry，使另一位Agent的检索能立即读到新结果。

## 一致性与权限

原生编译/真实时序优先以 pin源码、实际runtime分支和冻结配置为准；图注的 SRC/OBS/OPEN代表其具体命题的证据身份。源修订变化先提出图差异并定位真正改变的节点，保持历史实验自身身份，不直接修改旧收据/JSON/SHA。

**Agent使用图不会获得实验许可。** `ACTIVE_RESEARCH.json` 是唯一活跃目标与授权源；目前 E0，全部已登记原生范围完成、0待执行新作业。新的 GPU/模型/视频/训练须用户另外指定；计算图注解与独立研究讨论保持为分析工作。

**协作完成标准**：新 Agent 能从一个图节点检索到真实证据及当前开放问题，提出一项带图锚点的不同机制与最低成本判别；另一 Agent 能沿相同节点验证并指出具体争议；整合者能更新图/图谱的注释而不把建议误画成已执行算子。
