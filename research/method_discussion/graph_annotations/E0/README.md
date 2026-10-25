# E0 · Agent 与 G0–G4 计算图关联的研究注解

当前 Agent 每次**实质性 FFN 科学研究**都应使用 [图↔Agent协作协议](../../../agent_protocol/GRAPH_AGENT_COLLABORATION.md) 与 [稳定节点索引](../../../agent_protocol/GRAPH_AGENT_BINDINGS.json)。图的原始来源为 [G0–G4](../../../../docs/98_端到端计算图与证据注解_20261025.md)。

这里记录原始贡献，不是算子实现或实验排队：
1. 研究者选择**恰好相关的**节点/边，从源代码和上游/下游查证，允许反向从视频现象或一篇新论文定位节点；
2. 将 [GRAPH_CLAIM_TEMPLATE.json](../../GRAPH_CLAIM_TEMPLATE.json) 拷贝为唯一名称的 `<claim-id>.json`，填入 `anchor_nodes`、来源、机理、反解释与判别式，依据实际身份标注 `SRC/ALG/OBS/NATIVE-OBS/NATIVE-ACT/SYN/VIDEO/SYS/OPEN`；
3. 独立 Agent 用另一份记录或既有 `reviews/<method-id>/vN/` 文件提出实质证实、反例或争议，引用精确claim ID；
4. 整合者核对图ID、实际代码/证据路径与评议后，决定更新 G0–G4 原图的文字证据、OPEN边或实验谱系；保留原研究讨论的所有不同意见。

一条新科学定理、一个对新表示的反例，或一个发现原图忽略了重要源码分支的分析均为有效科研贡献。若用户未许可新模型执行，则记录下一实验需求而不自动运行。**这里暂不包含任何已获得独立认可的新图注解。**
