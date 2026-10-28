# M-B 独立评议｜2026-10-28

本目录独立保存对 FFNDS **M-B（gate/suffix-sensitive coordinates）v2** 的评议，区分（1）原方法科研及部署潜力，（2）推广到前 39 个 Block 的严格数学结构与多步动力学。评议保持与权威方法快照、正式 Agent 投票、实验登记和 ACTIVE 运行授权分离。

## 阅读顺序

1. **[INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)** — M-B v2 的首份独立评议：source/state 配对敏感度、末层 LN/head Jacobian、65 统计量、残差增量学生、A0 竞争基线、真实成本及初始分流。
2. **[EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md](EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md)** — 前层与四步推广（二阶独立审查，256 行）：逐 token 通道平移等变性的条件证明；任意前层 Flow Jacobian 的秩/核维度上界；Self-/Cross-Attention 传播差异；scheduler 的终端敏感度和跨位置有符号交叉项；CoBRAS、非线性微分 Gramian、2ndMatch、OBS-Diff、SVD-Cache 原论文的可迁移结论；最小实验矩阵与事实/假说边界。

## 合并评议结论

**CONDITIONAL_CONTINUE。** M-B 的配对敏感度具有数学价值，现阶段尚无同在线成本、独立来源完整视频和原生 SYS 的实际收益证明。前层推广提示了三项可并行研究子题：

- **B-GAUGE（ALG）**：在 Pre-LN、Cross-LN、无额外直读分支、固定条件的连续模型中，逐 token 的通道共同平移穿过后续 Block 而被最终 LN 消去；应优先核对 FFNDS 冻结 runtime 分支与 BF16 有限差。
- **B-OBS（ALG/OPEN）**：用可观测性、adjoint snapshot、JVP/VJP sketch 描述跨全层和四步状态传播；同来源配对度量与成熟边际基线分别比较。
- **B-JOINT（ALG/OPEN）**：从 E0 的有符号全步有限反事实提出跨 FFN 位置相互作用假说，并以分组混合模型有限干预检验；E0 不是跨位置交互的已通过证明。

**特别保留**：一个状态下 $\dim\ker J\ge(5120-64)M$（零度至少占 98.75%）是线性代数维数结论，**不能宣称 98.75% FFN 通道可删除**。稳定部署还需来源间可共享表示、学生可预测性与实际算子省算。末层径向近似不变性不自动扩展到前层。

当前科研执行目标仍为可用稀疏/紧凑 FFN：已验证 A0 6016 在四站点的原生局部恢复；先完成真实全深度、自由视频行为和 SYS Pareto，再对新方法声明部署收益。其它 A/C/I/F/C 路线继续并行。

## 身份、已有结果与原论文入口

- **本目录文件身份**：ALG / UPSTREAM-SRC / REPO-EVIDENCE / PRIMARY-LIT / OPEN。写入评议没有执行新模型、GPU、视频、微分谱测量、学生训练或系统测速，不代表正式 PEER_SUPPORTED / NATIVE / VIDEO / SYS 通过。
- [M-B v2 方法快照](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) · [整合方案 v2](../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) · [A0 四位置原生局部恢复](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md) · [E0 Whole-Flow](../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) · [研究导航](../../RESEARCH_NAVIGATION.md)。
- 论文和公开实现的逐项 URL、适用假设、实验入口及限制详见扩展评议第 6–7 节。独立新子题是**讨论标签，不自动改变** ACTIVE、QUEUE、原方法快照或历史 RUN。
