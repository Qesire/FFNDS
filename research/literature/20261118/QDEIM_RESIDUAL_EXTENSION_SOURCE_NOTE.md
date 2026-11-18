# Q-DEIM 与 FFN 残差延拓：原文阅读记录

日期：2026-11-18。稳定文献 ID：L110。登记者：`/root/coherence_objective_review`。基点：`fa66269c6749d1843c02efa13e0429403c810650`。

## 来源与实际阅读范围

Zlatko Drmač、Serkan Gugercin，*A New Selection Operator for the Discrete Empirical Interpolation Method — Improved a Priori Error Bound and Extensions*，SIAM Journal on Scientific Computing 38(2), A631–A648 (2016)。

- [作者 arXiv v2](https://arxiv.org/abs/1505.00370v2)，[PDF](https://arxiv.org/pdf/1505.00370v2)，[期刊 DOI](https://doi.org/10.1137/15M1019271)。
- 已读摘要与书目信息、§1.3–1.4、§2.1 的 Theorem 2.1 及选点论证、§2.1.1 实现段落。
- 此登记未审阅数值实验、§3 随机扩展或外部作者代码；不标记全文或作者实现复现。

## 原文所支持的联系

§1.3 解释非线性求值的 lifting bottleneck。§1.4 式 (1.6)/(1.8) 定义从选中坐标重建非线性函数；Lemma 1.1 的式 (1.9) 将误差分成子空间遗漏和选点放大。§2.1 使用正交基转置的列主元 QR 选择坐标；§2.1.1 讨论实现。

## 在本仓库中的使用

当前 FFN 读出可写为静态结果加上由 64 行真实残差驱动的全域延拓。这个表示允许分别讨论特征表达不足、观测混叠与正则偏差。对应算子和误差分解由本仓库当前定义自行推导，见[成本与算子分析](../../20261118/READOUT_COST_COHERENCE_REASSESSMENT.md)。

当前特征随状态变化，保留静态先验，使用超定采样、岭正则和 BF16 执行；它与原 Q-DEIM 不是同一算法。文献联系不证明 Wan 视频协调性或部署加速。这里也不把原文 QR 选点与仓库的贪心 log-det 选点写成相同算法。

G14 使用既有焦点 T05/F06（非线性表示）与 T16/P20（重建目标）。本次未新增模型运行、校准、视频、任务树或实验计划。

## 导航同步

固定基点的主索引与焦点图已有 L99–L109，但绑定计数仍为 98，可见图缺少这 11 个节点。此次由既有映射补齐这些节点及边，再加入 L110，将计数同步为 110；旧记录的事实和阅读深度保持原值。这是派生导航修正，不构成对 L99–L109 的重新原文审查。

## 静态验证范围

原工具的 `validate_paper_focus` 实际通过：110 篇主记录、20 篇补充、16 个焦点、371 条论文到焦点映射。另已逐对象确认旧 109 条记录及阅读深度、旧焦点和补充记录保持不变。

完整 `graph_agent_bridge.py validate` 尚未闭合：本地只读快照缺少旧 `CHIPMUNK_STUDENT_OBSERVATION_RESULTS_20261030.json`（8,422,612 字节），contents 接口未返回正文，blob 接口重复报 `Transport closed`。未用空文件代替该报告，也不将本次焦点验证写成完整历史图验证通过。
