# FFNDS · 深入文献与原始代码检索

**AutoResearcher v2.4 alignment:** This FFNDS-specific protocol extends the upstream [Web Research Discovery / WARD](../../core/WEB_RESEARCH_DISCOVERY.md) and its [pinned Skill](../../.agents/skills/web-research-discovery/SKILL.md). Exact source revision and per-file blob hashes are recorded in [ACTIVE_SOURCE_MANIFEST.json](ACTIVE_SOURCE_MANIFEST.json). Follow the shared breadth, depth, citation tracing and primary evidence method; specialize each search to actual Wan numerical/data-flow questions.


**适用：** 所有 FFNDS 科研探索、机理推导、方案构想、文献查重、方法评议及科研方向调整。当前阶段由 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 确定为 `E0_OPEN_DISCOVERY`，搜索服务于**发现真实现象和竞争性解释**。

## 默认科学行为

每次实质性科研工作主动开展**广泛、深入、多轮迭代的 Web Search**。先建立相关科学领域与已有原理的地图，再沿最有价值的机制追溯原论文、核心公式、消融、实验设计、原始实现与可迁移的失败条件。

研究成果应回答：**外部文献揭示了什么我们尚未考虑的结构？最接近的机制在哪些数学和计算边界上相同或不同？接下来哪一项证据会改变我们的认识？**

## 广度：围绕科学问题连接不同知识簇

- **模型与任务：** Wan、DiT、视频生成、Flow Matching、少步蒸馏与跨步/跨层/时空表示。
- **FFN 与数学：** W1→GELU→W2、residual/gate/LayerNorm、方向抵消、有效子空间、特征几何、条件计算、信息价值。
- **量化机制：** HiF4 QDQ、W4A4、E2M1/FP4、MXFP4/E8M0、NVFP4/E4M3、scale/code/exponent、层级共享尺度与量化边界。
- **系统实现：** GPU Tensor Core、原生 FP4 GEMM、scale layout、内存流量、kernel fusion、动态稀疏与实际物理 tile。
- **邻近理论与反例：** 视觉 registers/高杠杆 token、角度重要性、误差传播、扩散视频稳定性、跨步共享、近似计算和失效机制。

同时拓展经典理论、当前预印本、近年顶会、工程报告；使用中英文、多种术语、同义称呼与计算流程描述动态扩展检索。探索目标会随新发现而改变，搜索空间也应随之拓展。

## 深度：把文献变成可比较的机制

1. **宽检索：** 建立经典/前沿/相邻机制图，识别最接近的工作和解释张力。
2. **原始阅读：** 精读核心方程、方法顺序、实际输入输出、损失/目标与关键消融；进入作者正式论文和官方代码检查实现。
3. **引用追踪：** 沿 backward/forward citations、作者连续工作、模型与算子版本更新继续搜索，补充不同于第一批结果的先例。
4. **对抗性检索：** 主动寻找能否定当前主张的论文、严格条件、负面实验与另一种机制解释。
5. **来源交叉验证：** 将学术版本、原始代码、硬件/格式规范与 FFNDS 冻结的 Wan 源码对照。记录观察和推导在各实现中实际对应到哪一层。
6. **回到新问题：** 从搜索中提炼最有启发的未解规律、可证伪预测或新的数学表征，挑选下一项有区别力的检查。

常见入口：arXiv、OpenReview、ICLR/NeurIPS/CVPR/ICML 官方会议、作者网站、Google Scholar、Semantic Scholar、OpenAlex、GitHub 原始仓库、OCP 和 NVIDIA 官方格式/算子文档。可使用 [Semantic Scholar Academic Graph](https://api.semanticscholar.org/api-docs/)、[推荐与引文](https://api.semanticscholar.org/api-docs/recommendations)、[OpenAlex 搜索](https://help.openalex.org/api/searching/) 与 [筛选](https://help.openalex.org/api/filtering/) 追踪来源。

## 机制级比较与查重

对每个最相近的方法，比较以下内容：

| 比较维度 | 需要获得的原始信息 |
| --- | --- |
| 实际对象 | 模型、任务、视频/少步蒸馏、量化状态、所用硬件 |
| 信息时序 | 方法在何时知道什么，哪些统计是原本保存的、前置可读的或后验标签 |
| 数学机制 | 核心方程、非线性/尺度顺序、等价条件、主要反例 |
| 物理动作 | 真实更改的 GEMM/QDQ、数据搬运、kernel 布局与代价 |
| 实验证据 | 数据来源、质量指标、基线、消融、真实计时和适用条件 |
| 与 FFNDS 的关系 | 同构方法、共享基础组件、相邻但不同的机制、可迁移启发或仍待检查的联系 |

将最近的先例与旧 Proxy+HWG、静态/动态组选择、FP4 LUT、跨步缓存及新的 E0 残差径向现象作**源与方程级**比较。论文标题与关键词主要用于定位，研究结论基于真实数学关系和计算流。

## 输出可复核的检索依据

研究记录给出：

- 检索日期、实际使用的搜索途径、代表性关键词族、引用追踪路径；
- 重点论文作者/年份/版本和原始链接、关键公式/源码位置、最近的竞争方法；
- 哪一条来源改变了科学解释，新现象与已有工作究竟在哪一处不同；
- 证据类型：`PRIMARY_PAPER`、`OFFICIAL_CODE_OR_SPEC`、`ARCHIVED_WAN_MEASUREMENT`、`MATH_CONDITIONAL`、`SECONDARY_INDEX` 或 `CONJECTURE`；
- 尚值得继续检索或实际观测的领域。

若当前研究节点的网络受限，记录可访问的本地论文/源码与已经完成的数学分析，并将在线检索的未覆盖范围单独标注。学术查重表述与实际核查范围相匹配。

## 多 Agent 的互补探索

不同 Agent 各自深入量化与硬件、FFN 数学、蒸馏时空、后续质量几何等领域，同时沿新发现跨域连接。独立 reviewer 亲自追踪最可能同构的原始实现与反例。整理者聚合**有差异的证据链和竞争机制**，使共同结论建立在真实相互独立的研究上。

Web Search 所产生的新算子想法作为 `idea seed` 保留；当前 E0 的主要交付仍是**新的现象解释、可复核的数学关系和区分性下一观察**。

**资源权限：** Web 检索、原始资料和已获准保存数据的研究按当前 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 执行；新的 CPU/GPU/model 作业由用户另行明确授权。
