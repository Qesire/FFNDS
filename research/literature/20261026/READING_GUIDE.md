**G9条件计算与跨步预测（2026-10-27）：** [L²P / FFF / FFN↔attention训练期迁移一手研究](../20261027/G9_CONDITIONAL_EXECUTION_TRAJECTORY_AND_FUNCTION_REALLOCATION.md)和[逐论文证据账本](../20261027/G9_METHOD_SOURCE_LEDGER.json)已加入[计算图 G9 K28–K37](../../../docs/98_端到端计算图与证据注解_20261025.md)。L²P已读取CVPR官方发布与作者FLUX历史预测源码；FFF已核原始v1方法公式/消融；Sparsity Moves Computation只有作者摘要，需深读方法正文。研究比较分清“缓存整个模型输出”“新训练的树状条件函数”“训练期全网角色迁移”三种不同的省算机制；现实Wan四步执行仍暂停。

**G8 methods and method-status audit (2026-10-27):** [RT-Lynx, Dense2MoE, SVD-Cache and Chipmunk computations](../20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md) + [per-paper source tier](../20261027/G8_FOUR_METHOD_SOURCE_LEDGER.json). Dense2MoE and SVD-Cache now have original v1 method equations read, correcting earlier abstract-only and no-method claims. RT-Lynx official main-paper method remains unavailable; use author HF post as primary author statement, not verified formula/code. When reviewing other related works, attach each to G7/G8 or create a distinct branch with tensor shape, legal runtime signal and costs.

# FFNDS · 文献Agent阅读与升级证据契约

**第二轮可复核深读：** [五篇近邻方法审查](FIVE_PAPER_METHOD_AUDIT.md)与[5篇证据等级账本](METHOD_AUDIT_LEDGER.json)。后续Agent首先补齐RT-Lynx/Dense2MoE完整原始方法源码，Activation Concentration阈值/列的严格定义，以及DiffSparse原Wan分项VBench；Chipmunk已亲读MLP源，但没有Wan四步实验。

**研究版本演进：最初51项（其中18个追加近邻），G9后更新至54项。**先读[新研究地图](ADJACENT_METHOD_EXPANSION_20261026.md)和[已检查原始源码与公开代码状态](ADJACENT_SOURCE_METHOD_REVIEW_20261026.md)。下一轮方法级重点扩至Dynamic-in-Few-Step、FIS-DiT、Mobile Video DiT、NerVE、SVD-Cache、FastLightGen、PARE，再结合RT-Lynx/Chipmunk/Dense2MoE/Activation Concentration/DiffSparse。阅读顺序按科学重叠和证据变化选择，而非要求固定一条算法路线。

**当前优先级：原始文献查验与竞争机制分类。** 完整54项记录在[持续维护的文献索引](PRIMARY_LITERATURE_INDEX.json)，研究总图见[Atlas](PRIMARY_LITERATURE_ATLAS.md)，关系见[矩阵](MECHANISM_COMPETITION_MATRIX.md)。

每个paper ID深读时分别填写：原论文及版本、实际阅读章节/图、中心计算方程和前提、方法究竟使用什么已知输入及何时可知、训练/校准类型、online重算/缓存/GEMM与显存成本、原生kernel和真实时延、分项视频指标、失败消融、竞争论文与本项目现有机制的同构/不同，以及能区分两种解释的最低成本观测。

**阅读深度升级**：OFFICIAL_ABSTRACT仅说明已查到作者在摘要报告的机制。只有读过方法和消融才能升级为PAPER_METHOD_CHECKED；只有核对原始可执行实现、commit及数据shape才能升级为CODE_CHECKED；只有真实可绑定Wan模型/输入/硬件RUN才算WAN_NATIVE_VERIFIED。切忌以读过PDF标题代表读完方法，或以参考实现benchmark冒充WanFFN省算。

建议四个并行但独立专题：①直接DiT FFN：RT-Lynx/Chipmunk/Dense2MoE/OBS-Diff/Activation Concentration；②条件复用及四步：DiffSparse/TeaCache/ToCa/HyCa/背景反例；③表征和数学：Geva/superposition/FLAP/SliceGPT/TinyFusion；④物理与评价：GPU稀疏tensor tile/NVFP4、VBench/VMBench、注意力类比的严格转移条件。Agent间可互相质疑，但研究不需要强制收敛到某算法或评分。

**对外声称方式**：PRIMARY PAPER作者结论、已有WAN实验事实、TRANSFER HYPOTHESIS分别书写；对不同模型、步数、双专家、Quant/QDQ、CFG状态给出源代码层面的适用条件。依照AutoResearcher原则提出开放探索和可证伪机制，避免表述变成冗长限制清单。
