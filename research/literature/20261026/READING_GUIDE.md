**G8 methods and method-status audit (2026-10-27):** [RT-Lynx, Dense2MoE, SVD-Cache and Chipmunk computations](../20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md) + [per-paper source tier](../20261027/G8_FOUR_METHOD_SOURCE_LEDGER.json). Dense2MoE and SVD-Cache now have original v1 method equations read, correcting earlier abstract-only and no-method claims. RT-Lynx official main-paper method remains unavailable; use author HF post as primary author statement, not verified formula/code. When reviewing other related works, attach each to G7/G8 or create a distinct branch with tensor shape, legal runtime signal and costs.

# FFNDS · 文献Agent阅读与升级证据契约

**第二轮可复核深读：** [五篇近邻方法审查](FIVE_PAPER_METHOD_AUDIT.md)与[5篇证据等级账本](METHOD_AUDIT_LEDGER.json)。后续Agent首先补齐RT-Lynx/Dense2MoE完整原始方法源码，Activation Concentration阈值/列的严格定义，以及DiffSparse原Wan分项VBench；Chipmunk已亲读MLP源，但没有Wan四步实验。

**研究新阶段：51项，含18个追加近邻。**先读[新研究地图](ADJACENT_METHOD_EXPANSION_20261026.md)和[已检查原始源码与公开代码状态](ADJACENT_SOURCE_METHOD_REVIEW_20261026.md)。下一轮方法级重点扩至Dynamic-in-Few-Step、FIS-DiT、Mobile Video DiT、NerVE、SVD-Cache、FastLightGen、PARE，再结合RT-Lynx/Chipmunk/Dense2MoE/Activation Concentration/DiffSparse。阅读顺序按科学重叠和证据变化选择，而非要求固定一条算法路线。

**当前优先级：原始文献查验与竞争机制分类；非新算法实现。** 正式记录在[33篇索引](PRIMARY_LITERATURE_INDEX.json)，研究总图见[Atlas](PRIMARY_LITERATURE_ATLAS.md)，关系见[矩阵](MECHANISM_COMPETITION_MATRIX.md)。

每个paper ID深读时分别填写：原论文及版本、实际阅读章节/图、中心计算方程和前提、方法究竟使用什么已知输入及何时可知、训练/校准类型、online重算/缓存/GEMM与显存成本、原生kernel和真实时延、分项视频指标、失败消融、竞争论文与本项目现有机制的同构/不同，以及能区分两种解释的最低成本观测。

**阅读深度升级**：OFFICIAL_ABSTRACT仅说明已查到作者在摘要报告的机制。只有读过方法和消融才能升级为PAPER_METHOD_CHECKED；只有核对原始可执行实现、commit及数据shape才能升级为CODE_CHECKED；只有真实可绑定Wan模型/输入/硬件RUN才算WAN_NATIVE_VERIFIED。切忌以读过PDF标题代表读完方法，或以参考实现benchmark冒充WanFFN省算。

建议四个并行但独立专题：①直接DiT FFN：RT-Lynx/Chipmunk/Dense2MoE/OBS-Diff/Activation Concentration；②条件复用及四步：DiffSparse/TeaCache/ToCa/HyCa/背景反例；③表征和数学：Geva/superposition/FLAP/SliceGPT/TinyFusion；④物理与评价：GPU稀疏tensor tile/NVFP4、VBench/VMBench、注意力类比的严格转移条件。Agent间可互相质疑，但研究不需要强制收敛到某算法或评分。

**对外声称方式**：PRIMARY PAPER作者结论、已有WAN实验事实、TRANSFER HYPOTHESIS分别书写；对不同模型、步数、双专家、Quant/QDQ、CFG状态给出源代码层面的适用条件。依照AutoResearcher原则提出开放探索和可证伪机制，避免表述变成冗长限制清单。
