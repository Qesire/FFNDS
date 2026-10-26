# FFNDS · 学术近邻与竞争解释矩阵

**方法级核查补充：** [五篇核心机制差异与实际实验模型](FIVE_PAPER_METHOD_AUDIT.md)。DiffSparse的token缓存预算在离线训练中优化，Wan2.1为25步、FLUX-schnell为4步；Chipmunk的跨步MLP缓存在真实源码中先支付blockmean+W1的选择成本。这改变了与FFNDS的四步合法比较基础。

**第三轮新增的直接竞争先例：** [18篇研究与Wan四步差异](ADJACENT_METHOD_EXPANSION_20261026.md)、[源码级实际边界](ADJACENT_SOURCE_METHOD_REVIEW_20261026.md)。特别注意Dynamic-in-Few-Step的四步Wan阶段结构、FIS-DiT的frame×layer计算、Mobile Video DiT的FFN channel静态缩窄及NerVE的非线性函数谱。它们分别阻止“联合少步动态剪枝/训练自由帧稀疏/FFN channel压缩/非线性谱”被当作无已知强先例的新颖机制。

本矩阵用于**机制查重**，不是结果晋级或强制算法清单。先查[33篇原文入口与研究深度](PRIMARY_LITERATURE_INDEX.json)。

| 研究对象 | 代表先例 | 与FFNDS既有工作的相似点 | 必须保持的不同假说 |
|---|---|---|---|
| DiT激活N:M | L08 RT-Lynx、L12 Activation Concentration | 曾测GELU/Q2稀疏与近似均匀的token×channel散点 | 2:4原生kernel收益、列/tile闭包、同一步活跃≠跨步变化，补偿和LoRA属训练/额外计算 |
| MLP跨步delta | L09 Chipmunk、L14 TeaCache、L16 ToCa | 旧跨步、value向量与输出状态观察 | 仅四步，high/high/low/low；历史缓存容量、状态输入差、原native Q2不能假设与其它DiT一致 |
| 结构化条件容量 | L10 Dense2MoE、L19 DiffSparse | 曾做分组、动态决策和全40层近似 | 带训练的专家/稀疏调度有更强表征容量，不等于固定bank零训练掩码 |
| 表征、共享函数 | L01 Geva、L02 superposition、L04 FLAP、L06 SliceGPT | 四anchor固定函数oracle、Affine54原函数先验与ridge | 共享anchor自我重建≠neighbor新增信息；正交基一般不能穿GELU和Q2 |
| 时间与功能敏感性 | L11 OBS-Diff、L13 TinyFusion、L29 VBench、L30 VMBench | 两source全模型有限反事实与可见视频偏差 | 局部SSE/频带有符号项不是运动、照明和细节的因果真值；小后训练与零训练目标不同 |
| 可执行稀疏与低位 | L12 Activation Concentration、L31 NVFP4 | HiF4 QDQ→BF16和SM120已测合成packed GEMM | 数字格式加速≠少MAC；GPU sparse tile、重排、scale及在线量化成本需独立核对 |
| 静态vs动态规律 | L24 Sparse-vDiT、L25 SVOO、L26 DFSAttn | 关注跨source/层/step稳定性 | QK注意力块并非FFN神经元，粗块与细粒度发现可能互相挑战 |
| 动态场景与终端视频 | L15 AdaCache、L17 ProfilingDiT、L18 FasterCache | 历史背景/小物体/光照不稳、motion异质 | CFG在当前Wan关闭；背景看似静态仍可能承担全片稳定性 |

**后续同行任务**：从不同类别分别寻找一个最强近邻与一个能产生相反预测的反例；补读原始方法图、完整源码、训练和显著消融。用实时计算图给每一条新方案判断已知原理迁移、方法同构、部分重叠或新机制，而不是用论文标题进行新颖性判定。当前无新GPU作业或论文实现复现。
