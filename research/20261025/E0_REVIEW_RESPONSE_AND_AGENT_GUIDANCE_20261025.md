# E0 — Skill / Web 复审后的机制研究指引（2026-10-25，修订版）

**独立复审与完整理由：** [E0_SKILL_WEB_REAUDIT_20261025.md](E0_SKILL_WEB_REAUDIT_20261025.md)。依据 `main@1e1a0e0` 的独立评议、源脚本及保存结果，并按 AutoResearcher v2.4 的 mechanism-research、research-expansion、problem-reformulation 和 WARD 原始技能重新检验。保留先前观察值，更新对象精度、竞争机制和讨论分工。

## 核心问题

**低成本可观测信息在 Wan FFN 的哪个真实计算边界出现，如何被残差与后续网络读取，是否存在不同于昂贵 dense FFN 的可执行计算路径？** 先从现象提出不同解释和可区别预测，再拓展表示/操作/时序；算子只是可能的研究结果。

### 三个已观测现象的恰当含义

- Block0 的原始理想 LN 径向平方比例：正确配对 **19.794%**，错配 **20.542%**。跨位置去均值后是 **8.839%** 对 **3.590%**，但后者改变了用于 Jacobian 计算的原始 `x` 与 `delta`，是表示探针，并非原生 nextnorm 的消去比例。
- 其他五来源对当前来源 residual 的均值方向 cosine **0.99645**，是相同 seed42、相同固定 32 行采样及 step/block 设置的六来源内部结果。32行样本均值既不等于全部 patch token 的全局均值，也不是固定网络 bias 的直接证据。
- 四臂 star-null 距离和不一致是在检查**共同参考、交叉矩为零**的特定二阶模型；不是两组动作非加性的定义，亦无法独立把变化归因于 BF16。

## 首先确认真实计算对象

当前研究是 LightX2V 绑定的 Wan I2V D-Q/HiF4 四步实现；必须复建其 `norm2→AdaLN→Q1/W1→GELU→Q2/W2→gate→residual`，以及下一 block `norm1→self-attention` 与 bypass 的相关分支。参考实际版本和源 SHA：`research/20261025/E0_RESIDUAL_GEOMETRY.json`、`research/20261022/R12_E0_runtime_reconstruction.json`；[官方 Wan 源码](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py) 是交叉检查，不替代本次 LightX2V 绑定。

保存脚本构造的 `delta_{i,g}=diag(gate_i)W2_g q2_{i,g}` 是 **FP64 decoded panel group contribution**，不是完整原生 BF16 有限动作或精确写回增量。所有结论标记这一区别。Block39 为末层，不存在通常意义上的下一个 block norm1。

## 五个互补科学问题（Agent 自选独立切面）

### A · 四臂交互和数值边界

从原始 D/A/B/AB 保存结果出发，先构造 `a=A-D, b=B-D, c=AB-A-B+D`，再报告 `c`、响应 Gram、全部六距离、原 star-null 配对和、非负零模型拟合。做最小反例：`c=0` 不意味着 `S1=S2=S3`；把 **动作非加性**、**误差相关性**和 **BF16/连续传播**分开。检查原生已存 flow/post 是否真的能定位舍入边界；若无法定位，明确需要的最早额外张量边界及其两种机制预测。

**交付**：精确向量恒等式与跨 source/step/subspace 对照，一个足以推翻混合归因的例子，下一步最短判别观察。

### B · 真实状态的固定 Jacobian 与功能路径

对每个 source/step/block/group 的固定采样位置 `x_i`，令 `delta_i` 为已存 decoded FP64 group contribution；在**原始同一 x_i** 的 ideal `J_i=J_LN(x_i)` 下分解 `delta_i=mu+eta_i`，独立计算 `||J_i mu||², ||J_i eta_i||², 2〈J_i mu,J_i eta_i〉`、signed covariance 与来源异质性。与原 `J_LN(x_i−mu_x)` 的 centered probe 明确区分。

在项目固定 LightX2V 真实 norm1/norm2、epsilon、affine、AdaLN、FP32/BF16 cast 顺序下说明 ideal J 的适用边界；针对真实 residual identity、后续 attention/FFN 和四步轨迹构造至少一个与局部 LN 消去不同的预测。下游 finite differences 为另外一层证据，不由理想 J 自动给出。

**交付**：真实 operand 字段映射、固定 J 三项表、原先 raw/centered 反转的签名解释与反例。

### C · 采样公共方向、seed 与条件时序

从实际 32 行采样协议、patch 坐标、共同 mean 的构造检查跨来源 cosine 的解释力；比较同/不同 step、expert、gate、AdaLN、pre/post、attention 的可获取字段与结构预测。区分场景固定方向、共享 seed/位置协议、条件驱动的动态分量；以来源为聚类单位。明确不同 seed/相机/checkpoint 所能区分的机制，只有获新实验授权后才执行新采集。

**交付**：机制 × 预测 × 已有字段矩阵，不以“六来源高 cosine”推断功能性或可缓存性。

### D · 量化码字/尺度的信息出现时机与执行单元

从 HiF4 Q1/Q2 的具体 native QDQ 与 BF16 GEMM 路径，逐层核对 codes、scales、amax、Smooth 参数是在 W1 或 W2 **之前还是之后**产生，哪些读数需要额外遍历、去量化、同步或已完成昂贵计算；与 MXFP4（32组）/NVFP4（16组及两级缩放）官方格式区分。提出定量信息/动作关系或数学可重排结构，指出对应可物理省略的 K/N panel、权重读写或 kernel 次数，以及真实 A100 与未来 Blackwell 的代价差别。

**交付**：信息时间和成本地图，至少两种相互竞争的可执行自由度。探索种子欢迎大胆，但与已实现原生性能分层记载。

### E · 开放问题重构、跨领域先例与净信息价值

从 `Q1` 上游可用信息、`Q2` 净信息价值、`Q3` 实际可省算出发，主动用 WARD 深搜论文/官方代码：包括各向异性/rogue dimensions、massive activations、[SVD-Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)、[HyCa](https://proceedings.iclr.cc/paper_files/paper/2026/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html)、[RFC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4019681ef05e8e347c3067d7f973a364-Abstract-Conference.html)、[DiffSparse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html) 和[官方 NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)。逐一对比信息何时出现、依赖历史步数、是否训练、所省运算和终端验证。

通过 PRD 尝试比“公共方向 + 局部变化”更有解释力的表示或计算时序重写；保留还未具备完整性能证明、但已有明确可证伪机制的 idea seeds。

**交付**：改变研究决策的一项机制级近邻比较、至少一个新的表示/问题重构、对应分歧预测及最低成本检验。

## 独立讨论与创造性写作

各 Agent 只参与**方法、机理、数学、已存证据及文献讨论**，在 `research/method_discussion/discoveries/E0/` 独立署名记录；审查者独立读源/推导，不能以一致口吻代替实质同意。欢迎超出上述五类的机制解释，每项报告首先给出**新认识或有力假说**，随后是来源、最强竞争机制、区别性观察及潜在计算自由度。

默认使用 pinned AutoResearcher v2.4 的 mechanism-research、research-expansion、problem-reformulation、WARD。多轮 web 搜索包括原论文方程、反例、官方源码与前后向引用，写明已真正阅读的范围。方法讨论可以提出大胆研究种子，无须先证明最终省算；推进原生算子验证时再完整计算部署代价和质量。

**允许的当前工作**：阅读源码/Skill/原文、数学证明、在已有权限范围内复算保存资产的离线 CPU 数值分析。**新范围须单独授权**：模型/GPU/新生成/训练、扩大的原生干预、真实完整视频质量和 speed benchmark。研究阶段保持 `E0_OPEN_DISCOVERY`，候选选择和同行状态不自动改变。

## 下一次分歧判别的优先次序

1. B：先让 fixed-J 与原 centered-probe 的数值对象严丝合缝，不能用新 centered-J 冒充真实网络响应。
2. A：区分 star-null、`c` 与 Gram，明确哪种非线性归因能用已有边界定位。
3. D：在真实 Q1/Q2 计算点调查已有低成本元数据，恢复研究起点的量化信息方向。

C/E 同时开展独立开放探索。只有日后进入候选算子验证时，再要求明确计算节省、信息代价、matched competitor、全四步视频质量及真实端到端 latency。
