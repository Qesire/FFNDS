# FFNDS · E0 科研纲领重置：FFN 本征计算结构优先（2026-10-25）

**地位：当前研究纲领。** 本文规定 E0 科学问题和证据解释，不修改历史协议、checkpoint、量化实现、结果、运行权限或作者署名。唯一活跃状态仍为 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)；可执行的第一项任务为 [E0_FFN_FIRST_TASK_20261025.md](E0_FFN_FIRST_TASK_20261025.md)。

## 研究命题

少步视频生成 DiT 的 FFN 是否存在**随输入、位置、层、时间步和上下文而变化的计算需求**？这种需求由何种真实计算机制决定？能否利用这一结构，在控制真实生成轨迹质量的条件下减少 FFN 的必要执行代价？

**研究优先级：FFN 本征结构 → 功能机制与计算自由度 → 观测/决策的信息成本 → 量化与硬件中的实际部署。** 这是研究重心，不是所有 Agent 必须依次完成的流程。静态与动态、稀疏与重参数化均可产生启发；论文目标侧重真正有条件适应性的 FFN 节省。

**量化定位**：本轮采用工作假设——现有 HiF4/MXFP4/NVFP4 元信息主要反映幅值、动态范围、舍入及块统计，通常不足以独立决定 FFN 的有符号功能贡献与下游质量价值。此为有根据的优先排序，**不是一条宣称元数据在所有格式都只有幅值的定理**。有机制证据出现时仍可研究量化元数据；不能再以 HiF4 scale/code、QDQ、特定 64 组或 FP4 硬件作为默认算法设计入口。

## 一、真实 FFN 的数学对象：先从 FFN，而非量化器出发

对某真实 token i，省略必要时的 bias、modulation 细节，记
`u_i = norm2(residual_i)` 经 AdaLN 得到实际 FFN 输入 `z_i`，
`a_i = z_i W_1^T + b_1`，
`h_i = GELU(a_i)`，
`y_i = h_i W_2^T + b_2`，
`r'_i = r_i + gate_i ⊙ y_i`。

这里的矩阵方向为概念数学；具体 LightX2V 向量/批量 layout、dtype、Smooth/Q1/Q2 插入位置和 W1/GELU/W2 次序，应依据 [原始源码绑定](../20261023/C0_coupling_offline_results.json) 与 [实际 E0 数据](E0_NATIVE_OBSERVATIONS.json)重建。当前形状 `5120→13824→5120` 来自特定 Wan 双专家运行配置，不是所有模型的统一参数。

对每个 hidden 单元 j，有 `d_{i,j}=h_{i,j}(gate_i ⊙ W_{2,:,j})`，于是除了 bias 外 `gate_i⊙y_i = Σ_j d_{i,j}`。这是理解有符号通道作用和贡献交互的一个**准确实数分解**，还不是稀疏算法；非线性、量化和真实累加顺序要另行区分。

本征探索应区分四个对象：

1. **激活分布** `h`：幅值、支持集、有效秩、token/层/step 条件性。
2. **FFN 净作用** `d_j` 与 `Σd_j`：W2 方向、符号、组间 Gram、共同成分与局部创新，不能从 `|h|` 直接推出。
3. **功能后果**：residual bypass、norm1/attention、后续 block、跨 step trajectory，以及视频里的主体/背景/光照稳定性。
4. **可执行动作**：跳过或重新组织的 W1、GELU、W2 运算、可缓存项、可共享项、实际 GEMM tile/访存；完整读数/索引成本。

## 二、五个开放研究机制（不是五条既定方法）

| 机制入口 | 相互竞争的解释 | 关键区别性问题 |
|---|---|---|
| **M1 条件贡献集中** | 真正可用的 FFN hidden/unit/group 贡献集中 vs 仅元素很小、硬件 tile 仍密集 | 强度/有符号 W2 投影/整 tile 支持率、不同 source/step 的次序是否一致？ |
| **M2 有符号交互与表示自由度** | 少量方向具有独特作用 vs 大量互相抵消/同向重叠、净输出处于较低维结构 | 贡献 Gram、对齐、谱随输入如何变化？重表示是否能减少真正的算术而不靠事后完整 `h`？ |
| **M3 时空条件变化** | 绝对 FFN 输出相近 vs 输出增量/方向随 step、expert、token 变化；静态共同分量不等于可复用 | 同专家相邻步、跨专家、真实空间局部/非局部相似性及时间相关性谁决定必要计算？ |
| **M4 功能敏感性与传播** | 低局部误差对应低视频作用 vs 小规模特定误差通过 residual/attention/scheduler 影响背景光照和动作 | 在同局部误差条件下，方向、位置、时间哪一种因素决定真实下游排序？ |
| **M5 数学/执行重构** | 跳过部分 unit vs 共享子空间、重排计算、条件低秩/分块/局部增量或另一种 FFN 表示 | 能否明确列出执行前可得的信息、被移除的真正计算及近似质量的竞争解释？ |

**研究扩展自由**：新论文、异常现象或数学重写可独立开第六条机制；没有要求所有发现都必须依附径向几何、通道稀疏、缓存、补算、Proxy 或 64 分组。一次漂亮的 LN Jacobian 分解不代表优先研究方向已经锁定。

## 三、研究方法：AutoResearcher v2.4 默认机制

- **NOTICE/EXPAND**：从 FFN 源码、旧失败、数学、真实视频质量、前沿论文发现一个值得解释的结构，允许先提出有内容的猜想。
- **MAP_FLOW**：对所声称的机制复建 decision-critical 张量和计算顺序（真实 Wan 配置、dtype、gate、residual 和下游），不用熟悉的标准结构代替真实实现。
- **DERIVE/PRD**：使用准确代数、贡献 Gram、条件协方差、时空创新或跨表示的 E/T/R 问题重构；区分恒等式、条件推论、现有观察、机制假设。
- **WARD**：广泛深入的原始论文/源码搜索；追溯前后向引用，寻找最强的近邻和最有价值的反例，主动修订假说。
- **DISCRIMINATE**：优先数学、源码、既有资产；明确两种机制有何不同预测。新原生实验、GPU/模型/视频生成需独立用户授权。
- **CARRY**：独立 Agent 保存发现/挑战，不要求已证明可部署、也不以相互签名取代证据。整理者只基于真实独立贡献更新讨论汇总。

### 四个不可混同的尺度

`h` 的近零率 ≠ `d_j` 的净贡献 ≠ 下游因果价值 ≠ 实际可加速的 GEMM 结构。保留从这四者寻找桥接机制的创造性空间；已有 A5+HWG 的光照/小物体波动是重要的真实问题提示，不是推动恢复 Dense 计算的预设答案。

## 四、把旧证据放回正确位置

- [E0_RADIAL_PAIRING](E0_RADIAL_PAIRING.md) 与 [独立评议](../method_discussion/discoveries/E0/gpt6-independent-radial-pairing-judgment-review-20261025.md)：表示/局部 norm 的结构，**不是全局省算率或首选算子**。
- [E0_FOLLOWTHROUGH](E0_FOLLOWTHROUGH.md) 与 [E0_SHARED_REFERENCE_MOMENTS](E0_SHARED_REFERENCE_MOMENTS.md)：在特定 QDQ/BF16 运行下的有限动作与传播，不推出所有精度与模型下的因果边界。
- [E0_TEMPORAL_CALIBRATION](E0_TEMPORAL_CALIBRATION.md)：候选时序现象；固定共同均值/scale 的事后吻合不证明可前置缓存。
- [实验优先级目录](../EXPERIMENT_EVIDENCE_INDEX.md)：旧负结果只淘汰相同算子/预算/数据设置下的具体命题；不得推广为 FFN 一般不可稀疏。
- [量化专项旧引导](../20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md) 及 [二轮评议](E0_SKILL_WEB_REAUDIT_20261025.md) 继续可检索，但不再作为活跃任务与优先项。

**证据条件迁移**：量化无关的数学结论可以跨格式使用；在量化路径上观测到的大小、排序和质量只能按原数值边界陈述。BF16 对照是可检验的另一数值环境，不应默认与当前 D-Q checkpoint 或输出相同。

## 五、成功的科研推进是什么

下一阶段首先找到一项与量化元数据无关或较少依赖的 FFN 结构：精确描述分布和条件，给出至少两种不同机制解释、一个能够真正区分它们的最小对照，以及可能打开的计算自由度。不要求立即提交 kernel。

当研究进入可执行方法阶段，再逐步要求：明确真实可删/可共享的 GEMM 工作、信号出现时点、路由/索引/访存代价、matched static/dynamic/published 基线、原生 FFN 和整段延迟及四步自由生成视频的独立质量。当前 E0 保持方法讨论，不自动发布候选或实验。

## 六、相关近邻科学：与 FFNDS 比较的出发点

参见配套 [FFN 本征机制文献地图](E0_FFN_INTRINSIC_PRIOR_ART_20261025.md)。尤其核对 Chipmunk 的 MLP cross-step sparse delta；RT-Lynx 的激活 N:M；Activation Concentration 的 element-vs-column 差异；DiffSparse 的少步条件；ICLR 2026 Activation Sparsity 的适用对象。借用结构和反例，但不能把其它模型/硬件或论文的结论转写为本 Wan 已实测。

**完整重置不意味着删掉旧数据，而意味着**：研究问题、默认入口、证据排序、Agent 激励和查新视角都以 FFN 本身为第一研究对象。
