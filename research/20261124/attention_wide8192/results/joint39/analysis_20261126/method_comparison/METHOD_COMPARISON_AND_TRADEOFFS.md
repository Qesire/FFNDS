# 与 Chipmunk 等方法比较：同一计算预算买到什么信息

日期：2026-11-26。当前方法证据固定于 FFNDS 研究分支 154fe8620724bf1fbe6f0f9b6c5950e154468af4。当前对象是 Wan2.2-I2V-A14B 四步蒸馏、HHLL、wide8192/P64，并包含 low attention 的 mass-centered 状态。新的执行包括原始计时/质量记录的 CPU 复算、实际形状的成本计算、条件代数核验；本轮未新增 GPU 推理。

## 1. 研究判断

我们已经形成了一个有明确机制区别的研究对象：先用当前结构和历史状态形成全位置响应，再用少量当前原生输出恢复剩余变化。与 Chipmunk 相比，新增能力集中在未被 selected 计算覆盖的变化如何更新；与 ICC 相比，差别集中在固定低秩权重近似之外，怎样利用当前完整模块响应；与 SparsePR 相比，差别集中在全位置历史状态如何降低当前恢复任务的难度。

当前结果支持的定位是：在已经压缩到四步的固定模型上，以有成本的当前观测和可解释的状态解码，换取完整周期中的质量保留与额外加速。完整 C 已有六来源相对同四步 Dense 的 GPU 1.227×、完整请求 1.156×；当前 centered 另有两来源自身轨迹、成本执行和用户整体画面可接受反馈。[E1]

Chipmunk 的成熟能力同样清楚：细粒度动态支持、专用稀疏内核、缓存写回融合，以及更长轨迹和多个模型的系统证据。两者的得失应由相同模型、质量目标和计时边界比较。作者的几倍系统倍率与当前四步增量倍率，分别回答不同分母下的问题。

## 2. 最近机制及其实际信息来源

| 方法 | 命中时真正做的事 | 与当前 FFNDS 的关系 | 对方的优势与我方付出的代价 |
|---|---|---|---|
| Chipmunk FFN | 用当前 token 块均值预测变化大的隐藏列，重算其激活增量，更新旧输出 | 当前输入、稀疏增量和历史响应是共用构件；我方增加完整线性主项和 P64 当前输出恢复 | 对方动态局部支持、较轻 selected 算术和专用内核；我方付完整 d×d 投影与恢复 |
| Chipmunk attention | 当前 selected 内归一化输出，加历史 dense−selected 输出残差 | 我方 low 底座改变 selected 质量权重，并在当前 value 坐标读取历史；再用 P64 更新 | 对方缓存读取与命中步骤更简单；我方付统计、当前原生 query 和读出 |
| ICC | FC1 低秩增量、当前 GELU、FC2 低秩增量；权重分解由通道统计校准 | 两者都由当前输入增量驱动；我方额外观测当前完整模块输出 | 对方以 rank 控制规则 GEMM；我方把预算用于 selected 非线性、全输入预测与输出约束 |
| SparsePR | 当前分组/路由、P64 全 key 输出、加权 affine ridge 与 probe 覆写 | 与我方 high OUT-ABS 共享当前绝对残差恢复机制；low 的历史运输与创新目标不同 | 对方无完整历史输出残差字段；我方保存全场状态并支付锚点成本 |
| MC-Sparse | 精确概率选 token 支持，缓存分组和 dense−sparse 残差，命中重算当前 sparse | 同底座/路由下，静态输出残差复用与 Chipmunk 同属近邻；我方再运输并观测创新 | 对方支持更细、选取更准确、命中便宜；精确 two-pass 锚点依赖长段摊销 |
| DisCa | 训练轻量 DiT predictor，在蒸馏模型上替代部分完整 DiT 调用 | 同样研究蒸馏后的额外加速，但预测器来源、替换范围和当前观测不同 | 对方可替代更大计算范围；我方保留现有权重，使用校准及在线小规模求解 |

Chipmunk、ICC、SparsePR、MC-Sparse 的公式和实现核查见三个附录报告。[C1–C3] DisCa 原文的 11.8× 是 Restricted MeanFlow 与可学习缓存组成的系统结果；官方仓库还提供 HunyuanVideo1.5 I2V 四步示例。它是“蒸馏兼容与合并系统指标”已有直接先例。[L5] 本轮也核对了 Cache-DiT 作者公开的 4/8-step 支持说明；因此四步可用本身应作为实验条件和实证能力，贡献的区分来自具体操作。[L6]

## 3. 与 Chipmunk attention 的精确差别

固定层和 head，采用行存 token 的约定。完整输出为 A_t∈R^(N×d_v)，B_t 是在 selected key 集合内重新归一化的当前 attention 输出，a 表示锚点。作者代码的刷新过程实际计算一次完整 attention 和一次 selected attention，并保存 A_a−B_a；命中为

\[
\widehat A_t^{Chip}=A_a+(B_t-B_a).
\]

这已经同时使用历史与当前 Q/K/V。其便宜近似是让完整−selected 的输出残差保持历史值。作者 FFN 也确实计算当前 selected 激活，并非直接冻结所有输出。[L1]

当前 low centered 定义 query 向量 rho_a、value 通道均值 mu_t 和非零标准差对角阵 D_t：

\[
Z_a=
\{A_a-\operatorname{diag}(1-\rho_a)B_a-\rho_a\mu_a^\top\}D_a^{-1},
\]

\[
b_t^{centered}
=\operatorname{diag}(1-\rho_a)B_t+\rho_a\mu_t^\top+Z_aD_t.
\]

于是，对相同 selected 支持有精确差式：

\[
\boxed{
b_t^{centered}-\widehat A_t^{Chip}
=-\operatorname{diag}(\rho_a)(B_t-B_a)
+\rho_a(\mu_t-\mu_a)^\top+Z_a(D_t-D_a).
}
\]

三个变量分别是 selected 变化的质量权重、当前 value 均值、历史场在当前 value 尺度上的读出。随后本方法还计算当前 P64 原生响应：

\[
\widehat A_t=b_t^{centered}+K_t(A_{t,P}-Pb_t^{centered}),
\]

并在 P 行覆写真值。这把核心差异落实到“历史状态如何更新”和“当前未建模变化从哪里取得”，而不是用共同的加速目的区分方法。

### 一个可证明且可区分的变化类

固定完整/selected attention 权重与支持，令

\[
V_t=V_a\Gamma+\mathbf1c^\top,\qquad \Gamma>0\text{ 为对角阵}.
\]

非退化尺度下，当前 centered 底座精确给出当前完整 attention。Chipmunk 的误差为

\[
A_t-\widehat A_t^{Chip}=(A_a-B_a)(\Gamma-I).
\]

纯 value 平移时两者都精确；新增能力是历史残差的尺度运输。当前 Q/K 变化时，P64 提供剩余创新的当前观测，读出据此近似恢复；若支持改变，则按新支持重建锚点状态后使用上述差式。

本轮 FP64 合成构造核验了差式和精确类，最大差式误差4.44e−16。尺度构造中 centered 相对误差2.95e−16；Chipmunk 的0.1393只是该构造的见证值，未被解释成真实视频收益。[C1] 实际模型中，上一轮两个来源八个站点的 attention 误差均下降、七个同输入块出口改善，提供了对应的模型证据；整片观感按用户实际反馈解释。[E2]

## 4. 当前观测究竟买到了什么

对固定恢复器，令 L=M_{\bar P}(I-KP)，其中 M_{\bar P} 清除最终被原生值覆写的探针行。当前绝对残差恢复与历史状态支持的恢复分别有

\[
e_{absolute}=L(A_t-G_t),
\]

\[
e_{state}=L(A_t-G_t-R_a^{tr}).
\]

历史状态的有效性取决于它减少了多少 L 保留下来的误差。若两个底座相差 delta_b，恢复后的输出差恰为 L delta_b。这个关系解释了为什么 centered 的底座改善经过 P64 后被部分吸收，也把下一步判别对象定位到恢复后及后继模块。

在理想无正则情形，若剩余创新属于字典 Phi 的列空间，且 P Phi 满列秩，则有限观测足以恢复该创新。锚点保留的全位置响应可以远比 P64 的新更新复杂。实际 ridge、有限精度和未建模方向共同决定剩余误差；当前观测的价值是提供新的约束，而不是把所有历史场压缩成64行。

SparsePR 已经建立当前 probe 拟合的直接先例，Chipmunk/MC-Sparse 已经建立历史 sparse 残差先例。我们的机制主张应集中于：利用当前低成本信息解码历史，让观测预算用于运输之后仍未解释的变化，并证明这种资源分配在目标短轨迹里有效。[L1–L4]

## 5. 接近的 FFN 算术预算下，三者的分配不同

以真实 N=31668、d=5120、m=13824、selected k=2560 计算，一个乘加计2 FLOPs。完整 FFN 为 F=4Ndm。

我方主要 hit：

\[
F_H^{ours}=2Nd^2+4Ndk,\qquad F_H^{ours}/F=10/27.
\]

ICC 两个线性层均用 rank r：

\[
F_H^{ICC}=4Nr(d+m),\qquad F_H^{ICC}/F=r/d+r/m.
\]

ICC r=1408 对应 hit 剩余37.6852%；我方主要 hit 剩余37.0370%，精确相同的 ICC rank 为1383.7838。两者主要预算接近。ICC 的原生 GELU 在低秩 FC1 更新之后实际执行，不能把它描述成整 FFN 固定 Jacobian。

| 方法和实际配置身份 | 完整 FFN 周期剩余主算量 |
|---|---:|
| 当前 wide8192，主要 GEMM | 68.5185% |
| 当前 wide8192，加入已列 P64/读出 GEMM | 68.9033% |
| 11/17 ICC，CD r1408/4S | 68.8426% |
| 11/17 Chipmunk Author Safe，按真实逐块列数 | 69.0282%–69.1082% |

Chip 的11片真实选中均值为4741–4765列；账本还包括前两层 Dense、76anchor/76hit、52尾token和块均值W1代理。其他 pointwise、索引、packing、非线性、求解、访存均未据此假定等时延。当前我方与两基线来自不同阶段，表格是配置成本辨析。

由此可作出的肯定判断是：**三类 FFN 都处在约31%的周期算量节省区间，研究差异主要在同类预算下获取什么信息。** Chipmunk 将预算更多用于动态 selected 非线性；ICC 用于两级低秩增量；我们用完整线性主项、较少 selected 非线性与当前输出恢复。[C2–C3]

### 历史严格对照已经提供的实证

11/17 的执行是作者 FFN 机制在 Wan 四步上的明确适配，attention 保持原生。ICC 已按 noisy clean latent、真实一步续步、作者四边界统计和 CD-SVD 执行，使用 r1408/4S；Chip 保留作者 blockmean selector、Triton 内核及动态缓存，并对真实宽度下的 selector 问题作同输入验证的安全修补。两者都具有强制 Dense 的原生轨迹一致性证据。[E3]

495条逐视频记录重新聚合得到：

| 历史 FFN 臂 | 主体一致性 | 背景一致性 | I2V主体 |
|---|---:|---:|---:|
| Dense | 0.96942 | 0.96621 | 0.98733 |
| Chipmunk Author Safe | 0.92048 | 0.93356 | 0.94657 |
| ICC CD r1408/4S | 0.87289 | 0.92425 | 0.94511 |
| 旧Q1r2 | 0.95335 | 0.95823 | 0.98092 |
| 历史Global-P64 | 0.96146 | 0.95768 | 0.97286 |

这组11开发来源的结果支持继续研究我方响应恢复方向，旧Q1与Global-P64各有取舍。历史 Global-P64 用当前真实行统计隐藏激活、决定全局支持；当前 P64 则观测原生模块输出并拟合恢复，同名不等于同一操作。当时用户仍观察到 Global-P64 的画面损坏，推动了后续逐位置锚点设计。这张历史表不是当前 wide8192＋centered 的直接胜率表。

历史诊断墙钟含大规模磁盘缓存读写、同步与观测；它们不进入当前对手速度排名。当前完整方法的成熟速度证据来自独立的同四步 Dense 配对：GPU1.227×、预热完整请求1.156×。[E1,E3]

## 6. 状态、路由与生命周期上的得失

### 状态是我方购买恢复能力的重要成本

同一真实形状、40层、一个 active expert、BF16全驻留时，指定FFN逻辑历史字段为：

| 状态布局 | 指定字段容量 |
|---|---:|
| 当前我方：输入、输出、selected预激活，N(2d+k) | 30.201 GiB |
| Chipmunk FFN：完整激活与输出，N(m+d) | 44.697 GiB，另有约0.255 GiB块均值 |
| ICC作者四态：两层各自输入输出，2N(d+m) | 89.395 GiB |

上表统一按40层比较状态布局；历史 Chipmunk 端口的前两层保持 Dense，实际38层单专家的缓存序列化约42.635 GiB。

我方指定 FFN 槽确实更紧凑，但仍承担明显显存成本。ICC可改状态表示/重算/offload，Chipmunk有异步offload；这些数是具体布局容量，均不是必要下界或测得峰值。当前我方 attention 还保存完整历史场与特征，需另外计入。Chipmunk 作者视频主表关闭FFN稀疏，因此不能把这份FFN外推当其视频实测显存。[C1–C3]

### 四步适合把预算放在单次复用收益上

记 alpha 为 selected 密度，beta=64/N。当前 high OUT-ABS、low centered 的四步 attention 核心比为

\[
r_{ours}=(2+3\alpha+2\beta)/4.
\]

如果把 Chipmunk 机制适配为两个锚点均当次产生支持、各服务一次命中，两个锚点都还要计算selected，则其对应核心比为

\[
r_{Chip,adapted}=(2+4\alpha)/4.
\]

作者默认先两次 Dense 获取旧LSE与当前近似列和；在HHLL中如何首锚产生支持需要明确适配并付费，不能静默删除初始化成本。

MC-Sparse还在每个锚点多扫描一次完整QK以精确选取，因此同一短生命周期的核心比为

\[
r_{MC,short}=(3+4\alpha)/4.
\]

在alpha约12.3%–12.5%时，三者分别约59.3%–59.5%、62.3%–62.5%、87.3%–87.5%。这些是机制/作用范围指定的算术构造，不是三个完整算法的同质量实测；同密度也不等于同选择质量。MC-Sparse作者50步中有长段摊销，精确token支持因而更有价值；当前P64更适合把少量新信息直接投入每次命中的恢复。[C1–C2,L4]

Chipmunk的细粒度支持与内核是可以借鉴的独立资产：它决定用哪些交互构造便宜底座；我方的运输和当前观测决定如何补足底座。更换kernel/router时，重建历史状态与同一P64预算，可以研究两者互补，而不是只把它们视为互斥名称。

## 7. 系统指标与新颖性如何表述

作者报告的 HunyuanVideo 2.16×为Chipmunk本体、3.72×含step cache；Wan2.1 1.56×为Chipmunk+STA、2.67×还含step cache。主表为50步、H100；step cache在中间30步每4步计算一次。视频主实验仅使用attention稀疏。官方Triton已明确支持Ampere/Hopper。[L1]

SparsePR的Wan2.2 1.80×在原文§5.2标为estimated full-generation latency，设备H100、N=75600；该版没有明确四步蒸馏身份。MC-Sparse的Wan I2V 1.52×是50步去噪区间，排除编码/VAE。DisCa的11.8×包含蒸馏与学习式缓存。各自都是有价值的系统结果，各自的模型、时间边界、作用范围和质量目标应保留。[L3–L5]

我方可以报告蒸馏与当前方法组合相对原模型的总加速，同时报告同一学生上的额外加速。Chipmunk和DisCa的组合结果提供了已有先例；本方法的贡献仍以同一蒸馏模型条件下新增的质量/成本交换明确归属。

最有区分力的主张是：**在短复用周期中，按当前value坐标解码历史响应，并用少量当前原生观测恢复剩余创新，实现固定蒸馏骨干上的进一步加速。** “稀疏＋缓存”“FFN/attention都是加权和”“P64＋ridge”“兼容蒸馏”分别已有先例。当前有价值的研究增量来自状态、更新法则、观测对象及其完整周期成本。

## 8. 如何据此收束工作

现有证据支持继续推进，但推进目标可以更具体：建立同质量、同硬件、同四步骨干的比较曲线，让额外恢复所保护的信息兑现为时间或状态收益。

首先，以当前最终版本对作者式Chipmunk attention的明确四步适配做同批比较；FFN单独保留ICC r1408和Chip Author Safe的既有语义，并用当前稳态执行边界计时。现有旧诊断资产负责机制与来源，新的运行只补当前缺少的对照。

其次，用共用selected底座比较静态历史残差、当步绝对P64恢复、centered历史＋P64，回答历史和当前观测各承担什么。这是具体贡献的辨析，而不是重复做大量正则或探针数扫描。

另一个值得评估的系统简化是只保留当前attention方法、FFN恢复原生。Chipmunk视频主表采取attention-only提供了实际动机。按当前账本，这将少省约7.0个基准主算量百分点，同时释放当前约30.2GiB的FFN锚槽及相关准备。最终速度与画面需要真实联合轨迹；attention误差与后继FFN的相互作用决定该取舍，不从不同批次的5%和13.5%墙钟直接相减。

已有“先P拟合、再分块一次读出”的等价重排继续作为执行简化。它与上述机制比较共享当前方法版本，不增加新的恢复概念。[E1]

## 复算资产

- [Chipmunk原文与源码审查](chipmunk/CHIPMUNK_PRIMARY_AUDIT.md)
- [ICC / SparsePR / MC-Sparse机制审查](neighbors/NEIGHBOR_MECHANISMS.md)
- [仓库严格复现与真实结果审查](evidence/REPOSITORY_COMPARISON_AUDIT.md)
- [条件attention关系核验](chipmunk/CHIPMUNK_RELATION_VERIFIED.json)
- [495条原始指标与逐块预算复算](evidence/STRICT_COMPARISON_VERIFIED.json)
- [ICC增量与实际维度预算](neighbors/STRUCTURAL_COST_COMPARISON.json)
- [短生命周期成本及attention-only范围](LIFECYCLE_TRADEOFFS_VERIFIED.json)

四个脚本分别位于上述子目录与本目录，可在CPU复算。合成数组只验证代数；静态预算只计算指定运算；实际模型、计时与质量证据均保留原始Git来源。

## 来源

- [E1] [当前速度与核心报告](../speed_and_core/SPEED_LIMITS_AND_CORE.md)，FFNDS 154fe862。
- [E2] [当前centered真实结果](../mass_centered_review/MASS_CENTERED_RESULT_ANALYSIS.md)及[联合推导](../mass_centered_review/MASS_JOINT_DERIVATION.md)。
- [E3] [11/17严格复现质量与成本](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/reports/20261117/ffnds_strict_reproduction/QUALITY_AND_COST.md)。
- [C1] 本目录chipmunk/CHIPMUNK_PRIMARY_AUDIT.md及固定源码身份表。
- [C2] 本目录neighbors/NEIGHBOR_MECHANISMS.md及固定源码身份表。
- [C3] 本目录evidence/REPOSITORY_COMPARISON_AUDIT.md及原始Git blob身份表。
- [L1] [Chipmunk原文](https://arxiv.org/html/2506.03275v1)、[官方固定仓库](https://github.com/sandyresearch/chipmunk/tree/b5e8d1c3da6bddc7ca9007c48e91b82087c11490)。
- [L2] [ICC原文](https://arxiv.org/html/2505.05829v1)、[官方固定源码](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py)。
- [L3] [SparsePR v2](https://arxiv.org/html/2608.18484v2)、[官方仓库](https://github.com/PardisTaghavi/SparsePR/tree/e028c78e6b40a29cb9683a9f90f7d7fa422b2b0e)。
- [L4] [MC-Sparse全文](https://arxiv.org/html/2610.06801v1)、[官方固定实现](https://github.com/dodododddo/mcsparse/tree/880f4859faf3621e70ec736546d9839420b76ab4)。
- [L5] [DisCa全文](https://arxiv.org/html/2602.05449v1)、[官方项目与四步示例](https://github.com/Tencent-Hunyuan/DisCa)。
- [L6] [Cache-DiT作者在Wan仓库的4/8-step与系统组成说明](https://github.com/Wan-Video/Wan2.1/pull/534)；[当前官方工程](https://github.com/vipshop/cache-dit)。
