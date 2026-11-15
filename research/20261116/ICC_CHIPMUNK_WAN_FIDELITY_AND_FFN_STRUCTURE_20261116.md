# 从原论文与作者代码重新审视 ICC、Chipmunk、Q1r2 与 Wan FFN

日期：2026-11-16（UTC+8）。仓库证据基线：`Qesire/FFNDS@e7a1e6b7a6babb5cf742072cc07a9961af854416`。

本文先核对原方法、固定作者代码及实际 Wan 适配，再解释现有结果。文献依据包括 ICC 正文和补充、Chipmunk 正文和算法附录、TEAL、DMD2，以及 Wan、LightX2V、Llama 的官方代码。实验依据复用 G19 历史视频、R1b、A、BW、BWS。新增模型运行、拟合、回放、视频生成和指标评分均为 0。

## 1. 核心判断

**当前证据足以支持一次有边界的机制比较，还不足以把所有历史基线概括为“完整、逐位复现原论文后失败”。** ICC 的四状态 FFN 核心与作者代码一致；实际主结果的双残差形式在实数域等价，BF16 与成本不同。Global-P64 是明确的 Chipmunk 变体。作者式 Block128 端口的历史结果存在，登记机制也与上游相容，但其历史 producer、编译 selector 与视频产物之间尚缺本轮可独立读取的完整绑定。

本次原始资料核对改变了五处解释：

1. **Chipmunk 原论文的视频主表没有启用 MLP 稀疏。** 附录 D.1 明确，HunyuanVideo/Wan2.1 只启用 sparse attention；70% MLP sparsity 用于 FLUX。因此这里研究的是 Chipmunk FFN 机制向 Wan2.2 四步 I2V 的迁移，不能称为复现并推翻作者的 Wan FFN 视频结果。[P2]
2. **历史锚点是三者的共同机制。** Q1 相对优势应解释为增量表示、输出拟合目标及缓存方式的差别。静态/锚点消融说明动态历史有价值，不能单独解释 Q1 为什么胜过同样使用历史的 ICC/Chipmunk。
3. **ICC 保留真实 GELU；CD-SVD 也已经考虑激活变化。** 我们的发现不能表述为作者“忽略非线性”或“没有意识到激活重要”。更具体的区别是分离的加权权重近似，与完整 FFN 门后有限增量目标并不相同。
4. **Global-P64 在最新局部回放中优于 Q1，完整主 MAC 又近似相同。** Q1 的证据优势集中于历史 I2V 主体保持、较小缓存及规整执行；现有结果不支持全面精度或速度优胜。
5. **蒸馏模型身份和官方 DMD2 推理标签已确认。** 现有权重×状态结果更支持复合状态条件的影响；完整训练配方与单独的蒸馏因果效应仍未确认。

## 2. 首先确定比较对象与宿主接口

### 2.1 对象与版本

| 对象 | 本次直接核对的版本或身份 | 比较范围 |
| --- | --- | --- |
| ICC | CVPR 2025 / arXiv:2505.05829v1；`ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4` | 作者 DiT/PixArt 图像方法中的 FFN 分支 |
| Chipmunk | arXiv:2506.03275v1；`sandyresearch/chipmunk@b5e8d1c3da6bddc7ca9007c48e91b82087c11490` | 稀疏 FFN 增量机制；论文视频配置未启用该分支 |
| 当前学生 | `Wan2.2-I2V-A14B-LightX2V-Distill-4step`，BF16 全权重 | HHLL 四步、无 CFG、40 层/专家，当前实验网格 21×29×52 |
| Q1r2 | G19 输出感知联合斜率拟合的精修版 | r2 是版本/精修标记，不是秩 2；A 可以满秩 |

采用列向量。原生 Wan FFN 为

\[
u_{t,\ell}=(1+s_{t,\ell})\odot\operatorname{LN}(x_{t,\ell})+b_{t,\ell},
\qquad F(u)=W_2\phi(W_1u+b_1)+b_2,
\]
\[
x^+=x+G_{t,\ell}F(u),\qquad G_{t,\ell}=\operatorname{diag}(g_{t,\ell}),
\]

其中 \(\phi\) 是 `GELU(approximate='tanh')`，\(d=5120,m=13824\)，每视频 \(N=31668\) 个 token。三个近似都应接收**当前**归一化及调制后的 pre-W1 输入，返回 pre-gate 输出，由原生路径乘当前 gate 并加残差一次。gate 可以进入离线损失，但不能因此在在线路径重复乘入，也不能用旧 gate 代替当前 gate。[C1,S2]

### 2.2 必要适配与实验选择不能混写

| 项目 | 性质 | 对结论的影响 |
| --- | --- | --- |
| 保留真实 pre-W1、偏置、tanh-GELU、pre-gate 边界 | 宿主语义所必需 | 错一个边界就可能不再是待比较的方法 |
| high/low 分别绑定权重、因子和缓存；请求间清空 | 宿主语义所必需 | 跨专家沿用旧缓存不合法 |
| 无 CFG 时按实际单次调用计数 | 宿主语义所必需 | 不能照搬作者 Wan2.1 双 CFG 调用计数 |
| 128 分组的尾 52 行用有效行数归一、正确逆排 | 非整除网格所必需 | pad 后仍除 128 会改变 proxy，而非单纯执行优化 |
| 原生 BF16、A100 kernel、融合后处理 | 合理数值/硬件适配 | 不保证原作者 kernel 逐位一致或原论文速度 |
| FFN-only、固定 D4、全部 40 层、r512/K5120、CAL 规模 | 实验作用范围和预算选择 | 可以公平回答指定 Wan 工作点的问题，不能称作者最优配置 |
| ICC 双残差、Global-P64、exact-K 或 Block-P64 | 缓存改写或方法变体 | 必须独立命名、计成本及验证数值 |

固定宿主任务与预算，是一种合理比较设计；“必要适配”不能覆盖所有主动选择。当前比较针对**给定四步生成器上的 FFN 替换**。它与原论文同时调整步数、作用模块和稀疏度所得的整系统质量—时延前沿，是两个不同问题。

## 3. ICC：核心算子忠实，双残差与历史 CD 调用链需要分别说明

### 3.1 作者真正定义的算子

锚点原生计算

\[
z_s=W_1u_s+b_1,\quad h_s=\phi(z_s),\quad y_s=W_2h_s+b_2.
\]

令 \(B_1,B_2\) 为两个权重的低秩近似。作者 FFN 分支在命中步执行

\[
\widehat z_t=z_s+B_1(u_t-u_s),\qquad
\widehat y_t=y_s+B_2[\phi(\widehat z_t)-h_s].
\]

固定作者 `models_sp_step_decomp.py` 与本仓 `icc_ffn.py` 的 `author_four_state` 对应：缓存 u/z/h/y，保留真实 GELU；偏置在锚点内，增量不再加偏置；hit 不改 anchor cache。因子默认将奇异值并入左因子，先右投影、再左投影。当前模块的四状态实现因此是**忠实的 FFN 算子移植**，不是完整论文系统复现。[P1,C2,S2]

作者 `sched_PERIOD` 的 p=2 就是 anchor/hit 交替，没有另设 warmup 或强制最后一步 dense。我们的 HHLL 0/2 锚点、1/3 命中与该形态一致；新专家首步重建缓存是正确适配。不能把 Chipmunk 的调度默认套到 ICC 上。

### 3.2 CD-SVD 的“channel-aware”具体是什么

作者固定代码对 u/z/h/y 四个线性边界分别统计：真实相邻调用差的绝对值 → token 均值 → 每样本跨通道 min-max → batch 均值与累计平均 → 平方根 → max-ratio 下限截断。默认左右 `max_ratio=1000`。然后对 \(LWR\) 做截断 SVD，再撤销左右缩放。对于固定正对角 L/R，它最小化

\[
\min_{\operatorname{rank}(B)\le r}\|L(W-B)R\|_F^2.
\]

本仓 `delta_stat`、`DeltaAccumulator`、`factorize` 支持上述数学流程；但 `weights(max_ratio=None)` 默认不做截断。**当前公开材料没有历史 CD 因子 producer 的完整调用链**，因此无法独立确认实际 u/z/h/y 配对、校准汇总方式及是否显式传入 1000。现有结论是“基础函数符合作者定义，历史 CD 结果存在”，尚非“历史 CD 端到端执行逐项已核实”。[C2,S2,S3]

另外，作者论文用 256 个校准图像；本仓登记的移植方案沿用历史 CAL4、真实配对 token 面板及专家分离校准，历史 CD 实际统计来源与汇总参数仍须由 producer 核实。这些选择改变统计预算，不能因校准源更少就直接判实现错误，也不能忽略它可能影响 CD 质量。幅值对角权重既不是完整增量协方差，也没有直接拟合当前 gate 后的完整 FFN 输出。原论文 DiT 使用 TF32、PixArt 使用 FP16；Wan 的 BF16 属于明确的数值适配。[P1]

### 3.3 r512 是否客观

作者 DiT-XL/2 为 \(d=1152,m=4608\)。两层同秩 r 的命中主 MAC 与原生 FFN 之比为

\[
\rho=\frac{2r(d+m)}{2dm}.
\]

| 配置 | r/d | 命中主 MAC / Dense FFN |
| --- | ---: | ---: |
| 作者 DiT r128 | 0.111111 | 0.138889 |
| 作者 DiT r192 | 0.166667 | 0.208333 |
| 作者 DiT r256 | 0.222222 | 0.277778 |
| Wan r512 | 0.100000 | 0.137037 |

r512 接近作者 r128 的相对计算量，具有合理迁移依据。但主文也使用 CD-r192、SVD-r256、PixArt-r64，且会调整步数以匹配总 MAC。我们的 r512 相比其较高预算配置更激进，不能称为作者推荐值或已找到的最佳工作点。[P1,C2]

### 3.4 双残差的等价范围与工程动机

定义

\[
q_1=z_s-B_1u_s,\qquad q_2=y_s-B_2h_s,
\]

则可重写为 \(\widehat z_t=B_1u_t+q_1\)、\(\widehat y_t=B_2\phi(\widehat z_t)+q_2\)。这是实数恒等式，缓存从四态降为两态，但锚点新增两级低秩投影。BF16 的减法、乘法和写回位置改变，不能声称逐位等价。

历史 CAL 直接比较提供如下约束：[S1]

| 因子/专家 | 四状态增量 relL2 | 双残差增量 relL2 |
| --- | ---: | ---: |
| SVD high | 0.636711481 | 0.636722151 |
| SVD low | 0.784308339 | 0.784398980 |
| CD high | 0.586572847 | 0.586590470 |
| CD low | 0.750282258 | 0.750342076 |

这不支持“整个质量差距主要来自双残差重排”的简单解释。但四状态零增量误差为 0，双残差 high/low 最大绝对偏移分别为 0.25/0.125；当前没有已核完整视频的四状态对照，所以不能由平均 CAL 接近推出自由轨迹完全相同。

在 N=31668、BF16、一个专家的 **40 个层缓存槽同时驻留**时，四态逻辑缓存 89.395 GiB，双残差 44.697 GiB。四态仅缓存就超过 A100-80GB，说明改写有很强工程理由；offload 等其它方案仍可能保留四态，但会改变搬运和时延。

### 3.5 实现审计的最终范围

BW/BWS 具有源代码、因子 SHA/形状/dtype、重载、运行收据和原始风险绑定，验证的是**普通 SVD-r512 双残差局部回放**。历史七源 SVD/CD 视频有逐源汇总及上游结果 SHA；本轮尚未取得旧 `icc_runner_all80.py`、CD producer、逐视频身份与 neutral trace 的完整内容。不能用 BWS 的局部执行绑定替代旧视频 adapter 的全链验证，也不能反过来把历史已完成 CD 说成“从未评价”。

建议名称：**ICC-FFN / SVD-r512 或 CD-SVD-r512 / two-residual / BF16 / Wan-D4-HHLL**。

## 4. Chipmunk：两种主实现不是同一算法的两个后端

### 4.1 原论文、固定代码与原视频配置

Chipmunk 的 FFN 机制对选中神经元重算当前激活、减旧激活、经 W2 写回旧输出。其 Algorithm 6 与代码一致；§3.3.2 的一个显示公式把缓存减法放进了激活括号，与算法及代码不一致，应按后两者核对，不应照抄那一式。[P2,C3]

对紧接 dense anchor 的第一个 hit，组 B 的作者 proxy 在实数域为

\[
s_{B,j}=\left|\operatorname{mean}_{i\in B}\Delta z_{i,j}\right|,
\]

选择 \(S_B\) 后更新

\[
\widehat y_{t,B}=y_{s,B}+W_{2,S_B}\big[\phi(z_{t,B,S_B})-h_{s,B,S_B}\big].
\]

代码在每个 sparse hit 更新选中激活及输出缓存；选中 proxy 只在重算索引时更新（step<10时每次重算，之后可以复用mask）。因此多 hit 时是各列不同更新时间的历史状态，不能总写成固定 dense-anchor。HHLL 每专家只有一个 hit，才允许上述简化。

论文选用 C=192；固定公开 BF16 模块默认 BM=MBM=128。后者使用128是沿公开代码的选择，不是论文写了128。固定代码还默认 `first_n_dense_layers=2`、`full_step_every=10`。Wan 示例关闭 MLP，FFN 仍是原生 Sequential，并非一个改 YAML 就能启用的现成 Wan sparse-MLP wrapper。[C3]

### 4.2 作者 selector 不等于精确 Top-K

固定 `topk_indices.cu` 从每行**前 1024 列**估计分位阈值，再扫描全行：满足阈值或随机条件的列被选中，数量向上补到256倍数。随机数种子还依赖该行 activation 的位表示。默认 `top_keys=.3, random_keys=.05` 既不保证 K5120，也不严格等于70%稀疏；忽略阈值估计及对齐，保留比例已约为 \(.3+.7\times.05=.335\)。历史平均 K≈4764.83 与此相容。[C3]

R0 记录使用 compiled author selector，**没有证据指控历史端口用了错误替代品**；只是本轮没有读到对应 .so、producer 和运行身份。若另一个端口使用全列 `torch.quantile` 或普通随机序列，只能说明机制近似，不能说选择索引数值复现。

### 4.3 逐项命名

| 实现 | proxy 与支持 | 方法身份 |
| --- | --- | --- |
| **FFNDS Global-P64** | 64 个真实 token 上 post-GELU 增量 RMS；全 N 共用 exact K5120 | 明确变体，当前 BW/BWS 直接评价此臂 |
| **作者式 Block128 端口** | voxel 排序后有效行块均值的 preactivation 差；分位阈值、随机补充、256对齐 | 历史 `AUTHOR_QUANTILE_RANDOM`，平均 K≈4764.83 |
| Global-Author / Block-Author exact-K | 作者评分家族、固定 K5120，改变全局/分组支持 | 支持粒度和选择器消融 |
| Block-P64 exact-K | 每组真实行 P64 代理、固定 K5120 | 更昂贵的强代理消融，非作者默认 |

Global-P64 保留抽样token的非线性变化幅值，但支持全局共享；作者端口汇总各组的输入均值，并局部选列。两种观测不是严格的信息包含关系。在相同分组粒度下，均值proxy比每组P64便宜；当前全局P64只算64行W1，而作者式mean需要248个组均值行，不能反过来说作者端口总代价更低。Block-P64结合分组支持与真实行RMS并支付代理成本。论文的 `[192,128] block sparsity` 指隐藏列块，与我们名称中的 token 分组 Block-P64 不是同一概念。

### 4.4 Wan 集成与数值核验做到哪里

本仓 Global-P64 可读代码明确使用 `(expert,block)` 缓存键、原生调制/gate、80 anchor+80 hit、命中后释放。旧898/96两源有 Dense/hook-Dense/cache-Dense 中性结果；packed 与本仓 A100 Triton 在同 mask 下有20对输出差0。这支持该变体的集成正确性。[S4]

但已核对的上游BF16 Triton参考kernel在 FP32 累加后直接 GELU再转BF16，W2增量先转BF16再加缓存；本仓 native 路径在W1后先BF16，并在W2累加加旧Y后统一写BF16。保持宿主数值边界合理，不能把本仓内部 zero-diff 称为上游 kernel 逐位复现。

作者式端口的 R0 登记包含 `chip_faithful_core.py`、compiled selector、voxel(4,6,8)、block128/尾52；七源历史结果也存在。其实际 runtime、.so、全列恢复/neutral、七源配对身份尚未完整取回。尤其全部40层是否覆盖作者默认保留的首2层、refresh与尾块实现，仍应由原 producer 最终确认。不要用 Global-P64 的中性证据代替作者端口的证据。

## 5. 现有结果支持怎样的质量与成本比较

### 5.1 历史完整视频：Q1 保真较好，运动指标不是全胜

七个已暴露来源，每源 seed43/44；先源内平均，再来源等权。独立来源数为7，不能把14段视频当14个独立来源。下表复用已归档历史汇总，未重新评分。[S1]

| 方法 | 完整 FFN 主 MAC / Dense | I2V 主体保持 | MUSIQ | RAFT 方向余弦 |
| --- | ---: | ---: | ---: | ---: |
| Q1r2 | 0.685185 | **0.982884** | **72.5356** | 0.808267 |
| ICC SVD512 双残差 | 0.637037 | 0.764362 | 58.9711 | 0.237157 |
| ICC CD-SVD512 双残差 | 0.637037 | 0.764391 | 56.5810 | 0.286290 |
| Global-P64 | 0.685690 | 0.974949 | 71.8515 | 0.756219 |
| 作者式 Block128 | 0.674297 | 0.948236 | 69.8552 | 0.625275 |
| Block-P64 exact-K | 0.810390 | 0.979636 | 72.0911 | **0.811679** |

Q1 的 I2V 主体对 Global-P64/作者式端口为7/7来源更高；对 Block-P64 为6/7。Block-P64 的 RAFT 模长比0.912017也比Q1的0.901732更接近Dense，其部分一般一致性、美学和平滑度指标更好。Q1不能称所有质量轴最好；这些指标也不等于人评非劣证明。

历史 CD 确实将 CAL 的 high/low relL2 从0.63672/0.78440改善到0.58659/0.75034，却没有恢复视频主体保持。这个观察排除了“没有做CD”作为完整解释，尚不能排除校准预算、参数选择与尚未闭合的历史执行细节。

### 5.2 最新局部结果必须保留其反向排序

BW 的同一D4状态、own-W目标、CAL11拟合与DEV11回放给出：[S5]

| 权重 | Q1 门后局部风险 | ICC | Global-P64 | Global-P64 相对 Q1 改善 |
| --- | ---: | ---: | ---: | ---: |
| 原始 W_O | 0.014902516 | 0.021427943 | **0.011515711** | 22.73% |
| 蒸馏 W_D | 0.014899847 | 0.021438563 | **0.011497516** | 22.83% |

两个权重下均为11/11来源均值 `Chip < Q1 < ICC`。不能继续把 ICC 与 Chip 合写为“都不如Q1”。局部样本、拟合、GEMM形状与历史视频 cohort 不同，因此两表不能直接作为局部误差导致视频排序反转的因果证明。

### 5.3 完整锚点—命中周期成本

令 \(F=2dm=141557760\)、\(a=d^2=26214400\)、\(i=2r(d+m)=19398656\)，单位为每token主MAC。

| 路径 | anchor | hit | 一对 / 两次Dense | BF16历史缓存 B/token |
| --- | ---: | ---: | ---: | ---: |
| Q1 单残差 | F+a | a | **0.685185** | **10240** |
| Global-P64 K=d | F | 2a+proxy | **0.685690** | 37888 |
| ICC 双残差 | F+i | i | **0.637037** | 37888 |
| ICC 四状态 | F | i | **0.568519** | 75776 |

Q1锚点需计算 \(C_s=F(u_s)-Au_s\)，不能漏掉一次A投影。Global-P64的hit主体确为Q1两倍，但在当前每专家一个anchor+hit的完整周期里，两者主MAC几乎相同。ICC双残差算术反而更少，问题不能笼统归为算得更多。

按40层驻留逻辑缓存，Q1为12.080GiB，Chip/ICC双残差44.697GiB，约3.7倍；Q1另有80张BF16 A约3.906GiB，Chip另有proxy、mask和packing，ICC另有低秩因子。逻辑payload不等于GPU峰值或测得时延。作者有缓存融合/offload，不能用低效通用gather实现代表其最佳系统性能。

Block-P64主MAC0.810390与每128行采至多64行、尾52行全取时的15860行proxy相符。这仍是**成本一致性推断**，尚非尾块runtime的直接验证。原作者式mean proxy exact-K的主MAC为0.687143。

若同一anchor后有q次hit、K=d，忽略proxy等额外成本，Q1为 \(F+(q+1)a\)，Chip为 \(F+2qa\)。q=1时相同；q>1时Q1的主MAC优势扩大。此为摊销算术，不是多hit质量预测；Chip此时逐列更新历史，Q1固定anchor，质量变化须另证。

实际整模型速度还受 FFN 占时约束：\(S_{total}=1/[(1-f)+f/S_{FFN}]\)，f应在目标原生图上测量。旧G0的历史配对速度不能替代Q1r2与ICC/Chip的独立公平速度比较。[S6]

## 6. 以共同有限增量解释差异

### 6.1 三者压缩的对象

令 \(v=u_t-u_s\)、\(\Delta z=W_1v\)、\(\Delta h=\phi(z_s+\Delta z)-\phi(z_s)\)。逐坐标定义端点割线

\[
\beta_j=\frac{\Delta h_j}{\Delta z_j},\qquad
\Delta F=W_2\operatorname{diag}(\beta)W_1v.
\]

零分母处可取导数，不影响乘零后的恒等式；GELU割线不必在[0,1]内。这是有限增量的精确式，不依赖“小步Taylor”假设。

| 方法 | 实数域预测增量 | 保留/约束 |
| --- | --- | --- |
| Q1 | \(W_2\operatorname{diag}(\alpha)W_1v\) | 无显式低秩输入投影，结构化原子联合系数；固定A可以满秩d |
| ICC | \(B_2[\phi(z_s+B_1v)-\phi(z_s)]\) | 保留非线性；最后输出增量落入固定至多r维空间 |
| Chip 单hit | \(W_2M_S\Delta h\) | 选中项用真实非线性变化；遗漏项置零增量，支持全局/分组 |

三者均利用历史锚点。Q1既不是FFN全局线性化的证明，也不是秩2矩阵；ICC不是去掉非线性的模型；Chip不是把当前大激活简单剪掉。

### 6.2 ICC：末级输出子空间与目标错配

上述实数域ICC算子严格满足 \(\widehat{\Delta F}\in\operatorname{col}(B_2)\)，这里不含BF16舍入项。给定gate，该算子的任意预测都受下界

\[
\|G(\Delta F-\widehat{\Delta F})\|^2
\ge\|(I-P_{GB_2})G\Delta F\|^2
\]

约束。保留GELU不能突破末级秩限制，CD可以旋转子空间但不取消秩上限。这一事实不等于“实际必需秩大于512”：也可能是同秩子空间方向或校准目标不合适；固定512行样本本身更不能证明所需秩超过512。

令 \(E_j=W_j-B_j\)，T为真实/近似预激活端点间的逐坐标割线，有

\[
\Delta F-\widehat{\Delta F}_{ICC}=E_2\Delta h+B_2TE_1v.
\]

R1b将同一路径的实际BF16输出先升F64再定义误差两项并统计，普通SVD512六站点的第二阶段平方项份额约64.3%–82.5%，交叉项约−3.0%至+7.5%。第二项包含末级低秩与双残差数值项，并非直接把实数 \(E_2\Delta h\) 当成实测值。在low19/34，即使数学上代入原生hidden，余下第二阶段风险仍约为Q1的1.278/1.269倍；其余四站点则低于Q1。[S7]

这将主要局部问题定位到**末级读出与真实增量的不匹配**，同时说明第一阶段仍有贡献。该分解不是“修W2必恢复64%–82%视频质量”的因果承诺，也尚未给CD同样的阶段归因。原ICC理论只对单linear给界并忽略累积；完整两层及轨迹分析正是我们补充的层次。[P1]

### 6.3 Chip：便宜proxy、局部支持、输出方向是三个不同问题

均值proxy有可精确构造的信息损失：一组token的 \(\Delta z_j\) 一半+a、一半−a，组均值可为0，各token的真实变化却非0；经过GELU后，基点又改变割线。P64观察真实post-GELU变化，减少这一盲区，但仅64行且仍不直接优化输出风险。

历史固定K消融：[S1]

| 变化 | I2V主体保持 | 各源方向 | 主MAC变化 |
| --- | --- | --- | --- |
| Global-Author → Block-Author | 0.940714 → 0.956854 | 7/7改善 | 0.687143不变 |
| Block-Author → Block-P64 | 0.956854 → 0.979636 | 7/7改善 | 0.687143 → 0.810390 |

按已保存设计，这分别支持局部支持与更丰富当前信息有价值。旧Global-P64布局不同，不能与后三臂拼成严格2×2析因；实际producer缺口也应随结论保留。

已有true-M四案例中，本仓全局均值proxy→P64→full-hidden RMS的gate-relL2为0.376113→0.329515→0.311198；这里全局均值臂不是作者Block128端口。固定P64支持、加有符号遗漏均值校正可到0.314917，low/96有例外。[S8] 因而代理信息确实重要，但“看全hidden并按RMS选”仍非最优输出选择，遗漏项也不是完全随机噪声。

### 6.4 Q1为何有用：联合输出拟合与小状态表示

对隐藏误差q，局部注入误差为

\[
\|GW_2q\|^2=q^\top K_Gq,\qquad K_G=W_2^\top G^2W_2.
\]

此Gram包含写回方向的有符号交叉项；秩至多d，所以隐藏维m中至少有m−d=8704维在该瞬时读出下不可见。重要的是误差在输出度量中的作用，不是每个hidden坐标都要逼近得一样好。这是架构代数性质，不是新测到的低秩残差。

Q1的正规矩阵，在吸收源码归一化常数后为

\[
H=\mathbb E[(\Delta z\Delta z^\top)\odot K_G]
+\lambda[(W_1W_1^\top)\odot(W_2^\top W_2)].
\]

13,824个系数共同拟合真实门后有限增量，并受矩阵先验约束。α通常不能逐个解释为该神经元的平均GELU导数；它们可以利用相关读写方向补偿输出。在线折叠成A后，一次规整GEMM与d维残差缓存完成命中，不需要在线proxy、选支持及按索引packing。[S2,S6]

它付出的代价是离线CAL与拟合、A存储、锚点A投影，并非没有额外成本。其优势是**当前质量—状态容量—执行形式的组合**；BW/BWS已经说明它不保证最小局部MSE。

## 7. 视频 FFN 与 LLM FFN 的区别应怎样表达

ICC与Chipmunk本来就是扩散模型方法，不能把它们的迁移困难解释成“将LLM算法直接搬到视频”。LLM比较的价值在于辨别计算图、重复结构和硬件工作点。[P1,P2]

| 维度 | 当前Wan视频DiT | 典型Llama类LLM | 含义 |
| --- | --- | --- | --- |
| 非线性图 | 两Linear夹tanh-GELU | 三矩阵SwiGLU | 门乘结构不同；也不能把所有LLM都归为SwiGLU |
| 时间重复 | 相同潜网格坐标跨去噪步重算 | decode每步处理新token，历史KV复用 | 视频天然有同坐标历史FFN锚点；不是物体跟踪对应 |
| 每次矩阵行数 | 本任务31668 | 小batch decode很少；prefill/大batch很多 | 本任务是大M GEMM；小batch decode常受权重带宽限制，prefill/大batch需另判 |
| 动态条件 | timestep shift/scale/gate，high/low切换 | Llama无此噪声时刻调制；MoE一般按token路由 | 缓存和误差目标需绑定真实宿主条件 |
| 稀疏对象 | 跨步变化Δh | 常见activation方法作用于当前h或输入 | 大而恒定的h适合缓存，却不适合直接剪掉 |
| 反馈 | 层和求解步后到latent/视频 | 层到logits，选token又改变后续输入 | 两者均有传播，不能说只有视频累积误差 |

TEAL明确区分小batch decode的memory-bound与prefill等工作点，并指出不同输入偏好不同支持。在其Llama-2-7B特定层例子中，保持同误差时，batch1到4使Wdown稀疏率约60%降至38%。这支持“跨多行共用支持有代价”，不证明所有视频FFN必然compute-bound，也不能将该比例外推到31668个视频token。[P3]

Llama3官方FFN为 \(W_d[\operatorname{SiLU}(W_gx)\odot W_ux]\)。若 \(a=W_gx,b=W_ux\)，端点算术均值记横线，则精确恒等式

\[
\Delta h=\overline b\odot\Delta\operatorname{SiLU}(a)
+\overline{\operatorname{SiLU}(a)}\odot\Delta b
\]

给出两个读入分支。Wan的 \(W_2\operatorname{diag}(\beta)W_1\) 不能原样套用，但“真实有限增量＋联合读写几何”的分析方式可以迁移。这是推导，尚无Q1在LLM上的有效性实测。[C4]

## 8. 蒸馏：已经确认什么，哪些因果解释仍不成立

### 8.1 官方权重和推理路径已经找到

本仓B_BINDINGS的expected/actual SHA与 `lightx2v/Wan2.2-Distill-Models` 官方文件页一致：[C5,S2]

| 文件 | SHA256 |
| --- | --- |
| `wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors` | `39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa` |
| `wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors` | `a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2` |

两份各28,577,915,880字节；这里比对官方pointer与既有实验哈希记录，本次没有重新下载权重。它们是BF16全权重，不应混称另一Lightning LoRA发布。

LightX2V固定版本 `9cb6929d83814ff28c4e1386532298fe93a96af1` 的匹配配置明确 `distill_method=dmd2`、四步、CFG关闭、boundary_step_index=2、sample_shift=5。scheduler将名义sigma变为 \(5\sigma/(1+4\sigma)\)，故名义[1000,750,500,250]对应实际 **[1000,937.5,833.333…,625]**。[C5]

真实更新为

\[
x_{i+1}=x_i+(\sigma_{i+1}-\sigma_i)v_\theta(x_i,t_i),\qquad\sigma_4=0.
\]

这不是DMD2图像论文逐步加独立噪声的采样式。已确认官方DMD2标签和推理语义，尚未确认该checkpoint完整训练loss、GAN比重、教师CFG与训练状态配方，不能泛称CausVid或Self-Forcing。

### 8.2 三条作用途径

蒸馏可能经**权重/函数**、**访问的状态与条件**、**复用与求解预算**三条途径影响近似。四步学生并非必须逐样本走教师轨迹的稀疏采样；DMD2的分布目标及学生状态模拟也没有保证内部FFN的Jacobian、稀疏度或割线保持。[P4]

当前每专家只用一次anchor和一次hit，限制成本摊销，没有同一FFN缓存多次命中产生的近似历史累积；此前层/步的误差仍会进入后续状态。少步可能增大单次位移，却没有“步数越少，Δu必越大或缓存必失败”的一般定理。

实际sigma差的绝对值依次为0.0625、0.104167、0.208333、0.625。因而末步**整模型flow预测误差**在状态更新中直接乘0.625；这是求解器的确定系数。某层FFN误差还须经过后续网络映射，且各步误差幅值/方向不同，不能据此推成“low误差必然放大十倍”或已经解释全部运动损失。

### 8.3 BWS更支持状态通道，未把蒸馏判为无关

BWS固定每套权重/站点的混合CAL3拟合，跨D4/O40状态评分，用共同D4参考gate；DEV是3个来源、6站点、每单元512行。216是嵌套测量单元数。W_O下：[S9]

| 原生状态 | Q1 | ICC | Global-P64 | ICC相对Q1超额 |
| --- | ---: | ---: | ---: | ---: |
| D4 | 0.015418118 | 0.020118910 | 0.010994489 | 30.49% |
| O40 | 0.007277827 | 0.008344406 | 0.005087691 | 14.66% |

W_D得到接近图景；三者绝对风险下降约53%–59%。BW共同D4条件中，切换权重引起的方法gap变化只有原gap绝对值约0.20%（ICC）和0.46%（Global-P64）。BWS中状态变化明显大于测试的权重切换，交互相对状态差也很小。[S5,S9]

准确结论是：**在各权重自己目标、每套权重一次混合校准及当前样本条件下，复合原生状态改变局部近似难度，权重切换的平均影响较小。** 这不表示W_O与W_D是相同函数；Chip两权重的mask SHA还全部不同。D4/O40同时改变轨迹、间隔、CFG等，且使用共同参考gate，不能把状态效应单独命名为“蒸馏步长效应”。low专家ICC相对差距约53.7%并未随O40明显缩小，也不能说所有站点都变容易到相同程度。

## 9. 真正揭示了哪些结构

### 9.1 低维的是哪一部分，必须区分

“少拟合参数”“低次数”“窄输入子空间”“低矩阵秩”“小历史状态”不是同一个性质。Q1用m个原子系数构造可满秩A；ICC用两个rank512矩阵；P64仅指在线观察64行，不表示输出函数rank64。当前证据更支持**保留宽输入与有效输出方向，同时压缩更新规则和状态**。[S6,S10]

### 9.2 读写方向的配合具有实证依据

既有Gaussian参考测度分析中，W2列随机翻符号严格保留其奇异值与列范数，却将一阶Hermite整体能量相对神经元能量和的中位比1.906187降至约0.999956。较小相对高阶尾主要来自低阶增强，绝对二阶尾基本不变。它直接说明独立权重谱不足以刻画函数结构；它不是训练为何形成该结构或所有native状态都会如此的证明。[S10]

### 9.3 原生弦上的响应可规则积分，但大非线性分量仍重要

既有F64分析对真实端点之间的虚拟直线积分，在Dq/40层/10来源uniform面板中，起点导数、中点、两点Gauss–Legendre的平方风险比分别为high 92.7851%/3.6553%/0.071593%，low 32.7542%/0.4677%/0.011184%。它支持弦上的合成方向响应可用少量积分点逼近，未证明低成本部署或全局线性。[S10]

tanh-GELU恰可分成线性奇部 \(z/2\) 与偶部。high增量中两部分能量相对总增量分别约3.1990和2.0763，交叉项−4.2753；大分量抵消才形成实际响应。因此Q1成功不能解释为“非线性可以删掉”。有限割线拟合与单点导数近似的差异，在这里有具体证据。

### 9.4 Q1仍有可迁移的线性剩余

最新A实验的可折叠线性修正L，使Q1局部风险0.01489947降到0.01429259，约4.073%，DEV11来源全部改善。修正矩阵秩至多32，不代表完整残差只有32秩。L用了64个线性输入PC，D/E为32线性+32非线性，非嵌套，不能由L获胜推出非线性普遍无用。[S11]

端点增量还有可加性：\(\Delta F_{01}+\Delta F_{12}=\Delta F_{02}\)。Q1及端点差型近似保留它，任意位移非线性一般不保留；这是基本代数约束，目前没有测出它对四步单hit视频差异的独立贡献。上述结构有助于选择表示，不应把每条恒等式都包装成新数学定理或已经完成的新颖性证明。

## 10. 当前可以形成的论文叙述

可以提出这样的主张：

> 在Wan2.2四步I2V的既定FFN替换预算下，我们比较了ICC的两级低秩增量机制、Chipmunk作者式局部稀疏更新及Global-P64变体，并明确记录缓存、精度和作用范围适配。结果显示，单矩阵加权低秩目标、局部/全局隐藏增量选择和完整门后函数增量拟合之间存在可解释差异。Q1以结构化、可满秩的固定映射与小输出残差缓存，取得有竞争力的I2V保真和状态容量；Global-P64的局部回放风险更低，按完整N=31668锚点—命中周期计算时与Q1的主MAC接近。较强分组proxy进一步改善部分视频轴但增加代理成本。权重×状态结果将主要解释转向真实轨迹上的有限增量及输出几何，而非将现象简单归因于蒸馏权重。

上述完整部署成本与局部回放分开解释：BW/BWS执行的是512行面板，并独立重算其P64端点；这些回放没有测出0.685690这一完整部署预算对应的真实时延，也不是按完整部署成本执行的等成本速度实验。

仍不能提高为：所有原论文方法在视频上失效；r512代表ICC最佳前沿；Global-P64等于作者实现；Q1全质量/全速度优胜；当前结果证明蒸馏造成某种内部数学结构。

实现基础剩下最有价值的补充，是**找回已有历史producer与产物绑定**：ICC CD四边界统计/截断调用，作者Chip端口的selector二进制/配置/tail/neutral，及七源视频配对记录。它们解决“实际运行的是否就是已审公式”，不需要先重跑模型。若这些记录无法找回，对原方法的结论应继续限定为已登记的迁移配置；对BW/BWS普通SVD与Global-P64的局部分析则已有完整执行基础。本报告不新增采集、拟合或参数搜索任务。

## 原始资料与仓库证据入口

- **P1**：Chen et al., [ICC原论文及补充](https://arxiv.org/html/2505.05829v1)，重点§3.3–3.4、§4与补充§7–8；[CVPR官方PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.pdf)。
- **P2**：Silveria et al., [Chipmunk原论文](https://arxiv.org/html/2506.03275v1)，重点§3.2–3.3、Algorithms5–7、附录D.1及limitations。
- **P3**：Liu et al., [TEAL原论文](https://arxiv.org/html/2408.14690v2)，重点§5.4.4和limitations。
- **P4**：Yin et al., [DMD2原论文](https://arxiv.org/html/2405.14867v1)，重点§4.4–4.5；只用来解释分布目标/状态匹配原则，不据此补造Wan训练配方。
- **C1**：[Wan2.2原生block固定代码](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py)。
- **C2**：[ICC固定作者仓库](https://github.com/ccccczzy/icc/tree/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4)：直接读`DiT/models/models_sp_step_decomp.py`、`models_stat_svd.py`、`DiT/adapters/sched_sp_step.py`、`DiT/func/func_decomp.py`、`DiT/utils/utils_decomp.py`、`DiT/gen_decomp.py`。
- **C3**：[Chipmunk固定作者MLP](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/modules/mlp.py)、[selector](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/csrc/indexed_io/topk_indices.cu)、[配置](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/util/config.py)、[Wan配置](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/chipmunk-config.yml)；并读同版本`copy_indices.cu`、两级BF16 Triton、Wan model。
- **C4**：[Llama3官方FFN](https://github.com/meta-llama/llama3/blob/main/llama/model.py)，`FeedForward.forward`。
- **C5**：[官方high文件与SHA](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors)、[low文件与SHA](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors)；LightX2V固定版本的[匹配配置](https://github.com/ModelTC/LightX2V/blob/9cb6929d83814ff28c4e1386532298fe93a96af1/configs/distill/wan22/wan_moe_i2v_distill_model.json)、[实际scheduler](https://github.com/ModelTC/LightX2V/blob/9cb6929d83814ff28c4e1386532298fe93a96af1/lightx2v/models/schedulers/wan/step_distill/scheduler.py)、[factory](https://github.com/ModelTC/LightX2V/blob/9cb6929d83814ff28c4e1386532298fe93a96af1/lightx2v/models/schedulers/wan/scheduler_factory.py)。
- **S1**：[历史视频external逐源汇总](../../reports/20261115/g20_concentrated_analysis/LEGACY_VIDEO_EXTERNAL_SUMMARY.json)、[来源记录](../../reports/20261115/g20_concentrated_analysis/LEGACY_SOURCE_RECORD.json)；ICC四状态/双残差的CAL数值也在同一汇总的`cal_rank512`中。
- **S2**：[BWS实际源码与绑定](../../reports/20261114/g20_two_track_core/BWS/v1/source)：`icc_ffn.py`、`icc_decompose.py`、`q1_fit_core.py`、`b_common_replay.py`、`chipmunk_wan_packed_pregate.py`、`B_BINDINGS.json`；[ICC接口规范](../20261107/ffn_baseline_reproduction/ICC_FFN_PORT_SPEC.md)。规范与实际执行证据分开使用。
- **S3**：[R0资产定位](../../reports/20261112/g20_r1_screen/R0_ASSET_INVENTORY.md)。
- **S4**：[Global-P64真实视频集成结果](../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.md)、[同mask数值核对](../20261030/CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.md)及[实现与旧资产说明](../20261107/ffn_baseline_reproduction/CHIPMUNK_REUSE_AND_COMPARISON.md)；实际代码为`tools/chipmunk_wan_global_p64_tap_v2.py`、`tools/chipmunk_wan_global_p64_driver_v2.py`、`tools/chipmunk_wan_packed_pregate.py`。
- **S5**：[当前结果索引与BW](../../reports/20261114/g20_two_track_core/README.md)、[BW Chip逐源](../../reports/20261114/g20_two_track_core/BW_CHIP_SCORE/v2/output/BW_CHIP_SOURCE.csv)、[BW聚合](../../reports/20261114/g20_two_track_core/BW/v3/output/BW_AGGREGATE.json)。
- **S6**：[G19方法推导、Q1精修、缓存和历史计时](../20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md)。
- **S7**：[R1b实际BF16操作数误差分解](../20261112/G20_R1B_NUMERICAL_RECHECK_RESULTS_20261112.md)。
- **S8**：[P64代理对照](../20261030/CHIPMUNK_FULLM_CAUSAL_PROXY_RESULTS_20261030.md)、[有符号遗漏补偿](../20261030/CHIPMUNK_SIGNED_COMPLEMENT_RESULTS_20261030.md)。
- **S9**：[BWS固定设计与执行](../../reports/20261114/g20_two_track_core/BWS/v1/README.md)、[BWS原始四格](../../reports/20261114/g20_two_track_core/BWS/v1/output/BWS_AGGREGATE.json)。
- **S10**：[此前结构集中分析及Gaussian/native原始入口](../20261115/G20_METHOD_DIFFERENCES_AND_FUNCTION_STRUCTURE_20261115.md)，§9与S14–S15；[V4表示比较](../../reports/20261114/g20_v4_repr_density/README.md)。
- **S11**：[A残差修正原始结果](../../reports/20261114/g20_two_track_core/A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_READOUT.json)、[特征实现](../../reports/20261114/g20_two_track_core/A_Q1_RESIDUAL/v1/source/a_q1_residual_repr.py)。
