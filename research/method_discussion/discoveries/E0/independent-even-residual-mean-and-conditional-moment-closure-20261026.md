# GELU 偶残差的条件均值：全局功能、矩闭合与竞争解释

日期：2026-10-26。作者：`whole_flow_native_review`。类型：独立机制发现／条件数学／原始源码与一手文献；未选择下一方法。

**DISCOVERY — 新的正向结构。** GELU 的平均导数与平均输出携带不同的信息：对任意对称预激活分布，连续 GELU 的平均导数始终为 `1/2`，而平均输出中的偶残差可随方差和尾部显著变化。因此，“旧共享 slope 很稳定”和“signed FFN 均值仍随分布变化”能够同时成立。条件矩闭合值得研究的对象，是这个随当前输入分布变化的偶残差均值，不是将末态低频比例直接变成一项修复规则。

竞争解释是输入投影协方差、非 Gaussian 混合／稀有 token、动态 Q2，以及后续 attention／非线性把其他模式转成均值。最便宜的区分动作，是先在已经保存的同一 32／128 行面板上比较“已付 W1 后的边缘矩控制”和“只用 Q1 均值／对角方差的预测”，再逐层拆开 tanh、Smooth2、Q2 与 signed W2；本文件只设计这些观察，没有执行它们。

## EVIDENCE：现象与实际计算对象

root 提供的新观察是：两来源末态 **逐 latent channel 的时空均值组成的 16 维 DC 子空间**约承载 26% 状态差能量。本研究没有重算该张量结果。它指 16×21×58×104 latent 网格的 channelwise TYX DC；这不是 RGB 亮度坐标，也不是“26% 失败由 FFN 均值造成”。已经发布的 [whole-flow 结果](../../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) 和 [频谱解释](../../../20261026/E0_SAVED_FLOW_SPECTRUM.md) 保留了继承偏移、有限 dense 响应和当前候选注入的区别。

实际 FFN 的顺序是

\[
\text{token LayerNorm/AdaLN}
\to q=Q_1(\operatorname{BF16}(u/S_1))
\to a=\operatorname{Linear}_{BF16}(q,W_1,b_1)
\to h=\operatorname{GELU}_{\tanh}(a)
\to r=\operatorname{BF16}(h/S_2)
\to c=Q_2(r)
\to y=\operatorname{Linear}_{BF16}(c,W_2,b_2).
\]

`infer_ffn` 还先把 attention 输出加入 residual，再作 norm／调制。实际 `post_process` 是原生 `x.add_(y*gate)`；本分支保存过的 gate 为 `[1,D]`，沿 token 广播。矩的输入应是实际 **post-Q1 q**，这个信息出现前已经支付了全 token norm／调制／Smooth1／Q1。它不同于原始 latent 的均值，也不同于无需接触 M×D 数据的标量 observer。

图锚：F03（post-Q1、before-W1）→F04（W1 GEMM）／F06（tanh-GELU hidden）→F08R（HiF4 shared scale tree）／F08（post-Q2 系数）→F09（W2 投影），再经 F12／P24／P25。条件矩到末态 DC 或视频功能的边均为 conditional／OPEN。

## 1. 矩闭合真正能够精确表达什么

先取**实数解码权重、连续 erf-GELU、未作 Q2 和最终 BF16 舍入**这一清楚的子问题。令 token 在一组指定坐标上均匀抽取，q 的均值／协方差为 \(m,\Sigma\)，W1 第 j 行为 \(w_j^T\)。那么

\[
\mu_j=w_j^Tm+b_{1j},\qquad
v_j=w_j^T\Sigma w_j.
\]

这两个等式对任何有限二阶矩分布都成立。若每个边缘 \(a_j\sim\mathcal N(\mu_j,v_j)\)，定义 \(s_j=\sqrt{1+v_j}\)、\(k_j=\mu_j/s_j\)，则

\[
F(\mu,v)=\mathbb E[X\Phi(X)]
=\mu\Phi\!\left(\frac{\mu}{\sqrt{1+v}}\right)
+\frac{v}{\sqrt{1+v}}
  \phi\!\left(\frac{\mu}{\sqrt{1+v}}\right).
\]

推导：\(\mathbb E\Phi(X)=\Phi(\mu/\sqrt{1+v})\)，而 Gaussian integration by parts 给出 \(\mathbb E[(X-\mu)\Phi(X)]=v\mathbb E\phi(X)\)，其中 \(\mathbb E\phi(X)=\phi(k)/\sqrt{1+v}\)。于是理想部分的均值为

\[
\widehat{\bar y}=W_2\big(F(\mu,v)\oslash S_2\big)+b_2.
\]

**只预测 W2 输出均值，不需要 hidden 单元间的协方差。** 线性 readout 的期望只消费各系数均值；需要的是每个 W1 投影的正确边缘方差。输入 q 的 off-diagonal covariance 因此仍然重要。联合 Gaussian 是充分条件，各投影边缘 Gaussian 已经足够；“hidden 独立”是比这个均值命题更强的额外假设。

这个闭式是已知数学组件，已与 [Kuang–Lin，2026，Appendix D Lemma D.1](https://arxiv.org/html/2601.22307v1#A4) 的原式核对，不主张公式创新。文献还区分单层 Gaussian 矩精确性与多层重新 Gaussian 化的误差。它改变了这里的问题：先问边缘闭合及协方差信息是否成立，再问如何部署。

### 与旧 Gaussian slope 的严格区别

[GELU 原论文](https://arxiv.org/html/1606.08415v5) 给出 erf 定义与实际使用的 tanh 近似。两种连续函数都满足

\[
g(x)=\tfrac12x+r(x),\quad r(-x)=r(x),\quad
g'(x)+g'(-x)=1.
\]

当 X 对称时，\(\mathbb E g'(X)=1/2\)，但 \(\mathbb E g(X)=\mathbb E r(X)\)。特别地，零均值 Gaussian 有
\(\mathbb E g(X)=v\phi(0)/\sqrt{1+v}\)，随 v 改变。平均 slope 完全相同并不固定偶残差 DC。

旧 `run_e0_statistical_gelu.py` 的 `gaussian_moment_gain` 通过 quadrature 计算 **期望导数**，再用另一个来源的 32 行 post-Q2 均值拟合 offset，最后评估逐行输出 SSE。新研究对象是当前 q 分布预测的 **输出均值**；没有理由把旧 slope 失败扩大为这一对象已失败，也没有理由把旧 offset 校准误称为当前输入条件矩预测。

## 2. 能直接推翻过强闭合主张的反例

### 对角输入方差即使在完全 Gaussian 时也会错

取 \(q=(G,-G)\)、\(G\sim\mathcal N(0,1)\)，\(w=(1,1)\)、b=0。真实预激活恒为 0，因此 GELU 均值为 0；对角估计却给 \(\tilde v=1+1=2\)，预测 \(F(0,2)=2\phi(0)/\sqrt3>0\)。这是 Gaussian 输入下遗漏协方差的 O(1) 反例，而非尾部或采样问题。把两输入改为同号又能反向低估方差。

此外，GELU 不是全局凸函数：

\[
\partial_v F(\mu,v)=\frac{\phi(k)}{2\sqrt{1+v}}
\left[2-\frac{v}{1+v}-\frac{\mu^2}{(1+v)^2}\right].
\]

故“少估方差一定少估输出均值”也不是通用符号规律。即使各系数误差符号已知，W2 signed value vectors 仍可相消或增强。

### 同样的均值／方差不能识别非 Gaussian 的非线性均值

令 X 以 \(1-\epsilon\) 的概率为 0，以各 \(\epsilon/2\) 的概率为 \(\pm1/\sqrt\epsilon\)。所有 \(\epsilon\in(0,1]\) 都有均值 0、方差 1；但 erf-GELU 均值为

\[
\sqrt\epsilon\,[\Phi(1/\sqrt\epsilon)-1/2]\longrightarrow0.
\]

同矩 Gaussian 的预测却固定为 \(1/(2\sqrt\pi)\)。对 tanh-GELU 同样有趋零结论。前两矩对任意尾部分布不存在这样的统一相对误差保证。LayerNorm 是每 token 跨 channel 的约束，不保证跨 token 的各投影 Gaussian；训练后的固定 W1 与相关输入也不能仅凭 D=5120 自动套用独立求和 CLT。

### Signed 输出使“小的逐系数均值误差”也可能重要

若两个值向量为 v 与 −v，而真实系数均值为 a 与 a+η，真实部分输出只有 −ηv；一个相对 a 很小的系数误差可以远大于这个净输出。应直接评分 \(\|\operatorname{diag}(gate)W_2\Delta\bar c\|^2\)，同时保留绝对能量和近零分母，而不是仅看每个 neuron 的百分比误差或对角加权和。

## 3. 实际 tanh／Smooth2／Q2／gate 的边界

Gaussian 闭式用于 erf-GELU，不是原生 `approximate='tanh'` 的精确期望。可以分别比较同一 saved z 上的 erf 连续均值、tanh 连续均值和保存的 BF16 GELU；不必先把三个误差混成一个拟合系数。

Smooth2 在激活**之后**，故实数常量除法允许 \(\mathbb E[h/S_2]=\mathbb E[h]/S_2\)，但不能把它移到 GELU 内。实际 h、除法 cast、Q2 和最后 W2 都有数值边界。一般有 \(\mathbb E Q_2(r)\ne Q_2(\mathbb Er)\)；HiF4 Q2 的 64 通道根组及 8／4 子组 scale 还依赖同 token 的其他系数。相同的单 neuron 均值／方差不能决定这个联合量化均值。

因此 native 输出均值可写成

\[
\bar y_{native}=W_2\bar c_{postQ2}+b_2+\bar\eta_{W2},
\]

其中 \(\bar c_{postQ2}\) 与 Gaussian continuous 系数预测的差包含 Gaussian 闭合、tanh、W1／GELU 舍入、Smooth2 cast 与 Q2。只有实际字段足够时才逐项分开。M128 archive 没有保存 q2_pre 时，Smooth2 cast 与 Q2 的差应合并标记，不把它伪装成纯 Q2 项。

当前广播 gate 允许在理想实数局部均值中移出 token 平均；真实 postgate／residual BF16 写回仍有自己的舍入。若其他模型分支使用 token-dependent gate，还要增加 \(\operatorname{Cov}(gate,y)\)，不能照搬此简化。

## 4. 更强的竞争解释与替代表示

**后续 attention 可以保留 DC，也能生成 DC。** 固定 row-stochastic attention A 满足 \(A\mathbf1=\mathbf1\)，所以常量 value 更新能穿过该 token mixing；但通常 \(\mathbf1^TA\ne\mathbf1^T\)，故零均值 value 更新也可能输出非零均值。Q/K 随状态变化时，还出现 attention-weight 的有限变化项。均值为零的 \((a,-a)\) 通过 GELU 偶残差也会生成共同正均值。末态 DC 因而可能是 spatial／token 变化经后续算子整流、重加权后产生，而不是早期 FFN DC 直接遗失。

[Dong–Cordonnier–Loukas，ICML2021](https://proceedings.mlr.press/v139/dong21a/dong21a.pdf) 的 stochastic path 表示提供了相邻机制；其纯 attention rank-collapse 条件不能直接推成带 residual／MLP 的 Wan 结论。这里使用的是明确的 \(A\mathbf1=\mathbf1\) 与均值不保守的代数事实。

**稀有 token／少数 neuron 可能承载全局功能。** [Jiang 等，2025，§3](https://arxiv.org/html/2506.08010v1) 在 OpenCLIP／DINOv2 发现少数 MLP neurons 与高 norm token、后续 attention sink 的联系。这使“Gaussian bulk 估得很好但漏掉少量功能性尾部”成为值得区分的解释；它不是 Wan 存在相同 register 机制的证据，也不引入额外 token 方法。

**分布混合可以比单 Gaussian 更合适。** 对预先固定的 token 区域 B，条件 Gaussian 假设会给出 \(\sum_B\pi_BF(\mu_B,v_B)\)，不同于用 pooled 两矩计算一次 F。这是条件信息增加的解释路线，代价随区域数量增加；空间组不是语义 mask，观察到某个分组有效也不等于发现相机／主体机制。

**均值保护可先作为不依赖 Gaussian 的能力参照。** 对局部误差 \(e_t=f_D(q_t)-\tilde f(q_t)\)，理想常量校正 \(\tilde f+\bar e\) 满足

\[
\sum_t\|G(e_t-\bar e)\|^2
=\sum_t\|Ge_t\|^2-M\|G\bar e\|^2,
\quad G=\operatorname{diag}(gate).
\]

这是局部 token-DC 的最优常量校正和精确 SSE 分解；\(\bar e\) 是付过 dense 计算的 oracle，不能拿来声称在线获得目标。它先区分“均值这一动作本身是否有局部容量”与“moments 能否预测目标”。这种均值保护与 [Nagel 等，ICCV2019，§4.2](https://arxiv.org/pdf/1906.04721) 的 bias-correction 共享已知组件；后者在固定权重误差下利用 \(\Delta W\mathbb Ex\)，这里面对当前输入、非线性系数与动态 Q2，没有相同的自动精确性。

另一条竞争路线是预定少量 token 的 residual sample mean／control variate。在真正独立的均匀无放回抽样设计下它可以无偏估计全 token 均值，并有有限总体方差；固定历史 random32、一个 contiguous128 或 source 内重复 step 不能据此随意增加独立样本数。若只需要实数 decoded W2 均值，可先平均 hidden coefficients 再做一次 W2 matvec，减少 sampled readout 的计算；native 每行 W2 舍入不会与平均精确交换。这是另外一种付费 observer，不是本文件选定的实现。

更一般地，[Chorin–Hald–Kupferman，PNAS2000](https://math.huji.ac.il/~razk/Publications/PDF/CHK00.pdf) 的条件投影说明粗变量动力学还依赖未解析状态。这提供一个表示层面的提醒：16 维末态 DC 不是天然闭合的四步状态，前一层均值／方差并未囊括后续的空间、attention 和高阶矩。这里没有新增实时或缓存任务。

## 5. 真实信息、权重读与 MAC 成本

对 D=5120、H=13824，\(DH=70,778,880\)。理想 observer 做
\(W_1m\)、\((W_1\odot W_1)\operatorname{diag}\Sigma\)、\(W_2\hat m_h\)，共 **212,336,640 MAC**。均值 readout 保留全部 signed W2 列，不是选少数 neuron。

最理想融合扫描两块 BF16 权重也触碰 **283,115,520 字节**；未融合的方差投影可能再次扫描 W1，预存 FP32 的 W1² 又增加存储与访存。是否能复用已有解码／residency 的 HBM 读取须实际核对，不能把逻辑字节直接冒称实测 DRAM 流量。

M=31668 时，独立扫描完整 BF16 q 还要读取 **324,280,320 字节**，加上均值／平方和 reduction 和全局同步；融合进 Q1 producer 是未来实现选择，仍有累加／同步成本。完整 dense W1+W2 是 **4,482,851,143,680 MAC**，所以这个 observer 的 matvec MAC 仅占约 **0.0047%**，但它主要是低算术强度工作，不能从 MAC 比例推出相同比例的 wall overhead。

若只在 K=32 行保存面板上算矩，获得的是该 panel 的矩，尚非真实 full-M 条件信息。若使用 K 个真实 W1 sample 做 projected-covariance observer，成本又从单个对角 matvec 变为 K 行 W1；它获得了更多信息，应单独计费。单独均值 observer 本身没有省掉原 FFN 的 M×D×H 运算，只有与另一个确实少算的功能表示结合才可能改变 frontier。

## 6. 最小 saved32／128-row 判别设计：先测假设，不先修复

建议预先固定 **block10、high-noise step1、source0／12**，各使用已有 all-block tangent archive 的原 random32 与中心 contiguous128，保持真实 RUN seed（该较新 archive 公开记录为 42／54），共四个面板，全部 H 坐标／实际 W2。更早 shared-tile 报告另有 seed42 描述，不能仅按 source 名称混接。先验证精确 RUN、tensor、row IDs、权重 shard／Smooth2／gate；重叠行须去重或明确标为重复观察。

| 控制／预测 | 使用的信息 | 能区分的问题 |
| --- | --- | --- |
| `g(mean(z))/S2` | 已保存 native z 的均值 | 忽略偶残差的 plug-in baseline |
| `F(mean(z),var(z))/S2` | 已付完整 W1 的边缘矩 | Gaussian 边缘闭合控制；不是在线 pre-W1 observer 或 loss 最优下界 |
| 实际 saved z 上的 erf／tanh 均值与 saved GELU | 同一批完整 H 样本 | Gaussian 误差、函数近似与实际 GELU 数值边界 |
| `F(W1 mean(q)+b1,(W1²)diagvar(q))/S2` | post-Q1 矩＋静态完整权重 | 目标 cheap observer 的协方差／数值缺口 |
| saved post-Q2 均值、实际 native output 均值 | 真实后验标签 | Smooth2 cast／Q2 与 W2 舍入是否留下 signed mean 误差 |

第一步甚至不用重做 W1：saved z 已给出各边缘矩，先将闭式与同一面板的 empirical erf-GELU 均值比较，再经实际 signed W2／gate 评分。若这个 post-W1 控制就失败，调整 pre-W1 对角矩不能补上 Gaussian 模型缺失的高阶信息。若边缘控制好而 pre-W1 对角预测差，下一项才是付费 sample W1 replay／projected-covariance 对照；不能仅凭 saved-native-z 与理想线性预测的差就把全部误差归给 covariance，BF16 W1 也在其中。

连续均值可用而 post-Q2／native 输出均值不可用，则定位到格式／数值边界；两者都可用，也只建立该面板上的均值预测能力。可预先加入中心128按坐标固定切成两个64的条件矩控制，检查单 Gaussian 与分布混合，但应记录它使用更多条件信息，不进行看误差后选分组。

主要量直接用 current-gate signed output mean 的**绝对误差、方向／dot 和范数**，辅以原始／centered 输出能量，保留近零情况；不把 diagonal neuron 误差或同一来源的行数当作充分结论。跨 source／step 的固定 offset 比较须明确其原校准来源和已支付成本。

这些面板能判别 Gaussian／covariance／Q2 条件，不能证明 full-M DC、四步轨迹或视频收益。当前冻结 TRACE 保存完整 SSE／dot，但没有逐 site 的全 M channel-mean vectors；不能从范数倒推出该均值。正在运行的 frozen phase1 不改；若面板支持机制，后续单独有界研究才考虑同时观测 full-M FFN 均值、postgate 均值及 suffix 的 DC／非DC 传输，再比较同预算固定替代和完整视频。

## 检索记录与 EXECUTION

2026-10-26 使用实时 web search／primary open，多轮覆盖：Gaussian GELU moments、PBP／assumed-density filtering、analytic bias correction、row-stochastic attention／token uniformity、MLP register neurons、非线性粗变量闭合。实际重点读到：GELU 原论文 §2；Kuang–Lin2026 §1、Appendix D.1 与 adversarial section；PBP2015 §3.3 的 diagonal Gaussian 传播；Nagel2019 §4.2；Dong2021 §2 stochastic path；Jiang2025 §3；Chorin2000 projection 推导。PBP 是[已知 Gaussian moment-propagation 近邻](https://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf)，不是 Wan token 分布成立的证据。扩展检索改变了初始解释：均值闭合是已知 ingredient，真正尚未确认的是当前数据分布、signed numerical boundary 与跨层功能，不能靠一个 Gaussian 公式宣称新方法或其必然成功。

执行边界：本轮仅数学推导、原始源码／公开旧报告与一手文献阅读，以及整数操作量核算；未加载或计算任何 saved tensor、未执行模型／GPU、未改运行源码／协议、未作拟合或候选视频。上述 saved-data 对照是待独立登记的设计。新文件不改变 ACTIVE、实验状态、原 frozen 记录或质量接受。

读取的关键本地源 SHA：

| 对象 | SHA-256 |
| --- | --- |
| `run_e0_statistical_gelu.py` | `ec01b29d19b4defd0fc4864ed87b5684614225d89a066e53f6eecc03e8355c26` |
| `e0_function_basis_math.py` | `01447d9550a308e0735e3d9a837aebaffe25ee7e12513ccbdab9dba4a3a24e4f` |
| `e0_paired_ffn_observer.py` | `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` |
| native Wan `transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| HiF4 `lightx2v_wan_distill.py` | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |

对 FFN compute／完整视频的含义：偶残差均值提供了一个独立于旧 slope 的候选功能坐标；是否需要保护它、是否能从当前便宜信息预测它、是否能在少算后保留真实视频，分别由上述竞争性证据决定。26% latent DC 是打开这个问题的现象，不是预选均值修复的结论。
