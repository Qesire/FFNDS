# Chipmunk 原始机制与当前 FFNDS 的得失

核查日期：2026-11-26。作者仓库固定到 `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`。本次阅读原论文 §3.3、§4.1、Appendix B/C/D.1，并沿作者模块、算子、Triton sparse attention、配置和存储源码核对。没有执行 Chipmunk GPU 实验。原始实现来自 MIT 仓库，来源清单见 SOURCE_IDENTITIES.json，LICENSE 随证据保存。

## 1. 应当正面承认的已有能力

Chipmunk 已将 attention 和 FFN 共同表示为非线性系数对 value 向量的加权和，并通过稀疏差分保留历史响应；其价值同时包括列稀疏模式与实际 GPU 内核。把“从 FFN 扩展到 attention”“缓存加稀疏更新”“两次矩阵乘之间引入稀疏”本身作为我们的主要新颖性，区分力较弱。我们可强调的是当前真实输出观测参与恢复，以及 attention 概率质量、value 坐标与历史残差之间的具体处理。

公开实现已完整包含上述两个模块。README 2025-06-09 更新写明支持 Ampere/Hopper；CUDA 原实现针对 Hopper，另外有 Triton 实现。当前公开配置含 Wan 的 `provider: triton`，所以 A100 应以该路径核查，而不是依据早期 CUDA 要求排除该方法。实际 A100 性能仍由匹配运行确定。

来源：
- [原论文](https://arxiv.org/html/2506.03275v1)，§3.3、Appendix B.1。
- [作者仓库](https://github.com/sandyresearch/chipmunk/tree/b5e8d1c3da6bddc7ca9007c48e91b82087c11490)。
- [Wan Triton 配置](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/configs/triton-test/chipmunk-config.yml)。

## 2. 恢复真实计算公式

### Attention：重算归一化 selected 响应，保留其输出残差

省略 batch/head，令 \(A_t=\operatorname{softmax}(Q_tK_t^\top/\sqrt d)V_t\)，固定锚点选出的支持 \(S\)。令

\[
B_t=\operatorname{softmax}(Q_tK_{t,S}^\top/\sqrt d)V_{t,S}.
\]

这里 \(B_t\) 在 selected key 集合内重新归一化。它不是使用 full-attention 分母的未归一化部分质量。

作者刷新步计算

\[
C_a=A_a-B_a,
\]

随后命中步返回

\[
\widehat A_t^{\rm Chip}=C_a+B_t
=A_a+(B_t-B_a).
\]

源码确认：`modules/attn.py` 刷新时先 `dense_colsum_attn` 取得 full 输出、近似列和与当前 LSE，选支持，再额外执行 `csp_attn(..., -1)` 构造缓存；命中时 `csp_attn(...,+1)`。Triton 内核 `attn_csp.py` 在 selected key 循环内计算 \(l_i,m_i\)，输出 \(\mathrm{acc}/l_i\)。full LSE 不进入 selected 输出的归一化。

动态支持的信息时点：attention 的 top-k 来自刷新步的 query-group 列和，并缓存给后续命中；并非每个命中步重新搜索全 key。每组 192 个 query 可以选择单独的 key 列；公开配置还可以混合局部与随机支持。相较之下，我们目前的 key-block 粒度更粗，但利用 P64 当前全-key 教师提供额外信息。

精确定位：
- [attention 模块](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/modules/attn.py)：155–217。
- [selected 内核](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/triton/attn_csp.py)：157–186。
- [算子包装](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/ops/attn.py)：67–82。
- 论文 §3.3.1、Algorithm 2–4、Appendix B.2。

### FFN：当前块均值路由，逐坐标更新激活缓存

令 \(a_t=\sigma(X_tW_1^\top+b_1)\)，\(\widetilde a\) 为每个 token-neuron 最近一次计算的值，\(M_t\) 为当前选中坐标掩码。Chipmunk 执行

\[
\widetilde a_t=M_t\odot a_t+(1-M_t)\odot\widetilde a_{t-1},
\]

\[
\widetilde Y_t
=\widetilde Y_{t-1}
+\bigl[M_t\odot(a_t-\widetilde a_{t-1})\bigr]W_2.
\]

当前实现先计算每 128 个 token 的输入均值，经完整 \(W_1\) 投影，比较缓存块均值 preactivation，选变化大的 neuron。选中的当前精确 activation 减去旧 activation，再通过对应 \(W_2\) 行更新输出；activation 与 output cache 都随命中更新。因此不同坐标可以有不同年龄。当前源码可按 block_mask_cache 缓存支持，但前 10 步每个 sparse hit 都重新计算支持。

论文正文 §3.3.2 的一行公式把旧 activation 写进 \(\sigma\) 括号内，和 Algorithm 6、Appendix B.3 及作者源码不一致。比较应采用后者的“先激活，再相减”，避免把排版错误当算法。

这比当前 FFNDS 更轻的主要算术路径，是 selected 两次投影；我们的全输入折叠 \(d\times d\) 投影及当前 P64 校正，提供了 selected 子集之外的响应建模，同时付出相应计算。

精确定位：
- [FFN 模块](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/modules/mlp.py)：52–115。
- [FFN 算子](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/ops/mlp.py)：18–20、47–49、65–70、84–102。
- 论文 Algorithm 5–7、Appendix B.3。

## 3. 当前 centered attention 的具体区别

使用同一路由与同一 selected 响应 \(B_t\)，令 \(\rho_a\) 为锚点未覆盖的 attention 概率质量，\(\mu_t,D_t\) 为当前 value 的均值和逐通道标准差。当前 FFNDS 编码

\[
Z_a=\bigl[A_a-(1-\rho_a)B_a-\rho_a\mu_a\bigr]D_a^{-1},
\]

并预测

\[
b_t=(1-\rho_a)B_t+\rho_a\mu_t+Z_aD_t.
\]

假设对应通道尺度非退化，锚点重建成立，所以有精确差式

\[
\boxed{
b_t-\widehat A_t^{\rm Chip}
=-\rho_a(B_t-B_a)+\rho_a(\mu_t-\mu_a)+Z_a(D_t-D_a).
}
\]

这将共同成分和新增操作分开：

1. 双方都保留历史输出信息并重算 selected 响应。
2. 我们对 selected 变化按锚点概率质量加权。
3. 我们用当前全场 value 坐标重新读取历史残差。
4. 我们另有 P64 当前全-key 真值和特征读出校正；上式只比较校正前底座。

“用当前 value 坐标运输历史残差”有可严格辨识的条件性质。固定 QK 与支持，若

\[
V_t=V_aD+c,\qquad D\text{ 为正对角矩阵},
\]

则 \(\rho_t=\rho_a\)，当前均值和标准差按同一 \(D,c\) 变化，centered 底座恰好得到 \(A_aD+c=A_t\)。Chipmunk 的输出误差则为

\[
A_t-\widehat A_t^{\rm Chip}=(A_a-B_a)(D-I).
\]

纯平移时 \(D=I\)，两者都精确，所以区分点是历史残差的尺度运输，不宜泛称 Chipmunk 不能处理 value 平移。

CPU FP64 检查见 `verify_chipmunk_relation.py` 和 `CHIPMUNK_RELATION_VERIFIED.json`。该脚本使用合成数组核验恒等式，三组差式误差最大 \(4.44\times10^{-16}\)；正尺度＋平移构造中 centered 相对误差 \(2.95\times10^{-16}\)，Chipmunk 为 0.1393。后一个数仅是条件构造的数值见证，不是模型质量收益。现实轨迹还会改变 QK、支持质量和高阶形状，P64 恢复的作用应由模型数据判断。

## 4. 四步生命周期和可比较成本

作者默认 attention 生命周期的前两步均为 dense：step0 只计算 native 输出和 LSE；step1 利用旧 LSE 近似当前 column sums，选支持并建立 \(A_a-B_a\) 缓存。其余按 full_step_every 或手工 schedule 刷新。这个设计服务连续的长轨迹。

当前 HHLL 每个 expert 只有两次调用。把每个 expert 独立初始化后照搬“前两步 dense”，会用完两次调用；因此四步适配的关键是让首个锚点就产生可用支持，并计入其初始化开销。这是具体的初始化/支持识别适配问题，不构成方法无法用于四步的结论。

若约定两个锚点均当次得到支持、每锚额外计算一次 selected 响应，两个命中各一次 selected，则 attention 核心算量比例为

\[
r_{\rm Chip,4}=(2+4\alpha)/4.
\]

我方 high OUT-ABS 不在 high 锚点建立差分 cache，low centered 在 low 锚点建立；加两个 P64 教师后为

\[
r_{\rm ours,4}=(2+3\alpha+2\beta)/4,\qquad \beta=64/N.
\]

这两个式子只是同支持密度的算术比较。Chipmunk 第一锚点的路由如何形成、是否需额外 QK、是否改用其他廉价路由，必须明确；作者原实现的 old-LSE column-sum 近似不能凭空免费初始化。也要分别计入我方拟合、特征和读出，与两者保留的 QKV/O 等共同部分。

精确定位：[LayerCounter](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/util/layer_counter.py) 27–35；attention 模块 155–202。

## 5. 状态容量：我们的逻辑 FFN 缓存可以更小

BF16 路径的作者源码明确要求完整 activation cache 和 output cache 都为 BF16；存储类保留传入 dtype，没有额外量化这些缓存。

同 \(N=31668,d=5120,m=13824,k=2560\)，40 层、单一 active expert、全部逻辑状态同时驻留时：

| 主状态对象 | 容量 |
| --- | ---: |
| 我方 \(X_a,Y_a,Z_{a,\mathrm{selected}}\)：\(2N(2d+k)\times40\) bytes | 30.20096 GiB |
| Chipmunk \(A_{\rm cache},M_{\rm cache}\)：\(2N(m+d)\times40\) bytes | 44.69742 GiB |
| Chipmunk blockmean：\(2\lceil N/128\rceil m\times40\) bytes | 0.25543 GiB |

这是指定状态槽的容量比较。整体峰值还包含权重、我们的折叠矩阵与打包权重、作者的转置 W2 副本、indices、临时量和调度；Chipmunk 提供异步 CPU offload，实际 GPU 驻留量可以大幅降低。论文视频主表又关闭了 FFN 稀疏，因此上述预算用于“将其 FFN 机制适配到我们形状”的分析，不能当作作者视频的峰值显存。

来源：`modules/mlp.py`、`ops/mlp.py`、[存储实现](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/util/storage/layer_storage.py)；论文 Appendix B.4。

## 6. 加速数字的正确分母

| 作者公开结果 | 配置与含义 |
| --- | --- |
| HunyuanVideo 2.16×，1030→477秒 | 50步，720×1280×129，H100，Chipmunk 本体 |
| HunyuanVideo 3.72×，1030→277秒 | 同模型，Chipmunk＋step caching |
| Wan2.1 1.56×，1357→870秒 | 50步，720×1280×121，H100，Chipmunk＋STA |
| Wan2.1 2.67×，1357→508秒 | Chipmunk-73%＋STA＋step caching |
| 当前 FFNDS 1.156×完整请求／1.227×GPU | Wan2.2 I2V A14B，四步蒸馏，A100，对同权重四步 Dense |

作者视频主表只用 sparse attention deltas；其 Appendix D.1 解释 MLP 占墙钟较小，因此关闭 MLP 稀疏以获得其最佳速度质量折中。当前作者 Wan/Hunyuan 配置也将 `mlp.is_enabled: false`。

作者的 Chipmunk＋step caching 使用固定中间跳步调度；论文示例为中间30步每4步计算一次。它与单独列出的 TeaCache 自适应基线分开。当前仓库的跳步集合可以比论文示例稍有区别，应按锁定配置比较。

因此，“论文2.67×和我们1.156×”反映了不同模型、序列长度、步骤数、硬件与组合范围。其数字仍提供一个有用事实：大倍率来自充分摊销、注意力占比与模块内稀疏/整步复用的组合。我方的特定能力是在少步蒸馏剩余调用中获得可接受质量与额外收益。谁在这一条件下形成更好速度质量曲线，仍由匹配适配结果决定。

来源：[作者 README 主表](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/README.md)；论文 Table 4、§4.1、Appendix D.1。

## 7. 最有用的比较判断

**Chipmunk 的优势**：动态细粒度支持；省略未选坐标更新；较轻的 FFN hit 算术；专用内核、稀疏 gather 与缓存写回的共同设计；成熟的多模型长轨迹证据。

**我们当前获得的能力**：低噪 attention 的概率质量与 value 尺度运输；每个命中用当前 P64 全响应观测校正未建模变化；FFN 对未选通道仍有全输入线性预测；四步双 expert 的完整周期证据；更小的指定 FFN 主缓存槽。

**相应代价**：我们计算完整 \(d\times d\) 折叠响应、小探针真值与全场读出，只有一次 hit 的锚点摊销有限；当前实现的动态支持粒度和专用内核优化空间需要继续评估。P64 的信息利用效果取决于剩余变化能否由当前读出表达，不能从代数接口直接推出质量。

**新颖性应落在**：如何让有限历史与少量当前原生观测，在变化较大的短轨迹内形成有效恢复；其中 probability-mass accounting 和 value-coordinate transport 提供可证明的机制差别。通用的“预测＋校正”式子用于统一分析，具体贡献来自状态、观测、运输与恢复算子的构造。

