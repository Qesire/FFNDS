# FFNDS方法讨论：量化胞腔与组级共享

日期：2026-10-24（Asia/Shanghai）  
作者：codex-ffnds-method-01a16d3d  
研究源：FFNDS @ 00617012688e70c53e645618ffdae7f4a27f91be  
方法论：AutoResearcher @ 5bd7ec537ae63f2b6a6dd579f0e886aff58a177c  
状态：条件数学构造与源码核对；未进行数值实验、模型执行或性能测量。

## 本轮得到什么

可把FFN共享单位从“整个token输出”改成“同一原生量化胞腔内的完整64通道组”。只共享不变组，其余组逐token计算，因此可以保留token间差异。所需结构是**量化后的组结果相同**，不要求整个FFN输出相同，也不要求下游归一化把误差消掉。

这一构造的主要未决问题不是代数等式，而是：在目标token的W1之前，是否存在足够便宜、足够紧的相等性证书。若只能在完整W1后判别，理论可省工作量减半。下文明确区分两者。

本提案未获其他agent评议；不能称为PEER_SUPPORTED。另有一份独立parity评议，结论为REVISE：保留其理想恒等式，补足Smooth坐标与完整组量化影响。

## 1. 真实计算边界

按列向量记号，固定同一层、去噪步、expert、权重与数值配置：

\[
v_t=Q_1(\operatorname{BF16}(q_t/s_1)),\quad
z_t=\operatorname{Linear}_{BF16}(v_t,A,b_1),
\]
\[
a_t=Q_2(\operatorname{BF16}(\operatorname{GELU}_{tanh}(z_t)/s_2)),\quad
y_t=\operatorname{Linear}_{BF16}(a_t,B,b_2).
\]

最后对每个token使用其当前gate与残差。d=5120、m=13824、G=216个完整64组。Q1仍需处理全部token；它的成本不算省掉。

已核对的关键事实：

- packed_ffn_reference.py保留原始完整组；s2随被选通道一起选择，除scale发生于Q2之前。
- native_hif4_activation_qdq调用hif4_qdq_tensor，沿末维分组。
- backend使用hifx4与float QDQ，矩阵计算是BF16 GEMM。
- QTensor.py在force_fp32时先转FP32，优先选择CUDA实现，最后转回原dtype；Python fallback不是未经核对即可替代的数值定义。
- hifx_quant_cuda_inner以64坐标共同最大值确定BF16/E6M2舍入尺度；内部8坐标及4坐标子组还有离散尺度选择。相同尾数码而尺度不同，不代表相同反量化值。

源码只证明这些语义。当前没有读取完整checkpoint张量，没有获得目标Wan中cell重复率。

## 2. 条件组共享定理

令\(\Psi_g\)含GELU、Smooth除法、BF16以及原布局的完整组Q2；记：

\[
a_g(v)=\Psi_g(A_gv+b_{1,g}).
\]

此处A_gv+b使用理想实数第一投影；原生第一投影的舍入在下一节单列。在实数输出组合下：

\[
F(v)=b_2+\sum_{g=1}^{G}B_ga_g(v).
\]

对同一步的一小块T个token取anchor a，选择集合S。若

\[
\forall t,\forall g\in S:\quad a_g(v_t)=a_g(v_a),
\]

则

\[
F(v_t)=b_2+
\underbrace{\sum_{g\in S}B_ga_g(v_a)}_{\text{每tile计算一次}}+
\underbrace{\sum_{g\notin S}B_ga_g(v_t)}_{\text{每token分别计算}}.
\]

证明是逐组代入，没有近似。只要求S内相等，S之外可以不同，所以整个F(v_t)仍可不同。该条件允许共享非零、甚至很大的组贡献；它不是“小值置零”。

**原生适用边界：** 相同Q2操作数保证相同数学W2贡献。把原整宽GEMM拆为两个GEMM再加和会改变归约及BF16舍入，不能据此宣称与原native输出逐bit相等。真实bit identity和最终视频质量均未证明。

R13给出的整行共同输出下界要求两个token被写入同一个完整输出C。本构造只令部分组贡献相同，因此该前提一般不成立。R13也没有提供本方法所需的逐组Q2状态标签。

## 3. 前置证书：一个充分条件，尚非低成本算法

对anchor实际已得到的z_ag，定义量化胞腔：

\[
\mathcal C_{ag}=\{z:\Psi_g(z)=\Psi_g(z_{ag})\}.
\]

取已被证明包含在胞腔内的L∞球半径\(\mu_{ag}\ge0\)，定义

\[
\kappa_g=\max_{j\in g}\|A_j\|_1.
\]

理想实数W1有

\[
\|z_{tg}-z_{ag}\|_\infty
\le\kappa_g\|v_t-v_a\|_\infty.
\]

故严格小于\(\mu_{ag}\)即可保证整个组Q2输出不变。原生版本必须满足

\[
\kappa_g\|v_t-v_a\|_\infty+\epsilon_{gt}<\mu_{ag},
\]

其中\(\epsilon_{gt}\)是anchor与目标第一投影浮点误差之差的可靠上界。**未推导出实际CUDA/BF16路径的有效且紧的epsilon；不能把它直接设零。**

mu必须同时涵盖Smooth、GELU、BF16、64组共同尺度、8/4子组尺度及尾数rounding边界。GELU负区间并非全局单调，不能只取区间端点像而忽略内部极值。这里使用胞腔包含定义避免假定单调性。便宜求mu的方案仍待其他agent推导。

可在anchor付费后为每组建立保守半径，把其余token压成单个输入距离，再取tile最大距离。这样目标侧候选判别可为O(Td+TG)，但anchor半径构建、可靠舍入界和选择成本必须另外计费；复杂度表达不意味着已有实现。

禁止循环论证：计算目标W1或Q2后再判断相等，仅构成事后判别，不能计入前置W1省算。

## 4. 可执行成本的数学要求

令b=64|S|。共享组算一次、非共享组算T次，故两投影名义MAC为

\[
C_{\rm share}=2d[b+T(m-b)],\quad
C_{\rm dense}=2Tdm,
\]
\[
C_{\rm share}/C_{\rm dense}=1-\frac bm(1-\frac1T).
\]

该计数已经包含anchor完整W1：共享b在anchor做一次，非共享m-b在包括anchor的T个token上做。

若所有目标都先做W1/GELU/Q2，仅W2共享，则

\[
C_{\rm late}/C_{\rm dense}
=1-\frac12\frac bm(1-\frac1T).
\]

因此对有限T，仅W2共享的两投影MAC比例严格大于1/2，即使b=m。它无法在这个MAC指标上超过一个理想半宽直接FFN；两者质量可能不同，且此式不是原生延迟比较。若前置证书允许两侧都省，则名义MAC低于1/2需

\[
b/m > \frac{T}{2(T-1)}.
\]

例如T=4时需共享超过2/3的hidden宽度，额外开销会让实际要求更严格。这是一个能改变路线选择的条件：只发现少量cell碰撞还不足以宣称有竞争力。

物理构造讨论采用同一tile共同S，让S合计输出仅存一个d维向量，不默认缓存216个d维group贡献。少量预打包S可限制权重搬运，但bank冗余、命中率、同步、split GEMM、广播、原生QDQ和完整gate/residual均须计费。S为空时使用原算子。该构造没有实现。

## 5. 三种有区分力的反例与预测

1. **小距离跨边界：** 一个组最大值跨共同E6M2尺度边界，可改变其他未变化坐标的反量化值。余弦高、L2小或尾数相同均不是证书。
2. **重复存在但证书失败：** 某方向实际A_gΔv很小，而\(\kappa_g\|\Delta v\|_\infty\)很大。此时应修正证书几何，不能据此否认重复结构。
3. **信息过晚或过贵：** 知道目标Q2后可以精确复用W2，但W1账已经花掉。前置判别与事后oracle必须作为不同成本对象。

完整64组状态的交集也可能极稀少。在假想独立且每坐标保持状态概率p的模型里，整组概率为p^64；共同尺度还增加依赖。这个模型仅说明交集障碍，不是Wan统计估计。

下一步讨论最有价值的是：从已读HiF4舍入源码推导可计费的保守mu，或者证明某类廉价范数证书在实际维度上必然过松。现有OP2清单保存V1及整行输出，没有提供本方法的逐组Q2状态；不能从整行误差标量推断命中率。实验维持暂停。

## 6. 近邻方法与原创性边界

- [ToMe for Stable Diffusion](https://arxiv.org/abs/2303.17604)：合并冗余token。这里研究保留token身份、只共享部分原生量化组。
- [DeepCache](https://arxiv.org/abs/2312.00858)：复用相邻去噪阶段的高层特征。本提案首先限定在同一步同层，避免依赖跨步平稳性。
- [QuantCache](https://arxiv.org/html/2503.06545v1)：联合层级缓存、动态精度和结构剪枝。当前构造聚焦固定原生精度下、可证明相等的组结果。
- [Neuron-Level Fuzzy Memoization](https://upcommons.upc.edu/server/api/core/bitstreams/b0a14faa-0e9f-4151-bbff-66616382c287/content)：以较便宜预测器决定神经元复用，是本方向必须正面对比的先行机制；本提案尝试充分条件而非预测近似输出。

这些差异是初步机制定位，不构成完整新颖性证明。缓存、区间界和分组线性代数均为既有原理。

## 7. 精确来源

本地读取源码的SHA256与FFNDS的upstream_audit/SOURCE_MANIFEST.json核对：

| 文件 | SHA256 | 状态 |
|---|---|---|
| HiFloat4/ptq/adapters/wan_i2v.py | d5a9be8ddb777fe360e66ba8e4ccde033b4a2969d2a0b853c7aefa0f45c44293 | 与C0原生绑定相同 |
| quant_cy/base/QType.py | aec909792bbeb59a1a441bd0f3677585bc9597389c10519185e00a706b56e0fc | 与C0及归档manifest相同 |
| quant_cy/base/QTensor.py | ac5061523cfb87c203cb38a250fffc590d2248f4b7e303f671ba36b48d0eaee2 | 与归档manifest相同 |
| quant_cy/base/QFuncs/hifx.py | 6a8d1f1b8ea49ce099439cff251309b4d9b3eeeab6a7ee0a83bdcf85b24704f7 | 与归档manifest相同 |
| quant_cy/base/cusrc/hifxg_quant_cuda.cu | 07120a5d2519e3ed3c8782acb8897a44e31103f93dd84d353ae7bb86b043e00c | 使用本项目修改版，不能只引上游原文件 |

其中后三者匹配2026-10-20归档，未独立核验当前服务器安装或编译二进制。没有连接服务器或启动任何实验。

源码定位：
- [packed参考](https://github.com/Qesire/FFNDS/blob/00617012688e70c53e645618ffdae7f4a27f91be/tools/packed_ffn_reference.py)：prepare、forward。
- [Smooth与native QDQ adapter](https://github.com/Qesire/FFNDS/blob/00617012688e70c53e645618ffdae7f4a27f91be/upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/wan_i2v.py)：native_hif4_activation_qdq、WanPreparedHiF4Linear.forward。
- [原项目修改的CUDA量化](https://github.com/Qesire/FFNDS/blob/00617012688e70c53e645618ffdae7f4a27f91be/upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu)：hifx_quant_cuda_inner，229–287行。
- [QTensor分派](https://github.com/alvinshao0313/HiF4_Sp/blob/9aa2d3be6de265910731aa7c0d053e9c773cf556/HiFloat4/hif4_gpu/quant_cy/base/QTensor.py)：get_cuda_func、quant_dequant_float。
- [FFN与残差顺序](https://github.com/ModelTC/LightX2V/blob/ed146db64d0b6130c669285fc1aa0b280a6a4823/lightx2v/models/networks/wan/infer/transformer_infer.py)：infer_ffn、post_process。
