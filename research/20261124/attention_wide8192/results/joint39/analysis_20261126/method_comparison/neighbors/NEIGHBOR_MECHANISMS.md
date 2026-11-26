# FFNDS 的近邻机制：ICC、SparsePR 与 MC-Sparse

核验日期：2026-11-26。当前 FFNDS 指四步 HHLL、wide8192/P64、low mass-centered 方案。本文为机制和静态成本研究；新 GPU 计时为 0。来源身份见 `SOURCE_IDENTITIES.json`。

## 1. 直接结论

最有区分力的三个邻居分别承担三个对照角色：ICC 比较“固定权重低秩增量”与当前原生观测驱动的 FFN 恢复；SparsePR 比较“当步绝对稀疏残差拟合”与历史状态支持的创新恢复；MC-Sparse 比较“静态 dense−sparse 残差复用”与当前 value 坐标运输加探针校正。

我们获得的是少步情况下的当前响应适应能力，以及让有限探针拟合更小、更有结构目标的机会；付出的是原生探针、历史字段、读出和锚点成本。收益应以同质量下可删减的计算衡量。P64、ridge、残差缓存各自已有直接先例，具体状态与更新方程才决定与近邻的关系。

## 2. ICC：原始非线性仍然执行

ICC 全称 **Increment-Calibrated Caching**，CVPR 2025 论文 *Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition*。原文 §3.3 Eq.(3–4) 在每个线性层使用

\[
\widetilde y_t=y_a+\widetilde W(x_t-x_a).
\]

作者实现 `ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4` 的 `DiT/models/models_sp_step_decomp.py` 中，`SpMlp.forward_upd` 保存两线性层各自的输入与输出，`forward_inc` 在第一个增量后执行当前 GELU，再进行第二个增量。不能把它描述成整个 FFN 的固定 Jacobian。

采用行向量约定，令

\[
z_a=X_aW_1+b_1,\quad h_a=\phi(z_a),\quad Y_a=h_aW_2+b_2,
\]

则作者路径为

\[
\widetilde z_t=z_a+(X_t-X_a)\widetilde W_1,
\quad\widetilde h_t=\phi(\widetilde z_t),
\quad\widetilde Y_t=Y_a+(\widetilde h_t-h_a)\widetilde W_2.
\]

**由该代码过程直接推出的误差恒等式**为

\[
Y_t-\widetilde Y_t
=(h_t-\widetilde h_t)W_2
+(\widetilde h_t-h_a)(W_2-\widetilde W_2).
\]

第一项是 W1 截断误差经过当前非线性后的作用，第二项是 W2 对激活增量的截断误差。因此两者都由实际输入增量驱动；将其归因于“没有使用当前输入”或“忽略 GELU”会误判原方法。

CA-SVD 用输入/输出激活幅度加权，CD-SVD 用步间增量幅度加权；作者采用 256 张图的校准集。原文实验对象是 DiT-XL/2 和 PixArt-alpha，主要计算指标为 MACs。论文跨不同采样步数匹配计算预算；例如 30 步 ICC/CD-SVD 与 35 步 DDIM 的比较，并非四步视频模型上的延迟结果。作者的 attention 增量路径保留完整 QK/AV，低秩化的是 Q/K/V/O 投影。

### 同预算 FFN 对照

当前 Wan 维度 \(N=31668,d=5120,m=13824\)。令 ICC 两个线性层都用 rank \(r\)，其 FFN hit MACs 为 \(2N(d+m)r\)，Dense 为 \(2Ndm\)，所以

\[
q_{ICC}=r/d+r/m.
\]

我方主 hit MACs 为 \(Nd^2+2Ndk\)，\(k=2560\)，剩余比例 \(q_{ours}=10/27\)。令两者相等得到 \(r=1383.7838\)，因此选择对齐 rank 1408 是合理的近等预算构造。具体数值见 `STRUCTURAL_COST_COMPARISON.json`。一锚点一命中的周期比例均为 \((1+q)/2\)。

这是主矩阵乘加对齐：不含 GELU、FP32/BF16 转换/舍入、缓存读写、低秩权重准备、gather、路由、kernel launch，也没有据此假设两者质量或延迟相同。ICC 源码中的四个 FFN 历史字段在上述形状占 \(2N(d+m)\) 个元素；BF16、40 层直接常驻对应约 89.39 GiB。这是忠实照搬作者状态的静态容量外推，既不是必要状态下界，也不是适配实现的测得峰值；重算、融合或 offload 可以改变这个代价。

**得失判断：**在相近乘加预算下，我方保留 selected 非线性响应、折叠线性主项以及当前 P64 真值；ICC 保留低秩近似后的全隐通道非线性，执行更规则、可用 rank 直接控制计算。匹配对照要回答：当前观测带来的适应性，是否能比固定权重低秩分配更有效地保护当前输出与后续模块。

来源：

- [ICC 原文 §3–4](https://arxiv.org/html/2505.05829v1)
- [CVPR 2025 正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html)
- [ICC 固定源码](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py)
- [校准入口](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py)

## 3. SparsePR：当前 P64、ridge 和覆写的直接近邻

SparsePR v2 §4.2、附录 A.3 每个 attention 调用在 64 个 query 行计算全 key 原生输出，用标准化 sparse 输出作为特征，对 dense−sparse 残差做 coverage 加权的 affine ridge；最终将 probe 行覆写为原生值。默认不做输出子空间截断。它还根据当前响应构造 K/V 与 query 分组，从而同时改善共享路由和残差的可拟合性。

它的输出恢复没有跨步历史输出残差字段；当前分组计算可以复用此前 centroid/factor 作为迭代初值。不要将“无历史残差”误写成完全无状态。

原文在同一 H100、BF16 模型上报告 Wan2.2 I2V 1.80×，\(N=75600\)、实际 pair density 21.97%。§5.2 将 1650→917 秒明确称为 estimated full-generation latency；C.3 说明同 checkpoint/input/seed/schedule、计入在线开销，但该版正文与附录没有列出具体 step 数或标识为四步蒸馏。官方仓库截至本轮仍是两个 commit、仅 README、代码待发布，因此结果属于作者报告，本轮未执行其完整实现。

**机制关系：**我们的 OUT-ABS 与它共享“当前稀疏底座＋原生 probe 拟合”的结构。区别应落在特征、分组、拟合目标和成本上；对 low centered，更大的区别是完整历史状态先按当前 value 统计解码，P64 拟合剩余创新。

固定同一 P 与恢复算子 K，令 \(r_t=A_t-G_t\)，\(R_a^{tr}\) 是运输后历史响应。未采样行的两种误差为

\[
e_{absolute}=M_{\bar P}(I-KP)r_t,
\qquad
e_{state}=M_{\bar P}(I-KP)(r_t-R_a^{tr}).
\]

这是本次从共同接口推导的辨析式。历史状态的价值在于降低恢复器未能表达的剩余变化，不能只由 probe 数或总残差能量判断。相应代价是完整锚点和历史存储；SparsePR 的吸引力是直接当步恢复、无完整历史输出字段。

来源：

- [SparsePR v2 §4.2、A.3、C](https://arxiv.org/html/2608.18484v2)
- [作者项目页](https://pardistaghavi.github.io/SparsePR-website/)
- [当前官方仓库](https://github.com/PardisTaghavi/SparsePR/tree/e028c78e6b40a29cb9683a9f90f7d7fa422b2b0e)

## 4. MC-Sparse：静态残差复用与昂贵锚点的精确先例

MC-Sparse §4.3 Eq.(3–4) 用 \(R_a=A_a-G_a\) 缓存全位置 dense−sparse 残差；命中重算当前 Q/K/V 与 sparse 输出，输出 \(G_t+R_a\)。它还缓存 query 分组与根据真实 attention 概率选出的 token-level KV 支持。与 block 绑定相比，token 粒度给它更细的计算分配。

附录 A.2 明确锚点采用 two-pass：完整 attention 返回 output/LSE，再重算全 QK 以恢复精确概率并聚合选择；构建残差另需一次 sparse attention。源码 `TokenSparseCache`、`_full_cal` 与 reuse 分支吻合。当前默认分支 master 仍为 `880f4859faf3621e70ec736546d9839420b76ab4`。

表 9 的 Wan2.1 I2V/T2V 都是 50 步、10 步 dense warm-up、anchors {10,26}；其 I2V 去噪加速 1.52×，计时包括 warm-up、排除条件编码/VAE。H3 49 步对应最高 1.80×；设备为 Hopper，代码使用 CuTeDSL/WGMMA。长复用区间摊销精确选取与分组是其明确能力。

**得失判断：**MC-Sparse 得到更准确细粒度支持和便宜命中，付出较贵锚点。我们的低步数方案把计算更多分配给每次命中的当前观测和状态解码。相同 sparse 算子/路由下，关闭运输与 P64 后，历史残差复用与 MC-Sparse 属于条件重合；完整方法则改变了残差的状态表示和当前更新来源。

来源：

- [MC-Sparse 原文 §4.2–4.4 与附录 A](https://arxiv.org/html/2610.06801v1)
- [作者 token sparse 核心源码](https://github.com/dodododddo/mcsparse/blob/880f4859faf3621e70ec736546d9839420b76ab4/mcsparse/kernels/flash_mcsa/token_sparse.py)

## 5. 对当前 FFNDS 最有用的取舍

1. FFN 以 ICC rank 1408 为近等主乘加对照，比较当前精确观测是否改善同预算的输出保护；保留作者逐线性增量与当前非线性语义。
2. attention 以 SparsePR 的当步绝对残差恢复、MC-Sparse 的历史静态残差复用作为信息来源的两端，再定位 centered 当前解码与 P64 更新各带来的收益。
3. 将已验证的 mass-centered 改写视为一个状态设计原则：锚点保存哪些场，当前哪些信息可解析更新，剩余变化由哪些观察恢复。该原则比“我们也用了稀疏、缓存、校正”更有区分力。
4. 实际优势应落实为同画面/运动质量下更短完整请求、更少原生调用或更小状态。当前两源局部方向改善支持运输机制；它与六源完整方案速度一起确定下一轮对照对象，尚不替代同批原方法适配比较。

PI-Attn / ReaDy 的本轮检索命中同名 LLM 工作，未从历史引用恢复确切视频方法身份，因此未把它们混入此表。
