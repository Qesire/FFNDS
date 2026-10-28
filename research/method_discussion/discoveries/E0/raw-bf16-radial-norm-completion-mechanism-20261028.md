# 原始 BF16：先检验固定时刻范数偏差，再决定是否学习 state

**发现（NATIVE + saved-row OBS）**：完整 W1、head-span128＋weight-PCA384 的 rank512 W2，在原始 BF16 low/block39 中已把线性 head 误差压到约 3.4–4.2e−7，径向误差仍为 3.9e−4–1.66e−3。新保存行判别支持一个更简单的竞争解释：主要缺口可能是 step 固定的归一化偏差，而非必须逐 token 预测的新函数。

[12flow 原始结果](../../../20261028/ORIGINAL_BF16_LOW39_FOUR_CORNER_RESULTS_20261028.md) RUN `768a5218da2ac17914ffd7e6a3e1623adf848be679a71a941634596c2d1c0088`。新增 saved-row RUN `6e265ed29bb6a018e8fe2312259d29a9f12f6e26d38d7c6040870422ffe43a40`（HF `ffnds-derived/e0_raw_norm_rows_20261028/attempt1/worker/results/RUN.json`，actual exit0）：source0 拟合的 κ₂=1.039908414775213、κ₃=1.020973593727097 转移到 source12，把 256 行无权 ratio MSE 分别降低 92.87% /84.21%。这里 κ=σ_D/σ_S；需要的归一化输出 gain 是 **1/κ**。独立观察单位仍为两个已暴露开发图像簇，拟合一源、转移一源；这些数不是 native head SSE 或视频质量。

另一个具体结构是方差损失的符号。δ=z_S−z_D 时，Δv=2⟨z_D,δ⟩/D+||δ||²/D；四个 source×step 的平均 signed-cross 为 −0.261/−0.095/−0.226/−0.085，正平方项只有 0.00768/0.00231/0.00590/0.00208。实际缺口由相关性交叉项主导。仅补“被丢掉方向的正能量”会漏掉这一机制。

## 最有区别性的下一步：source12 的固定 κ 原生转移

复用原始 raw W1/GELU、同一已导出的 BF16 B/Phi/b2 和 source0 的两个 κ，不重新拟合。在原 final `norm.apply` **返回之后**、原 head modulation **之前**，把归一化输出乘 1/κ。原 shift、head bias 及后续算子原样执行；不把含常数项的完整 head 直接缩放，不改 postwrite u，也不让 norm 输出与 u 共用原地修改。κ=1 的 bare 模式走同一 wrapper，为实际新增接缝提供 neutral。

source12、seed42、末两步的固定八次完整 40-block flow：DD₂/DD₃ 两个 neutral，bare S₂/S₃ 两个 neutral，C₂(xD₂)/C₃(xD₃)，再 C₃(xC₃)/D₃(xC₃)。四个 neutral 的 flow/post 必须逐字节匹配已关闭 12flow 和原始 BF16 对照，通过后才运行 C；任一失败即停。最后两次仍包含明确付费的 Dense counterfactual。无视频、训练、拟合、QR/eigh；一次 setup，复用已绑定 raw CPU proof 并 stat 检查。

同状态报告实际 full-M head/flow SSE、q direct、**有效分母 κσ_S** 的 radial、signed cross 与 native remainder；raw-u σ_S 另列，不能把修正后的 head 继续按未修正 σ_S 解释。跨状态仍用 a=C(xD)−D(xD)、p=D(xC)−D(xD)、c=[C(xC)−D(xC)]−a 的完整 signed 四角。

若 native 同状态与自身状态都改善，固定条件 norm 偏差得到这一个 transfer 来源的直接支持；若同状态改善但自身状态收益弱，继续区分 Dense 状态传播与 state-dependent correction，而不是增加网络宽度。若 unweighted proxy 改善却 full-M head 不改善，优先查 q 加权/tail 行和离散读出余项。单末层局部成功只决定是否保留这条表示路线。

预算建议：1 A100、4 CPU、32 GiB、wall900/outer1200s、HF≤0.5 GiB；320 FFN、5 次 W2 override（635 次原 FFN linear＋10 次因子 GEMM）、328 次原 block-buffer load、3 次 1/κ gain，保留原 W2 预取成本。薄 wrapper 把 reciprocal 明确舍入为 FP32，原 LN 的 BF16 返回值先转 FP32、原地乘 gain、再 copy 回该 BF16 返回对象；每次增加648560640B临时缓冲，三个未融合算子的读写量下界合计3242803200B。κ=1走相同storage gate并直接返回；全部8次拒绝与原u的storage别名。仅该 W2 的 MAC 从70778880/row变为9699328/row，完整W1 70778880/row仍保留；没有当前系统加速主张。

## 固定 gain 的竞争者及合法信息时点

**state-dependent 范数补全（ALG/OPEN）**：保留全 W1 的 h 在线可用。记实际部署因子的实数矩阵为 A_S=BΦᵀ，E=(A−A_S)diag(g)P₀，则 z_D≈z_S+hE。在已有 z_S 范数之外，补全 2⟨z_S,hE⟩+||hE||² 才保留交叉项；它不需要在线 teacher Y。用预冻结共用随机 Ω[D,k] 可做控制变量估计：Δv̂=[2(z_SΩ)·(hEΩ)+||hEΩ||²]/(kD)。k=32 时每行增加 (H+D)k=606208 MAC、每 step FP32 sketch 权重约2424832B；离线构建 EΩ 的原权重/因子矩阵乘法必须收费。随机 sketch 的方差、负估计和 BF16 写回偏差均可能大于待修正缺口，不能预先称其准确；先用已付费新标签估计这些误差，再决定是否原生运行，不做 k sweep。

**B-H constrained q＋variance（ALG/OPEN）**：以 FFN 前的合法 V1/r/time 为输入，直接预测 q 和非负 slack，使 v≥qᵀ(KᵀK)⁺q/D；能跳过完整 W1 才有不同的计算目标。旧 HiF4 h128 的特征、目标及数值路径不同，不能替代 raw BF16 的可预测性证据。此时 raw 训练来源尚缺，q 投影、残差摘要、训练 teacher 及 own-state 标签都要另付费。当前跨来源固定 κ 已有强 proxy，因此下一次优先实际检验固定 gain，而非直接启动新的 B-H 网络。

本页只从已冻结源码/现存数据推导，未新增论文读深、G14 项或运行授权；未执行新模型/GPU任务。
