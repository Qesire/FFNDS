# 跨步 token 耦合与共同 mask 的区分：ALG 草案

原始两 token 交换构造由 `/root` 在任务中提出；`/root/sm120_full_quality_build_terra` 独立核查推导并新增共同 mask 反例；`/root/teal_full_llm_calibration_build_terra` 独立评议及复核。状态：**OPEN；修订版数学内容已独立复核**。参见[独立初评](chipmunk-cross-step-coupling-alg-20261030-independent-review.md)。以下是形式反例和对已冻结 scope 的预注册式解释，不是新数值实验、测得结果或实验授权。

令实际 post-bias preactivation 为 a，G(x)=x[1+tanh(sqrt(2/pi)(x+0.044715x³))]/2。实数运算中 tanh 的参数是奇函数，故 G(x)−G(−x)=x。两个**身份固定、跨步匹配**的 token，旧 a=(t,−t)、新 a=(−t,t)，t≠0，则 Δh=(−t,t)，该通道跨两行 RMS=|t|。旧、新每步的均值、方差和完整边际直方图都相同；对照的新状态取旧状态本身，也有同样两步边际，但其 Δh=0。即使每步 GELU 后的边际分布也相同，仍不能从边际恢复逐行变化。一般 M×H 矩阵中若仅该通道的两行变化，该通道跨全部 M 行的 RMS 是 |t|sqrt(2/M)，跨全部 M×H 元素的 RMS 是 |t|sqrt(2/(MH))；仅当 M=2 时分别为 |t| 和 |t|/sqrt(H)。原作者 mean-preactivation score 在此为零，独立跨步边际摘要也无法分辨；相反，成对的旧 h 和新 a 包含所需关联。

这是对**含 b1 的总 a**施加的对称条件，不能先忽略非零 bias 再使用公式。同一 expert 的 b1 相同才在两步比较中自然固定；其权重列非零及输入可达等条件使示例可由真实 W1 输入构造。必须保留同 source/seed/expert 和时空 raster 行身份，不能排序后重新配对。对单一变化隐层通道 j、当前 gate g，固定 W2 和 b2 时，缓存旧 h 的逐行门后输出误差为 ±t·(W2[:,j] ⊙ g)（符号取决于误差方向）；该向量非零时有非零作用。多通道可互相抵消；零 gate 或所涉 W2 列 W2[:,j] 为零也可使其消失（W2 shape 为 [5120,13824]）。b2 在同权重同 bias 的线性差中抵消；跨步门控或其他残差若也改变，额外项需单列，不能把这个局部差直接等同自由视频因果效应。实数恒等式不是 FP32/BF16 bitwise 恒等式：GELU 实现、W1 GEMM、输入和输出转换均有舍入，符号相反数差也不保证按该式逐字节相等。

## 两类竞争预测与现有范围的判别力

1. **有限采样覆盖**：若重要变化集中在少数空间位置，固定的4行可能漏掉、64行更易覆盖；在均匀无放回抽样且重要行数为 s、总行数 M 时，全部漏掉的概率是 C(M−s,k)/C(M,k)。实际范围只有固定 seed 42 的一个嵌套前缀，不能把这个概率当作已观测抽样频率。预期 P64 相对 P4 的每通道 full-RMS 分数相对误差下降、cosine 与所选 Δh 能量提升，且 gated 误差向同 K 的 full-RMS 靠近。这个等幅两行交换反例**不本身支持采样不足**：只要抽到任一匹配行，单行绝对变化已等于 |t|。需要稀疏或异质变化反例才能预测 P64 优于 P4。
2. **共同 mask 行动颗粒**：即使全 M RMS 排序准确，单一 K 通道集也要覆盖全部行。最简单反例 ΔH=[[u,0],[0,u]]，u≠0，W2 为单位映射且 gate 非零；每行单独选一个不同通道可无遗漏，而共同 K=1 必漏掉一个行通道，即便每行的名义选择数相同。这个逐行动作需要新的选择、索引、W1/W2 实现和付费，不由现有 scope 实现或免费证明。若 P4/P64 排序和所捕获能量已接近 full-RMS，三者的门后误差仍显著，则支持代理采样之外存在损失，但共同 mask、固定 K、W2 方向和通道交叉项仍相互混杂。

四个冻结诊断分别是两暴露开发 source × 两个同 expert 的 step pair；它们不是四个独立来源。代码 `tools/chipmunk_fullm_proxy_math.py` 固定 `M=31668`，4/64 行来自同一固定随机排列前缀，均使用相匹配的旧 h 和当前重算 preactivation；mean、P4、P64、固定 random、full-RMS 分支各用 whole-M 的 K=5120。全 current h 出现在在线分支之后；full-RMS 是付费事后参考，按 Δh 能量排序，不是输出最优或可免费执行的 causal oracle。`tools/e0_chipmunk_fullm_causal_proxy.py` 实际保存 gated relative L2、mask 索引 SHA、P4/P64 score 对 full-RMS 的 relative L2/cosine、各 mask 所捕获的 Δh 能量和 Dense 对原捕获 h/Y 的一致性；它**不保存 mask overlap 或逐行最优输出**，不能声称这些已由现有结果测得。

先检查原捕获 h/Y 的 Dense replay 和有限性；数值路径不匹配时分别报告该误差。若 P64 score 逼近 full-RMS、同时输出也明显靠近，支持采样覆盖是本组相关因素；若 score 逼近但两者都保留大 gated 误差，则仅说明排序采样之外仍有代价。若 P64 score 改进却 gated 误差不随之下降，消费者方向、跨通道抵消和固定 K 都是具体竞争解释。精确区分共同 mask 与这些替代机制，须另审有界的同预算局部/逐行选择或 gate·W2 方向/交叉项诊断，并支付其信息、选择和物理执行成本；不能以本四 scope 单独断言。
