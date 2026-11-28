# 将冻结 C 的遗漏响应随当前均值与尺度运输

本构造依据冻结 C 的真实 low_cache / approximate 数据流，Git blob 为 7e7c611ec2aa7e8c8754bd19963cbce6ed49ded4。它由 job 201276 的机制证据引出；本轮完成实数推导、SymPy 和 NumPy 合成核验，尚未在 Wan 上执行。job 201276 使用 selected + 两组摘要底座，其效果数值不能直接移给本构造。

## 1. 为什么需要重新定义缓存

固定一层、一个 head，query 轴记 i，key 轴记 j，value 通道记 c。完整 attention 权重为 \(S_{ij}\)，选中 key 上单独归一化的权重为 \(T_{ij}\)，两者的行和都是 1；\(A=SV\)，\(B=TV\)。理想锚点的遗漏概率质量为 \(\rho_{a,i}\)，已选质量为 \(w_{a,i}=1-\rho_{a,i}\)。

冻结 C 的 low OUT-D 保存

\[
u_{a,ic}=A_{a,ic}-w_{a,i}B_{a,ic},
\qquad
\operatorname{base}^{old}_{s,ic}=w_{a,i}B_{s,ic}+u_{a,ic}.
\]

实际源码把 \(u_a\) 写为 BF16，\(\rho_a\) 为 FP32。固定的 CAL scale_r 只是拟合标签单位；它不是当前 value 尺度。

定义 \(M_t=S_t-\operatorname{diag}(w_a)T_t\)，则 \(M_t\mathbf1=\rho_a\)。因此 \(u_a\) 携带非零质量。value 整体平移 \(\mathbf1 b^\top\) 会使它增加 \(\rho_a b^\top\)。只将 \(u_a\) 乘上标准差比率，无法同时正确搬运均值变化。

## 2. 一个可直接写出输入、状态和读出的构造

在 key 轴计算每个 head/channel 的均值与标准差：

\[
\mu_{t,c}=N^{-1}\sum_j V_{t,jc},\qquad
D_t=\operatorname{diag}(\sigma_{t,c}).
\]

锚点把现有 attention 历史槽替换为

\[
\boxed{
C_a=(u_a-\rho_a\mu_a^\top)D_a^+.
}
\]

命中底座为

\[
\boxed{
\operatorname{base}^{new}_s
=\operatorname{diag}(1-\rho_a)B_s
 +\rho_a\mu_s^\top+C_aD_s.
}
\]

统计量属于 head/channel；\(\rho_a\) 还保留 query 身份。这里没有把逐位置信息压成一个全局均值：完整 \(C_a\) 仍逐 query 保存。

对严格常量 value 通道，中心化缺陷为零；令对应标准化缓存为零。当前通道若恒定，均值项与 selected 项共同重建该常量。anchor 常量而 current 非常量时，历史不含该通道新形状，后续创新恢复仍承担变化。

## 3. 精确性质及其条件

**实数锚点恒等。** 当命中对象等于锚点对象时，

\[
\operatorname{diag}(1-\rho_a)B_a+
\rho_a\mu_a^\top+C_aD_a=A_a.
\]

**正对角仿射 value 演化的精确类。** 若完整权重 \(S_s=S_a\)、选中键归一化权重 \(T_s=T_a\)，且

\[
V_s=V_aG+\mathbf1 b^\top,\qquad G\succ0\text{ 为对角矩阵},
\]

则 \(\mu_s^\top=\mu_a^\top G+b^\top\)、\(D_s=D_aG\)，代入得到

\[
\operatorname{base}^{new}_s=A_aG+\mathbf1 b^\top=A_s.
\]

这里要求权重本身固定；只固定选中 block 的索引并不够。真实 C 的 Q/K 会随步改变，完整/选中权重和遗漏质量的变化继续进入创新，不能由此宣称一般命中精确。

**底座的独立坐标等变性。** anchor 与 current 的 value 各自作独立正对角仿射换坐标时，\(C_a\) 的理想标准化形状保持不变，新底座按 current 坐标变换。这是底座性质。若保留现有 OUT-D 未标准化特征与固定 CAL 正则化，整个恢复算法的独立两时等变性还需要额外证明。

**与原底座的已知差。** 在相同当前特征、P64、CAL 和 ridge 下，设 \(K\) 为探针标签到全场的线性提升，\(M_{\bar P}\) 为探针覆写掩码，则

\[
\delta B=\rho_a(\mu_s-\mu_a)^\top+C_a(D_s-D_a),
\]
\[
\boxed{
\widehat A^{new}-\widehat A^{old}
=M_{\bar P}(I-KP)\delta B.
}
\]

该式在实数、精确编码条件下成立，并单独隔离历史运输。原 OUT-D 特征只依赖当前/anchor selected 投影差，与 \(u_a\) 无关，因此维持同一特征是可执行的比较。P64 的当前原生标签继续复用，最终 P 覆写也保持。

## 4. 真实实现中的编码选择

本候选选择**替换现有历史槽**：anchor 直接从当时原生 \(A_a\)、selected \(B_a\)、\(\rho_a\) 和当前统计量形成中心化标准化缓存，再写入一份 BF16 \(C_a\)。当前均值与尺度在使用后即可释放，不新增第二份全位置历史。

旧 BF16 \(u_0\) 已有舍入误差，不能将“读取旧槽再除尺度”与“从 anchor 操作数新编码”视为逐位等价。特别是在常量或接近常量通道，除以尺度会改变数值残差的处理。新编码需核查 anchor 重建误差、解码误差、尺度阈值及活动通道变化和原生 P 的闭合。参考 moments() 对尺度 ≤1e−12 使用零逆值，其数值等变性见证限定在阈值不改变活动支撑的范围；实数 D⁺ 命题与此数值截断分开。源码中的 \(\rho\) 来自 LSE；小数值误差可导致极小负值，概率解释针对理想值，生产数值应忠实记录。

还有一种保留旧 \(u_0,\mu_a,D_a\) 并在 hit 解码的实现，能保留旧槽编码边界，但会增加约 1.5625 MiB 的 40 层统计量和相应运算。它与本次选定的替换槽构造属于不同有限精度实现，不能混写成同一已验证路径。

## 5. 为什么它值得作为下一具体候选

冻结 C 已经计算 low anchor/hit 的 selected attention，也已经保存 \(\rho_a\)、完整 \(u_a\) 和 anchor 特征。新构造把结构运输接到这些现有资产上：

- 使用原 selected 输出和原 P64 教师；当前特征、CAL、\(\lambda\) 可保持原值。
- 用同形状 \(C_a\) 替换原 \(u_a\) 槽；新增信息是当前 value 均值和尺度。
- 不要求 selected + 两组摘要的全 query 合并，也不额外建立 high attention 历史场。
- high OUT-ABS 没有对应的 \(u_a\) 槽。把本构造用于 high 需要另一套状态与成本设计，当前候选限定为已有 low OUT-D 路径。

这些是数据流上的成本机会；新候选的实际速度、质量和与完整运输的效果差别仍待实测。

真实尺寸 \(N=31668,H=40,d=128\)：每层历史场有 162,140,160 个元素，替换后持久全场净增为零；当前 \((\mu,\sigma)\) 的 FP32 工作量是 40,960 字节。40 个 low hit 的逐通道解码包含 6,485,606,400 次乘法。若统计量额外以一次独立 BF16 全 V 读取实现，low anchor + hit 共 80 站点至少增加 24.1608 GiB 读取；真实归约可能多遍或融合，不能把统计量小等同于计算免费。

每个 anchor 只服务一次 hit，验证周期应包含编码、统计、读取、解码和后继模块。成本参照继续使用已配对测量的冻结 C。

## 6. 本次实际数学核验

脚本 [verify_mass_centered_transport.py](verify_mass_centered_transport.py) 绑定真实冻结 C 源码 Git blob，并用明确合成的 \(2\) heads、\(96\) queries、\(768\) keys、\(128\) channels 进行数值核验，P64、32 特征、ridge 加项 6.4。

| 核验 | 最大绝对差 |
|---|---:|
| 实数 anchor 重建 | \(2.22\times10^{-15}\) |
| 固定权重下正对角仿射 value 演化 | \(1.42\times10^{-14}\) |
| 已知底座更新差 | \(2.22\times10^{-15}\) |
| anchor 标准化缓存的换坐标不变性 | \(5.72\times10^{-14}\) |
| 两时独立坐标下底座等变性 | \(1.77\times10^{-13}\) |
| 固定 P64 恢复后的更新差 | \(4.44\times10^{-15}\) |
| 当前常量列重建、P64 差 | 0 |

SymPy 同时验证 anchor 和仿射重建恒等式。合成例中 raw 底座误差 L2 为 158.79，省略均值质量修正的“只缩放 \(u\)”误差为 143.38，说明均值修正是这个精确类的实质环节；这些数字不是 Wan 实测。

~~~sh
OPENBLAS_NUM_THREADS=1 python verify_mass_centered_transport.py \
  --code ../../attention_online_structured.py
~~~

结果保存在 [MASS_CENTERED_TRANSPORT_VERIFIED.json](MASS_CENTERED_TRANSPORT_VERIFIED.json)。
