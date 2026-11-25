# 从未来读出推导充分缓存：固定作用折叠与动态状态

2026-11-25。依据 FFNDS `873e1a879b91d544fdae11894d68bbc6acc854d1` 的实际 wide8192 与 low OUT-D 执行代码。使用 operator-tensor-analysis 的算子族、近似误差与执行成本，以及 active-abstraction 的任务充分表示和更新闭合判据。本轮完成符号核验、实际 CAL 矩阵计算和静态预算；小尺寸数值实例明确使用合成操作数。没有运行 Wan 模型或测量候选性能。

## 1. 新的迁移条件：缓存由未来会读取的作用决定

六源 C/A 成片“基本没有差别”的用户反馈，使已执行的完整组合具备新的质量证据。下一层数学问题可以从“保存了哪些原变量”改成“未来真正读取哪些关于这些变量的函数”。

若未来读出属于已知线性族，历史变量可以通过该族的共同投影保存；若未来系数尚待当前观测确定，需要保存能够回答整个系数族的状态。低维系数族与完整输出场是不同对象。下面以实际 FFN 和 attention 得到两种不同结论。

## 2. wide8192 的实际历史依赖

固定层、专家和 CAL。用行向量约定，当前 FFN 输入为 U∈R^(N×d)，历史为 U_a、Y_a、Z_a。真实形状 N=31668、d=5120、selected k=2560、共享模式 r=16。

实际缓存为：

| 字段 | 形状/格式 | hit 中的使用 |
|---|---|---|
| u0=U_a | N×5120，BF16 | 当前差分乘 A0；当前差分乘 J |
| y0=Y_a | N×5120，BF16 | 加到底座 |
| z0k=Z_a | N×2560，BF16 | GELU 有限变化与非线性核特征 |

当前 A0∈R^(5120×5120) 为 BF16，J=jmap∈R^(5120×16) 为 FP32。设选中激活变化为 D_H，Z=zmap∈R^(2560×16)，选中 W2 为 W_K，共享输出方向为 V∈R^(16×5120)。真实 hit 的实数对应式是：

\[
B_s=Y_a+(U_s-U_a)A_0+D_HW_K,
\qquad
Q_s=D_HZ-(U_s-U_a)J.
\]

P64 拟合当前增益 γ_s 和核系数 C_s 后，输出写为

\[
\widehat Y_s
=B_s+Q_s\operatorname{diag}(\gamma_s)V
+\tfrac12 K_{sP}C_s.
\]

K_sP 由当前 selected 预激活、Z_a 以及 P 核特征构造。这里已经保留了 FFN 的真实结构：P 教师、16 个输出方向和非线性核项，而不是把它们统称为一个任意低维读出。

源码：[当前 fourstep_hit.py](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/tools/ffnds_fourstep/fourstep_hit.py)、[缓存与默认方法](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/tools/ffnds_fourstep/README.md)、[原生 anchor 写入](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/tools/ffnds_fourstep/fourstep_anchor.py)。

## 3. 一个可构造的充分状态：预先求出固定读出

在 anchor 已知 U_a、Y_a、A0、J 时，定义

\[
\boxed{C_a=Y_a-U_aA_0,\qquad J_a=U_aJ.}
\]

改存 (C_a,J_a,Z_a)，未来计算：

\[
B_s=C_a+U_sA_0+D_HW_K,
\qquad
Q_s=D_HZ-U_sJ+J_a.
\]

代入即可恢复上节的 B_s、Q_s，因此完整恢复器在实数域相同。未知的未来增益 γ_s 只作用在 J_a 的 16 维坐标上；无需保存整个 U_a 来等待 γ_s。这一重写没有缩小所允许的 γ_s，也没有降低 nonlinear kernel 的 2560 维输入。

这给出具体跨模块判据：**未来算子已经确定的历史作用，可以在 anchor 求值并保存；未来算子仍会变化的部分，保存其共享作用空间。**

这份充分状态不是对原数据的通用无损压缩，也不主张全局最小。它是为当前执行图的未来查询族构造的状态表示。固定层、专家、CAL 和支持属于等式条件；参数或支持变化后，缓存需要按新查询族建立。

### 3.1 沿隐藏通道构建同一作用的另一条路径

本地封存的真实 compiler `evidence/compile_sites.py` 给出：先在 CAL 上拟合每个 hidden 通道的 β，将 selected 位置置零得 β_om，再构造

\[
A_* = W_1^\top\operatorname{diag}(\beta_{\rm om})W_2^\top,
\qquad A_0=Q_{\rm BF16}(A_*),
\]
\[
J_* = W_{1,S}^\top\operatorname{diag}(\beta_S)Z.
\]

因此在连续实数对应模型中，令完整预激活 Z_full=U_aW1ᵀ+1b1ᵀ，则有含偏置的构建：

\[
C_a^*=
[\phi(Z_{\rm full})-(Z_{\rm full}-\mathbf1b_1^\top)\odot\beta_{\rm om}]
W_2^\top+\mathbf1b_2^\top.
\]

它等于 F(U_a)−U_aA_*。针对当前冻结 BF16 A0，还存在−U_a(A0−A_*)这一确定的差项，此外有原生 Z、激活、W2 的数值舍入。因而可以把隐藏构建视为另一种数值候选，但不能无条件声称与当前 A0 精确一致。

若仅利用 anchor 已得到的隐藏预激活，另算 omitted 线性投影再从 native Y_a 相减，需要额外2N(m−k)d FLOP；本模型为直接 U_aA0 的 **2.2倍**。另算完整残差 W2 则为 **2.7倍**。虽然同一权重/隐藏状态可以复用加载，native Y_a 仍必须输出，额外累加器与计算均存在。这里没有一个“已有Z，所以额外W2免费”的重写。

同理，可以从选中预激活计算 J_a^*=(Z_a−b1,S)diag(β_S)Z；用已舍入Z_a与冻结FP32 J时存在相应因子化和预激活误差。本文主预算保持直接 U_aJ 路径，以清楚对应实际固定J。

## 4. BF16 差分与乘法使数值路径需要单独定义

实际 wide8192 在计算 linear 时执行：

\[
\ell_{\Delta}=
\operatorname{GEMM}_{\rm BF16\ out}
\bigl(Q_{\rm BF16}(U_s-U_a),A_0\bigr).
\]

其中同时存在输入差分 BF16 舍入和 GEMM 输出 BF16 舍入。重排后不能把原式直接宣称逐位等价。

一个具体数值构造为：anchor 使用 BF16 操作数、FP32 输出计算 L_a≈U_aA0，再存 \(\widetilde C_a=Q_{\rm BF16}(Y_a-L_a)\)；J_a 使用 FP32；Z_a 保持原 BF16。hit 计算 \(\ell_s=\operatorname{GEMM}_{\rm BF16\ out}(U_s,A_0)\)。

相对于当前实现，同一当前输入上的底座误差精确记为

\[
\delta B_s=
\widetilde C_a+\ell_s-Y_a-\ell_{\Delta}.
\]

若将 anchor 的乘法误差记为 e_a=L_a−U_aA0，缓存舍入记为 e_C=\widetilde C_a−(Y_a−L_a)，当前乘法误差记为 e_s=\ell_s−U_sA0，差分舍入记为 e_Δ=Q(U_s−U_a)−(U_s−U_a)，旧 GEMM 输出误差记为 e_m=\ell_Δ−Q(U_s−U_a)A0，则有

\[
\delta B_s=-e_a+e_C+e_s-e_\Delta A_0-e_m.
\]

这明确了必须观察的数值对象。C_a 是 Y_a 与固定线性响应的差，两者通常相关；缓存误差不能通过“分别估计 Y_a 和 U_a 的独立噪声”来替代。是否具有更小动态范围，需要真实 C_a 数据回答。

J_a 重排也产生 FP32 舍入差：

\[
\delta Q_s=
\bigl(D_HZ-U_sJ+\widetilde J_a\bigr)
-\bigl(D_HZ-(U_s-U_a)J\bigr),
\]

实际各项按代码的 FP32 运算次序求值。其精确实数对应为零。

### 4.1 P64 原值保护把影响限制在输出读出

除新状态外，单独保留 P64 的 U_a、Y_a BF16 原值；Z_a 仍完整保留。在 P 上继续执行旧 operands 路径，所以固定当前 U_s 时：

- 原生教师 observed 相同；
- base_P、Q_P、X_P 相同；
- 当前 γ_s、C_s 的拟合完全相同。

非 P 行只改变 B_s、Q_s，X_s 不变。因此，在最终 dtype cast 之前，候选与当前恢复器的差为

\[
\boxed{
\delta\widehat Y_s
=\delta B_s+\delta Q_s\operatorname{diag}(\gamma_s)V.
}
\]

P 行仍直接覆盖原生教师。这个关系给出了比“缓存相对误差多大”更合适的读数：实际底座重排误差，以及通过当次输出增益传播的 Q 误差。它也提供明确实现顺序：先以完全相同的当前输入核查这条等式，再评价自身轨迹。

P 保护在给定当前输入时固定拟合。候选经过前面层或前面步后，当前输入仍可能改变；整条生成行为由自身轨迹判断。

## 5. 真实形状下的状态和计算代价

40 层活动槽在 high/low 专家之间按生命周期复用。新状态采用 C_a/Z_a BF16、J_a FP32、P64 原 U_a/Y_a BF16：

| 缓存净载荷 | GiB |
|---|---:|
| 当前 U_a、Y_a、Z_a | 30.200958 |
| 新 C_a | 12.080383 |
| 保留 Z_a | 6.040192 |
| 新 J_a，FP32 | 0.075502 |
| 精确 P64 U_a 与 Y_a | 0.048828 |
| 新状态合计 | **18.244905** |
| 节省 | **11.956053（39.59%）** |

这些是逻辑字段净载荷。临时 GEMM 输出、校准参数、allocator 行为和真正的峰值需分别计入。选择 C_a 全 FP32 会显著改变这个预算，不能同时声称 FP32 残差和表中 BF16 容量。

构建 C_a 需要 anchor 新增 U_aA0：每个 site 1.660315 TFLOP，每片 80 个 anchor 合计 **132.825219 TFLOP**。J_a 再增加 **0.415079 TFLOP/片**。

| 主 GEMM 静态算术 | 结果 |
|---|---:|
| 80 个原生 anchor 的 W1/W2 | 717.256183 TFLOP |
| 当前 anchor+hit 主 GEMM | 982.906621 TFLOP |
| 新增 A0 投影 / 原生 anchor W1/W2 | 18.52% |
| 新增 A0 和 J / 当前 anchor+hit 主 GEMM | **13.56%** |
| 当前主 GEMM / 全原生 FFN 主 GEMM | 0.685185 |
| 新构造对应比值 | 0.778067 |

主 GEMM 表计入原生 W1/W2、hit A0 和选中 W1/W2；小型拟合、核相关、编码、布局等不在这个分母里。因此表中 13.56% 是清楚定义的算术变化，不是预测实测延迟。

四步场景每个锚点只复用一次。这一构造应作为**省状态、增加 anchor 计算**的具体取舍评估。它的价值在于允许在显存条件下维持原恢复结构；其时间收益需要由实际系统边界决定。

## 6. low OUT-D 为什么需要另一种状态处理

真实 low attention 在每个 head/query 保存 u_a∈R^128、ρ_a∈R、φ_a∈R^31，其中

\[
u_a=A_a-(1-\rho_a)B_a,
\qquad
\phi_a=Q_{\rm BF16}(B_aP_vT).
\]

hit 拟合当前 (b_s,W_s) 后，非 P 输出为

\[
\widehat A_s
=(1-\rho_a)B_s+u_a+b_s+(\phi_s-\phi_a)W_s.
\]

对历史的读出为 u_a−ρ_a B_s−φ_aW_s。W_s 在当前 P64 拟合后才知道；因此 anchor 时无法把 u_a−φ_aW_s 求值为唯一固定向量。

在允许任意未来 B_s、W_s 的存储状态域上，若两个历史状态对所有这些读出都相同，取 B_s=0、W_s=0 得到 δu=0；改变 B_s 得到 δρ=0；逐个改变 W_s 的31个行方向得到 δφ=0。联合观测映射的核为零。一个显式160维观测矩阵已由 NumPy 验证满秩160。

这个证明限定于该未来读出族，不把它当作真实生成数据的内在维数测量。真实 low CAL 的 P_vT∈R^(128×31) 数值秩确为31，奇异值范围3.591672–3.828439，条件数1.065921；现有固定投影本身没有可直接消去的零方向。

因此 FFN 的状态折叠可以借助已知 A0/J，而 low attention 需要保留能回答当前未知 W_s 的历史信息。两者共享充分状态原则，具体可合并的字段不同。

### 6.1 对 additive cache 编码，当前教师能起怎样的作用

固定当前 Q/K/V、ρ_a 和特征 X_s=[1,φ_s−φ_a]。令

\[
K_s=X_s(X_P^\top X_P+\lambda I)^{-1}X_P^\top,
\]

这里 λ 为实际 solve 中的正则值，当前为64×0.1。若仅将 u_a 编码为 u_a+ε，当前 P 拟合后的预 cast 输出变化精确为

\[
\delta\widehat A_s=(I-K_sP)\varepsilon.
\]

教师会重新拟合缓存误差在 P 上暴露的部分。若单独保存原生精度的 P64 u_a，使 ε_P=0，当前系数保持原值；非 P 输出只增加各自 ε_i。

若也编码 φ_a，但保留其 P 原值，则当前系数仍相同，非 P 的新增误差为 ε_i^u−ε_i^φW_s。ρ 保持FP32时无需加入质量编码项；若压缩ρ，则另有−ε_i^ρB_i。这个式子给编码设计提供实际输出敏感度，而不要求重新搜索 P 或 λ。

作为静态容量例子：u_a 每行/head 使用128个INT8值、一个BF16 scale和另存P64 BF16原值，其他字段保持现状，low字段由15.194857降至9.273458GiB。若φ_a也用INT8、每行/head一个BF16 scale且保留P原值，则为7.910889GiB。它们尚未做当前 C 的成片或延迟验证。现有 wide8192 INT8 已经采用精确P缓存保护，但那一组合的实测不能直接替代 attention 新组合的验证。

源码：[low_cache 与 approximate](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)、[已有 FFN INT8 编码与P保护](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/reports/20261123/wide8192-cache/README.md)。

## 7. 与历史推导的关系

仓库已有直接前身，残差折叠本身不能作为新的首创主张：

- [2026-10-31 SICR §4–5](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261031/G15_SICR_CROSS_LAYER_STEP_CAUSAL_STATIC_QUANT_CACHE_REVIEW_20261031.md)已将固定 selected 作用折入单残差，并说明未来动态系数未知时需要保存额外历史响应；后续修订把“最小缓存”改为相应条件下的充分缓存。
- [2026-11-04 缓存推导 §1](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/CACHE_COMPRESSION_DERIVATION.md)已明确 C_a=F(U_a)−U_aA，未来使用U_sA+C_a。该历史对象是原40步线性预测器，数值与复用次数不同。
- operator-tensor-analysis 的 `derivatives-and-secants.md` 已给出静态作用折入C_s、低维动态读出保存P_s=X_sU，以及含bias的hidden构建恒等式。本轮使用这些已有数学关系，落实到当前wide8192实际字段、核特征和BF16顺序。

本地已有端点研究还包含更接近的实际实例。`cal_mean_anchor_producer.py` 从 `compact.capture(...,'intercept_features')` 构造逐位置 C=Y0−B(U0)、BF16标准化非线性端点f0和FP32的q0。该实验同时比较原实现、保留真实端点的actual_endpoint、用CAL均值替代端点的cal_mean；后者与本文保留完整逐位置状态的构造不同。

本轮重新读取封存的 `cal_mean_anchor.json` 与 `extra_cal_mean.json`，在layer19、11来源、high/low与两互补折上重算actual_endpoint/original。先按来源内两折取log均值，再来源等权：

| 历史端点重写 / 原实现 | high | low |
|---|---:|---:|
| 点与边联合J | 1.000002095 | 1.000015944 |
| point | 1.000000572 | 1.000012512 |
| edge | 1.000002863 | 1.000017666 |

这为端点式充分状态提供了真实局部数值证据。它不是本文保留Z_a与原P路径的具体候选实测，也没有对应全层成片或当前四步额外建锚计时。两文件与原始输入绑定SHA已记入机器可读结果，源码和历史记录的版本范围分别保留。

本轮新内容是把这条原则应用到**实际已验证的 wide8192 条件增益和核恢复**：保留16维固定读出的J_a和2560维非线性端点，以P原值保护当前拟合，并给出真实四步预算和数值重排误差；再用 low OUT-D 的未知未来W_s建立对照。它使跨模块迁移条件从类比变成可核查的未来查询族。

## 8. 已完成的核验及仍需真实数据回答的问题

[verify_sufficient_state.py](verify_sufficient_state.py) 已执行，结果见 [sufficient_state_results.json](sufficient_state_results.json)：

- SymPy 核验完整 FFN 读出在实数域的状态折叠；
- 符号核验固定设计下的 cache 编码误差残差化；
- NumPy 构造 low attention 的未来读出族并核验160维满秩；
- 实际冻结 low CAL 与 P_v 的31维秩及奇异谱；
- 历史11来源layer19 actual_endpoint/original 的真实面板评分重算；
- 合成实例验证保护P后gain/coeff保持一致，预cast输出误差等式最大差8.64e−16；
- 真实模型尺寸下的字段字节数和完整四步新增主GEMM算术。

历史面板的原始评分字段逐值提取到 [historical_endpoint_scores.json](historical_endpoint_scores.json)，包含44条记录和两个来源文件SHA，可用 `--endpoint-scores historical_endpoint_scores.json` 独立复算；真实CAL和P_v分别通过 `--calibration`、`--projection` 指定。已读本地源码的字节指纹见 [source_fingerprints.json](source_fingerprints.json)。

候选是否值得进入执行，集中于两个具体问题：C_a 的BF16舍入与差分重排在真实输入上经过当前增益后有多大；以及额外anchor投影是否符合所需显存—时间取舍。无须把这一步展开成新的rank、P或λ扫描。

### 发布后复算

解包同项目 results/replay_evidence.zip，获得 calibration/self_low_transform.npz 与 plan_assets/bases/P_v.npy 后，可运行：

```bash
python verify_sufficient_state.py --calibration /path/to/calibration/self_low_transform.npz --projection /path/to/plan_assets/bases/P_v.npy --endpoint-scores historical_endpoint_scores.json
```

[历史逐条评分摘录](historical_endpoint_scores.json)保留 actual_endpoint 与 original 的44条评分及两个原文件SHA；其中源回执的CAL-mean定义单独标识，不作为本次actual_endpoint评分的分子定义。[本地源文件身份](state_source_fingerprints.json)记录实际核对的compiler、端点producer与在线代码。仅运行符号和静态部分可省略以上三个输入参数。
