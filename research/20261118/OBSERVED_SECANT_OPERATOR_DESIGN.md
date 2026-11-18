# 当前响应驱动的 FFN 割线算子：完整设计

日期：2026-11-18。证据基点：FFNDS commit f89e4fae868ac8d21b3b8f0da47616f43b0c954a。

本设计给出两种明确的条件算子、离线参数生产规则和原生数值合同。NumPy FP64 参考核心与合成代数测试已实现；CAL 拟合、Wan BF16 绑定、模型回放、视频和性能实验尚未运行。[实验方案](OBSERVED_SECANT_EXPERIMENT_PLAN.md)规定本轮的有限执行范围；[参考核心](../../tools/ffnds_observed_operator/README.md)不是已验证的 Wan runtime。

**同日MSE筛选修正。** 两种构造都保留完整80位置的static/online视频比较，不再按局部MSE决定视频资格；总预算改为12条主视频、最多3条范围诊断，仍不超过15条。下面的通道选择、模式POD、prior和lambda保持原定义，是局部欧氏响应目标下的受检构造，不能称为已经对齐视频功能。推导、历史反例及输出加权可能从全输出回归中消去的条件见[MSE筛选复核](MSE_SELECTION_REASSESSMENT.md)。

**执行后更新。** `3e4f7ba` 已完成两类全 80 static/online 视频及六位置诊断；上段“尚未运行”保留初稿证据基点的历史状态。0392 全 80 在线读出已接近 Global-P64，用户补充 P64 自身也有全画幅损坏。新的[结果与算子分析](OBSERVED_SECANT_RESULT_ANALYSIS.md)说明核传播、配置交互和同状态 MSE 的证据边界，并记录模式基底生产实际使用 ΔH、部署使用 ΔZ 的差异。原有参数与已运行版本身份保留；清洁部署性能仍未测量。

## 1. 研究对象与推导依据

当前目标是利用视频 FFN 的状态、响应和执行结构设计自己的紧凑算子。Global-P64 作为机制参照，提示我们利用当前真实响应、保留有符号作用并区分全局选择与逐 token 非线性。

固定一个 expert、一个层及其原生权重。采用行向量约定：

\[
F(U)=\phi(UW_1^\top+b_{\mathrm{in}})W_2^\top+b_{\mathrm{out}},
\quad
U\in\mathbb R^{N\times d},
\quad W_1\in\mathbb R^{m\times d},
\quad W_2\in\mathbb R^{d\times m}.
\]

当前配置为 \(N=31668=21\times29\times52,d=5120,m=13824\)。这里21是latent时间token轴\(\tau\)的长度，29/52为latent空间行/列；解码视频帧轴\(t\)长度为81，去噪步轴\(k\)长度为4，三者不能混同。输入为 norm2/AdaLN 后、W1 前的 U；返回 pre-gate 输出；当前 gate 在宿主残差路径施加一次。此轮是原生 BF16 路径，不插入历史 HiF4 QDQ。

anchor/hit 为 HHLL 中的 0→1、2→3。每视频每位置只复用一次；每个候选在自身学生轨迹上生成真实 anchor。expert、权重、token 顺序或视频变化时清空对应状态。

共同基线矩阵 \(A\)：high/low 的 04、19、34 六位置使用已绑定的 L64；其余 74 位置使用严格 Q1。不存在未经拟合的“全 80 L64”。固定此规则，所有新构造与自己的静态控制共用相同 A。结构轨的A64是这份冻结部署BF16矩阵的FP64提升，不换成旧拟合器舍入前的另一张A。

已有结果的定位：

- 严格同目标 P64/Q1 的局部误差与响应方向，见 [RESULTS](../../reports/20261117/ffnds_strict_reproduction/output/RESULTS.json)；
- 固定输出 Q32 的原生特征实现，见 [features](../../tools/ffnds_state_operator/features.py)；
- 六位置 N32 的局部改善、0392 的视频失败和作用范围，见 [结果](../../reports/20261118/state_conditioned_ffn_operator/README.md)；
- 原六位置计划和结案保持原样；本设计不把未运行的终端反馈判为失败。

## 2. 构造一：观测条件读出

### 2.1 端点特征与缓存

固定选 16 个原生隐藏通道，索引集合 K。令 \(z_K(u)=uW_{1,K}^{\top}+b_K\)，定义总共 32 列：

\[
\Phi(u)=
\left[z_K(u),\ \phi(z_K(u))-\beta_K\odot z_K(u)\right]
\operatorname{diag}(\sigma^{-1}).
\]

beta 是 CAL 上逐通道的差分最小二乘斜率；sigma 是两块差分特征的 CAL 来源等权 RMS。beta 不是 Q1 的 alpha。两块合起来与原生 \([\Delta z,\Delta h]\) 同列空间，能够表达线性抑制和非线性恢复；不加截距、不广播无条件常数。

anchor 保存：

\[
C_0=F(U_0)-U_0A\in\mathbb R^{N\times d},
\qquad
Z_0=\Phi(U_0)\in\mathbb R^{N\times32}.
\]

生产缓存 C0 为 BF16，Z0 为 FP32。保存 FP32 特征是为了避免再量化很小的端点差，额外状态相对 BF16 Nd 为 \(2r/d=1.25\%\)。

hit 先计算：

\[
Y_{\mathrm{base}}=U_1A+C_0,\qquad
X=\Phi(U_1)-Z_0.
\]

从当前 token 中选 p=64 行 P，执行完整当前原生 FFN：

\[
R_P=F(U_{1,P})-Y_{\mathrm{base},P}.
\]

R_P 是实际基线遗漏的有符号全输出残差。只给 observer 传入这 64 行；不计算全 token 的真实 FFN。包括 probe 在内的全部 token 都使用同一个预测，不把 probe 真值拼回最终输出。

### 2.2 静态先验与在线求解

主配置使用同特征、CAL 拟合的完整输出读出 \(V_{\mathrm{stat}}\in\mathbb R^{32\times d}\) 作为先验。这样能直接比较“固定全输出表示”与“当前观测增加的信息”，不将解除旧 Q32 的收益归因于在线更新。

设 \(\Gamma_{\mathrm{CAL}}\) 为固定 CAL 特征度量，\(\Gamma_X=X^\top X/N\)。在线目标：

\[
\min_V
\frac1p\|X_PV-R_P\|_F^2+
\lambda\,\mathrm{tr}\left[(V-V_{\mathrm{stat}})^\top
(\Gamma_X+\tau^2\Gamma_{\mathrm{CAL}})
(V-V_{\mathrm{stat}})\right].
\]

对应：

\[
K_X=X_P^\top X_P/p+\lambda(\Gamma_X+\tau^2\Gamma_{\mathrm{CAL}}),
\]

\[
\boxed{
V_s=V_{\mathrm{stat}}+
K_X^{-1}X_P^\top(R_P-X_PV_{\mathrm{stat}})/p,
\qquad
\widehat F(U_1)=Y_{\mathrm{base}}+XV_s.
}
\]

采用 Cholesky/SVD 支撑上的线性求解，不显式求逆。几何和参数变换时，Gamma_CAL 与 prior 同时变换。有效特征支撑通过 CAL 白化后当前 X 的 SVD 定义；参考核心保留当前 RMS 奇异值大于 max(1e-12, 1e-10*sigma_max) 的方向。秩为零时只返回静态基线，无额外修正；生产 lambda 必须大于零。

### 2.3 近锚点数值修正

只用当前 Gamma_X 的正则有一个缺陷：若 \(X=\epsilon X_*\)，固定舍入噪声 eta 可以被拟合成 \(V-V_{\mathrm{stat}}=O(\epsilon^{-1})\)，使 \(X(V-V_{\mathrm{stat}})\) 不随位移消失。

加入冻结的 CAL 正则下界后：

\[
V-V_{\mathrm{stat}}=O(\epsilon),\qquad
X(V-V_{\mathrm{stat}})=O(\epsilon^2)
\]

对固定观测噪声成立。它保证小位移下的噪声修正衰减，不保证所有状态的视频误差下降。

tau 的一次定标规则：

1. 在冻结 CAL 的已选特征上，计算部署 BF16/FP32 特征差与同输入 FP64 特征差的差异；
2. 用 Gamma_CAL 白化该差异，按来源等权计算每特征均方能量 e_phi；
3. 冻结 \(\tau=\max(0.01,\sqrt{e_\phi})\)。

0.01 是明确的数值设计下限，不宣称为 BF16 理论误差界。tau 不用 DEV 或视频结果调节。lambda 从 {0.01,0.1,1.0} 中只通过 CAL 留一来源 query 选择；平均损失相同时选较大的 lambda。

### 2.4 表示能力与可辨识性

该构造压缩当前残差的 token 因子：

\[
\mathcal R_{snc}\approx\sum_a X_{sna}V_{sac}.
\]

单次残差矩阵秩不超过 r；输出方向可随状态变化，不再固定在 Q32。若 \(R=XV_*\)、probe 能辨识 X 的列空间，无正则数学极限可精确恢复全部 token 的 R。

先讨论没有截去非零方向的情形。令 \(R=XV_*+\Xi\)、\(X^\top\Xi=0\)。无正则满秩情况下：

\[
\|R-XX_P^\dagger R_P\|_F^2
=
\|\Xi\|_F^2+\|XX_P^\dagger\Xi_P\|_F^2.
\]

第一项度量特征表达不足，第二项度量观测混叠。带正则时还存在向 prior 收缩的偏差。完整误差项写为：

\[
R-XV_s=
\Xi-XK_X^{-1}X_P^\top\Xi_P/p
+\lambda XK_X^{-1}(\Gamma_X+\tau^2\Gamma_{\mathrm{CAL}})
(V_*-V_{\mathrm{stat}}).
\]

实际核心保留完整静态预测X_raw V_stat，只把在线delta限制在当前支撑。如果截去了非零方向，应先令 \(E=R-X_{\rm raw}V_{\rm stat}\)、\(Z=X_{\rm raw}T_{\rm support}\)，再分解 \(E=ZC_*+\Xi\)、\(Z^\top\Xi=0\)。令 \(H=T_{\rm support}^\top(\Gamma_{\rm raw}+\tau^2\Gamma_{\rm CAL})T_{\rm support}\)、\(K=Z_P^\top Z_P/p+\lambda H\)，则最终误差为
\[
\Xi-ZK^{-1}Z_P^\top\Xi_P/p+\lambda ZK^{-1}HC_* .
\]
丢弃方向的静态prior仍在最终预测中，不将它错误地一并删去。64 行拟合误差下降不是有效性结论；主要局部指标在不参与拟合的共同 query 上计算。

### 2.5 CAL 特征生产：一个有界、确定的流程

每位置只产一套参数，不遍历不同宽度或激活族。CAL11 的每个 P512 端点对按真实 global token ID 哈希划成两个 256 行的部分；具体划分与实验方案一致。步骤1—5用FP64数学特征与 \(F_{64}(U_1)-F_{64}(U_0)-\Delta U A_{64}\) 做结构选择；通道冻结后在实际窄BF16特征程序上重算beta、sigma和度量。步骤6—8及lambda定参使用部署Xb、缓存Ybase、真实64行BF16 observer与原始full-M捕获query标签。数学目标和实际部署目标各自报告，不把两轨数值拼成一张胜负表。

1. 计算所有原生通道的 delta_z、delta_h；逐通道拟合 beta，分母为零的通道保留 beta=0 并标记无位移信号。尺度为来源等权差分 RMS；完全为零的列将sigma置1并标记inactive，避免除零。两列都没有信号的通道不进入候选；只有一列有信号的配对可参加秩1评分，其零列由后续T_CAL去掉。零尺度处理在窄BF16特征重新定标时沿用。
2. 用每个通道配对列 \([\Delta z_h,\Delta h_h-\beta_h\Delta z_h]\) 对完整 gated residual 的单通道投影增益打分。对每个 CAL 状态计算两列的秩揭示 QR，使用 \(\|Q_h^\top R G\|_F^2/N\)，再来源等权。这里没有 Q32 输出投影。
3. 保留分数最高的 64 个通道作为候选池，完全相同分数以原生通道索引打破平分。候选池是本构造的明确边界，不因 DEV 结果扩大。
4. 从空集合起，贪心加入 16 个配对通道。每步枚举剩余候选，在 CAL 两折中用固定 64 probe、零先验、lambda=0.1、tau=0.01 拟合；用另一折 256 query 的完整 gated residual 损失排序。使用全部 512 无标签特征形成 Gamma 是允许的。每步最多64候选；不进行无限搜索。
5. 选择阶段只使用 CAL。可通过 probe/query residual Gram 重用评分，避免为每个候选反复乘 5120 输出通道。若 \(L\) 为候选特征产生的 query-to-probe 线性重建权重，则
   \[
   \|R_QG-LR_PG\|_F^2=
   \|R_QG\|_F^2-2\langle L,R_QG^2R_P^\top\rangle+
   \mathrm{tr}(L R_PG^2R_P^\top L^\top).
   \]
6. 在选定、重新数值定标的配对特征上形成原始Gamma_CAL。保留特征值大于最大值1e-10的方向，冻结正交有效基 T_CAL，并实际使用 \(\Phi_{\mathrm{eff}}=\Phi_{\mathrm{raw}}T_{\mathrm{CAL}}\)。所有缓存、Xb、静态V与在线Gram均在r_eff坐标中；重新计算部署Xb的Gamma_CAL并确认SPD，不能把奇异raw32 Gram交给参考核心。实际 r_eff<=32 单列记录。维度为零时记FEATURE_SPACE_EMPTY，不生成该位置的条件参数。
7. 以完整 CAL 拟合 V_stat，最小化来源等权 gated query-independent residual MSE，加 \(10^{-6}\mathrm{tr}(V^\top\Gamma_{\mathrm{CAL}}V)\) 的固定岭项。输出各列独立求解；CAL gate 随来源改变，不能把它换成一份平均 gate。
8. lambda 的三点选择采用固定已选传感器、留一 CAL 来源拟合 V_stat，再在被留来源的两折 query 上评价。特征池和传感器已经使用全 CAL，因此这不是未经暴露的泛化估计；其职责只是 CAL 内定参。最后在全 CAL 重拟合一次 V_stat，导出一份冻结参数。

若可用传感器不足16，记录 FEATURE_POOL_LIMITED，保留已完成结果；不从 DEV 补通道。零先验仅做局部机制消融，视频中只用上述静态先验。

### 2.6 观测位置

主视频使用确定的固定 64 行，以便隔离读出条件化。局部固定策略从当折候选池按 global token ID 排序后使用 seed42 的冻结排列；视频从全31668 token集合用相同规则产生自身64行，保存确切ID清单。不声称它们与历史Global-P64的64行相同。

局部额外计算一次几何选点：CAL 白化和当前 X 白化后，按贪心 log-det 信息增益选择64行，初始信息矩阵为I，近似平分取最小global ID。它只使用特征，不读真实 residual。参考核心提供该策略；本轮不把它另展开成视频臂，也不按每个 DEV 来源切换策略。

## 3. 构造二：少量状态模式控制满秩算子

### 3.1 数学定义

q=4 个隐藏系数模式 \(a^{(j)}\in\mathbb R^m\)，定义：

\[
B_j=W_1^\top\operatorname{diag}(a^{(j)})W_2^\top,\qquad
A(\theta)=A+\sum_{j=1}^q\theta_jB_j.
\]

每个 B_j 可以满秩；q 限制的是中心化算子族在状态轴上的维数，不限制单个矩阵的输入输出秩。

anchor 保存 BF16 U0 与 Y0=F(U0)。hit：

\[
\widehat F(U_1)=Y_0+\Delta U A(\theta_s).
\]

对64当前probe：

\[
e_P=F(U_{1,P})-Y_{0,P}-\Delta U_P A,\qquad
T_{j,P}=(\Delta U_PW_1^\top\odot a^{(j)})W_2^\top.
\]

先验 theta_stat 是同一4模式上 CAL pooled 最佳固定系数。在线求：

\[
\theta_s=\arg\min_\theta
\frac1p\|(e_P-\sum_j\theta_jT_{j,P})G\|_F^2+
\lambda_B(\theta-\theta_{\mathrm{stat}})^\top
\Lambda_{\mathrm{CAL}}(\theta-\theta_{\mathrm{stat}}).
\]

Lambda_CAL 是冻结 CAL 模式作用 Gram，按有效支撑归一化；不使用会随当前位移归零的唯一正则。输出 gate 对所有模式系数共享，真实改变其最优权衡。

静态控制使用相同 U0/Y0、相同矩阵编译顺序的 theta_stat，不用“theta=0的原A”替代。当前理论模板和实际 BF16 编译路径分别验收。

### 3.2 模式生产，不学习四个任意满矩阵

以下训练只用现存 CAL 端点，先六位置；进入全80时只为实际将执行的family补齐其余位置。

定义线性作用 \(\mathcal L_s(b)=(\Delta Z_s\odot b)W_2^\top\)。

1. 每个 CAL 来源分别求一个隐藏系数修正 b_s：
   \[
   \min_b\|\big(R_s-\mathcal L_s(b)\big)G_s\|_F^2/N_s+
   \mu_s b^\top D_s b.
   \]
   此结构选择使用FP64数学残差。正规作用的对角为 \(d_h=(N_s^{-1}\sum_n\Delta Z_{nh}^2)\sum_c g_c^2W_{2,ch}^2\)。d_h=0的通道系数固定0；其余设置 \((D_s)_{hh}=\max(d_h,10^{-12}\operatorname{mean}_{d>0}d)\)。选择 \(\mu_s=10^{-3}\)，PCG预条件器为 \(\operatorname{diag}(\mathcal L_s^*G_s^2\mathcal L_s/N_s)+\mu_sD_s\)，最多80迭代、正规方程相对残差1e-5。未收敛明确记录FIT_NOT_CONVERGED，不通过增加来源/视频修复。
2. 对11个 b_s 建立11×11的CAL作用Gram：
   \[
   H_{uv}=\frac1{11}\sum_s\langle \mathcal L_s(b_u)G_s,
   \mathcal L_s(b_v)G_s\rangle/N_s .
   \]
   保留特征值大于最大值1e-10的方向，取前4个非退化特征向量，按作用范数构造模式
   \(a^{(j)}=\sum_s b_s v_{sj}/\sqrt{\lambda_j}\)。
   这在真实输入/输出作用的度量中压缩状态轴，没有假定隐藏通道欧氏正交。
3. 模式冻结后，使用部署Db与FP32模板、原始捕获Y和当前gate拟合theta_stat，保存同一归一化的Lambda_CAL（目标除token数N，不除d）。固定读出正则为1e-6 Lambda_CAL。lambda_B从{0.01,0.1,1.0}以固定模式、留一CAL来源重新拟合theta_stat的两折query选择；probe为真实64行BF16 observer。实际动态矩阵编译输出必须参加BF16数值合同核验，局部MSE不作视频准入。模式已见全CAL，该选择是欧氏目标内的开发定参。
4. 模式退化时保存q_eff与尾部作用能量；q_eff<4不是自动造新模式的理由。

矩阵算子可以通过 \(\mathcal L_s\) 和其伴随实现PCG，不物化 m×m 或 N×m×d。输出残差的伴随收缩为 \(\sum_n \Delta Z_{nh}(R G^2 W_2)_{nh}\)。

### 3.3 在线收缩与编译

无须保存四个完整 B_j。先组合：

\[
\alpha_s=\sum_j\theta_{s,j}a^{(j)},\qquad
\delta A_s=W_1^\top\operatorname{diag}(\alpha_s)W_2^\top.
\]

probe 上共享 delta_Z，沿批维堆叠四组模板进行W2收缩。FP64参考核心验证这一恒等式。生产首实现采用明确的 FP32 模板与动态矩阵编译（TF32关闭），最终当前矩阵转BF16；因此静态MAC数不意味着与BF16 GEMM等时。

本轮使用“只存隐藏系数、在线编译”这一条路径，不在视频阶段另加预存四矩阵版本。预存版本只是成本参照。

## 4. 冻结的原生数值合同

### 4.1 观测条件读出

1. 原生 Dense anchor 输出 Y0 为BF16。
2. A为已冻结BF16矩阵；base0=BF16(U0@A)；C0=BF16(float(Y0)-float(base0))。
3. anchor/hit的窄W1+GELU必须调用相同feature程序；z、GELU结果遵从该原生BF16路径，beta/scale运算FP32，Phi0缓存FP32。首实现不直接截取另一GEMM形状的Dense hidden充当窄feature结果。
4. hit base1=BF16(U1@A)，Ybase=BF16(float(base1)+float(C0))。
5. X32=Phi1-Phi0；部署投影使用Xb=BF16(X32)。在线Gram和求解以float(Xb)为实际设计矩阵，保证拟合与执行看同一批特征值；Gamma_CAL/tau按这条路径绑定。
6. probe Dense用64行原生BF16；RP=float(Fprobe)-float(Ybase[P])。不能把旧捕获Y[P]直接当在线observer值。
7. 当前统计/RHS至少FP32，32×32/4×4求解FP64；Vstat导出BF16，在线结果投影前转BF16；correction=BF16(Xb@Vb)，out=BF16(float(Ybase)+float(correction))。
8. 静态控制用同一C0/Phi0和相同最后两次舍入，只将V固定为Vstat，并且不执行无用probe。以显式adaptation=False验证控制等价，不用“很大的lambda”冒充严格中性。
9. 返回pre-gate；宿主应用当前gate一次，沿用现有gate见证。

### 4.2 满秩模式

1. anchor保存原生BF16 U0,Y0；没有anchor U0A GEMM。
2. D32=float(U1)-float(U0)，Db=BF16(D32)。probe模板的输入为float(Db[P])，以匹配命中大GEMM的输入；记录该delta舍入相对FP64割线的差异。
3. 当前完整probe FFN为BF16；模板以FP32原生权重和FP32模式计算，TF32关闭；4维求解FP64。
4. alpha以FP32组合；deltaA=(W1.float().T*alpha)@W2.float().T，TF32关闭；Acurrent=BF16(A.float()+deltaA)。
5. increment=BF16(Db@Acurrent)，out=BF16(float(Y0)+float(increment))。
6. 静态控制以theta_stat走相同编译、delta和输出顺序；可缓存其编译矩阵，离线编译与在线编译成本分开记。在线theta得到的模板预测与Acurrent实际作用不宣称逐位相等。
7. 只有最终pre-gate输出进入宿主当前gate。

### 4.3 一次有限验收

沿用已存在的P512和首条生产视频见证，不新增完整视频smoke队列：

- 同一程序中的disabled adaptation与匹配static逐位一致；
- zero-delta特征修正为0；BF16总输出相对anchor的误差按真实缓存舍入记录，不要求错误的逐位Dense恒等；
- 64行Dense probe与既有full-M捕获同坐标结果的偏差，必须独立记录；小批量与大批量GEMM形状不同，不能假定逐位相同；
- 生产预测与独立按冻结舍入顺序重建的reference逐位一致；
- 另报告FP64数学输出、实际编译输出和full-M BF16 query目标之间的差异。核验对象为实际部署数值程序，不能仅凭无量化模板替代；有限局部MSE退步不取消视频比较；
- 大小位移和固定噪声的连续性由合成单测覆盖，真实CAL/DEV不额外构造无限扰动网格。

## 5. 成本与缓存

以两个Dense的4Ndm MAC为分母，以下是主要收缩而非实测延迟：

| 方法 | pair主要MAC比例 | 40层同时活跃状态 |
|---|---:|---:|
| Q1/L64单C缓存 |68.5185%|12.0804 GiB|
| Global-P64 |68.5690%|44.6974 GiB|
| 观测读出，保守按32输入投影列 |约68.794%|12.2314 GiB|
| 4模式，预存四矩阵（仅参照） |约59.455%|24.1608 GiB|
| 4模式，在线编译（本轮） |约63.674%|24.1608 GiB|

观测读出实际同16原生通道配对只需16输入投影列，故其GEMM MAC不高于表中保守数；Gram、SVD、类型转换、选点、teacher权重读取另记录。FP32特征缓存使新值为12.2314GiB，不再使用此前双BF16的12.1559GiB。

每层完整BF16 W1/W2约270MiB；FP32模式编译需要更宽权重与临时矩阵，额外流量和算力格式不得隐去。只存4个隐藏模式在80位置BF16约8.44MiB，生产模式FP32则约16.88MiB；预存4个BF16满矩阵则额外15.625GiB。

表中是状态payload，不是GPU总峰值。研究视频使用既有缓存provider，并明确其CPU/磁盘/设备驻留；static/online使用相同provider。不得把内存不足导致的路径变化解释成科学失败，或把带落盘研究路径用于加速宣称。

## 6. 可辨别的限制

- 观测读出：两个token具有同X但不同真实R时无法区分；固定Q32高漏出不证明r32足够。
- 满秩模式：即使矩阵满秩，同一状态对全部token共享theta；不同token的GELU局部斜率可能需要原生非线性特征。
- 两种方法都使用当前真实probe，只能在自身student输入上获取；旧Dense轨迹probe不能替代在线观测。
- 近锚点正则、共享gate、没有截距、没有probe真值拼回，均是方法定义的一部分。
- 本轮不叠加两family，不增加区域bank/动态hidden选择，不做terminal更新。它们保持独立未测方向，不由本轮结果自动排除。
- 当前特征与模式由局部欧氏目标选择；它们的oracle floor只限制该目标下的重构能力。若视频未建立收益，结论限这些具体表示/拟合规则，不排除其他任务敏感的低维状态表示。

## 7. 实现与复核

参考核心提供 paired_features、fit_observed_readout、observed_hit、native_mode_templates、fit_mode_coefficients、compile_native_modes、mode_hit 和固定/几何probe选择。它是NumPy FP64定义实现，不包含CAL producer或BF16/Wan runtime。

运行：

    python -m unittest discover -s tools/ffnds_observed_operator/tests -v

单测针对恢复、换基、零行、未观测方向、近零噪声、原生模式收缩、共享gate和同缓存static control。详细执行记录及独立复核见 [审查记录](OBSERVED_SECANT_INDEPENDENT_REVIEW.md)。

方法学习依赖POD/部分观测重建与带正则最小二乘等已有数学工具；本轮不声称这些公式本身具有文献新颖性。待研究的是具体FFN端点响应的条件表示、合法在线信息与真实质量/执行收益。

