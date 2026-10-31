# G19 · 权重诱导的 Hermite 谱、条件期望与可执行 FFN 重表示

2026-11-01 · **METHOD_SNAPSHOT_CANDIDATE / INDEPENDENT_MATH_REVIEW / THEORY_AND_SYNTHETIC_ONLY / REAL_WEIGHT_NOT_READ / NO_NATIVE_VIDEO_OR_SPEED_EVIDENCE / NOT_ACTIVE**

## 0. 与 G18/G17 的关系

G18 的 GELU 奇偶分解、神经元函数群和 Taylor 近似可作为代数工具，但不能将函数群的秩下降当成假设已证。G17 的完整视频行为、全深度 Dense/Student 张量及 A5/HWG 比较保持首要科学评价地位。本文件提出独立的 **Weight-induced Spectral–Conditional FFN (WSC-FFN)** 研究候选；不改 ACTIVE，也不授权 CPU/GPU 科学实验。

当前对 GitHub 的读取确认了研究文档与源码，但 **外部 Wan 完整权重文件未在本轮读取**。本文件没有任何“真实 Wan 谱衰减/有效秩/画质/加速”测量。完整权重的存在和其在用户计算环境中的可用性，不能替代本轮实际数值访问。

## 1. 问题结构与精度

每个真实 FFN（按真实 checkpoint 核验）：
\[
f(u)=\phi(uW_1+b_1)W_2+b_2,\quad W_1\in\mathbb R^{d\times m},\ W_2\in\mathbb R^{m\times d}.
\]
e5 gate 只在 post-FFN 后作用一次，学生实际 u 与 gate 会漂移。原研究典型 d=5120、m=13824；High/Low 两专家各40组**独立权重**，4-step HHLL ×40 构成160个**调用位置**，不等于160组不同权重。

理论解析积分和奇偶结果先限定精确 GELU \(\phi(x)=x\Phi(x)\)。若实际后端为 tanh-GELU 或 BF16/HiF4/NVFP4，其函数与量化路径必须另立 numerical contract；特别是 tanh-GELU 的下面解析高斯积分公式一般不适用，需可靠的一维数值积分或重新推导。基础分析优先原始 BF16，不混用 Q1/Q2 动态激活路径。

## 2. DERIVATION_GRAPH A：完整权重的非线性正交谱（PROVED under Gaussian reference）

设参考输入 \(u=\mu+\Sigma^{1/2}z,\ z\sim N(0,I)\)。对第 j 个神经元定义：
\[
m_j=\mu^\top w_j+b_{1j},\quad \beta_j=\Sigma^{1/2}w_j,\quad s_j=\|\beta_j\|,\quad p_j=\beta_j/s_j.
\]
s=0 的项为常数。记 \(v_j=W_2[j,:]\)，并令
\[
c_{jq}=\mathbb E_{Z\sim N(0,1)}[\phi(m_j+s_j Z)He_q(Z)].
\]
则 \(f-\mathbb Ef=\sum_{q\ge1}F_q\) 在 Gaussian L2 中成立，其中
\[
F_q(z)=\frac1{q!}\sum_j c_{jq}He_q(p_j^\top z)v_j.
\]
利用 \(\mathbb E He_q(p_i^\top z)He_r(p_j^\top z)=\delta_{qr}q!(p_i^\top p_j)^q\)，得到
\[
E_q:=\mathbb E\|F_q\|^2 =
\frac1{q!}\sum_{ij}c_{iq}c_{jq}(p_i^\top p_j)^q(v_i^\top v_j)\ge0,\quad
\mathbb E\|f-\mathbb Ef\|^2=\sum_{q\ge1}E_q.
\]
该表达直接包含 signed cross-term，并由完整 W1/W2、\(\mu,\Sigma\)、激活定义决定。可按块计算，不需显式形成高阶 \(d^{q+1}\) 张量。高阶谱截断的严谨上界：对任何整数 Q≥0，
\[
\sum_{q>Q}E_q\le \frac{1}{Q+1}\mathbb E\|\nabla_z f(z)\|_F^2.
\]
这是 Gaussian Hermite 导数等距关系 \(\mathbb E\|\nabla f\|_F^2=\sum_{q\ge1}qE_q\) 的直接结果。q=0 项另行保存均值，不把均值误算成非线性能量。

若所有 m_j=0，GELU 的 \(x/2+\psi_{\rm even}(x)\) 使 q≥3 的奇数 Hermite 系数精确为0；实际 AdaLN/preactivation 偏置可能使此条件不成立，需实测 m_j。

## 3. DERIVATION_GRAPH B：最佳低维统计量的精确条件表示（PROVED under Gaussian reference）

在白化坐标取秩 r 的正交投影 \(P=P^\top=P^2\)，定义 \(t=Pz\)。由于
\[
a_j\mid t\sim N(m_j+\beta_j^\top t,\ \sigma_{j,\perp}^2),\quad
\sigma_{j,\perp}^2=\beta_j^\top(I-P)\beta_j,
\]
使用 Stein 恒等式对精确 GELU 推出：
\[
\mathcal G_\sigma(h):=\mathbb E_{X\sim N(h,\sigma^2)}[X\Phi(X)]
=h\Phi\!\left(\frac{h}{\sqrt{1+\sigma^2}}\right)
+\frac{\sigma^2}{\sqrt{1+\sigma^2}}\varphi\!\left(\frac{h}{\sqrt{1+\sigma^2}}\right).
\]
因而所有依赖 t 的可测函数中最佳 L2 逼近**精确为**
\[
g_P(t)=\mathbb E[f(z)\mid Pz=t]
=b_2+\sum_j v_j\,\mathcal G_{\sigma_{j,\perp}}(m_j+\beta_j^\top t).
\]
这是消去输入方向后的解析函数，不是对原神经元粗暴截断。任何实现 \(\widehat g(Pz)\) 满足精确的正交误差分解：
\[
\mathbb E\|f-\widehat g(Pz)\|^2
=\underbrace{\mathbb E\|f-g_P(Pz)\|^2}_{\mathcal E_{\rm info}(P)}
+\underbrace{\mathbb E\|g_P(Pz)-\widehat g(Pz)\|^2}_{\mathcal E_{\rm representation}}.
\]
第一项是**丢弃输入方向的不可约误差下界**，第二项是给定保留方向下的执行表示误差。

设 \(\rho_{ij}=p_i^\top p_j,\ \rho^P_{ij}=p_i^\top Pp_j\)，有如下**无需形成高阶张量**的精确谱式：
\[
\mathcal E_{\rm info}(P)
=\sum_{q\ge1}\frac1{q!}\sum_{ij}
c_{iq}c_{jq}(v_i^\top v_j)
[(\rho_{ij})^q-(\rho^P_{ij})^q].
\]
有限 Q 截断的遗漏为非负，并受上节的谱尾上界控制。每项非负的结论指整个 q 阶双重求和，而不是其中每个 (i,j) 项。

## 4. DERIVATION_GRAPH C：能够先否决输入低维路线的证书（PROVED under Gaussian reference）

定义 Gaussian-whitened Jacobian \(J_z f\)，输入方向敏感 Gramian：
\[
G_{\rm grad}:=\mathbb E[(J_zf)(J_zf)^\top]\succeq0.
\]
由**条件 Gaussian Poincaré 不等式**，
\[
0\le\mathcal E_{\rm info}(P)\le
\mathbb E\|(I-P)J_zf\|_F^2
=\operatorname{tr}[(I-P)G_{\rm grad}].
\]
选取 \(G_{\rm grad}\) 最大 r 个特征向量，最小化此可计算上界，最小值为其剩余特征值尾和。这个 P 不一定最小化真实 \(\mathcal E_{\rm info}\)，必须用 B 节精确能量检验。

可从完整权重计算：
\[
G_{\rm grad}=\sum_{ij}(v_i^\top v_j)\,
\mathbb E[\phi'(a_i)\phi'(a_j)]\,\beta_i\beta_j^\top.
\]
期望由二元 Gaussian 积分或固定精度估计获得，具有 signed cross-coupling。若尾谱缓慢衰减且精确 \(\mathcal E_{\rm info}(P)\) 大，则舍弃低维静态共享输入表示，而不是继续盲选更小 r。参考分布不代表真实 Wan 数据；高斯下的证书不得转写为任意真实输入/视频质量保证。

## 5. DERIVATION_GRAPH D：张量耦合与新型执行算子（CONDITIONAL / OPEN）

第 q 阶谱对应向量值对称张量
\[
T_q=\frac1{q!}\sum_j c_{jq}\,v_j\otimes p_j^{\otimes q},
\]
并满足 \(\mathbb E\|F_q\|^2=q!\|T_q\|_F^2\)。特别是 q=2 对应带正负输出向量的二阶特征相互作用；检查 q=2/q=4 及不同 High/Low/site 的共同方向、联合 CP/TT 的有效执行秩，而不是先设定每神经元分组。

候选 1：**Project-and-Smear / 条件平滑求积**。先求 t=Pz（线性 d×r），再用 K 个可独立重参数化的「平滑 GELU」原子 \(\mathcal G_{\tilde \sigma_k}(\tilde a_k^\top t+\tilde b_k)\tilde v_k\) 逼近 g_P。其模型不局限原 W1 某子集；参数须通过完整权重给出的目标函数、解析核 Gram、或冻结校准样本优化。固定候选函数时，读出 B 可由最小二乘或下游固定权重型目标得到闭式解；选择方向/尺度是非凸问题，没有全局最优保证。理论主 MAC：dr+rK+Kd（不计逐元素特殊函数、quant/launch、gate/suffix）。

候选 2：**Projected Chaos Tensor / 低阶张量核**。若 q≥Q 的谱尾很小，且低阶 T_q 具有可执行的联合低秩结构，构造带有正交多项式或低秩多项式收缩的算子；必须同时记录输入投影、张量读取、输出投影、kernel fuse 和运行成本。不能将参数低秩冒充在线 FLOP 低。

候选 3：**Mixed Basis / 混合线性-非线性**。线性阶可直接折叠或在共享 P 中表示，非线性阶根据 q 的谱和输出方向分配预算；若实际 Hessian/high-q 成分对运动更关键，需保留其功能而非只追求最大能量阶。

纯理论示例 d=5120、m=13824、r=1024、K=4096：
\[
C_{\rm dense}=2dm=141{,}557{,}760,\quad
C_{\rm projected}=dr+rK+Kd=30{,}408{,}704
\]
即理想主 MAC 占比约21.5%。这是假设的 shape/cost **算术示例**；没有任何 evidence 证明 r/K 可保留真实视频或实现该速度。

## 6. DERIVATION_GRAPH E：全160调用位点的后缀传播（LOCAL_CONDITIONAL）

对给定 Dense 轨迹线性化，令某 FFN 的 gated output 扰动为 e_i，后续 state/video readout 对该位置的导数为 L_i。则有限时域终端描述的**一阶近似**为
\[
\delta y_T\approx\sum_iL_ie_i
\]
并且
\[
\|\delta y_T\|_Q^2\approx
\sum_{ij}e_i^\top L_i^\top QL_je_j.
\]
跨 site 有符号交叉项可以抵消或增强；逐层局部 SSE 的求和并非终端质量目标。L_i 需要真实后缀 JVP/VJP或合法冻结状态 probe，不能仅由 W1/W2 推断；输出 Q 的选择须与动作事件/语义的验证对象绑定。此目标仍是局部可微描述的代理，最终视频质量需 81 帧、4step×40 全路径自由生成验证，且要检查学生状态漂移和高阶项。

## 7. 首个有决策价值的完整权重审计（NOT EXECUTED）

针对**真实 High/Low 80 套权重**逐个执行，区分四步的 \(\mu,\Sigma\) 条件：

1. 核实 BF16 checkpoint W1/W2/bias/GELU/norm/e3-e4/e5 与量化数值族；报告 shape/units/weight finite/tied status；仅用实际路径确定 identity。
2. 先以 isotropic Gaussian reference 做权重闭式的 E_q(q=0..Q) 与尾部上界；再分别用 source-held-out 的真实 pre-W1 输入估计 Gaussian family/mixture，比较真实神经元联合输出矩与预测差。记录非 Gaussian 的失配，不允许用 surrogate 结果冒充真实值。
3. 对多个 r 和所有 High/Low/layer，计算 Poincaré tail bound 与独立的 exact conditional information loss；若 r 在现实预算中仍有明显下界，否决 projected branch。
4. 按 q=1/2/4/... 对运动相关输出对比方向、主体ROI与时间窗做 signed cross、tensor contractions、联合谱；缺失的 tensor/source 不以零替代。
5. 若存在明确低成本余量，比较 original-neuron compact、G18 Taylor group、signed HWG、R256成本对照、Project-and-Smear、Chaos Tensor、同预算 compact student。每种保留真实同状态、自由学生、理论 MAC 和原生 kernel latency 四种不同证据身份。
6. 在任何新科学 CPU/GPU 或视频任务执行前，交由不同 Agent 完成独立评议与 recheck，冻结方法版本/参数/来源，再进行完整视频、语义/事件/运动/背景/时间/成像评价。

### REUSABLE_ASSET_PLAN（方法侧）

- WSC_WEIGHT_MAP：expert/layer → 原始参数 shape、数值族、精确 checkpoint locator、原生 GELU/bias contract；
- WSC_REFERENCE_FAMILY：source/step/expert/layer/ROI 的 \(\mu,\Sigma\) 及估计条件、held-out 身份；
- WSC_CHAOS_SPECTRUM：q、E_q、尾部上界、输出 signed cross、Gaussian 假设适用范围；
- WSC_CONDITIONAL_RISK：P/r、exact information loss、Poincaré bound、失配与置信区间；
- WSC_TENSOR_SKETCH：q=2/4 所用投影/随机种子、不可复用的张量近似误差；
- WSC_OPERATOR_PARAMETERS：投影/虚拟原子/输出读出、理论和原生成本、精度及数值族；
- WSC_VIDEO_RESULT：Run/source/seed/full160/81frame/所有质量维度/系统指标/统计单位。

各资产均须登记 producer、run/version、input locator、shape/axes、units、state、dtype、numeric recipe、验证方式、允许复用范围及失效条件。不得以历史缓存身份替代可复现的本轮观测。

## 8. 最邻近先例与新颖性边界

- Active subspace/conditional expectation 的高斯 Poincaré 证书和 ridge approximation：成熟数学，不是本研究独创。参阅 https://link.springer.com/article/10.1007/s10915-024-02542-0 及 https://www.sciencedirect.com/science/article/am/pii/S0045782521005302
- SlimDiff (2025) 已使用 DiT activation covariance + functional group spectral compression：https://arxiv.org/abs/2509.21498
- ICML 2026 MFF/GFWSVD 已研究非对角敏感性驱动的权重分解：https://proceedings.mlr.press/v306/chekalina26a.html
- ICML 2026 RT-Lynx 已研究 DiT activation N:M 稀疏与原生 GEMM：https://proceedings.mlr.press/v306/cong26a.html
- GELU 基础定义：https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU

**新颖性仍为 OPEN**。不能以 Hermite、Stein、active subspace、conditional expectation、tensor low rank 或 virtual neurons 单个术语宣称创新。潜在可验证贡献是：借完整权重的**有符号多阶函数谱**确定何种维度/阶数可以丢弃，推导精确条件平滑目标并由新型低成本混合算子实现，再用相同 source 视频证明恢复运动事件和完整质量、拥有 native SYS 收益。

## 9. 独立代数数值自检（SYNTHETIC_ONLY）

本轮在独立小矩阵（d=4,m=9）和随机 Gaussian 输入上核验 Hermite q≤10 累积能量与 Monte Carlo 输出方差（差约0.8%）、条件信息损失（差约0.5%），并检查条件 Poincaré 右侧确实不小于观测条件风险。精确积分 \(\mathcal G_\sigma\) 的 Monte Carlo 数值亦一致。均非 Wan 权重证据；误差来自有限 Monte Carlo 与阶数截断。

**裁决：G18 函数群退为 strong baseline；G19 作为待真实权重审计的 METHOD_SNAPSHOT_CANDIDATE 保留。只有首先通过函数谱与条件风险的结构判据，才推进新物理执行算子。**
