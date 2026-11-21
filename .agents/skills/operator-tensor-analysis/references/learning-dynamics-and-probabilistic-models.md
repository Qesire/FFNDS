# 学习动力学与概率模型

从当前 AI 任务恢复参数、观测、状态、随机性和目标，再按目录调用工具卡。
本文件处理训练、生成、序列推理、图学习及 Agent 交互；数学对象的选择见 [AI 分析对象](ai-analysis-objects.md)。
除明确引述来源的结果外，下列矩阵化、误差界和反例均为在所列条件下的直接推导，不代表额外实验结论。

## 目录

| 工具卡 | 调用目标 |
|---|---|
| [LK01 参数到有限观测函数与经验 NTK](#lk01-参数到有限观测函数与经验-ntk) | 参数更新怎样改变所选输出；何时可以冻结核 |
| [LK02 含状态的训练与推理动力学](#lk02-含状态的训练与推理动力学) | 优化器、循环记忆和缓存怎样影响未来 |
| [LK03 随机目标及梯度估计](#lk03-随机目标及梯度估计) | 期望求导、score 项、路径导数与条件无偏 |
| [LK04 条件概率核与自回归](#lk04-条件概率核与自回归) | 采样、观测传播、分布传播与充分状态 |
| [LK05 Score SDE 与概率流 ODE](#lk05-score-sde-与概率流-ode) | 连续扩散生成的时间方向、边缘分布和数值误差 |
| [LK06 Flow matching](#lk06-flow-matching) | 固定耦合下的回归、概率路径与生成流 |
| [LK07 配对误差与分布及轨迹误差](#lk07-配对误差与分布及轨迹误差) | 蒸馏、条件生成和视频应比较什么 |
| [LK08 非光滑与离散机制](#lk08-非光滑与离散机制) | 激活分支、硬量化、路由与替代梯度 |
| [LK09 RL 与 Agent 交互](#lk09-rl-与-agent-交互) | Bellman 更新、长期回报、近似与部分观测 |
| [LK10 图与集合上的 AI](#lk10-图与集合上的-ai) | 消息传递、置换等变、图谱和边稀疏成本 |
| [来源与已读定位](#来源与已读定位) | 原始定义、定理范围及本文件的推导边界 |

## LK01 参数到有限观测函数与经验 NTK

**对象与目标。** 固定输入、条件及观测顺序，把 \(N\) 个 \(d\) 维输出堆成

\[
\mathbf f:\mathbb R^p\to\mathbb R^m,\quad m=Nd,\qquad
J_\theta=D_\theta\mathbf f\in\mathbb R^{m\times p},\qquad
K_\theta=J_\theta J_\theta^{\mathsf T}\in\mathbb R^{m\times m}.
\]

\(K_\theta\) 是该有限观测上的经验 NTK；\(u^{\mathsf T}K_\theta u=\|J_\theta^{\mathsf T}u\|^2\ge0\)。
检查损失的求和／平均约定；本卡把归一化全部放入损失，不再重复吸收到 \(K\)。

**公式与调用。** 对只通过 \(\mathbf f\) 依赖参数的可微损失 \(\ell(\mathbf f)\)，欧氏梯度流满足精确关系

\[
\dot\theta=-J_\theta^{\mathsf T}\nabla_{\mathbf f}\ell,\qquad
\dot{\mathbf f}=-K_{\theta(t)}\nabla_{\mathbf f}\ell.
\]

对 \(\ell=\tfrac12\|\mathbf f-y\|^2\)，令 \(e=\mathbf f-y\)，则 \(\dot e=-K_t e\)。
若沿整个区间 \(K_t\succeq\lambda I\) 且 \(\lambda>0\)，直接由 \(\frac d{dt}\|e\|^2\le-2\lambda\|e\|^2\) 得指数衰减；无需先假定核不变。
若沿区间 \(K_t=K_0\)，则 \(e(t)=\exp(-tK_0)e(0)\)；一般时变核不能直接沿用这个解式。

**成立条件。** 精确梯度流公式只需相应链式法则及解存在；显式参数正则项会另加 \(-J_\theta\nabla_\theta R\)。
有限步梯度下降还存在离散化及非线性余项。预条件更新 \(-P_\theta\nabla_\theta\ell\) 对应 \(J_\theta P_\theta J_\theta^{\mathsf T}\)，含动量的情况转 LK02。
冻结核可来自参数线性模型，也可作为有误差控制的局部近似；“网络很宽”本身不是充分条件。
[S1](https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf) 的极限结论包含特定参数化、初始化、宽度极限、激活正则性及有限时间训练方向控制。

**反例与边界。** \(J_\theta v=0\) 只说明所选观测沿 \(v\) 一阶不变，不说明有限参数移动后不变，也不说明全部输入上的函数相同。
多层 ReLU 网络即使激活模式固定，关于全部层参数通常仍非线性，不能因此冻结参数 Jacobian。

**AI 计算意义。** 用一次 VJP 和一次 JVP 计算 \(Ku=J(J^{\mathsf T}u)\)，避免物化 \(m\times p\) 的 \(J\) 或 \(m\times m\) 的 \(K\)。
用实际训练轨迹上的核漂移、方向残差和观测外误差分别检验冻结核近似与泛化主张。

## LK02 含状态的训练与推理动力学

**对象与目标。** 把所有影响下一步的量纳入 \(z_k\)，写为 \(z_{k+1}=\Psi_k(z_k,\xi_k)\)。
训练状态可含参数、优化器矩、计步器、归一化统计；推理状态可含循环隐状态、KV cache、位置和外部记忆。

**公式与调用。** 对固定步长的动量更新，明确采用

\[
v_{k+1}=\beta v_k+\nabla L(\theta_k),\qquad
\theta_{k+1}=\theta_k-\eta v_{k+1}.
\]

在二次目标的 Hessian 特征方向 \(\lambda\) 上，误差满足二阶递推

\[
e_{k+1}=(1+\beta-\eta\lambda)e_k-\beta e_{k-1},\qquad
r^2-(1+\beta-\eta\lambda)r+\beta=0.
\]

直接检验特征根。对 \(\lambda>0,\ 0\le\beta<1\)，\(0<\eta\lambda<2(1+\beta)\) 保证该方向两根模长小于 \(1\)；这是二次模型结果。
固定系数、零初始矩的 Adam 版本还需保存 \(m_k,q_k,k\)：

\[
\begin{aligned}
m_{k+1}&=\beta_1m_k+(1-\beta_1)g_k,\quad
q_{k+1}=\beta_2q_k+(1-\beta_2)g_k\odot g_k,\\
\theta_{k+1}&=\theta_k-\eta_k
\frac{m_{k+1}/(1-\beta_1^{k+1})}
{\sqrt{q_{k+1}/(1-\beta_2^{k+1})}+\epsilon}.
\end{aligned}
\]

**成立条件。** Adam 取 \(0\le\beta_1,\beta_2<1,\ \epsilon>0,\ m_0=q_0=0\)；除法、平方根按坐标进行，核对实际版本的系数日程、偏差修正、\(\epsilon\) 位置和 weight decay。
在固定随机实现的可微分支内使用 \(A_k=D_z\Psi_k\)；若 \(\xi_k\) 也是连续可微变量，一阶变分为 \(\delta z_{k+1}=A_k\delta z_k+B_k\delta\xi_k,\ B_k=D_\xi\Psi_k\)。
离散样本索引不取普通导数，其改变应作为不同更新或随机转移分析。
评估长程敏感度要研究 \(A_{K-1}\cdots A_0\)；各步谱半径小于 \(1\) 不足以保证任意时变乘积收缩。

**反例与边界。** 同一 \(\theta_k\) 配上不同 \(v_k\) 或 Adam 矩，会产生不同更新；因此单写 \(\theta_{k+1}=F(\theta_k)\) 会丢失机制。
二次方向稳定不推出非凸训练全局收敛；把优化器状态视为常量求导也会改变元学习或超参数梯度。

**AI 计算意义。** 明确训练恢复、缓存重用和截断反传保存／丢弃了哪些状态；比较成本时计入矩、缓存和重算的存储及访存。
Adam 矩的依据见 [S2](https://pure.uva.nl/ws/files/326512947/1412.6980v5.pdf)；本卡递推与状态 Jacobian 为直接推导，不调用该文的通用收敛保证。

## LK03 随机目标及梯度估计

**对象与目标。** 先写清随机源和分布是否依赖参数：\(L(\theta)=\mathbb E_{Z\sim P_\theta}\ell(\theta,Z)\)。
数据抽样、dropout、随机路由、策略动作及生成噪声可能对应不同求导问题。

**公式与调用。** 若 \(P\) 与参数无关，且可交换求导与积分，则

\[
\nabla L(\theta)=\mathbb E_P[\nabla_\theta\ell(\theta,Z)].
\]

若 \(p_\theta\) 对固定基准测度有可微密度和共同支持，保留分布变化项：

\[
\nabla L=
\mathbb E_{P_\theta}\big[
\nabla_\theta\ell(\theta,Z)
+\ell(\theta,Z)\nabla_\theta\log p_\theta(Z)\big].
\]

若 \(Z=g_\theta(\varepsilon),\ \varepsilon\sim\rho\)，且 \(\rho\) 与参数无关，可改用完整路径导数

\[
\nabla L=
\mathbb E_\rho\big[
\partial_\theta\ell(\theta,g_\theta(\varepsilon))
+D_\theta g_\theta(\varepsilon)^{\mathsf T}\nabla_z\ell(\theta,g_\theta(\varepsilon))\big].
\]

**成立条件。** 每次交换都核对局部可微性与可积控制，例如邻域中差商／导数有同一个可积上界；仅有逐样本可微不够。
score 公式还需密度求导的可积控制，且不能漏掉移动支持的边界项。
训练中需检验 \(\mathbb E[\widehat g_k\mid\mathcal F_k]=\nabla L(\theta_k)\)，其中 \(\mathcal F_k\) 是本次抽样前的历史，\(\theta_k\) 对它可测。
均匀新抽样可给条件无偏；一轮内无放回抽样在给定剩余样本后，通常不再对全数据梯度条件无偏。

**反例与边界。** 对 \(\theta>0,\ Z_\theta\sim U(0,\theta)\)、\(\ell(z)=z\)，真实导数为 \(1/2\)；仅使用支持内部 score \(-1/\theta\) 却得到 \(-1/2\)，漏项来自移动边界。
路径表示 \(Z_\theta=\theta\varepsilon\) 则给出正确导数。梯度裁剪、自归一化重要性权重和有限链采样也不自动保留无偏性。

**AI 计算意义。** 先选实际目标的正确估计器，再比较方差、采样成本、重参数化可用性与反传存储；无偏并不等于低方差或收敛。
score 估计与 STE 的区别见 [S3](https://arxiv.org/html/1308.3432v1)，可微重参数化见 [S4](https://arxiv.org/html/1312.6114v11)；一般公式及边界反例为本卡直接推导。

## LK04 条件概率核与自回归

**对象与目标。** 在可测状态空间上，条件核 \(K_\theta(A\mid x,c)\) 对 \(A\) 是概率测度，对 \((x,c)\) 可测。
固定 \((\theta,c)\)，取有界可测的可观测函数 \(\varphi\)，区分核对函数与对测度的两种作用：

\[
(K\varphi)(x)=\int\varphi(y)K(dy\mid x),\qquad
(K_*\mu)(A)=\int K(A\mid x)\mu(dx),
\]
\[
\int\varphi\,d(K_*\mu)=\int K\varphi\,d\mu.
\]

前者把未来观测拉回当前状态，后者传播当前分布；两者对所作用的函数／测度线性，即使核的参数化网络对 \(x\) 非线性。
有限状态下若 \(P_{ij}=K(j\mid i)\)，采用列分布，则 \(K\varphi=P\varphi,\ \mu'=P^{\mathsf T}\mu\)。

**公式与调用。** 自回归分解来自条件概率链式法则：

\[
p_\theta(x_{1:T}\mid c)=\prod_{k=1}^T p_\theta(x_k\mid x_{<k},c).
\]

这不要求 token 序列是一阶 Markov。若用状态 \(s_k\) 代替完整历史，必须验证

\[
\mathbb P(s_{k+1}\in A\mid\mathcal F_k,c)
=K_k(A\mid s_k,c).
\]

**成立条件。** \(s_k\) 应保留影响未来的历史信息；可采用完整前缀，或有依据的充分统计量。
状态与时间／位置可变时，核一般非齐次；只有具备相应齐次性与 Markov 性时才把多步传播写成固定 \(K\) 的幂。
若使用确定性网络 \(y=F(x)\)，对应核为 \(\delta_{F(x)}\)，其分布作用就是推前。

**反例与边界。** 两段历史末尾 token 相同而语义或 KV cache 不同，下一 token 分布可以不同；最后一个 token 并非充分状态。
同样的可见消息配上不同检索索引、工具状态或记忆，也不能自动共用同一个环境转移核。

**AI 计算意义。** 巨大词表／状态空间可通过采样和网络作用实现核，无须物化转移矩阵；温度、截断、拒绝采样会改变实际核。
验证 cache 等价时同时检查条件分布和状态更新。Markov 定义及核合成依据见 [S5](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf) §2.2；对偶公式由迭代积分直接得到。

## LK05 Score SDE 与概率流 ODE

**对象与目标。** 本卡限定 \(X_t\in\mathbb R^d\)、Itô SDE，以及仅依赖时间的标量扩散 \(g(t)\)：

\[
dX_t=b_t(X_t)\,dt+g(t)\,dW_t,\qquad
s_t(x)=\nabla_x\log p_t(x),\quad 0\le t\le T.
\]

\(W_t\) 为 \(d\) 维 Brownian motion。明确 \(t\) 增大表示从数据到噪声；条件生成时固定 \(c\)，使用相应的 \(b_t(\cdot,c),p_t(\cdot\mid c),s_t(\cdot,c)\)。

**公式与调用。** 使用递增的反向时间 \(\tau=T-t\)，\(Y_\tau=X_{T-\tau}\) 的倒向动力学为

\[
dY_\tau=\big[-b_{T-\tau}(Y_\tau)
+g(T-\tau)^2s_{T-\tau}(Y_\tau)\big]d\tau
+g(T-\tau)\,d\overline W_\tau.
\]

\(\overline W_\tau\) 是相对于反向滤过的 Brownian motion；它不是直接把原 Brownian 增量取负。
相应概率流 ODE 使用原本的递增时间坐标：

\[
\frac{dZ_t}{dt}=b_t(Z_t)-\tfrac12g(t)^2s_t(Z_t).
\]

由 \(\partial_t p=-\nabla\cdot(bp)+\tfrac12g^2\Delta p=-\nabla\cdot[(b-\tfrac12g^2s)p]\)，在下述良定性条件和相同初始分布 \(p_0\) 下，ODE 与 SDE 的每时刻边缘均为 \(p_t\)。
采样时从正确的 \(p_T\) 出发，把 ODE 从 \(T\) 积分到 \(0\)；不能在已换成 \(\tau\) 后再次漏改或重复翻转漂移符号。

**成立条件。** 在所用时间区间内核对正密度、系数／密度正则性、非退化扩散及时间倒向定理的条件。
一个便于使用的充分正则设置是：相关系数光滑，\(g>0\) 且远离零，前向／反向漂移及 ODE 速度满足适当的全局 Lipschitz 和线性增长界，且对应概率方程的解唯一。
数据端密度奇异、score 发散或终点不正则时，可先在 \([\varepsilon,T]\) 上分析，再单独处理端点极限。

**反例与边界。** 相同单时刻边缘不推出 SDE 与 ODE 的联合路径分布相同；SDE 的随机增量与确定性 ODE 轨迹具有不同路径性质。
仅时间依赖的矩阵扩散 \(G(t)\) 可用 \(a(t)=G(t)G(t)^{\mathsf T}\) 替换 \(g(t)^2I\)；空间依赖的 \(G(x,t)\) 一般还含 \(\nabla\cdot a\) 修正，需重新推导。

**AI 计算意义。** 分开检验 score 近似、初始先验偏差、时间截断、ODE／SDE 离散误差；同一个网络替换进两种求解器后不保证所得边缘仍相同。
核对网络预测的是 score、噪声、数据还是速度，再转换参数化。公式依据为 [S6](https://arxiv.org/html/2011.13456v2) §3.2、§4.3 与 Appendix D.1。

## LK06 Flow matching

**对象与目标。** 固定与模型参数无关的耦合 \(\pi\in\Pi(p_0,p_1)\)，采样 \((X_0,X_1)\sim\pi\)。
对有限二阶矩的端点，定义线性插值 \(X_t=(1-t)X_0+tX_1\)、监督速度 \(U=X_1-X_0\)，并独立采样 \(t\sim U[0,1]\)。

**公式与调用。** 平方回归目标及其在所有平方可积向量场中的最优解为

\[
\mathcal L(v)=\mathbb E\|v_t(X_t)-U\|^2,\qquad
v_t^*(x)=\mathbb E[U\mid X_t=x],
\]
\[
\mathcal L(v)=\mathbb E\|v_t(X_t)-v_t^*(X_t)\|^2
+\mathbb E\|U-v_t^*(X_t)\|^2.
\]

条件期望只在 \(dt\,p_t(dx)\) 几乎处处确定；有限网络类中的最优解还受表达能力限制。
对任意 \(\varphi\in C_c^\infty(\mathbb R^d)\)，直接沿插值路径求导得到

\[
\frac d{dt}\mathbb E\varphi(X_t)
=\mathbb E[\nabla\varphi(X_t)\cdot U]
=\mathbb E[\nabla\varphi(X_t)\cdot v_t^*(X_t)].
\]

这就是 \(\partial_t p_t+\nabla\cdot(p_tv_t^*)=0\) 的弱式，不要求先假设 \(p_t\) 有处处正的密度。

**成立条件。** 上述回归分解依赖平方可积性，弱式依赖可交换求导与期望；固定耦合和有界测试函数梯度给出可用控制。
由弱式进一步断言 ODE \(\dot Z_t=v_t^*(Z_t)\) 唯一地生成 \(p_t\)，仍需流的存在、唯一性及连续性方程唯一性。
例如空间全局 Lipschitz 常数随时间可积、速度有相应增长界时可调用经典 ODE 理论；奇异端点需另处理。

**反例与边界。** 端点相同而耦合不同，会改变中间分布、回归方差和速度场；独立耦合不自动是最优传输耦合。
监督样本沿直线移动，不推出边缘速度场的每条 ODE 轨迹也沿这些直线；同一 \(X_t\) 可对应多种端点对和速度。
CFM 原始损失的最小值可因条件方差而非零；小回归误差也不能绕过正则性和数值求解条件直接保证小生成误差。

**AI 计算意义。** 训练可直接采样 \(t,X_0,X_1\) 而不积分 ODE；生成仍需计入网络评估次数、步长和求解器误差。
对条件模型固定 \(c\) 后使用 \(\pi(\cdot,\cdot\mid c)\)。本卡固定耦合推导与 [S7](https://arxiv.org/html/2210.02747v2) §3 的条件／边缘向量场原则对应，未把该文正密度定理直接扩展到所有奇异情形。

## LK07 配对误差与分布及轨迹误差

**对象与目标。** 先选比较层级：同一输入与随机性耦合下的输出、条件分布，或整段轨迹联合分布。
令 \(X=G(Z,c),\ \widetilde X=\widetilde G(Z,c)\)，并确认两个程序对 \(Z\) 的消费顺序和含义确实对齐。

**公式与调用。** 若两分布在相同欧氏空间具有有限二阶矩，由 Wasserstein 定义直接得到

\[
W_2^2(P,Q)=\inf_{\gamma\in\Pi(P,Q)}
\int\|x-y\|^2\,d\gamma(x,y)
\le\mathbb E\|X-\widetilde X\|^2.
\]

右边采用当前配对构成的一个可行耦合；它是分布误差的上界，不是分布距离的恒等表达。
对共同条件分布 \(P_C\)，逐条件使用相同论证得

\[
\mathbb E_C W_2^2(P_{X\mid C},P_{\widetilde X\mid C})
\le\mathbb E\|X-\widetilde X\|^2,
\]

其中右边必须在相同 \(C\) 下配对。分布距离的依据见 [S8](https://arxiv.org/html/1803.00567v4) §2.3–2.4；这些上界由可行耦合直接推导。

**反例与边界。** 若 \(Z\sim\mathcal N(0,I_d)\)、\(X=Z,\widetilde X=-Z\)，则 \(P=Q\) 且 \(W_2=0\)，但配对均方误差为 \(4d\)。
若 \(C\in\{-1,1\}\) 等概率、\(X=C,\widetilde X=-C\)，无条件输出分布相同，条件输出却完全错配。
对两帧序列 \((Z,Z)\) 与 \((Z,-Z)\)，每帧边缘相同，跨帧相关性不同；逐帧分布分数不能独自检验视频运动一致性。

**成立条件。** 为整段序列选定统一的路径／序列代价，例如取 \(w_k>0\) 并使用 \(\sum_k w_k\|x_k-\widetilde x_k\|^2\)，再在该空间比较联合分布；若允许零权重，就无法区分只在那些位置不同的序列。
对连续轨迹，明确采用积分误差还是上确界误差，以及相应矩是否有限；边缘耦合未必能拼成保持时间依赖的轨迹耦合。
有限样本上的配对均值还包含估计误差；分布测度也不能自动替代条件正确性或任务表现。

**AI 计算意义。** 加速与蒸馏可分别报告配对保真、条件分布和任务质量；视频另测跨帧／轨迹量。
需要从局部更新误差推到最终误差时，结合 LK02 的状态传播或显式 Lipschitz 稳定性界，保留跨步依赖。
相同 seed 只标识一种实现相关的耦合，不能省略对实际噪声样本与调用顺序的检查。

## LK08 非光滑与离散机制

**对象与目标。** 区分连续但带折点的 ReLU／max、硬舍入／阈值的不连续映射，以及随机离散决策的期望目标。
先列出当前分支的激活模式、top-k 支持、路由、舍入阈值和停止条件。

**公式与调用。** 有限参数的 ReLU 网络对输入局部 Lipschitz，故在环境空间中几乎处处可微。
若 \(F\) 在路径附近局部 Lipschitz，\(\gamma:[0,1]\to\mathbb R^n\) 绝对连续，则 \(F\circ\gamma\) 绝对连续，因而

\[
F(\gamma(1))-F(\gamma(0))
=\int_0^1(F\circ\gamma)'(t)\,dt.
\]

只有在链式法则沿该路径几乎处处成立时，才把被积量写成 \(DF(\gamma(t))\gamma'(t)\)。
可直接使用的情形包括路径处于光滑区域，或分段光滑路径仅在有限时刻穿越分支边界。
环境空间零测的不可微集可能包含整段路径；不能仅凭 Rademacher 定理便断言路径上的环境 Jacobian 几乎处处存在。

**反例与边界。** 对 \(q(x)=\mathbf1_{x\ge0}\)，\(q'(x)=0\) 几乎处处，但 \(q(1)-q(-1)=1\)。
硬量化跨阈值的有限变化不能由普通导数积分恢复。若沿路径只有有限次跳变，正确拆分为光滑段积分与跳变量之和：

\[
\Delta F=\sum_j\int_{I_j}(F\circ\gamma)'(t)\,dt
+\sum_r\big[(F\circ\gamma)(t_r^+)-(F\circ\gamma)(t_r^-)\big].
\]

此式需各段绝对连续及相应单侧极限；端点值也按实际分支约定处理。
top-k 的索引、选中值及聚合结果可能有不同连续性，不能把“支持变化”统一当作同一种跳变。

**成立条件。** STE 用指定替代导数穿过硬算子，如把 \(q'\) 替换为 \(1\)；它定义训练更新规则，一般不是原硬前向目标的真梯度。
soft relaxation 改变前向目标；score 估计求随机决策期望的梯度；可微重参数化又有另一组条件，转 LK03。
例如取 \(\theta\in(0,1)\)，\(B_\theta=\mathbf1_{\varepsilon<\theta},\ \varepsilon\sim U(0,1)\) 的逐样本导数几乎处处为零，但 \(\frac d{d\theta}\mathbb E B_\theta=1\)；这里不能交换求导与期望。

**AI 计算意义。** 固定分支的 AD 可用于分支内敏感度；跨分支验收需记录触发率、跳幅和完整前向误差。
量化／稀疏训练同时说明训练用替代规则与部署用硬计算。STE 的原始说明见 [S3](https://arxiv.org/html/1308.3432v1) §4；路径公式与反例为直接推导。

## LK09 RL 与 Agent 交互

**对象与目标。** 先限定有限状态、有限动作、转移概率 \(P(s'\mid s,a)\)、有界期望奖励 \(r(s,a)\) 和折扣 \(0\le\gamma<1\)。
状态必须对奖励和未来转移充分；目标为 \(V^\pi(s)=\mathbb E^\pi[\sum_{k\ge0}\gamma^kr(S_k,A_k)\mid S_0=s]\)。

**公式与调用。** 策略评价与最优 Bellman 算子分别为

\[
(T^\pi V)(s)=\sum_a\pi(a\mid s)
\left[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right],
\]
\[
(TV)(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right].
\]

由概率权重归一化与 max 的非扩张性，直接得到

\[
\|TV-TW\|_\infty\le\gamma\|V-W\|_\infty,\qquad
\|V-V^*\|_\infty\le\frac{\|V-TV\|_\infty}{1-\gamma}.
\]

因此 \(T\) 有唯一不动点，精确 value iteration 收敛。若每步统一满足 \(\|\widetilde V_{k+1}-T\widetilde V_k\|_\infty\le\varepsilon\)，则 \(\limsup_{k\to\infty}\|\widetilde V_k-V^*\|_\infty\le\varepsilon/(1-\gamma)\)。

**成立条件。** 上界要求对全部状态的统一误差；训练数据上的平均拟合误差不直接满足它。
折扣为 \(1\)、无界奖励、连续状态或非平稳环境，需要重新建立所用空间、范数和收敛条件。
最优 Bellman 算子有 max，通常非线性；固定策略算子是仿射映射。

**反例与边界。** \(T\) 收缩不意味着神经网络参数优化或 \(\Pi T\) 收缩；投影／拟合可以放大误差，抽样和 off-policy 分布还会改变可识别范围。
部分观测时，同一观测可对应不同隐状态；在已知模型等条件下可使用历史条件下的信念状态，不能默认 RNN／LLM 摘要已经充分。
Agent 的实际状态可能包含模型记忆、环境／工具内部状态、未完成调用、预算及终止信息；可见对话只是其中的观测。

**AI 计算意义。** 区分策略输出、工具／环境执行、观测回传与学习更新，明确奖励发生时点、轨迹截断和终止规则。
比较 Agent 方法时把推理 token、工具调用、环境样本与长期回报一起统计；单步动作概率或价值拟合损失不能直接证明任务成功率。
有限折扣结论与近似边界对应 [S9](https://web.mit.edu/dimitrib/www/AbstractDP_ED3_TEXT_2021.pdf) §2.1、§2.3.1；此处采用奖励最大化约定并给出直接推导。

## LK10 图与集合上的 AI

**对象与目标。** 给定图 \(G=(V,E)\)、节点特征 \(h_i\) 和边特征 \(e_{ij}\)，明确边方向、邻居集合及共享参数。
一层 MPNN 可写为

\[
m_i=\sum_{j\in\mathcal N(i)}M_\theta(h_i,h_j,e_{ij}),\qquad
h_i'=U_\theta(h_i,m_i).
\]

**公式与调用。** 若消息／更新函数跨节点共享，聚合对邻居顺序不变，并同步置换节点、边和所有附加特征，则

\[
F(PH,PAP^{\mathsf T})=PF(H,A).
\]

这里省略了同步置换后的边特征参数。该等变性可按层归纳；再用 \(R(H)=\rho(\sum_i\phi(h_i))\) 等不变读出得到图／集合级不变输出。
对无向非负权图，取 \(A=A^{\mathsf T}\)、各元素 \(A_{ij}\ge0\)，令 \(D_{ii}=\sum_jA_{ij}\)、\(L=D-A\)，则

\[
x^{\mathsf T}Lx=\tfrac12\sum_{i,j}A_{ij}(x_i-x_j)^2\ge0.
\]

因此可用对称半正定谱结构分析平滑和图滤波；该恒等式由展开平方直接得到。

**成立条件。** 归一化 Laplacian \(I-D^{-1/2}AD^{-1/2}\) 需正度数，孤立点须明确另定约定。
带方向或负权的图不自动继承上述对称性与半正定性。节点编号特征、位置编码和随机采样也须按目标群作用一起处理。
点云的置换等变不包含旋转／平移等变；后者还需几何特征与相应变换律。

**反例与边界。** 给每个存储位置配不同且不随置换移动的权重，会破坏置换等变；对邻居按任意索引拼接也无此保证。
等变性只约束重标号行为，不独自保证区分所有非同构图、建模任意长程关系或保持深层表示可分性。

**AI 计算意义。** 若单边消息计算成本为 \(C_M\)、单节点更新成本为 \(C_U\)，消息与更新的计算量约为 \(|E|C_M+|V|C_U\)，再计入聚合成本。
对 \(d\) 维消息，求和聚合通常为 \(O(|E|d)\)；每边稠密矩阵、全连接边、动态图邻居搜索及 scatter／通信成本按实际机制计算。
预先可用的稀疏边支持才能直接减少执行域；学习 top-k 支持的选择成本与跨分支梯度转 LK08。
MPNN 定义与不变读出依据见 [S10](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf) §2；等变归纳、Laplacian 条件和成本表达为本卡直接推导。

## 来源与已读定位

只把下表列出的章节／定理作为已读依据；链接指向原始论文、作者公开书稿或正式论文库。
表内定位不是全文覆盖声明；本文件的条件化改写与反例按正文标为直接推导。

| 来源 | 实际核对的位置 | 支持范围与边界 |
|---|---|---|
| S1 — [Jacot、Gabriel、Hongler：Neural Tangent Kernel，2018](https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf) | §2 参数化与有限经验输入测度；§4 NTK 定义；§4.2 Theorem 2；§5 平方损失 | 有限宽时核依赖参数；冻结极限有初始化、缩放、正则性和有限时间条件，不推广为任意架构／优化器保证。 |
| S2 — [Kingma、Ba：Adam，arXiv v5，2015](https://pure.uva.nl/ws/files/326512947/1412.6980v5.pdf) | Algorithm 1（论文 p.2）；§2 矩估计与偏差修正 | 支持有状态更新；本卡取 Algorithm 1 的固定 \(\beta_1,\beta_2\) 版本，不援用其一般收敛主张。 |
| S3 — [Bengio、Léonard、Courville：Estimating or Propagating Gradients Through Stochastic Neurons，2013](https://arxiv.org/html/1308.3432v1) | §3 Theorem 1 的 Bernoulli 估计及基线；§4 STE | 区分期望梯度估计与启发式替代导数；本卡一般 score 公式由密度乘积求导另行推导。 |
| S4 — [Kingma、Welling：Auto-Encoding Variational Bayes](https://arxiv.org/html/1312.6114v11) | §2.3 Eq. (4)–(8)；§2.4 重参数化及 Gaussian 示例 | 支持参数无关噪声经可微映射采样；交换求导的控制条件和移动支持反例由本卡补充。 |
| S5 — [Pavliotis：Stochastic Processes and Applications，2015-11-11 draft](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf) | §2.2 Definition 2.3；Eq. (2.15)–(2.17) | 支持历史条件下的 Markov 定义与 Chapman–Kolmogorov 核合成；仅用于 AI 状态／采样过程建模。 |
| S6 — [Song 等：Score-Based Generative Modeling through SDEs](https://arxiv.org/html/2011.13456v2) | §3.2 Eq. (6)；§3.3 Eq. (7)；§4.3 Eq. (13)；Appendix D.1 Eq. (35)–(37) | 支持倒向 SDE 与概率流边缘关系；本卡明确限制空间常数标量扩散，不声称联合路径一致。 |
| S7 — [Lipman 等：Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2) | §2 流定义；§3.1–3.2 Eq. (6)–(9)、Theorem 1–2 | 原文条件／边缘向量场及损失关系；本卡任意固定耦合的回归分解和弱式单独推导，经典流需额外正则性。 |
| S8 — [Peyré、Cuturi：Computational Optimal Transport](https://arxiv.org/html/1803.00567v4) | §2.3 Remark 2.13–2.14；§2.4 Proposition 2.2 | 支持耦合约束、代价最小化与 Wasserstein 定义；配对上界及条件／视频反例为直接推导。 |
| S9 — [Bertsekas：Abstract Dynamic Programming，3rd ed.](https://web.mit.edu/dimitrib/www/AbstractDP_ED3_TEXT_2021.pdf) | §2.1 Proposition 2.1.1；§2.3.1 Proposition 2.3.2 与 Example 2.3.1 | 支持收缩、不动点／残差界及逼近放大边界；仅将有限折扣 MDP 结论用于所声明的 RL 模型。 |
| S10 — [Gilmer 等：Neural Message Passing，2017](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf) | §2 Eq. (1)–(3) 与 permutation-invariant readout | 仅提取通用图学习架构；不扩展论文中的独立领域应用，谱条件与执行成本由本卡另行推导。 |
