# 从整步有限响应到便宜的 flow 校正：先测可修复方向，再选择目标

2026-10-26；独立讨论 Agent `independent_e0_audit`。DISCOVERY / CONDITIONAL_THEOREM / SOURCE_RECONSTRUCTION / EXISTING_SCALAR_ANALYSIS；无正式方法 endorsement。

**新的研究对象：** 将整网误差改写成“dense 整步映射对实际状态差的响应 + candidate 在该状态上的注入”，再问已经算出的少数 flow/state 方向覆盖多少可修复误差。这既给出比点积符号更强的有限判别，也允许一种无需新增 FFN GEMM 的输出校准种子。现有两来源标量已足以证明一个具体边界：source12 的 step1，甚至允许沿真实 teacher 差方向任意反馈，理想下一状态误差的最小能量仍为 555.174，高于该步输入差能量 474.421。纯沿偏差方向的校正无法使这个状态对在这一整步内收缩。

执行边界：本次只读源码、原始论文/作者代码和既有 JSON，做普通 CPU 标量算术；没有模型/GPU/新拟合、科学张量或测试。下文校准、有限干预和伴随计算都是未来另行授权的探索种子，旧文件与冻结回执保持原字节。

## 1. 三个整网 flow 足以分开有限状态响应与当前注入

固定一个实际 step、相同 condition/time/expert，省略下标。令 teacher/student 在 step_pre 后的实际 BF16 latent 为 zD、zS；δ=zS−zD。原模型及候选的完整 40 层加 head 输出为 vD、vS。本次完成的配对记录已有 vD(zD)、vS(zS)。额外观察 **整个 dense 网络**的 vD(zS) 后，定义

\[
R=v_D(z_S)-v_D(z_D),\quad e=v_S(z_S)-v_D(z_S),\quad d=R+e.
\]

R 是这对有限位移上的 dense 函数响应，e 是同一 student latent 上的整网模型差；它们都无需 QDQ 可微或假设小扰动。现有逐 FFN shadow 评估只替换一个局部算子、输出被丢弃，不能提供 vD(zS)。

对实际正步长 h，令

\[
P=\delta-hR=F_D(z_S)-F_D(z_D),\qquad F_D(z)=z-hv_D(z).
\]

那么理想实数整步误差正好是

\[
\delta'_{ideal}=P-he,\qquad
\|\delta'_{ideal}\|^2=\|P\|^2+h^2\|e\|^2-2h\langle P,e\rangle. \tag{1}
\]

这比只查 `⟨δ,R⟩` 和 `⟨δ,e⟩` 的符号更有辨识力：

\[
G_D=\frac{\|P\|^2}{\|\delta\|^2}
=1-2h\frac{\langle\delta,R\rangle}{\|\delta\|^2}
+h^2\frac{\|R\|^2}{\|\delta\|^2}. \tag{2}
\]

δ·R<0 足以推出 GD>1，但 δ·R>0 也可以因切向响应或整步过冲产生 GD>1。candidate 相对于“当前 student 状态整步改用 dense”的理想净效应是 `h²||e||²−2h⟨P,e⟩`，而非仅 h²||e||²。即使 e 非零，该净效应也可能为负，说明这个具体 step 的候选误差部分抵消已有状态差；这不替更早的候选替换开脱，也不等于视频质量改善。

建议未来记录的最小充分标量为 `||δ||²,||R||²,||e||²,⟨δ,R⟩,⟨δ,e⟩,⟨R,e⟩,h`。于是 `⟨P,e⟩=⟨δ,e⟩−h⟨R,e⟩`、`||d||²=||R||²+||e||²+2⟨R,e⟩` 均可重建并对原 TRACE 做 signed 闭合。step0 的 δ=R=0 是 neutral equality control；GD 在此应为 undefined，而不是 0 或 1。

**native 边界还有一个便宜的精确版本。** 取得 vD(zS) 后，用原 scheduler 的 FP32 `z−sigma*v`、再加 `sigma_next*v`、cast 回 BF16 的真实顺序，在 clone 上做一次不写回轨迹的 shadow update，无需再做模型 NFE。记这些真实有限算术映射为 TD、TS，则

\[
P_{nat}=T_D(z_S)-T_D(z_D),\quad
N_{nat}=T_S(z_S)-T_D(z_S),\quad
\delta'_{actual}=P_{nat}+N_{nat}. \tag{3}
\]

分别保存两项能量及 `2⟨Pnat,Nnat⟩`，即能检查真实整步差。只记录式(1)时，GD 的对象是理想 Euler map，原生 step_post defect η 仍要单列。式(3)把同一 dense shadow update 用作共同参照，避免把 BF16 cast 的非线性全塞进一个推测的乘法 gain。

这是**给定实际观察状态对上的有限测试**。它识别此 step 的函数差和响应，不给出全局 Jacobian 稳定性、最初致错层、未来切换策略的长期回报或视频失败主因。不同的中间参考可改变分解项；dense 参照之所以有用，是它对应一个清晰、可执行且与历史强控制一致的整步反事实。

## 2. 一个无需拟合的新投影界：沿真实偏差方向能修多少？

先用现有记录问最有利的理想问题：假设可免费知道 δ，允许 flow 加任意 `u=γδ`。δ≠0 时，分解

\[
a=\frac{\langle\delta,d\rangle}{\|\delta\|^2},\quad d=a\delta+d_\perp,
\]
\[
\delta'_{ideal}(\gamma)=[1-h(a+\gamma)]\delta-hd_\perp.
\]

所以

\[
\gamma^*=h^{-1}-a,\qquad
\min_\gamma\|\delta'_{ideal}(\gamma)\|^2
=h^2\left(\|d\|^2-\frac{\langle\delta,d\rangle^2}{\|\delta\|^2}\right). \tag{4}
\]

这是标量恒等式，不需要控制实验、参数拟合或 Jacobian。仅把负 radial gain 消成 0，即 γ=−a，还会留下 `||δ||²+h²||d_perp||²`；要贴回 teacher，单看反向点积消失并不足够。

以下用两份实际 TRACE 的原标量代入；每行是各自**未修改轨迹**的一步 oracle 下界，不是将校正连跑四步的结果。

| source / seed | step | 已有 `||δ||²` | a | γ* | 最小理想 next SSE | 原 ideal next SSE 的剩余比例 |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 42 | 1 | 1731.584 | −13.49381 | 23.09381 | 1096.127 | 9.860% |
| 0 / 42 | 2 | 11125.326 | −5.67238 | 10.47238 | 3432.117 | 6.087% |
| 0 / 42 | 3 | 56392.662 | −2.82876 | 4.42876 | 22454.665 | 4.940% |
| 12 / 54 | 1 | 474.421 | −13.38705 | 22.98705 | **555.174** | 16.950% |
| 12 / 54 | 2 | 3283.147 | −5.43301 | 10.23301 | 2352.697 | 13.620% |
| 12 / 54 | 3 | 17278.609 | −2.68542 | 4.28542 | 13489.857 | 9.815% |

source12 step1 的切向下界已经超过输入差能量，即使真实 δ 可用也无法在此理想整步内收缩。另一方面，多数行的 ideal next SSE 有很大 radial 部分，提示“方向可修复性”值得研究。两来源 γ* 数值相近是已暴露两案例上的描述，不是跨来源可迁移反馈的证据。

δ 是 teacher 与 student 的差，部署时没有免费 teacher。沿 δ 的 radial control 与沿 zS、vS 的缩放是不同的控制族，式(4)不限制所有 flow 缩放、非线性校正或全视频策略，也不宣称这个 oracle 下界就是可达到的 native BF16 结果。

独立逐步算术见[同名 JSON](/home/xiang/projects/FFNDS/research/method_discussion/discoveries/E0/independent-whole-flow-control-and-value-geometry-20261026_v7.json)，SHA `fcee96cc4aa6a374c8bc6fb0833e7f36f689fa7a73b5df858a3946376b7eba9a`。它绑定两份原 TRACE 的精确 SHA，列全6个非零δ步骤、平行/正交能量、式(4)及原生 dtype。原更新 step1–3 的 `||η||²` 分别为 source0 的7.74673/5.22115/7.05916、source12 的7.70681/5.08822/6.33084；另以 `(actual−ideal−||η||²)/2` 恢复原 ideal–defect 点积。**这些 η 只属于未校正的原更新**；改变 flow 后的 BF16 defect 尚未测量，不能把原 η 直接加到 oracle 下界上来推断 native 控制结果。

## 3. 把可部署的低成本校正先化成一个投影问题

一个可研究的输出控制族是 `u=Bθ`，B 的少数列取自已经计算的 student 当前 flow、当前 latent、上一 flow，或 16 个 latent channel 的常量偏置方向。θ 可以按四个 step 分开；step2 切到 low-noise expert，上一 flow 的有效性需要单独检验。

希望单步回到 teacher 的理想目标为

\[
y=\delta/h-d=P/h-e,\qquad
\min_\theta\|\delta-h(d+B\theta)\|^2
=h^2\big(\|y\|^2-b^TG^\dagger b\big),
\quad G=B^TB,\ b=B^Ty. \tag{5}
\]

式(5)是已知最小二乘/正交投影原理在这个状态目标上的具体应用，不作新算法发明声明。它提供一个先于训练的**可修复性上限**：若这组在线可得方向的 oracle 投影都覆盖不了误差，后续把同一组方向的标量系数拟合得更精确也无法达到该单步目标；若覆盖较高，才值得研究按 held-out source/prompt/seed 校准的固定系数或便宜预测器。

特别地，纯缩放 student flow 的 `u=αvS`，其 SSE 对 α 在 0 处的导数是

\[
\left.\frac{d}{d\alpha}\|\delta-h(d+\alpha v_S)\|^2\right|_0
=-2h\langle\delta-hd,v_S\rangle. \tag{6}
\]

目前的 `⟨δ,d⟩<0` 并不告诉式(6)的符号。故不能由“流差反向”直接决定应增大还是减小 vS。未来 whole-flow capture 同时保存 B 的 Gram、`B^Tδ`、`B^Td`，只需少量归约标量，就能得到这个控制族的上限；当前 JSON 缺少这些点积，不能声称已经测试了缩放校正。

这给出与现有方法不同的两项预测：

- **系统性 gain/bias 机制：** 小 B 的投影覆盖高；独立 source/seed 的最优方向和系数相近；固定或小型校准能在不新增 FFN NFE 下带来收益。
- **任务创新/切向信息丢失机制：** 原始局部 SSE 虽小，y 在 B 外仍有显著分量；修正均值和能量后，运动、条件一致性等任务差异仍保留。此时更丰富的读出或保留必要计算比继续调同一缩放系数更有价值。

这两者可以共存。把整网 flow 调整放在 head 之后，在线不再支付 W1/GELU/W2，但也不自动减少它们；它只可能帮助保住已有压缩带来的收益。当前 latent 有 2,026,752 个元素，一个额外 FP32 历史 flow 为 8,107,008 字节，约 7.73 MiB。数个向量融合是 O(nr) 的内存/点算费用，实际延迟仍需未来测量。teacher δ 和离线 vD(zS) 的获取计入校准成本。即使闭式求 θ、不更新 backbone，也会产生新的校准参数/策略；“无骨干训练”与“没有拟合、没有新参数”需要分开。

## 4. 用下游价值选择误差方向，保留跨步 cross

另一个自由度是改变校准目标。若在某个选定、可微的参考 rollout 上研究小控制 u，设 Ak 为**被改变的参考更新映射**的 Jacobian，Tk 为从 step k+1 传到终点的 Jacobian 乘积。则

\[
\Delta z_4\simeq-\sum_{k=0}^3h_kT_ku_k. \tag{7}
\]

对终端目标 `L=1/2||z4−z4D||_W²`，伴随变量可定义为 `λ4=W(z4−z4D)`、`λk=Ak^Tλk+1`，从而一阶控制价值是 `−h_k⟨λk+1,u_k⟩`。这里预测 candidate 策略的控制变化，应围绕 candidate 的实际参考 rollout 求 Ak；围绕 dense 求导回答的是另一种近似问题，不能悄悄互换。

同理，一组小的算子误差 e 经参考后缀传递后的二阶能量包含

\[
\left\|\sum_kh_kT_ke_k\right\|_W^2
=\sum_{j,k}h_jh_ke_j^TT_j^TWT_ke_k. \tag{8}
\]

只给每步或每 FFN 一个正标量重要性，会漏掉 j≠k 的 signed cross。最简单反例是 T0=T1=I、相同步长、e0=a、e1=−a：终点一阶误差相消；只消掉其中一处局部误差，反而破坏相消。另一个例子是后缀 T=diag(100,1)：局部误差(.02,0)的 SSE 仅 .0004，却造成终端能量4；局部误差(0,1)的 SSE 为1，终端能量仍为1。局部最小 SSE 与下游价值的排序可以相反。

这使 downstream-value-weighted 重构成为有依据的训练/校准种子，而不是现有局部目标的充分性声明。昂贵处在离线：获取真实后缀敏感度或 reward/decoder 导数需要额外前后向、存储或有限干预。当前 native QDQ/BF16 路径并非光滑 ODE；一个 FP32 surrogate 的 JVP/VJP 要通过真实有限响应验证后才能代表原生作用。式(1)–(3)的三-flow 有限测试正好可以先做这种机制检查。当前全帧像素指标、Farneback、Laplacian 也没有自动给出可靠的可微语义 W。

## 5. 原论文与实际代码怎样改变了这个解释

本次按 pinned AutoResearcher/WARD `8b88a52d894568dc303db727decfbf13b5fabf3b` 独立检索。搜索从“correlated quantization noise / trajectory calibration / flow drift / solver statistics / adjoint control”展开，再沿 Q-Drift→PTQD/D²-DPM、DPM-Solver-v3→exponential integrator、Adjoint Matching→continuous/discrete adjoint、Bespoke→scale-time transformation，以及近期 trajectory-aware compression 追踪。以下只把实际亲读的原文/代码作为依据。

| 原始来源与亲读部位 | 最接近的数学/动作 | 对 Wan 问题的关系与差别 |
|---|---|---|
| [PTQD，NeurIPS 2023 原文 §4，Eq7–12](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)；作者 `analyze_error.py`、`ddim.py` | 输出相关误差的线性系数、channel bias、方差校准；代码以 linregress 求 slope，推理先减 bias 再除(1+k)，k 截到非负 | gain/bias 是已知原理。原文对 deterministic σt=0 另有方差吸收的限制；不能把 Wan 的确定性四步结构误写成有现成 stochastic variance budget。 |
| [Q-Drift，arXiv:2603.18095v1 §4、App.A.1/B.3](https://arxiv.org/html/2603.18095v1)；作者 flow scheduler 与 FLUX-schnell collector | 校准条件残差方差，使 dt 乘(1+c)；可选条件均值校正 | 很强的便宜竞争方案；目标是 marginal quality，flow 扩展原文明确为 heuristic。该论文还报告条件均值缩减可能损伤质量，提醒“最佳局部均方估计”与生成分布目标有张力。 |
| [DPM-Solver-v3，arXiv:2310.13268v3 §3.1–3.4](https://arxiv.org/html/2310.13268v3)；作者 EMS 与 sampler 实现 | 以 l,s,b 改变 ODE 参数化，降低离散截断误差；multistep/predictor-corrector 重用输出 | 共享“少量离线统计量 + 在线便宜向量计算”。其目标是同一 ODE 的离散化，Wan 这里还改变了 native flow 函数。代码的 EMS 获取确实使用 JVP/total derivative，离线并非只读几个现成均值。 |
| [Bespoke Solvers，arXiv:2310.19075v1 §2.1–2.3、App.E/F](https://arxiv.org/html/2310.19075v1) | RK1 的 `xnext=a_i x+b_i v(x,t_i)`，学习 scale/time 及传播加权的局部截断误差上界 | 与式(5)的低维 state/flow 组合非常接近，属于已知构件。原文的 Gaussian-path 等价以理想向量场为条件；改变四步蒸馏网络的时间坐标/专家切换不自动继承它。Bespoke 的 Lipschitz 常数还被当作超参数，不能当成已测 Wan 增益。 |
| [Adjoint Matching，arXiv:2409.08861v3 §4–5、Alg.1](https://arxiv.org/html/2409.08861v3) | terminal reward 的 SOC、lean adjoint 与回归控制目标 | 为式(7)的价值方向提供相邻机制；原方法训练新控制/向量场并使用特定 memoryless 训练噪声日程。这里的确定性 native finite test 或少参数闭式校准不因此获得它的分布保证。 |
| [Sampling-Aware Quantization，CVPR 2026 §4.1–4.2](https://openaccess.thecvf.com/content/CVPR2026/papers/Zeng_Sampling-Aware_Quantization_for_Diffusion_Models_CVPR_2026_paper.pdf)；[作者仓库](https://github.com/TaylorJocelyn/Sampling-aware-Quantization) | 用 higher-order teacher 中间方向作 first-order quantized 重构目标 | 共享改变目标而非只追同点 SSE 的思路。它校准量化参数，且两轨/中间时间不同；Eq(1)则固定 Wan 实际状态及同一步。亲读仓库当时只有 README，实际实现尚未取得。 |
| [TamedFlow，arXiv:2610.26025v2 §3.2–3.4、App.K](https://arxiv.org/html/2610.26025v2) | velocity alignment head、teacher-state anchoring、free-running endpoint loss与 teacher rescue | 直接的 compressed-student trajectory 竞争机制。其 rollout bound 是光滑假设下的上界，正文也区分 teacher-state velocity 与实际 rollout endpoint；它需要训练和 teacher fallback，成本/机制都不等同于固定 flow gain。未亲读作者代码。 |

**作者代码的实际细节尤为关键。** Q-Drift 的 flow 实现能够启用 conditional-mean bias，不能把所有配置都称为纯缩放。FLUX-schnell collector 虽在文件头及407行注释声称沿 quantized baseline，实际397/400行在同一个输入上各算一次原模型与量化模型，409行以 `scheduler.step(fp16_noise,...)` 推进；因此这份冻结源码在 dense 轨迹上校准。注释里的 student-trajectory 宣称不能代替实际调用。这也解释了为什么该实现没有回答当前缺失的 vD(zS) 整网有限响应问题。[实际 collector](https://github.com/sooyoung-ryu/Q-Drift/blob/118ea3ba35f55be15a70030dd2b7baaf91228f5d/experiments/svdquant/flux.1-schnell_w3a4/scripts/collect_statistics.py#L397)

个人读取的作者代码身份如下；通过只读 GitHub API/raw HTTP 获取，不 clone、不执行代码、不安装依赖。SHA256 是亲读文件的字节哈希：

| 仓库 commit | 文件 | SHA256 |
|---|---|---|
| PTQD `6b190f94b459ca869bf165509454c8676c3a62be` | `quant_scripts/analyze_error.py` | `771e22aba4c9fe79c4f242825206818cdfb164f1679f9e4a1bdda6b4fe183c58` |
| 同上 | `ldm/models/diffusion/ddim.py` | `a242c5ff253f928e901d4909df53ce6a605c947ce337e0ce4e3df3325e692233` |
| Q-Drift `118ea3ba35f55be15a70030dd2b7baaf91228f5d` | `schedulers/flow_match_euler_qdrift.py` | `ce0e33ea38f7c08e2ddc9783b7aa2b4a2358f182cc4e729b59aff4e46919fc79` |
| 同上 | `scripts/fit_scalar_gaussian.py` | `d4034f6c6e02bdfaf76be3a27163d7c02558a3bfcb39c65b1e8e362dd8d533aa` |
| 同上 | `experiments/svdquant/flux.1-schnell_w3a4/scripts/collect_statistics.py` | `045043eceea0e8f50b8516f797d39f4572e88c4e22b90934d924ae5ec6e4444c` |
| 同上 | `experiments/svdquant/flux.1-schnell_w3a4/scripts/evaluate.py` | `376382d36f698214bea2205f30001f5daad1ee838c595ef246940314d4997f8a` |
| DPM-Solver-v3 `b615d707e9df512d309c4e59ce228283a608eb8b` | `compute_EMS_scoresde.py` | `048c9e4ba7203cf9ff54c5b7463c60373c62b805b2e8a68f6bfd459247e5a1e3` |
| 同上 | `dpm_solver_v3.py` | `fe53d8f7aef836f41b1c868f2ae7a95c19d679e991d94614547bec3b21db8e95` |

检索后的解释改变是：把“修正流差的幅值”细分为条件偏置校正、marginal drift 校正、solver 参数化与 terminal-value 控制。它们可以使用相似的线性向量操作，却有不同目标、信息和反预测。D²-DPM 原论文/代码未亲读，本文只把其与 Q-Drift 的关系作为后者报告，不作完整等价或优劣判断；Bespoke/Adjoint/TamedFlow 的作者实现也未取得。没有全面新颖性保证或跨模型实证转移主张。

## 6. 最便宜的下一项判别及已有证据绑定

当前已完成的式(4)是第一个无需新运行的家族边界。下一项有信息的未来观察是：在保持原四步 teacher/student 全部历史 latent hash 的 replay 中，取得每步 vD(zS)，step0 做中性一致性，其余步骤记录式(1)–(3)及小 B 的 Gram。这比直接训练修复 head 更便宜，也能在同一轮区分：

1. GD 接近/小于1、candidate 净注入为正：优先研究当前状态上的模型缺陷与其可修复方向。
2. GD>1、candidate 净注入很小或为负：dense 的实际有限状态响应对该步扩大偏差更关键；重新贴轨或改变校准目标有独立意义。
3. 两项均大、cross 强：保留联合状态/模型解释，并测式(5)可用方向的覆盖。

这些是条件预测，不是本轮已经观察到的结论。四步每步的三-flow 账本会识别各观察状态上的响应；要判定某个修复策略的长期质量仍需其真实受扰 rollout。时空/条件 held-out 应以独立 prompt/source/seed 为单位；同一 clip 的数百万坐标不是同样数量的独立校准样本。若只改善 pooled moment、却损害动作或 I2V 条件响应，应以完整功能目标评价，而不以配对像素更远或更近单独决定质量。

一个基本反例说明目标差别：X~N(0,I)、Q 为非平凡正交变换，X 与 QX 的边缘分布和所有边缘矩一致，固定 seed 的配对位置仍可相差很大；再令两个等概率 condition 的 teacher 输出均值为±m、student 交换这两个 condition，pooled 分布仍可完全相同，条件功能却错置。统计矩校准可以是有效工具，语义/条件保持需要独立观测。

本轮从已完成 [两来源320-site报告](/home/xiang/projects/FFNDS/research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md) 出发；其结果 JSON SHA `1f1087d2d7fa111933fe2ce308ecbf4a9b7bd19f205d821973411a29115e16b2`。实际复算的原 TRACE 仍是 HF derived 相对根 `e0_paired_full_depth_20261026/v3_attempt1/dc_source00/TRACE.json`，SHA `26789cad90a54019df901838d6354fad40b229d12b652b9afc5ec4bd180c22c8`，以及 `dc_source12/TRACE.json`，SHA `fb486649c7d51c65a5abc73f9e39d05e502d1e79272d305057cb3e1f844723db`；源0/12实际 seeds 为42/54。协议 SHA `3027bc17588efdcdab32aacc5084cca6705b58d76f75529be43edeeda1b7bcc5`。它们已经证明318/320个 student-input 局部绝对误差更小而实际轨迹差增长；本文没有把这个现象重命名为某个已证实主因。

Graph 定位为 `F03/F09→F12→F13/G21→whole flow→scheduler→next latent` 与 `P23/P24`；校正方向及价值加权目标为 conditional/posthoc 的 OPEN 边。相关 `D09/P25` 仍要求独立完整视频功能证据。既有 strict/core PASS 是原始身份与计量闭合，正式 VBench/human acceptance 仍未执行；本发现未改变研究阶段、方法支持标签或执行范围。
