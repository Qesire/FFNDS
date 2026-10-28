# M-B 第四轮独立评议：从 B-QUOTIENT 回应到双轴误差分解与零额外算子偏置校正

> 2026-10-28 · FOURTH INDEPENDENT REVIEW / ALG + REPO-SRC + OBS-DATA + PRIMARY-LIT + OPEN。  
> **被评议对象**：[RESPONSE_TO_SECOND_AGENT_20261028.md](RESPONSE_TO_SECOND_AGENT_20261028.md)（第三轮回应，152行）；前稿：[EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md](EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md)。  
> 本轮核实了 [tools/compact6016_adapter.py](../../../tools/compact6016_adapter.py) 的 **92–137 行**、[A0 audit JSON](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.json)、[pilot driver](../../../tools/e0_compact6016_recovery_pilot.py)、[pilot report](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)、[A0 配置](../../../config/e0_compact6016_recovery_pilot_20261028_v1.json)。没有新学生执行、GPU、native 数值实验、四臂生成或仓库权威方法改写。  
> **方法判定**：**CONDITIONAL_CONTINUE**。同意第三轮回应的正确主张，采纳其 B-QUOTIENT 为同成本候选；首先用更强的已有代码诊断区分两种独立的中心化操作，并加入一个更便宜、已有成熟论文先例的 folded bias correction 对照。

## 0. 最重要的新发现：A0 已报的 mean fraction 不是 B-QUOTIENT 的规范误差

第三轮回应提出训练：

$$
\mathcal L_{\mathrm{quot}}(\phi)=\mathbb E\|P_D\,D_{g_s}(\hat y_\phi-y)\|^2,\qquad
P_D=I_D-\frac{\mathbf1_D\mathbf1_D^\top}{D}.
$$

这一目标的 gauge 代数**正确**。但是仓库已有的 A0 pilot 报告 **“source-row mean-error fraction”** 不度量这个通道方向。

[真实指标实现](../../../tools/compact6016_adapter.py) 116–136行明确先设置 `error64 = pre_error.double()`，然后：

~~~python
mean_error = error64.mean(0, keepdim=True)   # mean over 256 saved rows
mean_sse = float(len(error64) * mean_error.square().sum())
centered_sse = float((error64 - mean_error).square().sum())
~~~

$pre_error$ 形状为 $R\times D$（此 pilot $R=256,D=5120$）。`mean(0)` 是**沿 token/样本行平均**，最终是 $1\times5120$ 向量，**不是** B-QUOTIENT 要求的每个 token 内通道平均 `mean(1,keepdim=True)`。

原生 [A0 结果表](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md) 的 energy94 四位置最终 heldout（已暴露开发来源）的 `source_row_mean_error_fraction` 分别为：

| step / block | A0 energy94 gated relative SSE | A0 已存行均值误差份额 | 是否可作为 gauge 误差份额 |
|---|---:|---:|---|
| 0 / 0 | 2.349% | 3.393% | 不可 |
| 0 / 39 | 1.061% | 22.257% | 不可 |
| 3 / 0 | 2.546% | 13.396% | 不可 |
| 3 / 39 | 2.045% | 16.823% | 不可 |

这里的数据是**同源等权的 source 级 ratio 平均**，不是每行独立样本或整个 $M=31668$ token 场；此表仅已完成的 A0 student teacher-state local diagnostic，不能推断视频提升。

**直接反例**：$E_{p,c}=v_c$ 且 $\sum_c v_c=0$：行均值能量份额100%，通道 gauge 份额0%；$E_{p,c}=a_p$ 且 $\sum_p a_p=0$：行均值份额0%，通道 gauge 份额100%。两个指标在数值上彼此不能推导。

## 1. 双轴四正交分量：修正评议的最小完整计算对象

对每个 source、step、block 的 matched teacher/student postgate 误差 $E\in\mathbb R^{R\times D}$，令：

$$
P_R=\frac{\mathbf1_R\mathbf1_R^\top}{R},\quad
P_C=\frac{\mathbf1_D\mathbf1_D^\top}{D},\quad
R_R=I_R-P_R,\quad R_C=I_D-P_C.
$$

则精确直和：

$$
\boxed{E=P_REP_C+P_RER_C+R_REP_C+R_RER_C.}
$$

四部分 Frobenius 正交，因而误差平方和**严格相加**。

| 计算分量 | token 行结构 | 通道结构 | gauge / 可利用机制 |
|---|---|---|---|
| $P_REP_C$ | 行常量 | 通道常量 | **严格 gauge**；无需监督 |
| $R_REP_C$ | 行变化 | 每行通道常量 | **严格 gauge**；无需监督 |
| $P_RER_C$ | 行常量 | 通道非平凡方向 | **可观测偏置**；有机会用原 W2 bias 修正 |
| $R_RER_C$ | 行变化 | 通道非平凡方向 | **可观测剩余函数**；需要 FFN student/后缀建模 |

第三轮 B-QUOTIENT 关注的份额是：

$$
\eta_{\rm gauge}
=\frac{\|E P_C\|_F^2}{\|E\|_F^2}
=\frac{\|P_R E P_C\|_F^2+\|R_R E P_C\|_F^2}{\|E\|_F^2}.
$$

现有 A0 代码报的是：

$$
\eta_{\rm row}
=\frac{\|P_RE\|_F^2}{\|E\|_F^2}
=\frac{\|P_REP_C\|_F^2+\|P_RER_C\|_F^2}{\|E\|_F^2}.
$$

因此**不能从已有 JSON 的 $\eta_{\rm row}$ 恢复 $\eta_{\rm gauge}$ 或 4 项比例**；必须读取/重放实际 $R\times5120$ 输出和对应 gate。仅存于 JSON 的 per-source SSE 和 mean_SSE 标量不包含所需向量方向、交叉项。可以复用已存 CAL24 V1/Y/gate、A0 输出 packs，在已授权范围内重放**学生**而不必重新生成 Dense videos；如果已有完整配对输出缓存，先直接离线算四项。

### 1.1 先计算 B-QUOTIENT 最大直接省去的监督误差份额

记 $E_{p,:}=D_g(\hat y_p-y_p)$，则对同一个 teacher-energy 分母：

$$
\boxed{\mathcal L_{\rm gated}
=\mathcal L_{\rm quotient}
+\frac{\|EP_C\|_F^2}{\|D_gY\|_F^2}.}
$$

这是**固定模型、固定数据上的损失精确分解**。如果规范份额在当前 A0 已非常小，立即重训练 quotient 的预期机制收益有限；若份额明显，才有值得对比的训练信号。同时训练改变参数，因此这不是新最优 Student 对旧模型误差的简单数值上界，更不是视频质量上界。

对通道独立、各向同性误差，仅凭维度会预测 $\eta_{\rm gauge}\approx 1/5120=0.0195\%$；Wan 的真实误差高度相关，这仅是**对照零假设，不是观测**。建议同时保存 $p50/p90/max$、source 级值及四项比值，禁止把256个采样 token 当独立来源。

## 2. 比 B-QUOTIENT 先更省训练的机制：folded bias residual correction

原 A0 student 的 $b_2$ **固定为原始值**：[adapter](../../../tools/compact6016_adapter.py) 第37–52行，优化器仅 rank32 LoRA 两矩阵，第55–89行。因此 token 行方向的 channel-wise 误差均值若跨来源具有一致方向，可能通过对 $b_2$ 的小幅重新拟合被吸收，**在线仍由既有 W2 bias 加法执行，无新增 GEMM 或 Q2 observer**。

令 pre-gate $e_{s,p}=\hat y_{s,p}-y_{s,p}$、gate $g_s$，拟加一个**共享于同 expert 两个 step 的**固定 $\delta b\in\mathbb R^{5120}$。与 A0 保持源级归一化的加权 least squares：

$$
\delta b_c^\star
=-\frac{\sum_{s,t,p}w_{s,t}\,g_{s,t,c}^2\,e_{s,t,p,c}}
{\sum_{s,t,p}w_{s,t}\,g_{s,t,c}^2},
\qquad
w_{s,t}>0,
$$

其中 $w_{s,t}$ 要按原 `gated_loss` 的 source denominator 与相同来源等权规则明确冻结。分母0的坐标无从经 gate 识别；仅在训练来源/步骤拟合 $\delta b$。这是**固定已训 Student 在连续 FP32 gate 误差目标下的精确最优加性 bias 校正**，而不是新的 Wan 视频性能定理。

**重要部署条件**：四步模型 high expert 共用 step0、1 权重，low expert 共用 step2、3 权重。同一 expert 不能把四位置 pilot 中某个**单步特化**偏置直接当成合法跨步最终权重；实际 $\delta b$ 必须按真实共享的两个 step 联合拟合与验证，或者明确部署不同 step 的条件化偏置并计入实际在线条件/访存成本。

**科学可区分性**：
- 若 $\eta_{\rm row}$ 大且来源的 centered channel 方向稳定，folded bias 可能比 quotient 更快带来局部收益；
- 若各 source 的 $P_RE R_C$ 方向互相抵消，共享 bias 的净增益可能很小，需检测跨源向量相干而不是只看份额；
- 若误差主要落在 $R_RER_C$，应转向扩展函数表达/后缀敏感度，不把 bias 当作主方法；
- 由于该机制是成熟偏置补偿家族，**本身不是新论文创新点**。直接先例：[Nagel et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html) 的 bias correction，以及 [AdaRound, ICML 2020](https://proceedings.mlr.press/v119/nagel20a.html) 与经验 bias correction 的比较。FFNDS 的新对象应是其与 HiF4 合法 Q2、gated LN quotient、跨步 shared expert 的相互作用。

## 3. 对回应逐点裁定

| 对方主张 | 评议 | 修正后的下一结论 |
|---|---|---|
| 每 token 通道平移穿过连续 Pre-LN 后缀 | **接受（条件 ALG）** | 源码仍要验证 frozen runtime 的 $norm3$、adapter、post-process、head、bf16 有限差；LN shift-invariance 是已知数学性质。参见 [Brody–Alon–Yahav, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.895/) 的 LN 几何投影分析。 |
| $P_C D_g(\hat y-y)=0$ 是当前 Flow 不变的充分条件 | **接受（条件 ALG）** | 需要 current student/dense 与相同真实写回基底、固定上下文；这是功能等价，不意味着任何隐藏 Q2 group 可省 |
| A0 同成本 B-QUOTIENT 训练 | **接受为 matched ablation** | **先四分量诊断**，再 paired loss/bias 对照；目前$\eta_{\rm gauge}$ 未观测 |
| 98.75% 的局部核并不保证共享压缩 | **完全接受** | 80 个来源 64 维 rowspace 张成 5119 维中心化空间的构造在一般位置成立；限制到真实合法 student 函数才有部署意义 |
| 两种 JOINT 应严格区分 | **完全接受，必须改名** | $B-JOINT-STATE$ 为 $(G_s,C_s)$ 同态配对；$B-CROSSSITE$ 为不同位置终端误差相干性；不要复用同一裸名字 |
| 二因子四臂混合可区别非线性交互和 signed interference | **接受** | 对 $a=Y_{10}-Y_{00},b=Y_{01}-Y_{00},I=Y_{11}-Y_{10}-Y_{01}+Y_{00}$：$\Delta=a+b+I$，而 $\|\Delta\|_Q^2=\|a\|_Q^2+\|b\|_Q^2+2\langle a,b\rangle_Q+2\langle a+b,I\rangle_Q+\|I\|_Q^2$；$I=0$ 只排除此二因子有限交互，不排除其它站点/源轨迹高阶作用 |
| 先 quotient 后昂贵 VJP/JVP | **同意经济动机，但不作为全路线门槛** | 静态 gauge 与四分量、bias 的诊断可并行；B-OBS 的数学和 E0 历史回溯无需等待一次 quotient LoRA |

### 3.1 gate 中心化次序必须精确

学生目标为 $P_C D_g e$，一般**不等于** $D_gP_Ce$，因为：

$$
[P_C,D_g]=\frac{g\mathbf1_D^\top-\mathbf1_Dg^\top}{D}.
$$

所以实施 B-QUOTIENT 时，应先做 $E=(\hat y-y)\odot g$，再沿**通道轴**去均值；不可用已有的行平均、不可先 center pre-gate $y$ 然后 gate。优化时的 source-normalized denominator应与 A0 baseline 保持一致，至少同时报告原 gated loss、同分母 quotient loss 和真正 BF16 writeback 的 finite/control。

### 3.2 最末层还有更强但需要额外资产的 exact-head 对照

在 block39，若拥有同一 teacher residual $r$ 与 head 时间调制，可以直接计算**有限、非线性的真实连续 head 差**：

$$
z=P_C^0(r+D_g y),\quad
d=P_C^0D_g(\hat y-y),\quad
F(u)=T\frac{z}{\sqrt{\|z\|^2/D+\epsilon}}+c,
$$

其中 $P_C^0=I-P_C$，所以

$$
\boxed{
\Delta F=T\left[
\frac{z+d}{\sqrt{\|z+d\|^2/D+\epsilon}}
-\frac{z}{\sqrt{\|z\|^2/D+\epsilon}}
\right].
}
$$

它把方向、模长和当前 residual 显式纳入，**可直接作为 B-QUOTIENT 的同算子更强训练目标**；但已有 A0 report/JSON 仅有 sampled V1/Y/gate，尚未完成 B0 所要求的完整同状态 $r/u/head$ 捕获，不能假称目前可训练。若 B0 原计划 producer 完成，则比较 gated、quotient、finite-head 及其后续 full-video。

## 4. 可执行的最小分流：先诊断，再挑两种训练动作

| 次序 | 最小工作 | 需要资产与粒度 | 判别输出 |
|---|---|---|---|
| **0. 静态实现核验** | freeze $norm3$ 分支、adapter、actual FFN bias、gate broadcast、shared expert step；检查 LN & BF16 处理 | 代码/配置/manifest SHA | gauge 条件能否直接用于真实 runtime |
| **1. 双轴误差诊断（最高信息/成本比）** | 对 A0 已导出的 8 个 Student packs 与 CAL24 真实 V1/Y/gate 复算四正交分量，16 train + 8 暴露 heldout | 最初的 $R=256$ 行、D5120输出；同身份 teacher/student | $\eta_{\rm gauge},\eta_{\rm row}$ 与 source-level 方差、跨源向量相干；不得从 JSON 标量推断 |
| **2. 零参数/零训练的 bias 校正** | 先按两个共享 expert step 联合求 $\delta b^\star$，未训练来源只验证；同时尝试仅用历史四位置作**单步诊断，不可宣称共享部署** | 当前已输出 student & teacher pre-gate residual、gate | 相对于原 A0 的每来源 gated/quotient 改善与 BF16 folded bias 误差 |
| **3. 控制训练 B-QUOTIENT** | 同宽6016、同Q2、同source、同 rank32、同128步预算；A0-gated vs quotient，必要时加入已折叠 bias 正交对照 | 同一注册协议，独立 rank 参数与优化器设置 | 是否有**新增**非规范方向的学生函数学习收益，而不仅是训练 loss 定义变化 |
| **4. 更大作用空间** | 当前已有 B0 末层 finite-head；后续 2-site four-arm grouped hybrid | 相同 source/seed、fixed batch/initial latent、真实 QDQ/FFN、完整 4-step | 原生 finite 交互 $I_{AB}$ 与有符号内积，并行追踪视频动作/语义/系统成本 |

若 $\eta_{\rm gauge}$ 在不同来源非常小但 $\eta_{\rm row}$ 高且向量方向一致：先研究 bias/function correction；若 gauge 大而训练 quotient 同预算有稳定 heldout 提升：保留 B-QUOTIENT，继续完整模型多步；若二者都小而 rollout 仍坏：优先全后缀/四步传播（B-OBS/B-CROSSSITE），同时推进 M-A/M-C 独立算子路线。

## 5. 论文身份与“创新”边界

- [Brody, Alon & Yahav, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.895/) 已明确描述 LN 的通道均值投影和范数重标定。**通道平移不变性是基础几何事实**，不能单独作为投稿的 novel method claim；可创新的是以冻结 Wan 真实执行/量化/跨步行为形成可部署方法与新可观测性结论。
- [Nagel et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html) 偏置修正既有先例；用 $b_2$ folded 纠正 row-mean output bias 更应被作为强竞争 baseline，不能自行冠名成创新算法。
- [2ndMatch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html) 提供局部 $J^\top J$ 的随机投影匹配工具，但该度量不保留跨站点 $Je$ 的符号；B-CROSSSITE 是不同的经验对象，尚无新 Wan 实测。
- **贡献应最终落在**：法定信息可得时点、真实量化执行单位、可预测功能、跨层/跨步有限后缀、完整 I2V 主体/相机/时序行为，以及同预算 40×4 速度–质量 Pareto。一个源级 toy theorem 或失配案例仅是机制证据。

## 6. 结论与文件边界

第三轮回应值得保留：它对数学和机制的拆分是正确的；**B-QUOTIENT 同在线成本、只换 loss** 是容易判别的方案。此次再评议最有价值的增量，是发现 A0 的已存 `source_row_mean_error_fraction` 与 Channel Gauge **轴不同**，并由此构造四正交分量，把**严格不可观测的规范误差**和**有机会经 folded bias 修正的可观测偏置**分离。

**建议明确的下一动作**：不要直接把当前 $3–22\%$ 行均值百分比当作 B-QUOTIENT 的机会窗口，也不必立即启动同预算训练。先按源码正确轴计算四分量及跨源共享偏置潜力；结果再选择 B-QUOTIENT、bias 或终端敏感模型的下一最小实验。现有 M-B B0/B1、A0 可用 sparse student、M-C 与 I/F/C 并行不变；不自动注册 GPU，不编辑对方回应、原方法快照或运行授权。
