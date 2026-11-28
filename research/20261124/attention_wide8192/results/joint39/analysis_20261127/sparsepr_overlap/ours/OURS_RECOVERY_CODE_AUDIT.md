# 当前 FFNDS 恢复算子的实际边界

证据固定于 `Qesire/FFNDS@d3503514dfc1a747b07c2811c8f38d30b3ba2596`。本轮通过 GitHub 重新取回并逐字节校验五个文件，身份见 `SOURCE_IDENTITIES.json`。没有修改运行代码或执行 GPU。这里回答的是实际操作是否等价，不从不同任务名称推断新颖性。

## 结论

high OUT-ABS 的核心确实是“当前 sparse 输出形成特征，以 P64 原生行观测 dense−sparse 残差，做 affine ridge 并覆写探针”。SparsePR v1/v2 的 recovery 都有这个过程。当前 high 可以准确称为同一恢复范式的受限特征变体；仅凭 P64、残差目标、ridge、覆写不能建立新的核心贡献。

逐算子并不完全相等：当前 high 使用固定 CAL 输入特征降维、等权拟合、包含 bias 的 ridge；SparsePR 原文使用当前 sparse 全 value 通道作特征、按 probe coverage 加权并在线标准化，bias 不罚。是否值得作为方法增量，取决于这些改变带来的证据，不能由“不完全相等”直接升级为新机制。

low centered 的实质增加在全位置历史状态、当前 value 坐标解码和跨步特征。其拟合依然属于观测残差恢复家族。FFN 则使用另一种底座、16 模态增益与探针核的联合解，不能将 attention 的 affine recovery 直接当作完整 FFN 算子。

## 1. high OUT-ABS

源文件：`research/20261124/attention_wide8192/results/joint39/attention_online_structured.py`。

- 第 25–26 行：N=31668、H=40、d=128、query/key block=128、TOP=31。
- 第 59–81 行：载入固定 P_v 以及按 high/low 分别校准的基与 lambda。P_v 取 128×64。
- 第 84–97 行：以锚点 Q/K 的128行块均值打分，选31个 key block；命中在该支持中重算当前 Q/K/V 的归一化 attention。
- 第 166–197 行：OUT-ABS 的全部恢复操作。
- 第 246–278 行：当前 HHLL 先在 high step0 运行 Dense 并保存 route，step1 用 OUT-ABS；high 没有保存完整历史输出场。

固定一个 head，令当前 selected 输出 S∈R^(N×128)，原生完整输出 A，P 选64个固定位置。令 C=P_v U_CAL∈R^(128×31)。实际特征为

\[
\Phi=[\mathbf1,\operatorname{FP32}(\operatorname{BF16}(SC))]\in\mathbb R^{N\times32}.
\]

系数与预测是

\[
\Theta=(\Phi_P^\top\Phi_P+6.4I_{32})^{-1}\Phi_P^\top(A_P-S_P),
\]

\[
\widehat A_{\bar P}=S_{\bar P}+\Phi_{\bar P}\Theta,
\qquad \widehat A_P=A_P.
\]

运行代码在标签除以固定 scale_r 后拟合，再乘回 scale_r；实数域该标量约掉。实际 FP32/BF16 舍入仍按代码保留。

`attention_panel_fit.py` 第49–52、55–69行确认：追加常数列；`ridge*eye` 包括该常数列；不使用 coverage 权重。实际 CAL recipe 的 lambda=0.1，p=64，因此正则系数6.4。

该文件第223–247行确认 U_CAL 的来源：将 CAL 的64维 raw 输出特征二阶矩聚合，以逐维 RMS 归一化，取31个特征向量，冻结其符号与尺度；没有减去当前 probe 加权均值。已归档 high/low NPZ 中 `out_current_basis`、`out_delta_basis` 均为64×31。

忽略 BF16 舍入后，high 修正可写成

\[
\mathbf1\theta_0+SC\Theta_1.
\]

所以其函数类包含在SparsePR v2默认无输出截断的完整当前 S 的 affine 修正中。输入侧限制 C 的列空间、坐标中的正则项与权重不同，使实际估计器不同；这仍是该范式中的具体结构选择。SparsePR v1 的16阶截断发生在输出残差右奇异子空间，不能与我们的31维输入特征截断混为一谈。

## 2. low centered

源文件：`research/20261124/attention_wide8192/results/joint39/analysis_20261126/joint_action_review/mass_centered_runtime/attention_mass_centered_low.py`。

- 第20–34行：当前 V 的总体均值/标准差，以及 encode/decode。
- 第53–78行：锚点保存 route、rho、标准化完整响应状态、31维 projected anchor design。
- 第80–106行：命中解码、P64拟合、覆写。
- 第182–238行：固定 raw轨迹分析与 centered 自身轨迹配对的调用路径；高噪声方法保持原实现。

锚点历史为

\[
Z_a=[A_a-\operatorname{diag}(1-\rho_a)S_a-\rho_a\mu_a^\top]D_a^{-1},
\]

当前底座为

\[
b_t=\operatorname{diag}(1-\rho_a)S_t+\rho_a\mu_t^\top+Z_aD_t.
\]

这里 Z_a 在 BF16 中存储，D 的零尺度通道按原代码置零，rho 与 route 绑定锚点支持。

恢复特征同时具有历史依赖：

\[
\Phi_t=[\mathbf1,\operatorname{FP32}(\operatorname{BF16}(S_tC_{low}))-
\operatorname{FP32}(\operatorname{BF16}(S_aC_{low}))].
\]

拟合为同样32列、等权、全部列ridge的

\[
\Theta_t=(\Phi_{t,P}^\top\Phi_{t,P}+6.4I)^{-1}\Phi_{t,P}^\top(A_{t,P}-Pb_t).
\]

最终输出 b_t+Φ_tΘ_t 并覆写 P。历史状态既改变底座，也改变当前恢复特征。

可区分的实际信息能力：取两个未被 P 直接观测的位置 i,j；若当前 S_i=S_j、锚点 S_ai=S_aj、rho_i=rho_j，但 Z_ai≠Z_aj，则当前 Φ_i=Φ_j，而低噪输出之差保留 (Z_ai−Z_aj)D_t。仅以当前 S 为输入的同一个 affine recovery 对两位置给出相同值。该见证利用全位置历史所带来的信息，不依赖重命名拟合目标。

## 3. wide8192 FFN

源文件：`tools/ffnds_fourstep/fourstep_backend.py` 第20–53行把wide8192绑定至 `fourstep_hit.make_hit(8192,reuse_base=True)`；对应源码 `tools/ffnds_fourstep/fourstep_hit.py` 只有55行，完整读过。

第16–28行形成

\[
b=Y_a+\Delta X A_0+\Delta h_K W_{2,K},
\qquad q=\Delta h_K Z-\Delta X J,
\]

并通过 selected 预激活的割线特征得到 x。缓存是全位置输入、输出和 selected 预激活。P64 上实际调用完整原生 FFN（第30–32行），观测目标 T=Y_P-b_P。

令 k=2560，q_P∈R^(64×16)，x_P∈R^(64×k)，V∈R^(16×5120)，

\[
S=x_Px_P^\top/k+M^{-1}.
\]

第33–41行的联合求解为

\[
[(q_P^\top S^{-1}q_P)\odot(VV^\top)]g
=\operatorname{diag}(q_P^\top S^{-1}TV^\top),
\]

\[
C=S^{-1}(T-q_P\operatorname{diag}(g)V).
\]

第43–53行输出

\[
\widehat Y=b+q\operatorname{diag}(g)V
+\tfrac12(xx_P^\top/k)C,
\qquad \widehat Y_P=Y_P.
\]

该算子在固定 q,x,V,M 后对观测标签 T 线性，但通常耦合输出通道：g 的计算收缩了 T 与 V。它不是对所有输出通道共用同一左乘行提升矩阵的 attention ridge。P64观测与覆写是共享构件，底座、特征、约束和读出均另有结构。

## 4. 应如何描述重合程度

| 部分 | 核查后关系 |
|---|---|
| 当前原生64行、残差拟合、ridge、probe覆写 | 与SparsePR直接共享的方法组件 |
| high OUT-ABS | 同一当前 sparse 输出 affine recovery 家族的受限特征变体 |
| low 的残差恢复步骤 | 同一观测校正家族，恢复字典与底座由历史决定 |
| low 的全场 Z、rho 与当前value尺度解码 | SparsePR当前响应恢复之外的状态依赖操作 |
| wide8192 FFN底座与gain/kernel联合读出 | 共享观测思想，但算子不是SparsePR attention affine recovery |

三者是否在某次输入上产生同一最终输出，还取决于恢复器能否吸收底座差；这个条件由单独的等价性推导给出。代码操作不同、数学不完全等价和有足够论文新颖性，是三个不同强度的判断。

SparsePR原文依据：<https://arxiv.org/html/2608.18484v1>、<https://arxiv.org/html/2608.18484v2>。这里有关其恢复公式的定位由本轮独立原文审查提供；完整源审计另存 primary 子目录。
