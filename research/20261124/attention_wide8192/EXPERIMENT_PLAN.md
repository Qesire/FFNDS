# FFNDS 注意力扩展：有区分力的实验方案

**版本：2026-11-24，方案 v2（`wide8192` 运行身份更新）。** 本方案将注意力扩展推进到可实施的比较、数据契约和资产依赖。已经完成的工作是仓库与资产核查、采样坐标重建、数学对象的定义和静态资源计算；新注意力采集、拟合结果、完整视频和运行时间均待执行。

## 1. 本轮要形成的判断

本轮研究对象是：**在当前真实注意力计算中，用少量完整 query 观测校准内部有限变化特征，再按质量与中心化 value 矩的关系恢复省略贡献。** 现有 FFNDS 的锚点机制提供了可复用的研究结构；注意力扩展的具体特征、观测量和读出需要分别得到证据。

四个问题决定这条路线的下一步：

| 问题 | 决定性比较 | 结果直接支持的判断 |
|---|---|---|
| 内部坐标包含多少可恢复信息？ | 输出变化坐标与内部有限变化坐标；各自的纯输出表示下限与 P64 实得误差 | 表示收益、有限观测收益及其差距 |
| 当前 LSE 提供了什么？ | 完全相同的质量耦合函数族，分别令质量监督权重 μ=0、μ=1 | 质量标签对辨识和估计稳定性的增量价值 |
| 当前 value 方向的物理对应有何作用？ | 正确的当前方向与固定置乱方向；特征、参数量、标签和正则相同 | 质量变化沿正确输出方向作用的价值 |
| 局部收益能否成为可用方法？ | 候选自身轨迹、完整视频，以及四步 HHLL 的完整调用周期 | 与 FFNDS 组合后的质量、成本和状态开销 |

其中前三项共享一批真实模型状态。增加一个读出对照主要增加 CPU 分析，不增加一批模型生成。有限非线性自身的贡献另在完整 K/V 见证点上用精确切线与有限差分辨析。

## 2. 固定研究对象与现有检查点

### 2.1 身份与几何

| 对象 | 本轮绑定 |
|---|---|
| 仓库当前说明 | `Qesire/FFNDS@13ffb127e71a5d4d1519ecb0df9c7b4d03efe601` |
| 详细历史证据 | `Qesire/FFNDS@2c4aa753761194994ee25ea8caba76a29445912d` |
| 模型 | Wan2.2 I2V A14B，LightX2V 四步权重，high/low 两专家 |
| 生成配置 | 81 帧，832×464，16 fps，主 seed=42；原始 image/prompt |
| 视频 token 几何 | `(t,y,x)=(21,29,52)`，`N=31,668`，`id=t×1508+y×52+x` |
| 注意力维度 | 40 heads，`d_h=d_v=128`，拼接维度 5120 |
| 当前 wide8192 FFNDS | `wide8192`（BF16 缓存，命中块 8192 token 行），选中 2560/13824 FFN 通道，固定 P64，16 个共享模态，逐位置锚点，HHLL，steps 0/2 建锚，1/3 恢复 |
| 本轮观察层 | 0、13、26、39；两个专家均观察 anchor→hit |

high/low 权重 SHA、原始输入 SHA、调度器与当前 transformer SHA 见历史来源审计 `repository_binding_audit.json`；最新 FFN 执行身份见 `wide8192_binding.json`。实际 attention callable、LSE 约定、归一化/RoPE 位置、mask 与 context 分支在目标运行环境中绑定；`wide8192` 是当前 FFN 后端，8192 是执行分块行数，并非选中通道数；`fast512` 仅作为历史证据来源。

self-attention 的 Q/K 指实际归一化与 RoPE 后进入点积的张量；V 指进入加权求和的张量。cross-attention 用实际 context 分支的有效 K/V。固定文本长度 512 是当前静态预算的依据；采集时保存实际长度、mask 和 context hash。有独立 image-context 分支时，为该分支建立自己的模块身份与支持，不将两个 softmax 混成一个。


### 2.3 当前执行身份与缓存对照

`wide8192` 是 `tools/ffnds_fourstep/fourstep_backend.py` 的 `DEFAULT_VARIANT`，校验的检查点为 `fourstep-wide8192-185220`。两专家各 40 层，每片 80 次原生 FFN 锚点、80 次命中；命中路径每 tile 最多 8192 个 token 行。原生 W1/W2 和精确 P64 仍为 BF16。当前仓库 `research/CURRENT_HANDOFF.json` 将 `wide8192`、BF16 缓存设为默认。

`wide8192_all_int8_joined` 是已验证的独立容量变体：`u0/y0/z0k` 逐行每 128 通道共享一个 BF16 scale，保留 64 个探针位置的精确 BF16 值，W1/W2 保持 BF16，锚点 W2 输出合并编码与当前门控残差写回。缓存总量从 30.20 GiB 变为 15.40 GiB；独立运行的已分配峰值为 47.57 GiB 对 32.77 GiB。INT8 在同卡十源中 FFN+post 均值 4.83165→4.81399 s，整片均值 37.90009→37.93549 s；两种缓存产生有差异的视频，应分别进行真实配对和人工审片。

**本轮机制主对照全部使用 BF16 `wide8192`，保证同一缓存精度；INT8 留给条件性显存对照，且在同一 cache 变体内部对照 attention REF/NEW。** 训练/模型采集中的 `Dense` 背景始终运行原生 FFN。当前实验不将 INT8 的可用内存优势解释为注意力机制收益。身份与验证条件见 `wide8192_binding.json`。

### 2.2 采样与支持先固定

直接复用当前生产 P64。额外 H448 复用旧 `make_panel` 的 112 个不相交时空四点单元，形成 224 条空间边、224 条相邻 latent 帧边。两类 query 互斥，边的两个端点均属于 H448。`attention_panel_index.json` 已实际重建，包含完整 token ID、坐标、probe 顺序与邻边。

主支持是本方案自有的简单固定规则，以便隔离恢复机制：

1. 按真实展平顺序分 query/key block；最后一个 block 使用实际有效长度。
2. 在 anchor，以每个 head 的 block 平均 Q/K 的缩放点积评分。
3. self：query/key block 均为 128，248 个 key block 中取 top 31。
4. text-cross：query block=128、key block=32，512 keys 时 16 个 block 中取 top 2。
5. 同分按 key block ID 升序，anchor→hit 的支持保持固定；下一专家重新构建。

所有读出 arm 使用完全相同的支持和 P64。self 的有效 key 数每 query 介于 3,892 和 3,968，实际交互密度逐 site 记录；“12.5%”只表示块数设计比例。该规则用于机制辨析；SparsePR、RECAP 的完整作者方法复现分别保留自己的路由和调度语义。

## 3. 从信息流定义恢复对象

以下均按单个 query、单个 head 定义，端点用 a（anchor）和 s（current）标记：

\[
s_{ij}=q_i^\top k_j/\sqrt{128},\qquad
L_i=\log\sum_j e^{s_{ij}},\qquad
o_i=\sum_j\operatorname{softmax}(s_i)_jv_j.
\]

支持 S 上的归一化输出和 log-mass 分别为 \(o_{S,i},L_{S,i}\)。完整 query 返回逐头、投影前的 \((o_i,L_i)\)。定义省略质量与归一化一阶矩：

\[
\rho_i=1-e^{L_{S,i}-L_i},\qquad
u_i=o_i-(1-\rho_i)o_{S,i}.
\]

实现用 `-expm1(L_S-L)` 计算小质量。以 pair 内固定的 \(v_*\) 为 anchor 全键 V 均值，令

\[
w_i=u_i-\rho_i v_*,\quad
\mathcal B_{s,i}=(1-\rho_{a,i})o_{S,s,i}+u_{a,i},\quad
c_{s,i}=o_{S,s,i}-v_*.
\]

于是输出创新具有精确分解：

\[
R_i=o_{s,i}-\mathcal B_{s,i}
=\Delta w_i-\Delta\rho_i\,c_{s,i}.
\]

这给出两个可被直接观测的目标：P64 上的 \(m=\Delta\rho\) 和 \(R\)。所有 query 的当前 sparse 输出、当前 Q/K/V 的固定宽度统计与 anchor 小状态提供特征。H448 的完整输出只进入评分与表示诊断。

### 3.1 信息充分性具体检查什么

取 \(X\in\mathbb R^{n\times32}\)，定义参数 \(a\in\mathbb R^{32}\)、\(B\in\mathbb R^{32\times128}\)：

\[
\widehat m=Xa,\qquad
\widehat R=XB-\operatorname{diag}(Xa)C.
\]

当真实 \((m,\Delta w)\) 属于这个固定特征函数族、标签精确且 \(X_P\) 满列秩时，完整的质量标签先确定 a，再由输出标签确定 B。每 head 有 4,128 个系数，P64 的输出加质量给出 8,256 个标量观测；数量允许该族的恢复。实际模型的近似表示条件由表示下限检验，有限观测稳定性由受限秩和条件数检验。

一般地，设 \(\mathcal M_P\) 是 probe 观测映射，\(\mathcal A_H\) 是待恢复的 held-out 输出映射。准确恢复输出需要

\[
\ker\mathcal M_P\subseteq\ker\mathcal A_H.
\]

每个 site 报告 \(X_P\) 的奇异值、有效秩、\(\|X_HX_P^+\|_2\)，并对耦合映射使用矩阵作用或 Schur 补检查对应的目标增益。秩阈值同时登记 FP64 数学精度与实际缓存/标签扰动尺度。正则化改善解的稳定性，其作用与观测提供的信息分开解释。

该检查调用算子与张量 skill 的受限观测恢复、SVD/最小二乘、Schur 补和后向误差工具。求解器记录真实残差；小系统通过分解求解，不显式求逆。

## 4. 可执行的内部有限变化特征

### 4.1 原始 64 维特征

为使第一次 CAL 采集即可保留完整可复用特征，先冻结两个 128 维正交基 \(P_v,P_k\)。它们由登记的随机种子生成，矩阵直接随方案交付；所有 head 使用同一基，不将它们称作数据最优基。OUT 使用 \(P_v\) 前 64 列，FIN 使用下列 64 个统计：

| 维数 | 端点统计 \(f_t(q_i)\) | 信息与实际计算 |
|---:|---|---|
| 32 | \(o_{S,i}P_v[:, :32]\) | selected 输出的固定投影 |
| 8 | \(\mathbb E_{p^S_i}[k]P_k[:, :8]\) | selected softmax 下的 key 均值 |
| 8 | \(\operatorname{diag}\operatorname{Cov}_{p^S_i}(vP_v^{8},kP_k^{8})\) | 8 对投影 value/key 的混合矩 |
| 8 | \(P_v^{8\top}\operatorname{Cov}_{\mathrm{uniform}(K)}(v,k)q_i/\sqrt{128}\) | 全键未加权中心矩对当前 q 的作用 |
| 7 | \(q_i^\top P_k[:, :7]/\sqrt{128}\) | 当前 query 坐标 |
| 1 | \(L_{S,i}-q_i^\top\bar k/\sqrt{128}-\log|S_i|\) | 去除共同均值与支持长度后的 selected log-mass |

混合矩可在 selected softmax 在线归约中累计。全键矩的部署版本只需 8×128 投影矩；完整 128×128 的 CovKK/CovVK 只为研究诊断保留。原始特征不需要保存 N×|S| 概率、不需要完整 anchor Q/K/V。

从 CAL 的 \(\Delta f=f_s-f_a\) 计算逐列 RMS，做不去均值的 SVD，固定 31 维映射 T。实际

\[
X_i=[1,\ Q_{\mathrm{BF16}}(Tf_s)_i-Q_{\mathrm{BF16}}(Tf_a)_i].
\]

两端使用相同量化，等端点时变化坐标为零。主 rank=32 包含常数列。CAL 变换按 self/cross×high/low 共四个范围建立，汇总三个 CAL 来源、四个观察层和 40 heads，来源等权；同一范围的全部层/头共享变换。这样可直接在 40 层运行。未采层的实际表现由后续完整在线轨迹检验。系数 a、B 仍逐 call/head 用当前 P64 拟合。

### 4.2 精度契约

部署缓存为 `u_a BF16`、`rho_a FP32`、`Tf_a BF16[31]`、`v_* FP32` 与固定支持。主离线评估复现这套舍入顺序，R 的基线使用已经舍入的 u_a。相应 \(w_a\) 定义为缓存 u_a 减去 \(\rho_a v_*\)，从而读出恒等式仍精确描述实际缓存对象。

完整 K/V 见证点另以 FP64 重算数学对象，用于恒等式、切线及 finite secant 检查。由精确对象到 native kernel、由 FP32 状态到 BF16 缓存的误差分别保存。同一次原始资产足以比较这些数值路径。

## 5. 七个拟合 arm，两个零拟合参照

| ID | 特征/目标 | 读出 | 作用 |
|---|---|---|---|
| SP | 当前 \(o_S\) | selected 输出 | 省略后原始误差 |
| ANCH | 当前 \(o_S\)+anchor \((u_a,\rho_a)\) | \(\mathcal B_s\) | 已知 anchor 提供的恢复能力 |
| OUT-D | selected 输出变化，raw64→rank32；创新 R | 直接 ridge | 信息时点匹配的输出坐标对照 |
| OUT-ABS | 当前 selected 输出，raw64→rank32；\(o-o_S\) | 直接 affine ridge | 当前输出拟合机制参照 |
| FIN-D | 内部有限变化，rank32；创新 R | 直接 ridge | 内部坐标主效应 |
| FIN-C0 | 同 FIN；创新 R | 质量耦合，μ=0 | 固定耦合函数族的基线 |
| FIN-C1 | 同 FIN；创新 R 与 m | 同一质量耦合，μ=1 | 主候选 |
| FIN-CP | 同 FIN 与同标签 | μ=1，c 的 query 对应固定置乱 | 当前方向的物理对应 |
| FIN-ABS | 同 FIN 与 anchor 信息 | 拟合绝对 \((\rho_s,w_s)\)，基线 \(o_S\) | anchor offset 与创新目标的作用 |

Dense 是所有 arm 在相同输入上的真实目标。OUT-ABS 捕捉 SparsePR 的当前输出 affine 拟合机制，但本轮使用自己的支持、P64 与 rank 预算，所以它的名称和结果归属为自有机制对照。对作者完整方法的性能比较在候选形成后单独复现。

**主机制对比：** FIN-D/OUT-D；FIN-C1/FIN-C0；FIN-C1/FIN-CP。FIN-D/FIN-C0 解释函数族改变；FIN-C1/FIN-ABS 解释已知 anchor offset。OUT-ABS 与 OUT-D 中更强的执行参考在 DEV 结束后一次冻结。

所有 arm 共用 raw 信息时点、支持、P64、rank32 和 target 标度的定义。OUT 与 FIN 的特征计算量不同，成本单列；FIN-C0/C1/CP 的特征计算、参数量和成本路径完全相同。CP 的置换由 query ID 预先确定：同一 latent 帧内的面板 query 按 ID 排序，循环位移一位，全部 heads 使用同一映射，以保持跨头方向的联合分布。c 是所有 query 都可计算的当前 sparse 统计，因此可跨 P/H 交换；它们的完整输出标签不随之交换。控制自身无需成为视频候选。

### 5.1 共同求解规则

令 \(s_R\) 为标量，\(s_R^2=\mathbb E_{\rm CAL}\|R\|^2/128\)，\(s_m\) 为 CAL 质量创新 RMS，定义

\[
R'=R/(s_R\sqrt{128}),\quad m'=m/s_m,
\quad C'=s_mC/(s_R\sqrt{128}).
\]

拟合

\[
\min_{a,B}\frac1p\|R'-XB+\operatorname{diag}(Xa)C'\|_F^2
+\frac\mu p\|m'-Xa\|^2+\lambda(\|a\|^2+\|B\|_F^2).
\]

μ=1 对应一项平均向量误差和一项标量误差的等权。对直接读出保留同一输出缩放。λ 在 CAL 来源留一中仅考察 \(10^{-3},10^{-2},10^{-1}\)，每折的 RMS、SVD 与目标尺度由其余两个 CAL 来源重算，再以留出来源的 P64 拟合、H448 评分；选完 λ 后用 CAL3 重建最终变换。全部从已存 raw64 重算。核心 C0/C1/CP 共用 C0 选出的 λ，阻止标签增量与调参预算混杂。λ 的目标采用每个 query 的平均损失；含 p 行的求和正规方程使用 pλ，换用 448 行参考时同步调整。

零 RMS 的特征列使用零逆尺度并登记有效秩。若 CAL 的某目标尺度达到数值底噪，保留其绝对单位与观测底噪报告，按“该创新已由基线解释”处理；其规范化拟合状态明确登记，不以任意极小常数放大标签噪声。

可消去 B，得到一个 32×32 Schur 系统；最终仍记录未消元目标的残差。分析中的未约束线性 \(\widehat\rho\) 原样保留，并记录越界率。主执行版也采用同一线性配方；若引入 clipping 或 logit，需要另存 recipe 与同资产配对结果。

在 C1 内再保存一个无需新采集的代数诊断：使用完全相同的 Hessian/Schur 分解，将质量监督的线性项 \(X^\top m'\) 置零，其他项保留。C1/C0 衡量质量监督的整体增益；C1 相对该 zero-mass-RHS 解的差异进一步辨别真实当前标签的线性驱动与新增二次惩罚。它只是一列离线结果。

### 5.2 三层误差区分表示和估计

对每个函数族分别计算：

1. **逐头纯输出表示下限**：在 H448 上直接最小化逐头欧氏输出误差，系数自由，μ=λ=0。它刻画该函数族在该面板的 head-space 表达能力。
2. **全部 H448 标签下的估计器参考**：同一 μ、同一平均损失正则与求解规则，用 H448 拟合并在 H448 评价。它暴露联合监督与正则引入的偏置。
3. **P64 在线式估计**：只用 P64 标签拟合，在 H448 上评价。这是本轮主要可恢复结果。

C0 与 C1 的输出函数族完全相同，必须共享同一个逐头纯输出表示下限。在相同逐头指标中，估计器参考减去表示下限应非负（允许已记录的数值误差）；P64 减去估计器参考可以有任意符号。两项差值在代数上相加得到 P64 与下限之差，解释时保留采样、辅助监督和正则之间的相互作用。表示下限是同一 H 集上的诊断量，其数值不作为泛化评分。

将逐头最优系数送入真实 W_O/gate 后，得到的是“该 head-space 最优解的下游评分”。联合下游度量的最小误差需要另行联合求解，主表使用准确名称区分。在线 P64 结果的主要排序继续采用下游误差。

## 6. 真实模型上的最小采集批次

### 6.1 来源与输入

| 角色 | 来源 | 内容 | 本轮使用 |
|---|---|---|---|
| CAL | 0032、0304、0456 | 抽象画面、雪中街道、厨房 | 特征压缩、尺度与 λ |
| DEV | 0392、0800、1013 | 书墙客厅、海岸、人物演奏 | 机制比较、参考选择和完整视频 |

均使用原始 prompt 和 seed42；输入图、原始 prompt、source cluster 与 SHA 写入 `sources_locked.json`。主轮不追加 camera clause。0951 的 pan-left 跨 seed 专项沿用仓库的延后安排。

现有 24 源及旧研究已接触来源作为 DEV/replay 池。独立确认的六个新来源稍后按三类内容各取两个 cluster：室内/建筑、自然场景/植物、人物/动物。`freeze_confirmation.py` 按固定 hash 排序选择；它先解析历史 `VB_BOUND_INPUTS` 与 24 源并集的 54 个 clusters，加入 0905、1014 和后续访问记录，再选取 fresh clusters。确认来源在配方冻结后、产生候选结果前锁定，保存原始输入字节与配方 SHA；当前包只登记解析规则。

### 6.2 一次采集支撑全部主比较

初批：6 sources × 2 输入背景（Dense、当前 BF16 wide8192）× 4 layers × 2 expert pairs。

- self：96 个 site pairs。
- cross：96 个 site pairs。
- 每 pair 保存 512 queries × 40 heads 的轻量端点、特征与精确 O/LSE。
- 全部采集来自 12 条完整四步轨迹；保留对应完整视频、各步 latent/flow 原始文件及其 hash。原有索引里的 hash 仅作为指纹，实际重放需要可读取的 tensor 文件。
- attention 在这批公共输入轨迹中使用 Dense；wide8192 背景只改变 FFN，从而观察 FFNDS 轨迹下的注意力信息。两个背景分别汇总并报告配对差异。

首次接口校验直接使用 CAL0032、layer13 的 high/low 两 pair。该轨迹属于正式初批；字段正确后复用其产物。若接口修正改变数据语义，只重采受影响的资产。

主特征采用端点归约，因此可在每个端点直接输出 fixed-width 字段。无需为全部层保留 anchor 完整 K/V。writer 按 site 顺序落盘，有界队列和写入时间记入采集日志。

### 6.3 八个 self 见证 site 与 cross 的固定 K/V

self 原始见证集合预注册为：0032、0392 × Dense/wide8192 × layer13 × high/low，共八个 pair；每个保存 heads 0、13、26、39 的两端完整 K/V。Q 只保留 P64+H448。不能从这些 K/V 中再抽几个 keys 后宣称完成全支持 attention 的检验。

在 cross 中，同一个 source/expert/layer 的有效 K/V 若 hash 和 mask 完全相同，可跨端点及两个背景去重。预计 48 份 all-head K/V。先核验固定条件，再使用固定 K/V 数学。

见证计算至少输出：

\[
\ell_j=\frac{p_{s,j}-p_{a,j}}{\log p_{s,j}-\log p_{a,j}},\quad
\eta=\sum_j\ell_j,\quad \pi_j=\ell_j/\eta,\qquad
\Delta p=\eta[\operatorname{diag}\pi-\pi\pi^\top]\Delta s.
\]

在 \(p_{s,j}=p_{a,j}\) 时取连续极限 \(\ell_j=p_{a,j}\)，通过 log-domain 公式保持数值稳定，mask 上共同无效的 keys 不进入求和。再计算 self 的三个有限变化通道，其中 \(\bar p=(p_a+p_s)/2\)、\(\bar v=(v_a+v_s)/2\)，其余带 bar 的量同样取端点算术均值：

\[
\Delta o
=\mathbb E_{\bar p}\Delta v
+\eta\operatorname{Cov}_\pi(\bar v,\Delta q^\top\bar k/\sqrt{128})
+\eta\operatorname{Cov}_\pi(\bar v,\bar q^\top\Delta k/\sqrt{128}).
\]

同时求真实起点的一阶作用

\[
J_a\Delta=\mathbb E_{p_a}\Delta v
+\operatorname{Cov}_{p_a}(v_a,
(\Delta q^\top k_a+q_a^\top\Delta k)/\sqrt{128}).
\]

保存有限差分、精确一阶预测与余项，并保存三个通道的完整带符号 Gram 矩阵；它们的能量一般通过交叉项共同组成总变化。cross 进一步计算 \(\operatorname{Cov}_p(v,k)\Delta q/\sqrt{128}\) 与方向 HVP。只在预注册的两条实际方向及其缩放路径上计算，按 query/head 分块，不物化三阶张量。

这一组实验直接区分“有限步长中的曲率/双线性变化明显”与“一阶作用已覆盖当前实际变化”。它支持下一版特征选择；主 FIN/OUT 结果归因于整套内部坐标，有限非线性在最终候选中的独立因果贡献需相应的匹配特征消融。

## 7. 评分、统计单位和决策

### 7.1 主评分位置

每个 site 先拼接全部 40 heads，经真实输出投影 W_O；self 再乘该 call 的当前 gate。设该作用为 \(L_i\)，主要误差是

\[
\epsilon_i=L_i(\widehat o_i-o_i),\qquad i\in H448.
\]

在同一 site 上报告点误差总能量，并用该 site 的 SP 误差能量归一化。SP 能量低于已测的 native/重算数值差底噪时，保留绝对能量并标识归一化不稳定，不用任意极小常数制造巨大比例。

空间边和时间边分别报告 \(\sum_{(i,j)}\|\epsilon_i-\epsilon_j\|^2\)，用对应 SP 边能量归一化。点/空间/时间三个量分开呈现。head 空间的质量误差、中心矩误差及

\[
\|\delta w\|^2+\|c\delta\rho\|^2-2\langle\delta w,c\delta\rho\rangle
\]

用于解释机制。保留相对目标变化的带符号投影、正交能量和角度；目标变化为零的行单列。四头见证的 W_O 作用只代表这四头贡献，不冒充全头误差分解。

### 7.2 独立单位与预先冻结的效应

独立单位为 source cluster。head、token、layer、expert、seed、prompt 变体均为同一 source 内重复观测。三个确认性主估计量只取 self-attention：每个 source 内对两背景×四层×两专家的 16 个 site 配对 log 能量比等权，再在来源之间等权。分背景结果、cross 与 head/layer 分组作为机制解释性结果，不扩大这三个主对比的检验族。

DEV3 以逐来源表、分背景结果和效应方向作为探索证据；三个来源的区间只描述不确定性。fresh-confirm6 报告逐来源配对值、几何平均比与来源 bootstrap 95%区间。三条预注册主机制对比另给 simultaneous/Bonferroni 98.33%区间；单 seed 的结论明确限于 seed42 配对分布。需要跨 seed 推广时，在固定配方下追加配对种子，并按 source cluster 聚合。

### 7.3 结果如何决定下一项工作

| 观察到的组合 | 形成的判断 | 随后的唯一优先动作 |
|---|---|---|
| FIN 的表示下限与 P64 误差均下降 | 内部坐标包含并能提取更多目标信息 | 实现其固定宽度在线归约 |
| 表示下限接近，C1 的 P64 误差下降 | LSE 提供辨识/稳定性收益 | 固定 μ=1，优化 O+LSE 与小系统路径 |
| 表示下限改善明显，P64 尚未跟上 | 有价值的信息受限于当前观测 | 同面板分析受限谱，再评估预注册 probe 布局变体 |
| C1 优于 CP，质量/矩交叉项也一致改善 | 正确的当前方向承担有效结构 | 保留物理耦合读出 |
| ANCH 已覆盖主要改善，新增读出增量小 | 低成本 anchor 本身有可用价值 | 以 ANCH 进入同预算执行比较 |
| 同输入局部收益保持到学生轨迹与完整视频 | 已形成可推广的模块候选 | 冻结配方，进入 fresh-confirm |
| 局部收益保持而时延收益有限 | 恢复机制成立，当前执行受特定成本约束 | 针对实测主耗时项融合/复用 |

建议的工程推进标尺提前写为：DEV3 两个背景上候选相对冻结输出参考的 source-macro 点误差改善约 10% 以上，至少两个来源同方向，空间/时间关系误差没有稳定恶化；把它作为是否值得花集成成本的实用阈值，而非显著性或定理。完整视频中的可见结构、运动、遮挡、条件响应决定视觉判断。任何偏离该推进标尺但仍值得继续的情形，按上表记下所依赖的具体证据与下一项动作。

## 8. 从学生轨迹到完整视频与真实速度

主部署阶段仅扩展 self-attention 到全部 40 层；cross 先承担固定 K/V 的结构验证。这样首轮新增缓存与速度结论有明确边界。

视频主臂为六个：Dense、wide8192、Attention-REF、Attention-NEW、wide8192+Attention-REF、wide8192+Attention-NEW。REF 从 OUT-D、OUT-ABS 中在 DEV 一次冻结；NEW 初选 FIN-C1，若上节结果支持更简的同信息配方，冻结其准确 ID。

先在 DEV0392/1013 生成四个新增 attention 臂，共八条新视频；Dense/wide8192 视频复用同身份的初批采集。随后加入 DEV0800 的四条新视频。三个 DEV 最终形成 18 条可配对视频，其中六条来自已采集的两个背景。整个初批六源保留 12 条背景视频，连同这 12 条新增视频，共 24 条不同配置的完整生成产物。

每个在线 arm 使用自己的当前输入与 anchor。anchors 0/2 返回 native Dense attention；hits 1/3 用当前 P64，精确覆盖这些 probe 的输出，其他 query 用候选读出。额外 H448 full-query 作为独立 shadow 诊断，保持返回给模型的输出不变；只在诊断运行开启。对学生状态回放主对照，观察局部收益的保持程度。

完整视频保留原分辨率、全帧、相同初始噪声和原始提示，生成随机左右次序的同步对照索引。人工记录细结构、空间协调、运动幅度与连贯性、遮挡、颜色/亮度以及条件响应，也记录 Dense 自身的任务表现。PSNR/latent 差、时序边误差用于定位，视觉结论由完整视频审阅形成。

计时用关闭 capture/shadow 的同一冻结实现：先暖机，再以三个平衡顺序块遍历六臂和三个 DEV。这是候选通过后的 **54 次额外完整 HHLL 执行**，暖机另留每臂一次、最多六次；它与前述 24 个不同配置视频资产分开计数。第一阶段总执行计划上限为 24 次采集/诊断生成 + 54 次计时 + 6 次暖机 = 84 次；只有通过前段后才启动后段。既有 Dense 约 40 秒/条可作为约 56 分钟纯生成时间的粗略排程参照，实际还需加采集、加载、写盘及新方法开销。

报告 source 内配对时间、denoise 与完整请求两个口径、实测峰值、抖动范围。完整请求口径包含实际条件准备与 VAE decode/成片，不含进程首次加载权重；任何条件缓存策略在六臂一致并登记。若三次重测不足以区分观测到的微小差异，针对该不确定量追加重复，并说明重复的目的。时间运行不另造候选配方。

逐项计入 anchor sparse 特征、路由、全键矩、当前 sparse attention、O+LSE probe、特征投影、求解、恢复、布局转换、状态读写与专家切换。真正的计时结果以完整 HHLL 周期为主。


### 8.1 有条件的 INT8 缓存交互对照

当部署候选需要显存裕量时，在 BF16 wide8192 完成机制筛选并冻结 attention 配方之后，进行 2×2 的 **cache dtype × attention method** 配对：`wide8192` 与 `wide8192_all_int8_joined`，各自组合 `Attention-REF` 与 `Attention-NEW`。此外在相同 DEV3 再采 `wide8192_all_int8_joined` 单独视频，作为 INT8 自身锚点/运动基线。相较 BF16 已采的三臂，额外需要 3 sources × 3 INT8 臂 = **9 条视频**；若启动与主试验同样的三重复计时，额外为 **27 次 HHLL 执行**。主批 192 轻面板保持 6×2×4×2，不将 INT8 加入 CAL/机制比较以增加昂贵采集。

比较值必须同 source、seed、prompt、支持、attention 参数、P64 与候选版本，只改变 FFN 缓存格式。两个缓存格式的 BF16/INT8 缓存字节和变体收据分别校验，尤其是 `joined_y0=80`、`joined_fallback=0`。记录视频差异及 cache × attention 交互项；FFN 主调用计时、显存与端到端时间分别列出。

## 9. 资源预算与资产复用

`compute_asset_budget.py` 已按实际形状执行整数计算。初批 192 个轻量 pair，每 pair 约 50.410 MiB；八个 self 原始见证合计 0.966 GiB，去重后的 cross K/V 约 0.469 GiB。加入 self/cross 支持索引后，基础张量 payload 为 **11.004 GiB**；加入可选 W_O 快照后约 **11.786 GiB**。若学校同身份模型权重可直接读取，保留索引即可。

该预算按未压缩 tensor 计算，完整视频、latents/flows、文件头与日志另实测登记。可以按端点流式生成轻量统计，默认无需 2.416 GiB 的四层全头 anchor K/V 临时缓存。

主 self 候选每层缓存约 390.181 MiB，当前专家 40 层共 **15.241 GiB**。与BF16 wide8192 独立运行实测已分配峰值 47.57 GiB 简单相加为 **62.811 GiB**，这只是静态容量规划；实际联合峰值还受工作区、分配器、投影和临时副本影响。切换 high→low 时释放旧专家的 attention 状态，再构建新状态。若相同 attention 静态缓存估计搭配 INT8 joined FFN，静态相加约 **48.011 GiB**；两种容量估计分别加上实际工作区与峰值重叠，并重新测量联合峰值。

P64 占单次 self core 完整 query-key 对的 64/31,668≈0.2021%；H448 约 1.4147%，它属于诊断开销。以四步两 dense anchors、四步均构造 selected 统计、两次 P64 为基准，实际 core 工作比例约

\[
\frac12+\alpha+\frac{64}{2N},
\]

其中 \(\alpha\) 是实际 selected 交互比例；按本支持约为 62.39%–62.63%。该比例尚未包含投影、归约、拟合和内存开销，最终速度以完整运行测量。

现有资产的复用范围清楚区分：图像/prompt/seed/cluster 身份、P64/H448 坐标、FFN 检查点、旧完整视频与审阅可直接复用；旧 FFN U/Y/Z 面板用于 FFN 参照；注意力 Q/K/V、O/LSE、fixed-width 特征与支持按本契约新增。详细依赖和 producer 见 `ASSET_TREE.md`、`asset_registry.json`。

## 10. 方法贡献与最近邻的证据接口

SparsePR 以当前 sparse 输出为坐标，从当前完整 query 拟合输出残差；因此 OUT-ABS/OUT-D 提供与其读出原理直接相邻的比较。RECAP 缓存 omitted tail 的 value/log-mass refinement，并用当前 coarse 统计调节；因此 ANCH、FIN-ABS 与 C0/C1 分别衡量 anchor、当前创新与当前完整观测的增量。

本轮最有分辨力的贡献组合是：**内部有限变化坐标是否改变可恢复的信息；当前 O+LSE 是否提高有限观测质量；当前 value 方向是否构成有效的结构先验。** 每个主张对应单独对照。共同追求输出恢复不决定方法等价；机制筛查沿坐标、可用信息时点、恢复目标、耦合映射、估计规则和执行路径展开。

候选通过局部与完整视频后，再在相同真实模型、固定任务预算下复现作者完整方法，分别登记其原路由、额外校准、缓存与实际 executed pairs。当前自有 pooled 支持上的机制对照先回答本路线为什么成立。

## 11. 已交付与接续实施

此包交付方案、固定 P64/H448 坐标、原始正交投影基、六源身份、七 arm 协议、资产依赖、采集/结果 schema、预算脚本、fresh-confirm 冻结脚本与静态验证结果。新模型结果目录以状态字段登记待生成，不填入示例分数。

接续实施从原生 attention adapter 开始：定位实际 Q/K/V 边界及 O+LSE 后端；在第一条正式 CAL 轨迹上生成指定字段；随后同一 producer 完成余下五源和两个背景。主分析在这批固定资产上完成，得到一张机制效应表、一张表示/估计差距表和一份可运行候选 recipe，再进入上述少量完整视频。

### 来源与数学工具

- 仓库版本、来源、权重、旧面板与 release：`repository_binding_audit.md/.json`，包含逐文件 pinned URL。
- 本轮采样和资源整数计算：`attention_panel_index.json`、`asset_budget.json`。
- 上轮已验证的注意力重定义：`references/attention_redefinition.md`，其符号恒等式运行记录随原数学报告保存。
- 算子与张量 skill：类型/收缩语义；受限观测恢复；数值线代 N2/N3（SVD、伪逆与正则）、N5（Schur 补）、N7（条件数和舍入）、N13（算子作用与成本）。这些是本方案的数学操作工具。
- [SparsePR，§4.2 与 Appendix A](https://arxiv.org/html/2608.18484v2)。
- [RECAP，§3.2](https://arxiv.org/html/2610.26244v1)。
- [Wan2.2 官方模型源码，固定 commit](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py)。实际运行边界以本仓库绑定的 LightX2V 实现为准。