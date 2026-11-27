# 配对 value 新实验：算子与执行核查

日期：2026-11-27。证据固定于 `b26f644923646dcd9d853849cf58fdd122c55a54`，对照上一轮 `9dec5f1` 的候选。核查范围为归档源码、执行清单和收据；本次环境未安装 Torch，没有重跑 Wan 或实际 Torch 核。数学对偶式另以 NumPy 核验。

## 判断

本轮实际执行了所提议的 per-head 配对 value 通道运输。未发现会改变主要比较结论的算子实现偏差。现有 C/B 留出误差大于 1，应当按这组固定候选的实际效果解释。C/D 的一致优势说明位置对应有用；B 本身也保留了位置对应，因此 C/D 不能单独证明新增非对角自由度比现有对角运输提供更多有效信息。

## 配对矩阵与状态

`attention_paired_value_ops.py::paired_map` 使用 `[head, sample, channel]` 张量，逐 head 计算

\[
G=U^TU/J+\lambda I,\qquad
M=D_t+G^{-1}U^T(Y-UD_t)/J.
\]

`scale` 为 `[head,1,channel]`，`x*scale` 是逐列缩放，最后将相同 scale 加回矩阵对角线。因此它精确对应均式目标 `||UM-Y||²/J + λ||M-D_t||²`，λ=0.001。每个 head 分别拟合完整 128×128 矩阵，无低秩截断、无跨 head 混合。

anchor 的 U 与 Z 都在 FP32 中构建后写入 BF16；命中时转为 FP32 拟合和应用。当前 V 的均值/尺度由所有当前 key 行计算，256 个拟合位置及 64 个诊断留出位置固定、互不重合。三条开发来源和四个受测层共享预先固定的位置索引。全轨迹仅保存 256 个拟合行，诊断用 64 行不进入全轨迹候选。

该有限精度实现符合上一轮的部署候选；无正则、精确状态下的条件精确性定理仍是另一层声明。代码保留非活跃尺度阈值 1e−12，与既有 centered 实现相同。

## P64 校正与打乱对照

B/C/D 的 selected 支持、rho、当前 value 均值/尺度、CAL32 跨步字典、attention ridge、P64 标签和最终 BF16 覆写一致。当前与 anchor 的投影端点分别写入 BF16，再求差，符合已有方法的有限精度路径。唯一的历史解码分别为 ZD_t、ZM_C、ZM_D。

D 只打乱当前拟合标签 Y 的行，采用固定无不动点排列；U 和 Y 的各自边缘分布保持，anchor/current 行对应被破坏。该对照有效检验对应关系，却不是“仅删除非对角新增信息”的对照：B 已通过同位置的标准化历史保留这一关系。

固定轨迹中的 attention 标签来自公共轨迹原本执行的完整 native attention，P64 取其对应行；H448 只用于评分。全轨迹候选则实际执行 64 行、全部 keys 的原生 SDPA，并将 P64 覆写为本次真实输出。候选记录的 5120 个原生 query 等于两个命中步 × 40 层 × 64，未因配对 value 额外增加 attention query。

## SparsePR v2 恢复语义

`pfrr` 按当前 P64 sparse 输出求逐通道均值和标准差（方差加 1e−6），对 residual 减均值；对偶求解

\[
B=X_P^T(X_PX_P^T+I)^{-1}(R_P-\bar R)
\]

等价于 λ=1 的完整通道 primal ridge。输出加回 residual 均值，截距未受惩罚，无输出方向截断，最后覆写 P。输入保留全部 128 通道。旧面板和四臂 A 都使用 FP64 拟合后转回目标精度，B/C/D 的现有实现使用 FP32。

因此这是论文定义的 **等权、给定 S/P 的 v2 输出恢复**。旧路由与固定 P64 并非论文的 response-coupled partitioning、作者 probe 选择或覆盖权重；此次胜负支持恢复器移植的范围，不代表完整 SparsePR 系统已被复现并击败。完整通道在这批 P64 上表现较差，与源码语义相符，不能自动归因于实现漏掉输入通道或施加了输出截断。

## 下游联合作用与全轨迹

`JointTail.replay` 对各臂实际执行 W_O、self gate 与残差、cross-attention、FFN 前归一化/调制，再执行共享真实 P64 FFN 教师拟合的 wide8192 FFN。因为 attention P64 覆写，各臂在这些位置的 FFN 输入逐位一致；同一教师适用于各臂。

同输入 Dense 参考在 H448 上额外计算原生 FFN。相对 Dense 的误差由 candidate-vs-wide 偏差与 wide-vs-Dense 背景偏差的 Gram 交叉项合成，保留了二者的实际方向关系。它是同输入块参考，不等于完整 Dense 生成轨迹。

全轨迹 B/C 通过正常 `OnlineAttention.bind('OUT-D')` 路径在全部 40 层执行。step 0 为原生 high anchor、step 1 为相同 high OUT-ABS、step 2 为相同 low 原生 anchor、step 3 才分别应用对角/完整矩阵运输；FFN 都为 wide8192。收据检查初始噪声与前三步 flow/latent 逐位相同，第四步两臂各自完整运行。因此全轨迹确实测试了新运输在层间传播后的净作用。

两臂全轨迹使用同一新参考执行框架，其中对角路径也是直接 FP32 组合，不是此前独立计时的编译融合 MassCenteredLow 路径。此处保存全步轨迹、无正式计时设计；所记耗时可确认执行，不能据此替换以前的正式速度结果。

## 来源身份与核验

五个直接相关新源码的 SHA256 与学校执行清单完全一致：`attention_paired_value_ops.py`、`attention_paired_value_replay.py`、`attention_sparsepr_panel_control.py`、`attention_paired_value_end_to_end.py`、`attention_joint_action.py`。`verify_operator_semantics.py` 记录上述哈希以及 NumPy 中 primal/dual ridge 的数值一致性。

12 个站点各自对 40 个 head 取最大值后的正则 Gram 条件数落在约 794–35970。收据没有相同真实 U/Y 的 FP64 与 FP32 M 对比，因此该条件数只是数值背景，不足以把 C/B 的误差增量归因于浮点求解。主要实证仍是留出 value、P64 后 attention、同输入块及完整轨迹的实际比较。

## 对方法判断的影响

这轮已分别检验“历史信息能否补充当前输出读出”与“完整 value 通道识别能否提高现有历史运输”。前者得到给定 S/P 的实测支持；后者在固定 J=256、λ=.001、完整矩阵、BF16 状态下的局部与全轨迹证据没有展示稳定增量。保留现有 centered 对角状态有具体的实证依据。应把新增矩阵的结果记录为明确的机制边界，而不是继续把 C/D 优势理解为 C/B 应当胜出的理由。
