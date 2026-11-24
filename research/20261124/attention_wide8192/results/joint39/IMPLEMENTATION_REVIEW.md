# joint39 新实验的实现语义审查

证据提交：Qesire/FFNDS@b93ac8ee2ed54ccc7cb5b7cbe635fd5e0399e94b。范围为已完成的 DEV0392、两个背景、两个专家四个末层站点。源码审查未发现使已公布大幅方法排序失效的分支或标签泄露错误；下列区分会直接改变科学解释。

## 1. 四臂实际比较的是同一背景当前状态上的反事实

attention_joint39.py 267–314 行：

- 先从背景当前运行得到 X 与原生 self 输出 native。
- 冻结已有 attention 恢复器：high=OUT-ABS、low=FIN-C0，得到 independent；便宜底座 high=SP、low=ANCH。
- 三套 attention 写入分别执行实际 cross、LN2/AdaLN，得到 native/independent/cheap 的 R、T、U。
- 在 native 与 independent 的 U 上分别调用完整 FFN，形成 B00 与 B10。
- wide FFN 的共享系数只用 native attention 的 P64 输入和原生 FFN 标签拟合，然后用于 B01、B11_fixed。
- cheap 分支使用 FFN 原有底座，没有调用这套在线系数校正。

四臂的 target 统一为同一当前 X 上的原生块 B00。Dense 与 wide8192 两个背景的 X 和 FFN anchor 各有自己的身份；跨背景比较同时改变了上游状态与缓存。

ffn_operands（42–57行）的“折叠底座”包含锚点输出、折叠线性变化，以及选中非线性通道的变化写入，并非单纯线性层。FFN q16 是选中非线性与折叠响应的16维差分坐标。

## 2. P 覆盖发生在 flow 出口；refit 是特定反事实

267–313 行并未在 candidate attention、cross 或 FFN 的 P 位置强行覆盖；finish 的 341–344 行才把最终 independent flow 的 P64 覆盖为 teacher，joint_flow 也在 ridge_flow 121 行最终覆盖。

这符合本轮保留覆盖前误差的机制分叉。对 H448 而言，独立方案使用 native-P 拟合的固定 FFN 系数，仍对应原先 attention P 精确覆盖所产生的拟合信息；原生逐行 tail 使 P 端输出的延迟覆盖不改变 H 输出。

B11_refit（305、312行）则在未覆盖近似 attention 的 U_P 上重新拟合 FFN。RESULT 的 refit 字段对此有明确说明。其百分比回答的是“允许 P 输入随近似 attention 改变，再重拟合”的条件反事实，不宜称作原生产 P 精确覆盖时序中必然发生的反馈。在固定 X、cache 且 P 精确覆盖的原时序下，仅由这些 P 输入决定的拟合可保持相同。

## 3. 当前 joint48 同时改变了恢复边界和表示结构

独立 FFN 的读出（60–77行）为

\[
\widehat F_i=b_i+
q_i\,\operatorname{diag}(\alpha)V+
K(i,P)C/2,\qquad
K(i,P)=x_i x_P^\top/k.
\]

它包含16个 q 坐标和64个 probe-kernel 坐标。joint48 的 323–325 行仅保留

\[
\Phi_i=
[1,\;40^{-1}\sum_{h=1}^{40}X_{h,i,1:31},\;q_i^{\rm cheap}],
\]

即常数1列、跨40头均值31列、FFN q16。随后直接拟合64维 flow，系数矩阵是48×64。

因此这次 joint48：

1. 把 attention 的 head 身份压缩成均值；
2. 移除了 FFN 的64维 probe-kernel 读出结构；
3. 从 SP/ANCH + FFN底座重新恢复整块缺陷；
4. 把恢复目标移到 flow，使用单次、固定 λ=0.1 的P64岭读出。

结果针对这个具体函数族。它将“恢复边界后移”与“表示大幅压缩”联合起来检验，不能用其单独表现决定所有联合恢复构造。

attention_panel_fit.py 的 CAL 变换在 heads 之间共享，故跨头均值的问题不应写成“各head独立SVD坐标或符号未对齐”。它丢失的是头身份与头间对比方向。对共享坐标的扰动 \(\delta X_{1}=v,\delta X_{2}=-v\)，头均值不变；实际 W_O 的不同头块一般仍产生不同残差写入。后续 q16 可能补充部分信息，实际缺失量由联合表示诊断决定。

同样，flow输出维数从5120缩到64，并未自动把它跨query变化的函数复杂度降为48。末端目标维度与可观测特征族的维数属于不同轴。

## 4. 拟合、留出和表示下限的含义正确

ridge_flow（109–123行）只用当前便宜 P64 特征估计各列RMS，并用 P64 标签拟合；H448 没有进入此拟合。P64 与 H448 支撑互斥，评价边完全在 H 内。

audit_attention_joint39_assets.py 49–59行以同一48列特征，在H448上计算：

- 无正则最小二乘残差：固定仿射函数族在该H数据上的表示下限；
- 同 λ=0.1 的H448拟合：同正则的全H参考。

这两者在同H拟合与评分，身份是诊断。低噪声 wide 的下限7.55×说明该48列族自身还无法匹配现有独立修正；高噪声下限0.827×、P64结果1.618×则保留了当前族内改善观测/估计的空间。

P64特征矩阵四站点均满列秩48，条件数约36.5–64.3。它证明此坐标系在P处可辨识；表示余项如何被P拟合延拓到H，是当前误差的另一组成部分。

## 5. 数值和执行边界

主教师 attention 来自原生全行运行的P64/H448对应行。attention_joint39_probe.py 的八次独立检查实际查询P64且保留全31668个keys，验证了该注意力子查询。主四臂 tail 和 head 则在512行重放，未测一次端到端64行teacher的完整性能。

两个背景原有四步轨迹与解码输出由 equal_trace 核验。512行重放与全行背景的 flow 差异能量分别为：

| 背景/专家 | live flow 重放差异能量 |
|---|---:|
| Dense/high | 0.354060 |
| Dense/low | 0.347640 |
| wide/high | 0.357174 |
| wide/low | 0.368058 |

相对完整flow幅度约0.274%–0.287%。大幅 joint/independent 排序有足够的数值尺度分离；小交互项与0.07%–0.27%的重拟合收益应以当前BF16重放口径报告，避免将其直接解释为精确连续算子效应。上述 live 重放差异不是所有反事实误差的统一误差棒；wide 背景比较的是 B01 与原live输出，不能直接套作 B00 原生教师的已测噪声界。

joint flow 使用FP32输出，independent与原生head输出为BF16；本轮比较尚未统一候选最终输出dtype。现有 joint 大幅劣化并非因其精度更低导致，真实在线候选仍需绑定实际dtype与完整性能口径。

## 6. 可复用资产与下一证据

已封存向量包括当前X、各分支R/T/U、原生/候选/cheap attention O和投影A、各FFN/块/flow输出、48列特征、gates、P/H索引。它们足够继续分析：

- attention经过cross/FFN/head的实际方向变化；
- 三项Gram与补偿关系；
- 末端hidden到flow的目标几何；
- 当前48列的表示余项与观测延拓。

完整 FFN kernel 特征 x、原生 q16 以外的当前选中非线性特征与完整锚点缓存没有单列保存于这份向量文件。检验保留 kernel 或更丰富 head 特征的读出时，应复用原始 attention 面板与学校可用的FFN锚点/权重资产，按实际可用情况重建这些特征，不能把已有48列当作已经包含所有候选信息。

优先保留已经证实有效的 head 特异读出与 FFN kernel 结构，再检验它们如何在 flow 目标下组合或共享观测，是这轮证据直接支持的下一步。具体组合方式的效果由对应的特征/目标对照判定。

