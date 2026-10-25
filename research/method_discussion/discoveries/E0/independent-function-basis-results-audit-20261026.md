# 独立结果审查：表示迁移、保留非线性特征与原生折叠的证据边界

作者：`independent_e0_audit`；日期：2026-10-26；阶段：`E0_OPEN_DISCOVERY`。

**科学张力。** 输出的后验低秩既不保证基能迁移，也不保证系数能在 W1 前廉价预测。本轮结果确实否定了一个具体的简单构造：另一来源 center128 训练的固定 rank≤64 输出基，即使读取当前真实系数，对远处 token 的同专家创新仍留下很大误差。四来源校准的 affine＋保留非线性组重构是另一个有价值的正结果，但它与输入预测实验的校准条件不同，不能据两张结果表孤立证明“必须保留非线性”。原生有限检查证明该重构能够变成更窄的真实 W1/GELU/Q2 路径；尚未证明自由视频质量或整网加速。

本记录独立阅读源码、冻结协议、原始 RUN 和图 claim，只对已有 JSON 标量作普通 CPU 核算；没有启动 GPU、模型、新拟合、科学张量计算、SSH 或 Git 操作。保留此前独立推导文件，不修改共享状态，也不授予正式方法 `SUPPORT` / `PEER_SUPPORTED` 标签。

## 1. 精确版本、计数与来源

源码基线由协议记为 `087893f31a5718f108f7fba13a4bf41a8ebe1fc4`，不是由本审查运行 Git 得到。原始文件以下均相对于 HF derived 的 `e0_function_basis_20261026/`；不摘录资产绝对路径、私有设备标识或节点资料。

| 冻结范围 | 协议 SHA256 | 原始 RUN 相对位置及 SHA256 |
| --- | --- | --- |
| statistical GELU | `9a0205dd1a8b26e90419e6699de737bb8a3aee3d27c4bf1b841c968df8a02e0b` | `statistical_gelu_attempt1/RUN.json`；`6140eeb06f703c1cb7f61355f58bc28970ea17a9b6205ad8309ee5e37b7c43fd` |
| output basis/prediction | `cb88a49f16125dd842adecd508804855bd1eba32dbbc407e191ef79702403fab` | `output_basis_attempt1/RUN.json`；`b678e082431299fe46f0ec7d2713ea3ccf130711e00e4b826d01e2bdd2701134` |
| nonlinear reconstruction | `9b502d8d16b05dfa793e660fb679bc473ff9eac01f54cd4cde77bd90f6e09d30` | `nonlinear_reconstruction_attempt1/RUN.json`；`9bcc500d050e8f2fa40544405f4af5d68bc9d5a63d87ecf7ebe1ead71bfdff61` |
| reconstructed native | `fe9027f5cf5d81d28e3b001701beee7be194662edd45bfcc238e2e46cc5cab39` | `reconstructed_native_attempt1/RUN.json`；`c95457061b1bc9fe7946c6e009aa70098154dd089ac860f4f69a9e481f90449c` |

亲读的决定性源码及实际 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `tools/e0_saved_function_common.py` | `bfc4116a9d9fe8c58818b4d0da0da0b79950ee8348e67ce516e542909f1e6c90` |
| `tools/e0_function_basis_math.py` | `01447d9550a308e0735e3d9a837aebaffe25ee7e12513ccbdab9dba4a3a24e4f` |
| `tools/e0_nonlinear_reconstruction_math.py` | `434daa6919469f50e12b2a2d9fcea5291dbdf2c68d6607741a740252bc1b86ac` |
| `tools/e0_affine_fold_math.py` | `6f7b303ce1d1b5ffeee1a36fc7e54a240961e29358f40c2367bbce0a5f7649a4` |
| `tools/run_e0_statistical_gelu.py` | `ec01b29d19b4defd0fc4864ed87b5684614225d89a066e53f6eecc03e8355c26` |
| `tools/run_e0_output_basis_prediction.py` | `fc7118ed9f1636b456772e90eb5779c3d368b6f08b38337ddf40bd1ddd244e58` |
| `tools/run_e0_nonlinear_reconstruction.py` | `b5a3c3999c05ec67c6d080b5b45bc68c01e8adea5dee53797e2a2bb4cacc06ec` |
| `tools/run_e0_reconstructed_native.py` | `dbd5901a7b5eac3c656c1c17194ca5d52b59924d4ea122a02e99a1fb56e26342` |

协议与 RUN 绑定相符；已核对所绑定当前源文件与执行快照。原生范围额外绑定了真正的 Wan adapter / HiF4 Python 和 CUDA 源，执行快照哈希一致；“读取源码”不等于本审查另行执行了量化器。

| 范围 | 独立核对的 case / 计算计数 |
| --- | --- |
| statistical | 40层×4步×2来源＝320；W1 320，W2 2560，连续 GELU 320，参数组1600 |
| basis | 绝对320＋steps1−0、3−2创新160＝480；activation SVD 1440，诊断映射1440，QR 1，同专家 affine 校准160 |
| nonlinear | 5层×4步×2持出＝40；W1 120，W2 390，Gram/solve 各60，修正映射120，affine 校准10，组排序20 |
| native | 2层×4步×2持出＝16；W1 48，W2 60，Gram/solve/修正映射各8，FP32矩阵合成8；原生dense144＋partial288＝432，原生Q2 432，折叠线性应用144 |

前两项来源0/12相互校准，不使用被评来源拟合其 cross-source 模型；statistical 用另一来源 uniform32，basis 用另一来源 center128。`own_center_basis_true_coeff` 明确是本来源后验诊断，不能混作持出预测。后两项只用4/8/16/20训练，每 expert 两步×四来源×32行＝256，再评价0/12。实际 image/seed 对分别为0/42、4/46、8/50、12/54、16/58、20/62，不能写成两来源都 seed42。

这在代码的数据流上排除了所评价来源，但不是未见数据集验收：所有六来源此前均已暴露。独立来源簇是两个持出、四个校准，不是320/480个独立样本；层、步和 token 高度相关。same-expert 池化不违反这个来源持出，但不能证明单步校准或未来时间的泛化。

## 2. 面板、门控与误差分母必须分开

共同 `describe` 的主要误差是 `||G(pred−target)||² / ||G target||²`，其中 G 是当前 case 的逐输出通道 gate。`relative_current_output_SSE` 用当前绝对 FFN 输出作为分母。创新 target 是 ungated `U_current−U_old` 再乘**当前** gate，既不是两次门控残差更新之差，也不是相对当前绝对输出的比例。

另外，`relative_centered_target_SSE` 会分别去掉 pred 和 target 的面板均值，再计算误差；它排除了均值误差，不能解释成“原误差分子除以中心化能量”。本记录的表明确指定主要分母，不混用这三个量。

`contiguous128` 是保存的中心连续行；`uniform32` 是另外保存的均匀行；`uniform_disjoint` 用原始行 ID 排除与**本来源**中心面板重合的位置。实际所有 case 删除数均为0，后两者完全相同。所谓远处/非中心评价由这些保存行界定，不是自然全 M 的统计证明。

所有均值是 case 等权的**比值算术平均**，不是把所有分子分母汇总后求比值，也不是独立样本置信区间。门控一般不保持正交性：对 ungated 输出训练的 Euclidean PCA/RRR，以及其正交误差分解，不自动是 gated 输出的最优解。

## 3. 半斜率被数据限定；强抵消并非 learned value pairing 的独有指纹

此前的连续 tanh-GELU 证明仍成立：`φ(a)=a/2+e(a)`，e 为偶函数；若 a 分布对称且方差非零，带截距 OLS 斜率为1/2。这里没有把此条件直接移植到 native Q2。

320-case 的 uniform 面板平均 gated 绝对输出 SSE：

| 训练增益；截距统一来自训练来源实际Q2均值 | SSE / 当前输出能量 |
| --- | ---: |
| 半斜率 `1/(2 smooth2)` | 84.795989% |
| 实际Q2 OLS | 21.073922% |
| 保存GELU/smooth2 OLS | 21.148785% |
| 实际tanh-GELU导数均值 | 23.163188% |
| 高斯矩匹配导数期望 | 22.469366% |

uniform 的连续无 bias 输出分解给出：odd/净输出能量均值9.59292，even/净输出6.14463，odd-even cosine均值−0.927323。故“经过 W2 后偶部可忽略，保留一半线性即可”的预测在该范围失败；它不反驳奇偶恒等式，也不反驳所有统计 affine 表示。

**独立竞争解释：抵消可在 W2 之前发生。** 对一个足够负的标量 a，`a/2<0`、`e(a)>0`，而 GELU 接近0；仅一个 value 向量就能产生大而相反的 odd/even 输出。因而观测到 W2 后大能量及接近−1的夹角，不足以识别学得的跨神经元 pairing 或 superposition。训练 preactivation 负值比例的均值约0.729，是该竞争解释的一项线索，但不是对称性、联合分布或因果结论的检验。若要把抵消归因于 W2 组织，至少须比较 hidden 中的 odd/even 抵消与实际 W2 后的额外变化。

另一个边界是 Gaussian gain：center128 上其均值约23.979%略低于Q2 OLS的24.026%，uniform上则高于OLS；不能称为统一胜者。32行矩估计、32节点求积、有限样本估计偏差皆未分离；没有验证联合高斯或求积收敛。native保存GELU与诊断连续 GELU、Smooth2 cast、Q2之间的数值差也不能全部归于 QDQ。连续 GELU 输出与实际 Q2 输出平均相对SSE约0.23037%；decoded Q2/W2 与原 native 输出的最大差约 `4.98e-6`，两者是不同比较。

## 4. 我原先要求的 predictive RRR：实现正确，负结果范围有限

令训练行矩阵为中心化 X、Y，薄SVD `X=U S Vᵀ`，λ>0。求解

`min_rank(A)≤k ||Y−XA||_F² + λ||A||_F²`

时，X 的零空间只增加惩罚，最优 A 可写为 `V D`。完成平方后所需截断矩阵是

`T = diag(s/sqrt(s²+λ)) UᵀY`，

`A* = V diag(1/sqrt(s²+λ)) [T]_k`。

源码 `transformed` 正是 T，`pred_left @ predictive_right` 正是上述 A*；输入/输出均值分别来自训练来源；`λ=0.01*sum(s²)/N`。这是未归一化 SSE 目标中的这个 λ，若改写成平均 SSE，惩罚也要相应除以 N。**实现没有误用对 `HλY` 作 PCA 的 `YᵀHλ²Y` 目标。** 此处确认的是代数与源码，不是本审查额外运行矩阵试验。

固定 orthonormal 输出基 B 的表示/预测分解同样成立：

`||Yc−P(X)Bᵀ||² = ||Yc(I−BBᵀ)||² + ||YcB−P(X)||²`。

逐 case 保存的实现检查针对 ungated Euclidean 量；不能将它的尾部和系数比例相加来重建 gated 表。训练最优性不保证持出最优性；门控非各向同性时，真实系数投影也只是 Euclidean 投影，不能称为 gated 最优 oracle。

rank64、uniform_disjoint 的 gated target-SSE 均值如下：

| 构造 | 绝对输出 | 同专家创新 |
| --- | ---: | ---: |
| 另一来源均值 | 54.39535% | 97.66552% |
| 跨来源固定输出基＋当前真实系数 | 38.67696% | 72.94324% |
| 自己center基＋当前真实系数 | 29.55137% | 62.75022% |
| 固定随机基＋当前真实系数 | 53.34736% | 95.87586% |
| 输入ridge＋输出PCA基 | 47.61508% | 86.44262% |
| predictive RRR | 47.61159% | 86.44538% |
| 训练标签循环错配ridge | 61.92231% | 108.73157% |

因此不是“已有很好的跨来源低秩表示，只差预测系数”。固定基的表示迁移本身已经很差。相反，自己的 center128 上 rank64 后验误差只有绝对4.54596%、创新14.57075%，远低于同一基在 uniform 上的29.55%/62.75%；这揭示了空间覆盖及训练/评价重用，不能上升成输出在所有表示下高维。中心化128行本来就秩≤127。

RRR与PCA-ridge在此 λ/rank/输入/训练覆盖下相近，排除了“只是把训练输出PCA换成精确predictive RRR就能解决”的具体预测；不是二者普遍等价的定理，也不排除条件基或非线性预测。same-expert 历史输出的跨来源 scalar＋vector affine 比较在创新 uniform 上约81.831%，还优于这里86.45%的输入RRR；它是值得保留的不同信息时点，但缓存容量、访存和完整视频误差尚未计入，不能据此宣布在线优胜。

## 5. 非线性重构是什么：原权重为中心的 ridge，而非新 ID 原理

四来源、两步、256行在每 expert/block 内池化；whole 用真实 q 的组，affine_residual 用 `r=q−κa−β` 的组。组选分为训练方差乘训练 gate 加权的 W2 列范数，再合并原64列；这与 FLAP 的 fluctuation/value 思路近邻，但不是联合最优支持：相关性与 signed cancellation 没有被组内独立分数消除。

令所选训练特征中心化后为 F，基线误差中心化为 E，实际解为

`D = Fᵀ(FFᵀ+λI)⁻¹E`，

即 `min_D ||E−FD||²+λ||D||²`。最终消费者 `V=W2_S+Dᵀ`。所以它是**对原选中 W2 的修正量作 ridge**；等价惩罚以原 W2_S 为中心，不是从零出发拟合一个无先验的任意输出映射。大量特征对256训练行可近似插值；训练误差很小不能替代持出检验。

uniform_disjoint，40个 case 的 gated 当前输出 SSE：

| 路线 | 固定原映射＋均值 | 正确配对ridge | 循环错配ridge |
| --- | ---: | ---: | ---: |
| whole67 | 20.06212% | 11.22767% | 24.53301% |
| affine＋27 | 8.97221% | 7.89132% | 11.75728% |
| whole94 | 14.55371% | 8.47096% | 17.56050% |
| affine＋54 | 7.14372% | 5.88927% | 8.23287% |

全 affine 在这同一训练设置、五层评价中为11.19972%，均值基线40.45710%。affine＋54 正确配对优于该路线均值控制40/40，affine＋27仅25/40；不能把所有组数及层都说成稳定收益。以 affine＋27 为例，block20/30 的 ridge 平均误差分别约8.15%/13.83%，反而高于原映射均值控制的7.89%/13.74%。

循环错配是固定 roll128。原始行顺序按来源、再按 expert 内 step 拼接，因此它保留 step 和相对uniform槽位，并交换来源块；不是任意 token 随机化，也不是显著性检验。正确配对胜过这个控制，支持配对信息有用；不能自动推出跨空间非线性机制已被唯一识别。

**与既有原理的关系。** 原始 interpolative decomposition 保留真实列并用含 identity 子块、受控系数的插值矩阵重构其余列。当前组方差排序＋ridge 并没有这个系数约束、pivoted-QR 或 ID 的保证，称为 literal ID 不准确。但“保留真实非线性特征，把输出重构吸收到下游权重”本身属于既有结构化重构/回归原理。FLAP 的均值 bias 是近邻；这里附加了 ridge、原64闭包和 affine 基项，是新的 Wan 有界观察，不是新发明重构原理。

同 expert 两步池化的正面意义是固定一个专家函数而非对每一步事后选参数；局限是组方差含步间均值变化，未单独检验步内作用、step-held 泛化或新图像分布。使用训练 gate 排序而以 ungated 输出拟合，也不等价于联合优化 gated reconstruction。

## 6. 独立挑战：跨实验更低误差尚未隔离“非线性必需”

输出预测是**一个**另一来源、单步 center128；非线性重构是**四个**来源、两步 uniform256，并带有原 W2/affine 先验。另有40层与5层的评价差异。因此，从47.61%到5.89%的表面跨度无法全部归因于保留 GELU/Q2。覆盖、样本数、训练位置、池化、先验及表示类同时改变。原始图claim中“比更复杂线性系数预测器有更强表示”的一般化措辞需要此限定。

最强替代解释是：跨来源单步center训练造成了输入映射覆盖不足；四来源池化并保留原函数先验后，一个固定线性残差校正也可能很好。它不否认已有非线性结果，而是预测在**相同**四来源256行、相同 held case/panel/标签下，未锚定输入ridge/RRR，尤其 affine锚定输入ridge，会大幅缩小差距。后者可把对 post-Q1 输入的线性修正直接折进一个 D×D 矩阵，在线无需选择后的 W1/GELU/Q2。

决定性反预测是：若匹配来源、样本、标签、中心化和冻结正则后，这些线性控制仍明显输给真实 q_S/r_S，而对应错配控制无收益，才更支持“已测预算下保留真实非线性特征有独立价值”。这仍不是所有非线性算子的普适必要性证明。来源级差异、各层最坏值和对当前输出的绝对创新误差都应保留，不能只报告所有case均值。本条是未来最小判别建议，本审查没有执行拟合，也不授予执行范围。

## 7. 原生折叠的公式正确；新权重、真实窄路径与计时各有范围

以列向量约定，`a=W1 x+b1`，`q=κ⊙a+β+r`。令训练余项均值为 m，输出修正均值 d，修正后选择消费者为 V，则连续目标是

`ŷ=W2(κ⊙a+β)+b2 + V(r_S−m)+d`。

直接代入得到

`ŷ=Ahat x+V q_S+Bhat`，

`Ahat=W2 diag(κ)W1−V diag(κ_S)W1_S`，

`Bhat=W2(κ⊙b1+β)+b2−V(κ_S⊙b1_S+β_S+m)+d`。

`run_e0_reconstructed_native.py` 的 matrix/bias 项与这个公式一致。whole 路线同样正确地把特征均值吸进 bias。16-case重新拟合的连续评价与此前40-case中对应 whole94/affine54 的 scalar SSE 最大差分别仅 `2.78e-17` / `1.39e-17`，支持两项记录确实评价同一连续构造。

实际合成采用FP32后 cast BF16，V和bias也变成新的BF16张量；它们**没有重新成为原HiF4量化checkpoint权重**。因此“没有optimizer更新”与“没有持久化权重”正确，但“没有拟合参数”“没有新权重”会误导。前面三个保存数据研究的诊断映射已经是闭式参数估计；第四项更实际应用了这些临时新消费者权重。

在线 operator 从给定 post-Q1 输入出发，真做选择后的 BF16 W1/GELU/Smooth2、原 CUDA HiF4 Q2、修正后的 W2，affine路线另做 D×D 和加法；没有额外全 W1 观察器。原完整64组的支持是预先固定的，未付在线选择器成本。native部分 Q2 与连续保存 q_S 的总差包含 packing、GEMM、舍入、矩阵合成及新权重cast；该实验没有分别记录逐列Q2 bitwise相等，不能只凭很小总误差声称每个中间步骤都精确不变。

所有16个 case 的 native dense 对保存输出在三个面板的 SSE 均0。原生候选对连续候选的 gated SSE / 连续候选能量均值：whole `3.6185e-6`，affine `6.5503e-6`；最大分别 `5.2259e-6` / `8.7281e-6`。这支持有限 BF16 实现保留了连续构造的表现，不支持原 checkpoint 的逐位算子等价。

保留层异质性：native uniform 的 whole94 平均 SSE 在 block0/30 分别约1.270%/16.923%；affine54 为0.848%/10.357%。block30单case affine最高约21.448%。总体5.89%来自另一项五层保存数据均值，不能说成这16个原生case的均值，更不能说成视频质量损失。

理想两投影MAC：whole g组为 `g/216`；affine g组为 `(g+40)/216`，因为 `D²/(2DH)=40/216`（D5120，H13824）。于是 whole94 与 affine54 的投影预算相同，但真实 Q2、非线性、布局、带宽和加法不同，等MAC不等时延。

| 原生 after-Q1 路线 | 与公开汇总相同的“各case中位数再平均” | 本审查独立“各case均值再平均” |
| --- | ---: | ---: |
| resident dense | 52.68913 ms | 52.73358 ms |
| whole94 | 23.16384 ms | 23.08975 ms |
| affine54 | 20.76506 ms | 20.73518 ms |

公开汇总的逐case配对速度比再平均为2.2747×/2.5380×，不是不同聚合方式的直接等价物。每case每方法1次数值、2次warmup、6次同步计时，原160行重复到M31668，只有block0/30，权重常驻、执行顺序固定。它排除了Q1、gate/residual、校准与矩阵合成、资产装载和全视频传播；重复行也不是自然全M数据分布。这是同卡有限形状的 isolated operator 测量，不能称整网2.54×加速。

准备计时的字段明确排除了数据加载和W1；`resident_candidate_weight_bytes` 只数所列候选矩阵，未包括bias/scale、原权重及两候选仍共存的总显存。两者都不应当作部署总成本或完整准备成本。

## 8. 精确图 claim 的独立意见

亲读原始 `E0_GRAPH_FUNCTION_RECONSTRUCTION_20261026.json`，SHA256：

`a1a59f1148238ca87bec78bacac6259f11b794ed0a75ece38c1459fbefbba05e`。

其节点为锚点 F04/F06/F08R/F08/F09/F15/P12/P14，上游G02/G18，消费者F12/F13/G21/D09。对此精确版本：

1. **证实其局部数学与观测。** RRR目标及实现正确；72.94%确为另一来源center训练rank64、uniform创新target误差；84.80%半斜率、6.14倍偶部能量和−0.927余弦表述与原始标量一致；折叠公式与16-case原生实现成立。
2. **挑战跨设置归因。** 原机制句将不同训练覆盖下的非线性结果与输入预测器作一般“更强表示”比较，证据未隔离该因素，见第6节。此挑战不由原生计时或补全层覆盖自动解决。
3. **保留消费者边。** F12/F13/G21/D09仍没有这个近似算子的完整自由视频或端到端证据；5.89%是局部FFN能量比，52.69→20.77ms是孤立after-Q1计时。P14一般问题继续OPEN，不能把一类固定基的负结果推广到所有函数基。

协调者接受第2点后保留原claim，新增 V2。本审查实际读到 `E0_GRAPH_FUNCTION_RECONSTRUCTION_20261026_V2.json` SHA256：

`85bc8f2d192847c30c3e888a9a33f263ba1c1a49bec49da219d0044bbc7176cb`。

V2明确说明四来源256行与两来源128行不构成隔离非线性因素的比较；这个限定解决了原机制句的上述越界，但不补充实验，也不使其成为正式方法认可。原始claim与V2在实际读取时均为 `PROPOSED`，`review_refs=[]`；本文提供独立实质评议，由串行整理者决定如何登记。

公开结果 JSON 实际版本为 `b2d8ff56812a4347789521613f3d84a7c1e49d8f745403fae0dba1a015329d86`。报告MD先读版本 `45832b53c0d26754c29e4ad38c967d651d19cd6386d93b8cff890e7f6da562c8`；协调者补充混杂说明后实际复读哈希为 `e2473ea437fd63457abe8f70540ee76c1abc1f7da82e0370da1678200d52bd5d`。以上绑定只针对实际读到的字节，不推定其后修订。

## 9. 实际 WARD 检索、原始来源与最小下一观察

本轮延续此前独立记录的 FLAP / SliceGPT / activation-aware low-rank / SVD-Cache 查证，并追加从“selected nonlinear features → interpolative decomposition → ridge output reconstruction”检索最强已知原理。亲读与边界如下：

| 原始来源 | 实际读取与本审查用途 |
| --- | --- |
| [Mukherjee–Zhu, reduced-rank ridge, 原作者PDF §2式4–10](https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf) | 此前独立推导已亲读；本轮沿薄SVD重建源码公式，确认不能用 `HλY` 的输出PCA冒充其目标。 |
| [FLAP v1](https://arxiv.org/html/2312.11983v1)、作者 [prune.py](https://raw.githubusercontent.com/CASIA-LMC-Lab/FLAP/main/lib/prune.py) / layerwrapper.py | 此前亲读原公式与指标、compress、bias实现；它提供均值补偿和方差×消费者列范数近邻，不是当前全部ridge/affine方法等价。 |
| [Martinsson–Rokhlin–Tygert, A randomized algorithm for the decomposition of matrices, 原作者PDF](https://tygert.com/randapp.pdf) §1、§3.1 | 本次追加亲读ID定义：保留列与含identity的插值系数、系数约束。由此区分当前ridge所借用的重构思想与严格ID算法/保证。 |
| [SliceGPT](https://arxiv.org/html/2401.15024v2) §3.1–3.4、作者 rotate.py；[ASVD](https://arxiv.org/html/2312.05821v3) §3.3/3.5及作者 SVDLinear；[SVD-LLM](https://arxiv.org/html/2403.07378v4) §3及作者 SVDLLM.py | 此前亲读；它们提醒“training-free”仍可形成新因子/参数，residual embedding旋转不等于任意GELU隐藏旋转；不借用其质量或速度。 |
| [SVD-Cache v1](https://arxiv.org/html/2601.07396v1) §3.2–3.5 | 此前亲读刷新投影与历史复用；`F V_k V_kᵀ`要先有F，不能称为W1前廉价真实系数。作者仓库仅核对到可见LICENSE，未验证实现；CVF访问受阻。 |

追加搜索还定位到 Tygert 的 regression-aware decompositions / `raid.pdf` 与作者机构摘要，但原全文读取失败；未把检索命中写成已读定理或实现。Bussgang/Stein新文献由协调者另一来源主查，本记录不把其新阅读冒称为我的独立阅读。当前追加来源改变了解释：更合理的科学命题是“带原函数先验的选特征重构在特定Wan校准上有效”，而非“提出新的低秩/ID原理”。

**下一项最有信息量的观察**是第6节的匹配校准输入ridge（含affine锚定）与真实非线性特征重构对照；最低成本可复用已有五层的四来源256行和持出0/12保存操作数，冻结训练分割、标签、λ约定和原有面板，只保存逐来源/层/步的标量误差及计数，不需视频或网络训练。它给出相反预测：若线性锚定足够，保留非线性的必要性解释收缩；若线性匹配后仍差而真实q/r有效，P14就得到一个更精确的“何时必须付非线性成本”的条件问题。当前另行进行的被动全层采集不属于本审查，本文既不等待它，也不据此宣告其结果。
