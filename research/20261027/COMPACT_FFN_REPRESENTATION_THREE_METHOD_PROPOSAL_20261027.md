# FFNDS · 从旧候选的诊断材料到可恢复的新FFN计算结构：A/B/C三方法研究方案

> **2026-10-27 · METHOD_SNAPSHOT / DISCUSSION_ONLY / OPEN。** 基线版本：[FFNDS main 5f1fb066](https://github.com/Qesire/FFNDS/tree/5f1fb06681c35d44b8a53adb36bdb19754aaed87)。本报告把用户提出的三种候选整理为可证伪的数学和部署对象；完成了论文/作者代码与现存仓库证据核对，**本文件未运行新的模型、训练、校准、GPU算子或视频**。这里的优先级是研究方案建议，并非修改[ACTIVE_RESEARCH](../ACTIVE_RESEARCH.json)、注册实验预算、改变旧候选历史结论或替代Agent同行评议。
>
> **一句话判断：让原有UB1、Affine54、量化直方图和局部SSE成为构造新表示的基线与诊断资产；让能够重分配功能、具有可导出低成本算子的FFN成为候选研究主体。**

## 1. 研究框架和方法队列分别保持清晰

[统一I/F/C目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)与[现行路线证据组合](PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)仍是项目的科学入口；它们和本文件的A/B/C**不是一一对应的三个同义分类**。

| 新候选 | 主要关联研究入口 | 新的计算对象 | 优先定位 | 状态 |
| --- | --- | --- | --- | --- |
| **A 可恢复紧凑FFN** | C（执行）＋F（生成行为） | 以6016维非线性FFN取代“D×D仿射＋54组非线性” | **方法主线**：用成熟恢复技术建立强可部署比较器 | SPEC/ALG，恢复后质量OPEN |
| **B 敏感方向编译** | F（后缀响应）＋I（坐标）＋C（折叠） | 将低能但高行为作用的联合输出坐标编译入窄FFN权重 | **科学主线**：机制有意义则可独立成成果 | 数学构造ALG，方向是否存在OPEN |
| **C 编码直接求值** | I（编码信息）＋C（实际求值） | 小投影→离散码→输出系数表→全FFN输出 | **高风险备选**：输入信息本身承担输出求值 | SPEC/ALG，质量/算子性能OPEN |

实时生成、G13因果定位、FP4/MXFP4/NVFP4、Calibration V2属于跨线专题或实验方法；**不作为A/B/C的公共准入门槛**。三个候选可以独立成立。旧的Whole94、UB1、Affine54、R256只承担其已测范围内的诊断或强比较基线；原有RUN、数值阈值和失败版本原封保留。

共同原生对象为LightX2V Wan2.2 I2V A14B四步D-Q、两套high/low expert、每套40个block，FFN的实际边界依次为：post-Q1输入→BF16 W1→GELU(tanh)→Smooth2/HiF4 Q2(64→8→4)→BF16 W2→原调用者e5门控与残差。**去噪step不是视频帧时序**；81 RGB帧、21 latent时间槽与四个step的指标不同。

## 2. 研究判断所依赖的原生事实与成熟先例

**已完成的原生证据**：

- [Affine54](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)：两个已暴露来源×160位置的局部gated SSE均值8.54%，相同投影预算Whole94为11.67%；在指定源中FFN热态约2.182×、完整生成约1.120×，但六视频有模糊/运动与细节偏差。**局部拟合排序与完整视频质量不是同一目标。**
- [UB1 fixed-RR训练内consumer](UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)：24个训练来源的条件下局部gated误差约恢复83.01%；[两个已暴露开发来源六视频、八轴函数](UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)中连续运动部分恢复，但相对Dense仍有缺口及其它质量轴的取舍；[末步latent](UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)与Dense距离略增。以上不能单独识别“低能高敏”这一因果机制。
- [G12联合码结构](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)：固定方向可用离线 $p=W^\top c$、在线 $p^\top x$ 精确计算；单行码直方图的紧界不意味着跨行的联合输出最优。**这是编码求值方案必须面对的强低成本比较器。**
- [A100激活2:4布局](E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)得到预压缩合成SpMM在特定布局上的局部胜出；在线选择、压缩、QDQ与真实Wan全FFN收益尚未测得。A/B/C的成本论断必须包含这个物理先例所揭示的系统开销。

**经核验的论文构造**：

| 先例 | 已证实的方法作用 | 本项目应如何迁移 |
| --- | --- | --- |
| [TinyFusion，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html)；[固定作者代码](https://github.com/VainF/TinyFusion/blob/a147de7c33c2bc12872b9913f459119068b81ac8/prune_by_learning.py) | 搜索**恢复后的**结构，而非只按初始误差；源码233–257行分开优化gate/可训练权重，239–242行提供LoRA选项，444–470行重新载入原初权重后导出被剪模型 | A采用预算内可恢复性作为比较目标；TinyFusion是深度剪枝先例，不能把其层删减当作已验证Wan宽度操作 |
| [Dynamic-in-Few-Step，2026预印本](https://arxiv.org/abs/2607.06631) | 少步蒸馏、随step改变结构及rollout一起优化 | 直接比较少步视频的结构恢复；step-specific宽度本身已有先例，公平记录作者额外训练与新引擎 |
| [CoBRAS，SIAM JSC 2023](https://doi.org/10.1137/22M1513228) | 通过状态协方差＋梯度/adjoint协方差平衡重要坐标，包含斜投影/可学习降阶模型 | B借用“低变化量但高未来敏感”的机制；下述FFN代理目标是新适配设计，**不是其原定理在量化Wan上的直接套用** |
| [LUT-NN，MobiCom 2023](https://www.microsoft.com/en-us/research/publication/lut-nn-empower-efficient-neural-network-inference-with-centroid-learning-and-table-lookup/) | 学习centroid，直接以LUT替代所定义算子的部分乘法，并配套运行引擎 | C检验学到的码能否直接承担完整非线性FFN近似，而不止用码预测mask |
| [LUT-Diff，IEEE TMC 2025](https://chrisplus.me/assets/pdf/tmc25-lutdiff.pdf) | 扩散模型的输出感知码本、分层表配置与移动端CPU/GPU协同；主要近似线性/卷积MM | C必须与**完整非线性FFN输出**的码本、普通小MLP及原LUT-Diff式线性近似做实际对照 |
| [VideoJAM，ICML 2025](https://proceedings.mlr.press/v267/chefer25a.html) | 联合外观/运动的表示和训练目标、推理Inner-Guidance | A/B可借鉴动作/外观分别定义的质量读出，不把RAFT幅值增加或某个latent标量变小当作充分的生成能力恢复 |

外部论文指标只对作者各自任务、训练预算与硬件成立，不移植为Wan的加速数字。复现成熟方法本身可以形成有效工程基线；新增机制归因另需实证。

## 3. A｜可恢复的6016维紧凑FFN（方法主线）

### 3.1 精确的预算和原生输出边界

固定 $D=5120$、$H=13824=216\times64$；Affine54保留$54\times64=3456$个非线性hidden，另有$D\times D$仿射主干。只计每token两投影MAC：
\[
C_{\rm A54}=D^2+2D(3456)
=2D\left(\frac D2+3456\right)
=2D(6016)=61{,}603{,}840.
\]
原Dense主GEMM为 $2DH=141{,}557{,}760$；因此 **6016=94×64维普通FFN在主投影MAC上与Affine54完全等额，均为Dense的43.5185%**。A54的$D^2$项占原Dense的18.5185%，重分配后相当于增加2560个非线性hidden坐标。**等MAC不意味着等时延、等显存、等QDQ或等质量**。

定义$postQ1$输入 $x\in\mathbb R^D$，原调用者gate为$g$。学生函数为
\[
h_\phi(x)=\sigma_{\rm GELU(tanh)}(A_{e,l}x+a_{e,l}),\qquad
\widehat y_\phi(x)=B_{e,l}Q_{2,\phi}(h_\phi(x))+b_{e,l},
\]
\[
A_{e,l}\in\mathbb R^{6016\times5120},\quad
B_{e,l}\in\mathbb R^{5120\times6016},\quad
\widehat{\mathrm{residual}}=g\odot\widehat y_\phi(x).
\]
输入post-Q1和FFN pre-gate输出的边界与teacher一致；调用者仍负责其原数值顺序的e5门控与残差。**$Q_{2,\phi}$需建立学生自己可执行的Smooth/HiF4尺度及有限精度路径**；缩窄学生的量化标尺不可假定等于完整teacher原Q2。部署首版选择每expert×block一份权重（两expert共80份FFN），复用原high/high/low/low调度；是否按step另建bank是独立容量/驻留判别。

### 3.2 “有限后训练”须量化为真实预算

A/B两主矩阵**每个expert-block约6160万个矩阵参数**，80份共约**49.28亿**。若所有权重全量训练，优化器状态与反向显存和训练成本相当可观，不能以“小型适配”命名。预算受限版本应首先对齐：

- **初始化**：Whole94原生合法组选列、MPC/消费者初始化、其它结构蒸馏初始化；最终若均允许更新全部A/B，它们属于同一宽度函数类，初始化是实验因素。
- **有限适配**：固定窄模型基底，用LoRA或部分层/输出矩阵分阶段解冻；例如给A和B都加rank32 LoRA，80份矩阵的LoRA增量约$2\times80\times32\times(6016+5120)=57{,}016{,}320$个参数（只是参数账本，不包括实际训练开销）。初始化、rank和可训练参数范围需与最强同预算紧凑学生对齐。
- **双分布训练信息**：teacher同一真实post-Q1状态的局部pre-gate/gate监督，回答函数等价；student自己free-rollout状态及终端行为目标，回答轨迹稳定性。两者不能在训练统计与来源曝光上混记。teacher/后缀调用的离线代价进入适配总预算，部署在线不使用teacher。
- **结构选择**：预算$\mathcal B$内最大化可恢复生成质量，而不是以原模型未适配时局部SSE直接决定结构：
\[
\min_{\mathcal A\in\mathcal A_{\rm executable}}
\mathcal L_{\rm video}\!\left(\operatorname{Adapt}_{\mathcal B}(\theta,\mathcal A)\right),
\qquad C_{\rm deploy}(\mathcal A)\le C_0.
\]

**最强竞争解释/比较器**：按完全相同宽度、训练来源、参数更新范围、更新步数与计算预算恢复的**普通6016维学生**。未训练Whole94、冻结UB1和Affine54是历史对照，但不代替这个匹配适配后的强基线。若普通学生已经达到高质量–成本Pareto，它首先是成熟结构恢复的有效成果；新机制的额外贡献应另证。

**区分性预测**：相同部署MAC与实际适配预算下，全新可训练非线性容量可能比保留原支持的消费者恢复，更好维护来源间罕见运动/细节功能。若效果只来自训练步数、数据更多或未匹配的更新参数，则需将结论改为训练预算效应。

## 4. B｜下游敏感函数坐标编译（科学主线）

**研究假说**：在真实条件分布中，有些FFN输出方向的变化能量较低，却因gate、attention、后续层/去噪step对生成动作或细节具有较大有限作用。**低能高敏的存在、可重复性及能否低成本实现，目前均为OPEN**。

在指定expert、step、block、source/seed状态和预先定义的suffix读出下，令 $y\in\mathbb R^D$ 为一个token的pre-gate FFN输出（多token耦合另行定义），$C_y$ 为在**明确采样分布**上的中心化输出协方差；用局部响应/adjoint代理$G_y\succeq0$描述后缀读出对输出方向的敏感性。对秩$r$的投影$P$提出设计代理量：
\[
\mathcal E_{\rm proxy}(P)=
\operatorname{tr}\!\bigl[G_y(I-P)C_y(I-P)^\top\bigr],
\qquad P^2=P,\quad {\rm rank}(P)=r.
\]
若考虑双基，取 $\Phi,\Psi\in\mathbb R^{D\times r}$、$\Psi^\top\Phi=I_r$，则$P=\Phi\Psi^\top$可为斜投影。此式是一个局部二次近似/坐标选择**代理设计量**，不是Wan完整非线性、量化、时间轨迹上的已证明误差界；$G_y$的选择、分布平均、量化不连续性及跨token耦合必须实证核对。

为避免先跑完整teacher FFN再投影，在学生**可得**post-Q1输入$x$上计算：
\[
\widehat y=\mu+\Phi B_c\,Q_{2,\phi}
\!\left(\sigma_{\rm GELU(tanh)}(A_cx+a_c)\right),
\]
其中$B_c\in\mathbb R^{r\times h}$、$A_c\in\mathbb R^{h\times D}$。部署前把$\Phi B_c$离线折叠到最终$D\times h$矩阵：
\[
\overline B=\Phi B_c,\qquad
\widehat y=\mu+\overline B\,Q_{2,\phi}
\!\left(\sigma_{\rm GELU(tanh)}(A_cx+a_c)\right).
\]
因此在线仍可只执行普通两主GEMM，且$W_2$受秩$r$输出空间约束；原生BF16舍入与对折叠权重重新量化需单独校验。若$r$接近D或校准后质量不优，则退回A类普通窄FFN即可。

**关键可辨别对照**：固定$h$、$r$、同源训练曝光和计算，比较PCA/能量基、平衡基、随机/同秩基以及无低秩约束的普通窄学生；不仅看heldout局部SSE，还看真实same-state后缀有限变化、对生成行为的预定义读出和完整视频。比较时区分单token平均$G_y$与完整$M\times D$多token后缀；较高的局部敏感性不自动拥有“运动语义”标签。已有P/N标量无法反推出完整$G_y$。

**竞争解释**：轨迹问题主要来自Student状态分布迁移、纹理导致RAFT可观测差异或量化数值边界，而不是低能方向丢失；也可能高敏子空间高度随source变化，根本不能用同一共享基有效压缩。该路线允许以明确负结果建立机制边界。

## 5. C｜编码直接求值完整非线性FFN（高风险备选）

**新研究对象**：从便宜的预FFN输入投影学习足以**直接计算输出**的码，而不是把量化直方图当作后验神经元筛选器。对$x\in\mathbb R^D$：
\[
z=Rx\in\mathbb R^{r_i},\quad
k_b=E_b(z_b),\qquad
\alpha(z)=Lz+\sum_{b=1}^{B}T_b[k_b]
+\sum_{(b,c)\in\mathcal P}T_{bc}[k_b,k_c],
\]
\[
\widehat y=U\alpha(z)+b,\quad
U\in\mathbb R^{D\times r_o},\quad
L\in\mathbb R^{r_o\times r_i}.
\]
每个$k_b$有$K$个码字；$T_b\in\mathbb R^{K\times r_o}$，稀疏二组表$T_{bc}\in\mathbb R^{K\times K\times r_o}$。单组表表示常见码条件响应；少量二组表允许跨组非加性；$Lz$携带连续分量。它是**新训练的完整FFN输出近似函数类**，不是按输入组把GELU线性分解的恒等式。原HiF4-Q2可以作为teacher数值定义，而学生编码系统有独立真实部署格式。

固定编码$Z=E(X)$与监督目标$T$，平方损失下：
\[
\inf_d\mathbb E\|T-d(Z)\|_2^2
=\mathbb E\,\operatorname{tr}\!\left[\operatorname{Cov}(T\mid Z)\right].
\]
这只是经典条件均值性质，提示应减少**输出相关的码内歧义**，不意味着所构造的有限码本必然高效。若$T$是后缀/运动响应，必须说明编码时是否已经合法得到相应条件，而非暗用未来teacher信息。

朴素实现的在线复杂度近似：投影$O(MDr_i)$；$K$原型逐组搜索合计$O(MKr_i)$；连续分量$O(Mr_ir_o)$；表累积$O(M(B+|\mathcal P|)r_o)$；输出展开$O(MDr_o)$，另计索引、读表、同步、权重和码本带宽。表元素规模$(BK+|\mathcal P|K^2)r_o$，另外有$R,L,U$。这里$M$是实际Wan token数，大$M$的dense Tensor Core本身效率很高；MAC下降≠全系统提速。

**最强基线**：同输入投影/输出秩与实际成本的连续小MLP、低秩连续预测器、固定方向精确投影，以及LUT-NN/LUT-Diff的作者定义的算子近似。比较**输出目标码本 vs 输入重建码本**，并记录交互表大小和缓存未命中的真实成本。若非线性方向必须依赖复杂跨码交互、需$K^2$大量表或输出展开吞吐不足，则应改变编码粒度、混合连续计算，或回归A/B，不必继续重复直方图证书。

## 6. 三项决定、最小证据与可复用资产

| 决策 | 首个成本较低、能改变结论的对照 | 真正升级为候选所需的证据 |
| --- | --- | --- |
| **A：更多非线性容量是否可恢复？** | 静态核算A54与6016 FFN的原生形状/参数、可重用Whole94初始化及训练预算；分开teacher/student现有捕获 | 在**相同新训练预算**下，与matched普通6016学生、Whole94/A54/UB1和强论文适配基线比较原生QDQ、全深度生成与质量–速度Pareto |
| **B：是否存在低能高敏且可编译的子空间？** | 检查现存saved tensors是否具备pre-gate $y$、真实后缀readout及导数/有限差；先数学给出何时PCA与敏感基预测不同 | 同rank、训练和源曝光，比较PCA/平衡/普通紧凑学生；强关联必须用原生有限扰动＋新source的自然行为证实 |
| **C：编码是否能替代FFN真实求值？** | 对$K,B,|\mathcal P|,r_i,r_o$做表规模、带宽、连续/GEMM工作与下界核算，并比较简单连续预测器 | 自定义码本及完整FFN值恢复、实际硬件端到端在线成本、强LUT/MLP基线与独立完整视频效果 |

**共同数据合同**：[Calibration V2](CALIBRATION_V2_METHOD_SNAPSHOT.md)区分独立image cluster、seed和每来源的相关row；12×512、24×256、48×128只是在相同行预算下的候选设计，原CAL24旧PRNG与新方案不自动配对。所有来源标识训练/探索/正式隔离确认用途。同source/seed模型与原生噪声需真正重放，teacher_input/student_input在张量边界上分开；新方法改变$Q_2$必须记录student真实码/scale。官方VBench完整协议与自定义两来源八轴函数不等价，连续运动需补全ROI/相机和语义。

**REUSABLE_ASSET_PLAN（运行前登记）**：每方法预计复用：(1) teacher+student完整source/seed/step/expert/block原生post-Q1、post-GELU、pre-gate、gate/quant状态，(2) A窄模型初始化/冻结参数集合与LoRA增量、训练预算，(3) B坐标基和来源级敏感/协方差与有限差，(4) C码表、读表配置、真实布局与缓存流量，(5) full-depth视频、每轴source-level配对和实际算子/整模型成本。每件资产登记producer、方法版本、schema/shape/单位、来源/随机性、locator、复用条件和失效条件；本报告阶段这些新资产全为**NOT_PRODUCED / OPEN**，历史产物仅以原RUN版本复用。

## 7. 研究顺序与交付边界

**先A方法、并行B数学、保持C结构探索**是一个可反驳的资源配置建议，而非永久排序：若强等预算窄学生已足够，可先得到成熟迁移的系统成果；若B的方向规律可独立重现，能够形成科学成果；若C的物理带宽/交互核算有独特突破，可无需等待B便进入下一阶段。

下一位Agent首先完整重建每个方法真实的 **输入→数值算子→输出→gate→跨层/跨step生成** 边界，基于成熟先例与保存资产给出一份最小判别设计；提出的OPEN数学/函数/算子进入[独立发现目录](../method_discussion/discoveries/E0/README.md)，真正的方法提案由独立Agent自己按[QUEUE](../method_discussion/QUEUE.md)提交并经不同Agent评议。提案成立或本报告发表均**不直接触发GPU实验**，具体授权、数据冻结和预算以ACTIVE当前可执行范围为准。

**证据声明**：本报告仅整合已有原生NATIVE/VIDEO/SYS和外部LIT，新增的是ALG预算、SPEC方法和OPEN竞争假说；不声明三方法已有Wan质量收益或新算法首创。现有指标/视频/源码证据指向各自原始文件，保留历史反例为新函数设计和竞争比较器。
