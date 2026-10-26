# DISCOVERY · 固定四坐标是否预测了另外十二个函数

**四个 anchor 的连续 GELU 系数本来就是这四个输入投影的已知函数；可压缩信息的关键见证是另外12个神经元的 signed 输出能否由同一组坐标跨来源预测。** 将这两部分分开，再保留二者的 signed 交叉项，可以区分“非线性基主要重现已观察的 anchor”与“这些坐标携带额外函数信息”。最便宜的下一动作是同一128行上的分块归因加两个固定来源持出方向，复用已有 W2 Gram；保持当前两套15列基，不增加阶数或重新选坐标。

日期：2026-10-26。作者：independent_e0_audit。当前文件是设计和数学推导；未调用保存张量分析函数、模型、GPU 或 native QDQ。若协调者采用本设计，另行冻结工具/协议并登记一次 CPU 范围。旧协议、源码、RUN 和共享状态保持原字节。

## EVIDENCE · 实际结果提出的问题

本人读取了原工具、实际完成 RUN 的公开精确副本及其 HF 原件并核对 SHA。continuous 部分16输出的 raw 相对 SSE：常数85.5012%、四坐标 affine65.3436%、joint quadratic45.8738%、separable15为42.6921%，同rank15随机行空间75.3412%。减去共同128行均值后，两个非线性族分别残留53.6528%/49.9316%的创新能量。两者真实数值rank均15，条件数6.2864/13.5567。这些值来自读取现有标量，未重新分析操作数。

这说明固定输入坐标的非线性空间优于本次 affine 和随机空间。现有 RUN 没有保存 anchor4/neighbor12 的分块误差或跨来源拟合，因而尚未识别收益的归属。连续→pre-Q2 很接近、post-Q2略差，也没有完成该归因。

精确身份：

| 对象 | SHA256 |
| --- | --- |
| 原 RUN / `research/20261026/E0_FIXED_COORDINATE_BASIS_ORACLE.json` | `09e3acd05bf782405ebc39dddadb662cf8530677d24690d319520da1ca914c6c` |
| `tools/e0_fixed_coordinate_basis_oracle.py` | `d46c128716f7cf76b2b7a22e80ea875039590e16e3223f4548f4c3cef57bd793` |
| 原固定计划 `independent-fixed-coordinate-basis-oracle-plan-20261026_v1.json` | `8ad454c8dc359e99b6f10c9396da95e9981061ed981bd4566f550cf84efef2cd` |
| 本次亲读结果报告 `E0_FIXED_COORDINATE_BASIS_ORACLE.md` | `729fe00e7808dba2808a3be7dc89ebccaccbfa2c8c636144149376c414cdcf07` |

HF 原 RUN 的相对位置为 `e0_fixed_coordinate_basis_oracle_20261026/attempt1/RUN.json`。原 shard 继续通过绑定的私有 runtime manifest 解析，公开只记 `high_noise_model_block_10.pt` 与既有 SHA；本设计无新资产路径。

## DISCOVERY · 先把真正的新信息目标写出来

固定 anchor 集合

\[
A=\{0,4607,9215,13823\},
\]

以及原16集合中其余12个坐标

\[
N=\{505,1593,4367,5848,6958,7555,7802,8145,8860,10080,11914,13651\}.
\]

它们沿用原有选择，所有来源/步骤相同。令原 W2 的这些列为 \(V=[V_A,V_N]\)，系数为 \(C=[C_A,C_N]\)，固定基投影后的残差为 \(R=[R_A,R_N]\)。同一 raw 目标满足

\[
L_{16}=\|R_AV_A^T+R_NV_N^T\|_F^2
=L_A+L_N+2X_{AN},
\]

\[
L_A=\operatorname{tr}(K_{AA}R_A^TR_A),\quad
L_N=\operatorname{tr}(K_{NN}R_N^TR_N),\quad
X_{AN}=\operatorname{tr}(K_{AN}R_N^TR_A),\qquad K=V^TV.
\]

每份原 gate 只将 K 换成其已有 \(K_g=V^T\operatorname{diag}(g^2)V\)。因此两个正 SSE 与 signed 交叉项须共同保存；分别相除后相加不构成总误差。用 direct full16 Gram 能量独立核对重组，相对缺陷并列记录。

从 affine 换到非线性基 b 的收益亦可精确分块：

\[
\Delta L_{16,b}=\Delta L_{A,b}+\Delta L_{N,b}+2\Delta X_{AN,b}.
\]

首要观察是 \(\Delta L_N\) 和 N12 自身的创新归一误差，而非 full16 分母下的一个小数。即使交叉项使总误差更小，也应保留 N12 独立误差的真实变化。每个 neighbor 的系数残差/创新与 W2 对角加权能量另作标量，判断有符号输出收益是否集中于一两个坐标；这不要求每个坐标都同等重要。

连续 anchor 系数具有明确的解析基线：

\[
C_{A,j}(T)=\phi(T_j+b_{A,j})/s_{A,j}.
\]

若四个 anchor 投影已付费，这四个 GELU 是已知计算；令它们精确通过时，新近似误差只剩 \(R_NV_N^T\)。这是当前小范围连续函数的结构事实。保存 post-Q2 的 anchor 系数还携带原共享 scale/其它 hidden 依赖，不能把该解析旁路自动称为原生 Q2 等价或免费观察。

## 最小固定判别：一次读入、两种证据层

**保持数据范围**：high-noise block10，来源0/12、seed42/54，各 step0/1 原 uniform32，共128行；目标仍分别为 continuous、保存 pre-Q2、post-Q2。保持 constant、四坐标 affine、当前 joint15、当前 separable15，继续使用标准 eps 主数值rank与条件诊断。不同阶数、坐标搜索、ridge 参数扫描均不进入本设计。

第一层重建原 pooled128 的上述四个空间和预seed同rank随机行空间，将现有总误差分成 A4/N12/cross。该层仍是 `OFFLINE_FUNCTION_CLASS_ORACLE`，用于回答原43%–46%误差和非线性收益来自哪一部分；不把样本内投影称为预测。

第二层固定两个拟合/评价方向：

| 方向 | calibration | held rows |
| --- | --- | --- |
| 0→12 | source0 两步共64行 | source12 两步共64行 |
| 12→0 | source12 两步共64行 | source0 两步共64行 |

坐标中心/尺度、特征列中心/尺度、SVD rank 和 readout 都只从 calibration64 取得，原样应用于 held64。对每个固定基，训练特征 \(\Psi_{tr}=U\Sigma W^T\)，以预定标准 eps rank r 求暂存闭式映射

\[
B=W_r\Sigma_r^{-1}U_r^TC_{tr},\qquad
\widehat C_{te}=\Psi_{te}B.
\]

这是有标签闭式校准，数学上属于拟合；应记录诊断映射数，避免因不保存 bank 就记作“0 fits”。目标 C 的16列一起处理，然后按 A/N 分块评分；不为 A/N 另选基。由于 V 固定，对 C 的共同 raw 投影给出一个 raw-output 最小二乘解；gate 继续只评分此解，保持“非共同 gated 最优”的明确语义。rank不稳定、条件放大或训练秩亏情况下，迁移解释保持 OPEN，并报告 held 特征在训练可识别右空间之外的能量；不把截断失配算作族的严格排除。

held N12 同时报告 full partial-output 相对SSE，以及相对于 **calibration 均值预测** 的误差比

\[
E^{const}_{te,N}=\|(C_{te,N}-\mathbf1\bar C_{tr,N})V_N^T\|_F^2,\quad
R^2_{N,te}=1-L_{N,te}/E^{const}_{te,N}.
\]

R²允许为负。另保留每个 held step 当前 gate 的同一预测误差；不在评价来源上重新选截距或中心。增加一个几乎零乘法成本的 **训练来源按 step 均值** 对照：step 是已经给定的状态，若该简单对照已解释收益，就应将发现归于步骤统计，而不是额外 token 级函数坐标。

### matched-rank 容量与预测控制的区别

原 pooled 随机行空间保留为 A/N 分块的容量参考。它没有跨来源的函数映射，不能凭空在 held64 上再投影并称为预测。

两个非线性基各加一个固定训练 token 错配控制：对每个训练 step 的32行，用预先冻结的 Sattolo 单环置换（seed=`20261026+source+step`，记录真实行置换）**整行一起置换全部特征列**，标签 C 保持原序。评价仍使用真实 held 特征。若 \(\Pi\) 是这个分步置换，则

\[
\widetilde\Psi_{tr}=\Pi\Psi_{tr},\quad
\widetilde\Psi_{tr}^T\widetilde\Psi_{tr}=\Psi_{tr}^T\Psi_{tr}.
\]

所以训练 rank、奇异谱、常数项、特征维数和按 step 的输入分布完全相同；U 只需左乘同一置换，旧分解可复用。该控制打断 token-label 配对，保留 step 均值联系。整行置换同时保留四个坐标之间的相关结构；它不是逐列 feature importance 或因果删除试验。单seed控制不提供显著性概率。

4个普通空间 × 2方向 × 3目标 = 24个暂存 readout；2个非线性错配控制 × 2方向 × 3目标 = 12个，合计36个诊断映射，另有6份按step训练均值表。它们全部在进程内消亡，输出只存标量、rank/条件信息和行号，不保存部署 readout、预测张量或新权重。

## 强竞争解释、不同预测与预先决定的行动

一个反例为 \(Y=V_A\phi(T)+\epsilon V_N\zeta\)，其中 \(\zeta\) 与 T 独立。非线性空间会明显优于 affine/随机，并随 anchor 能量占比取得漂亮的 full16 SSE；N12 的可迁移信息仍为零。另一反例是 T 只标识已知 step、目标主要变化为 step 均值：pooled 两来源收益可能很大，训练按step均值与错配控制却可复现它。二者均与当前未分块结果相容。

本设计的决定规则是研究去向，非方法或质量接受：

- **anchor主导**：非线性相对 affine 的收益在 A4 或 cross，N12 pooled 自身改善很小；两个方向的 N12 预测均未优于训练均值/affine/同rank错配。结束这组固定四坐标、两套函数空间作为12-neighbor压缩路线的扩展，保留 anchor 解析作用；改问哪些额外输入坐标缺失。
- **暴露样本关系**：N12 pooled 确实有收益，source-held 不能保留，或只在一个方向/step有效。将这组共同来源无关 readout 保持 OPEN；优先输入支持/分布与条件异质性，不追加多项式阶数。
- **跨源函数信息**：N12 在两个方向同时优于 affine、按step均值和同rank错配，且各自数值rank稳定。建议预冻“至少降低各方向 calibration-mean 创新误差10%”作为值得扩大一次独立来源描述的实际效应门槛；该10%是尚未读取 N12 结果前提出的研究资源启发，不是统计或质量阈值。两来源内的32行与两步不是独立重复，不报告 token bootstrap 显著性。
- **数值分域**：continuous/pre-Q2 的 N12 迁移保留而 post-Q2 丢失，优先研究四坐标之外的 Q2 状态/scale 信息。它不直接授权删除原量化器。三种目标若均丢失，则本低阶函数描述尚未找到可迁移信息。

达到第三种预测，支持的是扩大到一个未用来源的同层独立保存数据观察；尚不支持全40层、多步替换或视频质量。来源0/12已在前序结果中暴露，两个 held 方向只隔离本次拟合标签，不是从未见过的独立最终验收集；其 source/image/prompt 与 seed42/54 同时改变，不能单独归因某一个因素。

## EXECUTION · 成本与证据路径

当前仅设计。未来最小 CPU 实现可复用本次 exact-SHA RUN 中的 **raw16×16与4个gate16×16 Gram**，保留原 W2/shard 身份；无须再重算这些 Gram，节省6,553,600 MAC。原固定16 W1重放仍付10,485,760 MAC，临时小特征投影/闭式拟合与分块评分可共用数组和分解；目标上限维持20M显式MAC、1线程/2GiB/30秒/1MiB，实际工具必须预扣并以源码审查确认，预算不足即失败收据而不扩范围。小特征 SVD/QR 另行计数，0 GPU/模型/nativeQDQ/模型权重分解/视频/持久化bank；单次独占输出目录，无自动重试。

应从 Gram 子块构造 A/N/cross，复用全部16系数预测；原完整 W2 的新投影、任何全FFN/native重放、PCA/ridge扫描都不必要。亲读原源码时确认旧 `features()` 每次在输入全体行计算中心/尺度；**它可以复现 pooled oracle，heldout 部分需要显式 train-only fit/transform，不能直接在128行上调用后再切分。**

图锚点为 F04（W1输入投影）→F06（连续GELU）→F08R/F08（量化依赖与真实系数）→F09（signed W2输出）。当前新增的是 posthoc/conditional 归因和转移预测边。对连续小簇，若四输入坐标加14个非恒定基能预测必要 N12 读出，理想两侧投影宽度由32降为18；保留精确4个anchor另加14个基时需把相应4个readout也计入，宽度成为4+4+14=22。二者都是形状预算假说，尚无原生打包、端到端时间或完整视频证据。

## 本轮实际阅读的一手来源与检索改变

2026-10-26 亲自检索和打开：

- [Cawley & Talbot, JMLR 2010 原论文](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf)，摘要、§1、§4.2相关图与§5.1：有限评价集上的二级选择也会过拟合。它促使本设计保留两套已冻结基并停止阶数选择，明确两来源已暴露和后续真正独立来源的角色。
- [scikit-learn 官方 data-leakage 文档](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)，§12.1–12.2：训练期确定变换并应用于评价集。由此明确旧 pooled `features()` 不能直接作为 heldout preprocessing。
- [scikit-learn 官方 correlated-feature permutation 示例及原代码](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)，数据拆分、原 permutation 调用与结果讨论：相关特征的逐列置换并不直接表示无信息。本设计因此使用整行配对错配、保留坐标相关性，仅作同rank负控制，不把置换差额称因果贡献。

本人也逐段读取了固定基工具、原结果报告、旧可辨识性澄清及独立源码review v7。本文提出的是已知最小二乘/持出/配对控制在该精确 FFN 问题中的 discriminator；未声明新回归算法或完整 prior-art 排他性，未搜索或执行新的部署算子。
