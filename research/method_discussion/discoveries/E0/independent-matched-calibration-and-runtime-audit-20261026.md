# 独立审查：同校准控制改变了结论；全模型原型仍须区分源码与执行

作者：`independent_e0_audit`；日期：2026-10-26；阶段：`E0_OPEN_DISCOVERY`。

**新认识。** 此前我提出的校准混杂挑战得到了真正的匹配控制。在相同四来源、同expert两步、256行训练以及全40层的同一持出标签下，affine锚定输入ridge的uniform输出SSE为15.3535%，affine＋54组重构为8.54194%，后者320/320案例更低。因此，“只是单来源center校准不足，加入相同覆盖与affine先验即可解释全部差距”的具体替代解释，在这个冻结ridge设置下不成立。这个更新不等于证明所有线性函数都不够，也不把局部SSE升级为视频质量。

本记录只读协议、源码、原始JSON与已保存标量，使用普通CPU求均值、最大值和哈希；没有新拟合、科学张量计算、GPU、模型、视频、SSH或Git操作。仅创建本文件；此前独立报告及其原SHA保留。原型审查是对准备运行的代码作静态检查，不宣告原型已运行，不授予方法标签或执行权限。

## 1. 证据身份与复核范围

| 对象 | 实际SHA256 |
| --- | --- |
| `config/e0_all_block_reconstruction_20261026.json` | `e108164bb36c24bf12b5f5bb080ccb04a1fa695815271aaca3def25b292b308a` |
| HF相对 `e0_function_basis_20261026/all_block_reconstruction_attempt1/RUN.json` | `27889f48aa2ce19ead820b09b69bc8f52b0b70a51cbd44fec950b82100b74b6b` |
| `research/20261026/E0_ALL_BLOCK_RECONSTRUCTION.json` | `b71e2094ee353c4929fa74ddc8bcd6f9bafdae1509eb5078a0454f9917e4a3c0` |
| `tools/run_e0_all_block_reconstruction.py` | `a80df9ab012a807607d473e8bbda4d90631683e2d1ee2f5b88e9d97fa4e8bbbc` |
| `tools/e0_merged_calibration.py` | `87ec0d760399d9015cc2dd912aea251224263a1dd02f014a3fdf7c9ab196f76c` |
| `tools/e0_reconstructed_runtime.py` | `957588f5dc3b32334982b6247064ddece1d21ab627267eb717bc72fb31003872` |
| `tools/build_e0_reconstruction_bank.py` | `a02eb8270d62d3ad956eb871c6fe3578a9a26e0e952801a4f01e1cf8f507175e` |
| `tools/ffnds_all_weight_residency.py` | `d19c14f57b17426f6f3497c2af1fd726c1fb9756a46cb559d60108e8103147d4` |
| 原 `lightx2v_wan_distill.py` adapter | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |
| 原 `transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |

video driver第一次亲读版本SHA为 `d9b0fb57787cab6c1e422c1073a83cc65d872800f2e00a7b68948b15e1eb2779`；反馈计时不对称后，亲读显式weight preload修订版SHA为 `b47582366f23e8ed6c40ea630eb8ad61e99cda698d5f2fdefb762c3a198025de`。这些是尚未冻结时实际读取的版本，不能替代后续冻结协议或执行快照；后文分清初版问题和已见修订。

随后实际读取输入绑定修订版 `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005`；该版本已补齐input/prompt与cache manifest哈希检查。仅用Python AST解析确认runtime、video driver与bank builder语法有效，未导入或执行它们。这个修订处理了第7节所记输入绑定问题。

已独立核对all-block协议与原始RUN一致、全部绑定源文件与 `EXECUTED_SOURCES` 一致。320个case无重复，80项expert/block校准，每项256行。计数逐项相符：W1 960，W2 3120，Gram/solve各560，修正映射1280，affine校准80，组排序160；0新生成、0原生FFN/QDQ、0optimizer、0持久化拟合权重。对旧五层40case的所有共有方法、面板和标量逐项比较，最大绝对差**确为0**。

来源分割仍是4/8/16/20训练，0/12持出；实际seed分别46/50/58/62及42/54。MergedCalibrationStore只在旧五层调用旧CalibrationStore，其余35层切换到有SHA绑定的新增被动采集；不使用0/12去补训练。六个来源均为历史暴露image/seed组合，独立持出簇仍只有2个。

## 2. 新输入控制的数学，以及它解决的具体混杂

X为同一256行的实际post-Q1输入，Y为同一256行保存的原FFN输出，中心化X记为Xc。共同dual ridge实现

`D = Xcᵀ(Xc Xcᵀ+λI)⁻¹E`，`λ=0.01*tr(Xc Xcᵀ)/256`。

direct控制取 `E=Y−mean(Y)`，输出为 `mean(Y)+(Xtest−mean(X))D`。anchored控制取 `E=Y−affine(X)`，再对E去训练均值，输出为 `affine(Xtest)+mean(E)+(Xtest−mean(X))D`。这正是我此前提出的两个竞争控制，没有悄悄换成已付全W1的当前观察器。

它们不是旧rank64输出PCA/RRR：这里没有输出基截断、没有activation SVD，直接拟合全部5120维输出，线性修正的秩上限来自256行中心化训练而非人为rank64。affine锚定项仍可离线折进D×D线性矩阵；保存数据评价中为了诊断计算的全W1，不是这种未来折叠实现必须付的在线观察成本。该线性实现本次没有原生计时或返回网络。

这次真正匹配了：训练来源、source-held分割、每expert两步池化、行数、uniform位置、输出标签、中心化约定、trace/N正则规则、评价层/步与面板；anchored控制还匹配了原来的affine函数先验。全40层消除了用原五层平均代替全层证据的问题。

仍未匹配或解决的是：

- 特征坐标的协方差及正则几何不同；同一trace比例只匹配固定规则，不是同一个函数空间惩罚，也没有证明该λ是最佳持出λ。不能把一个冻结估计器的负结果称为任意线性映射的不可表达性定理。
- 非线性路线另保留了原选中W2与真实q/r的函数先验、并用训练gate做组排序。对“这种付费特征是否有用”这是待检验因素本身；对“所有其他特征或支持是否更差”则不是穷尽对照。
- 输入D×D线性控制理论投影预算约40/216，affine54与whole94约94/216；这不是相同在线MAC的线性/非线性优劣试验。较低误差不能单独决定质量—成本前沿。
- 四来源每步同一uniform布局与128中心面板依然是小样本空间覆盖；来源、层、步并非独立样本，且持出来源此前已暴露。没有新seed、step-held或真正全M局部误差。
- 拟合是ungated Euclidean目标，评价是当前gate加权误差。不能把其最优性移植到门控后目标或下游视频。

错配仍是固定roll128，保留expert内step与相对uniform槽位、交换来源块；它检验配对有无用，但不是任意置乱、显著性检验或唯一机制识别。

## 3. 平均、来源与最坏case

以下均为case等权的 `||gate*(prediction−original FFN)||² / ||gate*original FFN||²`，不是RMSE、全局能量池化、残差相对误差或视频质量。uniform_disjoint的行数全部32；与uniform32相同。center128另列，不混合分母。

| 控制 | center128平均 | uniform_disjoint平均 |
| --- | ---: | ---: |
| 训练输出均值 | 48.28365% | 49.42618% |
| affine | 18.38658% | 17.28119% |
| direct输入ridge | 31.17316% | 27.76351% |
| direct错配 | 73.11501% | 69.17290% |
| affine锚定输入ridge | 17.39597% | 15.35352% |
| affine锚定输入错配 | 25.90948% | 23.30803% |
| whole94均值/原映射 | 19.37212% | 19.32746% |
| whole94 ridge | 13.15066% | 11.67327% |
| whole94错配 | 24.88216% | 23.13883% |
| affine54均值/原映射 | 11.18730% | 10.43545% |
| affine54 ridge | 9.63036% | 8.54194% |
| affine54错配 | 13.74130% | 12.44006% |

uniform上affine54胜anchored输入ridge **320/320**、胜whole94 **305/320**；40层均值中胜whole94为38/40，例外为block1/2。anchored输入ridge仅240/320胜全affine，所以“输入残差拟合总有收益”也不成立。direct正确配对胜错配320/320，anchored为319/320。

| 来源/实际seed | 控制 | uniform均值 | 此来源最坏case（step,block）及误差 |
| --- | --- | ---: | --- |
| 0/42 | direct输入ridge | 30.76445% | (0,28)，104.61868% |
| 0/42 | affine锚定输入ridge | 17.69305% | (0,7)，51.43430% |
| 0/42 | whole94 ridge | 12.85631% | (0,28)，32.82878% |
| 0/42 | affine54 ridge | 9.84145% | (0,28)，24.86325% |
| 12/54 | direct输入ridge | 24.76256% | (0,7)，66.06290% |
| 12/54 | affine锚定输入ridge | 13.01398% | (0,8)，42.82118% |
| 12/54 | whole94 ridge | 10.49022% | (0,26)，26.79938% |
| 12/54 | affine54 ridge | 7.24244% | (0,8)，22.35449% |

center的最坏case也不同：affine54为source0/step0/block11的27.68672%，whole94为source0/step0/block28的35.39643%，anchored输入为source12/step0/block7的53.95145%。不能仅凭uniform最坏位置设计全视频保护规则。

按step的uniform均值，affine54依次为12.66322%、8.80586%、6.72953%、5.96916%；whole94为15.59643%、11.23908%、10.44121%、9.41636%。这是本模型high/high/low/low的相关观察，不证明expert身份是因果来源。

全层更难：原五层affine54均值5.88927%，新增35层8.92090%，全40层8.54194%；whole94对应8.47096%、12.13074%、11.67327%。原五层结果没有错误，但明显低估全层平均难度。

## 4. 全40层的均值/最坏值表

每格为该层8个来源/步骤case的 `uniform平均% / 最大%`，不是置信区间。保留本表便于追踪层异质性，避免只摘总体均值。

| block | affine锚定输入ridge | whole94 ridge | affine54 ridge |
| ---: | ---: | ---: | ---: |
| 0 | 5.49/9.89 | 1.27/1.74 | 0.85/1.18 |
| 1 | 3.42/4.25 | 1.17/1.65 | 1.28/1.87 |
| 2 | 6.81/10.41 | 2.84/4.69 | 2.98/5.11 |
| 3 | 11.66/21.87 | 5.41/9.15 | 4.97/9.38 |
| 4 | 29.45/47.98 | 14.60/20.36 | 12.76/19.66 |
| 5 | 34.58/40.89 | 18.35/22.52 | 15.78/18.58 |
| 6 | 27.26/39.89 | 17.13/23.43 | 13.99/21.73 |
| 7 | 28.85/51.43 | 16.46/25.16 | 13.68/21.80 |
| 8 | 20.63/42.82 | 13.96/23.19 | 11.37/22.35 |
| 9 | 13.90/32.76 | 9.64/20.22 | 8.08/18.59 |
| 10 | 18.37/29.15 | 14.89/21.34 | 11.83/19.04 |
| 11 | 19.14/31.60 | 15.65/22.24 | 11.79/19.00 |
| 12 | 18.83/24.61 | 16.66/20.24 | 12.04/15.72 |
| 13 | 13.98/23.18 | 11.79/16.12 | 8.73/13.76 |
| 14 | 19.31/32.69 | 17.06/22.22 | 11.80/18.19 |
| 15 | 12.46/22.84 | 11.29/15.85 | 7.87/13.02 |
| 16 | 15.40/31.30 | 8.27/11.91 | 5.82/9.93 |
| 17 | 7.39/10.81 | 6.32/8.25 | 4.53/6.95 |
| 18 | 9.45/16.36 | 8.03/11.53 | 5.92/10.08 |
| 19 | 13.02/26.33 | 9.98/16.96 | 7.73/15.20 |
| 20 | 9.56/18.80 | 8.79/13.97 | 6.07/11.43 |
| 21 | 11.07/18.32 | 10.91/15.28 | 7.09/11.49 |
| 22 | 9.58/21.59 | 6.34/12.04 | 4.84/11.29 |
| 23 | 10.42/20.95 | 8.21/13.93 | 6.25/12.25 |
| 24 | 11.80/26.64 | 10.35/18.86 | 7.33/16.82 |
| 25 | 14.44/30.45 | 12.46/22.37 | 8.72/19.21 |
| 26 | 16.99/37.71 | 14.48/26.80 | 10.79/23.59 |
| 27 | 12.47/30.13 | 11.24/19.69 | 7.44/16.52 |
| 28 | 20.22/46.43 | 17.60/32.83 | 11.65/24.86 |
| 29 | 14.94/35.32 | 13.75/26.70 | 9.26/20.93 |
| 30 | 16.48/35.84 | 16.92/27.21 | 10.36/21.46 |
| 31 | 15.85/39.93 | 15.46/28.19 | 9.72/22.27 |
| 32 | 22.62/38.47 | 20.54/29.64 | 13.49/21.49 |
| 33 | 16.93/39.24 | 12.03/22.38 | 9.32/20.57 |
| 34 | 14.03/41.84 | 11.19/24.26 | 7.51/18.88 |
| 35 | 22.29/44.71 | 18.03/29.27 | 13.17/24.01 |
| 36 | 18.45/33.12 | 16.84/27.91 | 10.97/20.24 |
| 37 | 16.86/28.70 | 14.06/23.73 | 9.38/16.19 |
| 38 | 8.73/17.06 | 6.51/11.98 | 4.16/8.30 |
| 39 | 1.00/1.48 | 0.48/0.77 | 0.34/0.53 |

## 5. 全模型hook：原调用结构保留，线性读出确实改变

亲读原adapter的 `patched_mm_apply`，它只把目标MM调用交给runtime；`ReconstructedRuntime`又仅截获 `.ffn.0/.ffn.2`，其余模块调用旧runtime。FFNStageTimer只包装原 `WanTransformerInfer.infer_ffn`，没有重写其内部数学。因此源码层面仍依次执行：

1. 原 `infer_ffn` 的 `x.add_(attn_out)`、norm2、原modulation及dtype处理；
2. `.ffn.0.apply` 中，新hook做 `BF16(x.float()/s1)` → 原HiF4 Q1 → 选中W1/b1；
3. 原调用者的 `gelu(..., approximate='tanh')`；
4. `.ffn.2.apply` 中，做 `BF16(h.float()/s2_selected)` → 原HiF4 Q2 → 新V/bias，affine路线另加 `F.linear(q1, matrix)`；
5. 原 `post_process` 中的gate与BF16残差写回。

hook没有重做norm/modulation，也没有在V中偷偷预乘gate；原gate/residual只在原消费者发生一次。计时的infer_ffn边界包含前面的attn残差相加、norm/modulation及FFN，但**不包括后面的post_process gate/残差写回**；完整generation wall才覆盖后者。

关键量化顺序和坐标正确：s1为完整5120输入scale，s2按同一所选组/列次序切片；先float除scale、再cast BF16、再原生QDQ，与原runtime一致。affine矩阵应用的是hook保存的**post-Q1 q1**，而不是尚未Q1的modulated input，避免了折叠坐标错位。完整64组内部原次序被保留；全组选择给出原Q2依赖闭包的源码依据，但不能把静态依据替代实测partial数值检验。

`pending` 检查 up/down配对、expert/step/block和宽度；expert按steps0/1高、2/3低。pack的W1/b1来自选中原权重，V/bias及affine矩阵来自四来源闭式拟合，**是新的BF16参数，已保存为bank**，不是原checkpoint未改，也没有再HiF4量化。此前“0持久化拟合权重”只属于已完成saved研究和有限检查，不能沿用到这个bank范围。

bank builder的FP32组合与bias减项与此前独立折叠推导相符；矩阵项使用未最终BF16舍入的V来组合，再cast，所以不许宣称原生bitwise代数恒等。pack SHA及bank RUN SHA在video driver中被检查；实际冻结bank身份与运行回执仍应另行绑定。

## 6. Neutral对照能证明什么，总账应怎样数

当前neutral只用source0保存的modulated输入，在steps0/2×blocks0/30共4处，比较旧runtime完整宽度两层FFN与新hook完整宽度FFN的BF16输出位。它检验Smooth1/Q1、W1/b1、调用者tanh-GELU、Smooth2/Q2与W2/b2的hook路径；若执行并通过，这是有力的接线对照。

它不通过原模型norm/modulation/gate/residual，亦不验证partial packing、affine matrix、全40层实际bias/scale或自然全M形状；其函数docstring也明确了这个边界。因此只能写“有限full-width hook对照”，不能称“完整模型中性等价”。本审查没有运行neutral。

按当前计划每臂source0/12各一完整四步生成，单视频含160个FFN：

| 单视频 | 原runtime线性/QDQ | 新hook up/down及Q1/Q2 | 额外affine GEMM |
| --- | ---: | --- | ---: |
| dense | 各1600 | 0 | 0 |
| whole94 | 各1280 | up/down各160，Q1/Q2各160 | 0 |
| affine54 | 各1280 | up/down各160，Q1/Q2各160 | 160 |

两候选每臂neutral另有8次FFN、16次QDQ（4处×旧/新各一次）；它们发生在视频计数基线之前，不在各phase delta中。故三臂六视频的QDQ为9600，再加neutral32，**总9632**；FFN对为960＋16＝976；affine另320次线性应用。weight preload不做FFN/QDQ，不能算成零成本，但不能误加到这些调用数中。若冻结版本改变neutral范围，总账必须随之更新。

当前driver检查每phase原runtime数、hook数、160个step/block timer覆盖、四个latent step和dense历史视频/latent身份；这些检查对应实际完整执行后才有意义。建议总RUN另外保存视频与neutral的分项和总计数，失败时保留部分计数。`generations_started` 与最终 `len(phases)==2` 对固定source_sequence有用，但跨三进程的“只运行一次每臂、单卡串行”依赖外层冻结调度，单mode脚本自身不证明全局上限。

## 7. 计时与内存的运行前修订

初版把候选80个pack在计时外加载，而dense全量原权重在首视频内由AllWeightResidency按需传输；其source0墙钟存在不同的首代权重加载边界。我向协调者提出此实质问题后，亲读的preload修订版已对三臂都在首视频前显式加载其所有原常驻线性权重：dense全部，候选仅非FFN；pack也在计时外加载，并分记pack加载、原权重预载、完整setup时间。这解决了首代weight transfer的不对称，不增加FFN/QDQ。

仍应分别报告source0的fresh-process首代与source12的后续热态：候选先有neutral而dense没有相同kernel warmup，首代内核/allocator/模型缓存状态未完全匹配；每来源每臂只有一个视频，也不是重复计时统计。不能把旧cold source0数字与新preload数字拼成加速比。初始化、资产验证、bank构建、CPU/GPU权重传输与视频生成墙钟是不同成本。

AllWeightResidency在候选中不会取原FFN权重，因为hook直接消费pack；原非FFN常驻量加pack量的计算与该路径相符。pack_bytes数所有pack张量，比之前只计矩阵的有限检查更完整。原型还会保留q1直到down调用，增加真实瞬时存储；即使whole不使用affine项，当前配对状态也暂存q1。这些成本应反映在实际峰值中。

已见preload版在模型加载前检查主机MemAvailable阈值；GPU `retained_original+pack+原sampled_peak+2GiB` 则位于pipeline/pack/neutral之后，属于**generation前headroom估计**，不是GPU资产加载前的内存上限证明。原sampled_peak必须来自未重复计入全权重的对应参考，2GiB是margin而非全M激活峰值定理。AllWeightResidency的cap只约束原缓存权重，pack另计；主机开始可用量也不等于进程RSS硬上限。单进程/单GPU串行能避免多个完整加载过程叠加，但实际进程和物理峰值仍需运行回执。

运行前还提出了输入绑定问题：初版和上述preload版仅由通用preflight确认input文件存在，逐image/seed检查没有绑定prompt；旧all-weight driver曾另外检查input_default和cache_manifest SHA。后续亲读的7df86bdc版本已纳入这两个文件的精确哈希，此问题已在源码层面修正。其local tensor_sha对明确转换为CPU FP32的latent取连续字节哈希，与既有latent身份语义相符；不是新增模型数值检查。

显式预载后，7df86bdc版第158–159行**已经断言**phase `weight_transfers` 增量为0，否则抛出异常。这核验预载dtype/key与自然请求一致，不增加FFN/QDQ。我先前漏读该处、误写“未作为失败条件”，本次逐行复读后在本文件纠正；并非协调者随后才新增该断言。

### 冻结身份补记

已亲读完整冻结协议 `config/e0_reconstructed_videos_20261026.json`，SHA256为 `964acc9fc9d8aaf249593db710ff4301495b5c71dfa1e0ee84c0417b9f9b5fd4`。它绑定的driver仍为 `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005`，runtime仍为 `957588f5dc3b32334982b6247064ddece1d21ab627267eb717bc72fb31003872`。本次只读核对17项driver绑定、7项native源绑定、canonical input/prompt、cache manifest与bank manifest，实际文件哈希全部相符；不是执行preflight或模型。

bank公开manifest `research/20261026/E0_RECONSTRUCTION_BANK.json` 的SHA为 `9731a3dac0213a463337d6a3c9b51a7645b4a8ba2ecf75e5c664f127fb9d8918`；其HF相对 `e0_function_basis_20261026/prototype_banks_attempt1/RUN.json` 的实际SHA为 `fd38f0c148a9284da26adf35bc43c06e720ecb89b00c35500e5b4d375c32dc81`，与冻结协议一致。manifest有160包（每族80）、`held_sources_read_for_fit=[]`，已有CPU验证汇总记录全部参数有限、源快照一致。本审查核对这些JSON及其绑定，没有重新加载160个权重包或冒称重复完成其CPU张量验证。

冻结范围明确三臂、每臂source0/42及source12/54共六视频，最大单模型进程、单GPU串行；保留四GPU总上限。主机加载前可用量阈值192GiB，原权重常驻cap为57,982,058,496字节，原参考采样峰值21,390MiB，GPU估计另加pack及2GiB；每臂HF输出cap0.125GiB、磁盘保留80GiB。它明确区分source0的首代内核状态与source12的共同热态，单独计setup/pack/preload，且要求生成期间0权重传输。总9632 QDQ与9952 linear GEMM（含neutral及affine320次）与第6节独立总账相符。

协议另限CPU派生视频对照及像素/亮度/时序描述，明确不等于learned/official质量评价或人类完整视频审阅；0自动重试、任一生成失败后停止后续范围。以上是冻结范围与源码身份核对，尚不是实际六视频完成、资源峰值或质量通过的结论。后续实际RUN审查单独进行，旧独立记录不改。

## 8. 图锚点、竞争预测与研究交接

本次保存数据OBS位于 `G02/G18→F03/F04/F06/F08R/F08/F09`：匹配校准后，已付真实q/r特征的局部输出重构确有额外解释力。静态原型在 `F02/F03→F04→F06→F08→F09→F12` 提出conditional新执行边；源码仍保留F12原gate/residual调用，不代表这个近似输出的下游行为不变。`F13/G21→D09/P12` 需要实际全模型自由运行，不能由8.54%局部SSE或旧52.69→20.77ms形状计时闭合。P14广义函数基问题继续OPEN。

本次读取的canonical图有F03（原生Q1）节点，但registry未注册F03；这是检索索引缺口，不是源码缺少Q1。留给串行整理者补齐，本文件不改图或registry。

**对我前次假说的更新：** 同训练覆盖、去掉rank64输出瓶颈并加affine先验后，线性控制仍输给affine54全部320个case；应接受这个证伪，而非继续把旧混杂当作当前结果的解释。剩余最强竞争机制不是“实验仍没匹配来源”，而是“固定正则与小训练覆盖的估计偏差、原函数非线性先验及下游敏感性共同决定表现”。它预言：更强线性拟合或不同空间覆盖可能缩小局部差距；即使局部差距不缩小，全局视频对层/step/误差方向的敏感性也可能使局部最好路线不在最佳质量—成本前沿。

当前最有信息量的下一观察正是协调者计划的有界自由运行：在相同输入、真实原caller、匹配常驻边界下，dense先保持历史身份，whole94/affine54分别自由四步，观察误差是否沿norm/attention/residual累积。它能推翻“局部收益可直接转成可用完整视频”的额外假设。本文只审查该构造，不启动它，不预报质量结论，不赋予正式方法认可。

文献关系沿此前已亲读的RRR、FLAP与原始ID来源，详见保留的 `independent-function-basis-prediction-20261026.md` 与 `independent-function-basis-results-audit-20261026.md`。本轮新增贡献是针对实际匹配控制的源码/标量重建与运行边界挑战，没有把协调者新读论文或准备中的视频冒称为我的实测。闭式拟合仍是参数估计，原特征＋消费者重构仍是已知原理；新的证据是这个Wan设置中它在明确训练条件下的收益及代价。

## 9. 后续追加：已完成dense两视频与分析器静态复核

以下追加发生在dense完成、两个近似臂尚未全部完成时，不将它们的待执行结果写成事实。只读取已完成dense RUN、执行快照、历史reference、已有视频文件字节哈希与分析器源码；普通CPU算术，不执行分析器、不解码视频、不运行模型。

原始 `e0_function_basis_20261026/prototype_dense_attempt1/RUN.json` 实际SHA256为 `07abf016293897308c4efc576bce85b84c02b57e266d9a663c7e3911c56bf915`。协议仍为 `964acc9fc9d8aaf249593db710ff4301495b5c71dfa1e0ee84c0417b9f9b5fd4`；执行协议和17项源快照均匹配。独立核对0/42及12/54两个视频的实际文件字节SHA、各自四个latent记录，均与历史dense reference相等；不是仅相信RUN中的PASS字符串。每视频1600次原runtime线性/QDQ、0权重传输、0cache miss，160项step/block事件无缺失，事件和与总FFN计时相符。neutral为空；最终原权重retained_bytes=0。

dense物理显存采样峰值79,594MiB，采样间隔2秒、所有记录foreign process数为0、无violation；这是采样隔离和采样峰值，不能称连续显存上界。原权重实际常驻56,203,673,600字节，预载4.31357秒；setup字段139.18818秒。setup＋两个generation墙钟相加为242.20884秒，但仍不包括driver早期导入/协议检查、进程尾部及先前bank拟合，不是完整CLI wall。

亲读 `tools/analyze_e0_reconstructed_videos.py` 初版SHA为 `ff3d53bc9d7becb8856904331a4a79329d00b9ed5e0641ce3f0c5b69d7789146`。指出像素/亮度、常数总账与并发表述后，亲读修订版SHA为 `453ad18ea6956eb76958fd8024947cc86fa0d8b24b97c904fd9cc236c5d22a43`。后续意见绑定此修订版；分析器未在本审查中执行。

## 10. 分项与总账：修订版从实收计数累加，不由常数代替审计

分析器逐臂检查PASS状态、冻结协议、bank RUN、2个generation、执行源快照、资源采样、私有启动receipt的匿名同卡绑定；逐phase检查source/seed/order、4步轨迹、原runtime与hook调用、0generation weight transfer、160个FFN事件及和、视频文件SHA，dense还比对历史身份。具体检查与原driver口径一致。

初版直接返回9632/9952常数；修订版改为求和：

`generation_QDQ = Σ(runtime_QDQ + hook_Q1 + hook_Q2)`；

`neutral_QDQ = Σ(control.extra_native_QDQ_calls)`；

`GEMM = 2*neutral_FFN + Σ(runtime_linear + hook_up + hook_down + hook_affine)`。

再逐项与协议的9600 generation QDQ、32 neutral QDQ、16 neutral FFN、9632总QDQ、9952总GEMM比较。因为neutral是完整两线性层、没有affine项，这个GEMM公式正确。冻结源的实际调用结构与neutral声明对应；这些是未来全部实收齐备时应执行的检查，不是本追加已经确认候选总数。

修订版还核对4处neutral各48行。已从原source0 RUN的row_plan独立确认：这48行包括32 uniform及额外结构行，不能写成只有32行，也不能把192行跨4处当作192个独立来源。

同卡receipt配对能确认相同私有worker/card绑定；修订版进一步检查该同卡三个monitoring时间区间有序不重叠，将字段改成 `planned_maximum_simultaneous_model_processes=1` 与明确的监测区间声明。它比初版硬编码“实际最大模型进程1”准确。区间基于2秒采样的首末时间，不是所有CPU加载/进程生存期的连续审计；外层串行启动约束仍是单进程范围的另一来源。

## 11. 全81帧指标的数学含义与反例

静态实现同步解码三臂，要求FPS一致、每帧形状一致、全部81帧成功；在**原分辨率RGB**上先计算，再缩略到416×240制作三臂视频及0/20/40/60/80帧contact sheet。因此5帧contact不是81帧指标的采样替代；但生成81帧比较视频也不等于有人完整观看并确认画质。

每帧RGB除255后计算MSE/MAE，修订版补充dense均值、候选均值和paired signed mean RGB drift。它还用明确权重 `[0.2126,0.7152,0.0722]` 对**编码RGB**作加权，记录encoded-luma MSE和signed mean drift。这是有用的亮度/颜色漂移描述；没有逆传递函数，不是linear-light物理luminance，也不能据权重名称断言文件色彩元数据已经验证。

`mean_frame_RGB_RMSE` 是 `mean_t sqrt(mean_pixels,error²)`，不是 `sqrt(mean_t,pixels error²)`；前者由凹性不大于后者，失败少数帧可能被平均掩盖。完整逐帧数据保留，因此最坏帧、漂移方向与随时间走势可直接检查，而不能只摘一个平均RMSE作验收。

令解码后候选与dense为C_t、D_t，误差 `E_t=C_t−D_t`。时间差指标实际是

`||(C_t−C_(t−1))−(D_t−D_(t−1))||² = ||E_t−E_(t−1)||²`，

共80个相邻帧差。它不是独立的“闪烁评分”，也不是motion-compensated比较。

**强反例A：** 若 `C_t=D_t+c` 且数值不越界，恒定全局亮度或颜色偏移c能让该时间差误差严格为0，同时RGB误差、mean drift明显非零。故低时间差误差不能排除持续偏亮。

**强反例B：** 若 `C_t` 变成固定一帧，candidate_temporal_MAE可为0；这显然不能解释为更好的运动稳定性。与dense的时间差会暴露部分运动丢失，但背景、运动物体和相机运动仍混合。

**强反例C：** 两个都平滑且视觉有效、但有空间/时间相位偏移的视频，可有很大逐像素及时间差误差。因此大差异支持“未保持dense解码输出”，不唯一识别画质变差的类型。相反，小平均误差也可能漏掉小物体、局部细节或少数帧的失败。

这些指标来自分别编码后的MP4，包含模型输出改变与输入依赖的编码误差；它们不是未压缩输出或latent差。比较视频又经过缩略与重编码，但指标是在这一步之前计算，所以比较文件自身的压缩没有污染所存原分辨率指标。修订版明确 `human_full_video_review=NOT_OBTAINED`，没有从像素均值宣布感知/任务质量通过，范围合适。

## 12. 从dense实测独立推导条件Amdahl上界

定义T为包含common latent hash、视频编码/保存及末尾CUDA同步的generation host wall；F为160个CUDA infer_ffn边界事件之和；令算术余量 `R=T−F`。注意F包括attn残差相加、norm/modulation、Q1/W1/GELU/Q2/W2，不只主GEMM，且不包括随后gate/residual；R也不是已被单独测量的某个算子。

假设改变FFN后R不变、事件和能近似充当串行可删除组件，则FFN段快s倍时

`S_total(s)=T/(R+F/s)=1/[(1−f)+f/s]`，`f=F/T`。

| dense来源/状态 | T秒 | F秒 | R秒 | F/T | 即使F消失的条件上界 | 若整个F快2.54倍的总速度比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/42，fresh-process首代 | 52.270440 | 9.658217 | 42.612223 | 18.47740% | 1.22665× | 1.12616× |
| 12/54，共同热态第二代 | 50.750224 | 9.707139 | 41.043085 | 19.12728% | 1.23651× | 1.13118× |

共同热态若整个F快2.54倍，预测T约44.86479秒，即墙钟减少11.59686%；即使F全部消失，也只减少19.12728%的时间，速度比为1.23651。不能混淆“23.65%速度提升”和“19.13%时间减少”。此前孤立after-Q1的2.54×还没有加速F内保留的norm/modulation/Q1等工作，把它直接套到整个F已是乐观外推。

**这不是普适硬件上限。** CUDA事件与host wall有不同的时钟/边界；事件可包含排队/CPU发射间隙，也可能与边界外工作有重叠。缩小FFN还可能改变显存压力、allocator、搬运与同步，近似输出也可改变编码工作量。R还含attention、其他模块、VAE/编码、CPU与同步等未分解部分，不能仅凭41秒余量指定其中哪一项最大。dense单次FFN事件最大值约194–219ms，均值约60–61ms，也提示单次边界并非均匀裸GEMM。

修订分析器已经输出各phase `generation_wall_minus_CUDA_FFN_boundary_seconds`。候选齐备后，最小决定性核算是比较

`ΔR=(T_candidate−F_candidate)−(T_dense−F_dense)`。

若总加速超过上述条件预测，应先检查ΔR、内存/同步与计时状态，而不是立即宣布突破Amdahl或质疑全部结果；若总加速很小而F明显下降，约81%的其他时间占比就是有力竞争解释，不能反推“FFN省算无效”。此分解也不授予视频质量结论。当前只独立确认dense两完整视频；whole/affine的实际RUN、像素指标与质量观察仍待单独审查。

## 13. 最终追加：三臂六视频的实际回执与时间

本节在六视频与CPU分析均完成后追加，保留前文当时“尚待完成”的历史时点。实际公开结果 `research/20261026/E0_RECONSTRUCTED_VIDEOS.json` SHA256为 `f01554b974741191b39faffd31fc23badc7a395de87a621a95d98674a621e7c8`，分析器SHA为前述 `453ad18ea6956eb76958fd8024947cc86fa0d8b24b97c904fd9cc236c5d22a43`，与实际源码相符。

三个原始RUN均相对于HF derived的 `e0_function_basis_20261026/`：

| RUN | 实际SHA256 |
| --- | --- |
| `prototype_dense_attempt1/RUN.json` | `07abf016293897308c4efc576bce85b84c02b57e266d9a663c7e3911c56bf915` |
| `prototype_whole_attempt1/RUN.json` | `ed53a172f62cc9d3252b2ec8b905ba4c8814cc2ef6fc9618bcffbf0f62f9fc32` |
| `prototype_affine_residual_attempt1/RUN.json` | `fb0ff5dfa4927c0ce0af9f09b1bf0b2590c83658dcb1c823d1fcd94a20f3f7c0` |

逐一独立检查PASS状态、同冻结协议、每臂2次generation、协议及17项执行源快照、160项FFN事件与和、实际六个MP4字节SHA；均一致。候选8处neutral各48行、BF16 bit mismatch为0，hook计数各为up/down/Q1/Q2各1、affine0。实际累加为6视频、neutral16 FFN/32 QDQ、视频9600 QDQ、总9632 QDQ及9952 GEMM，与协议相符。所有phase的generation weight transfer与cache miss均为0。

三份资源回执采样隔离均通过、无violation、所有foreign计数为0，记录中的物理卡索引一致，三个monitor区间独立核对为有序不重叠；匿名worker/card的完整关联由公开分析所绑定的私有receipt检查提供，不在此公开设备坐标。采样间隔均2秒；采样物理峰值dense/whole/affine为79,594 / 63,870 / 60,796MiB，不能称连续最大值。

| 来源/实际seed | 路线 | generation T秒 | FFN边界F秒 | 余量R=T−F秒 |
| --- | --- | ---: | ---: | ---: |
| 0/42，首代内核状态不完全匹配 | dense | 52.270440 | 9.658217 | 42.612223 |
| 0/42 | whole94 | 46.692066 | 4.848414 | 41.843652 |
| 0/42 | affine54 | 46.219357 | 4.422429 | 41.796928 |
| 12/54，共同热态 | dense | 50.750224 | 9.707139 | 41.043085 |
| 12/54 | whole94 | 45.735768 | 4.879968 | 40.855799 |
| 12/54 | affine54 | 45.313464 | 4.448653 | 40.864811 |

共同热态whole的FFN速度比1.98918×、wall比1.10964×；affine为2.18204×、1.11998×。这是真实全模型中的FFN省时，但不是有用画质前沿已成立。

whole节省FFN约4.82717秒、wall约5.01446秒，ΔR约−0.18729秒；affine分别节省5.25849秒、5.43676秒，ΔR约−0.17827秒。因此热态收益主要由实测FFN下降解释，未出现需要额外巨大非FFN改善才能解释的结果；约41秒余量依然支配总时延，符合此前条件Amdahl预期。首代ΔR则约−0.77/−0.82秒，不能与共同热态混合估计普适速度。

setup另为dense139.18818秒、whole236.92193秒、affine237.08860秒；候选pack加载约100.42/101.13秒，均在generation计时外。小样本进程的初始化负担没有被约5秒generation收益抵消；是否可长期摊销是另一部署条件。此处不把组件相加冒称包含所有进程开销的CLI计时，也不把setup排除的wall比叫端到端加载收益。

## 14. 已保存81帧指标与本人实际查看的contact范围

分析产物记录每视频81帧、16FPS，实际解码图像形状为**464×832×3**。这是输出文件形状，不能按配置目标高度480重写。本人核对保存的81个frame索引、80个时间差、逐帧RMSE均值及时间差RMSE均值，全部与公开汇总相符；没有重新运行解码或像素分析。

| 来源 | 候选 | 81帧平均RGB RMSE | 80差分平均误差RMSE | encoded-luma signed drift均值 |
| --- | --- | ---: | ---: | ---: |
| 0 | whole94 | 0.1578974 | 0.0575771 | −0.0115839 |
| 0 | affine54 | 0.1541602 | 0.0564177 | +0.0066702 |
| 12 | whole94 | 0.1616454 | 0.0507843 | +0.0002772 |
| 12 | affine54 | 0.1305286 | 0.0651147 | +0.0151958 |

source12已出现实际排序差异：affine的平均RGB误差较低，但相邻帧差误差比whole高。其encoded-luma signed drift全部81帧均为正，范围约+0.01162至+0.03789；这只是编码亮度偏移，不是物理光照或画质评分。

source0的candidate相邻帧变化MAE为whole0.014636、affine0.008670，dense为0.023536；source12却为whole0.018540、affine0.029545，dense0.020195。不能统一把候选变化变小解释成“更稳定”，也不能把变化变大唯一解释成更强运动。误差、运动差异、纹理变化和压缩都混在该量中。

本人实际查看的只是 `prototype_comparisons_attempt1/source00_contact.png` 和 `source12_contact.png`：各含帧0/20/40/60/80、三臂同缩略过程。两PNG及两比较MP4的字节SHA均与汇总一致；全部派生文件9,216,830字节也与汇总相符。

- source0的whole列后几张有明显叠影、毛发/水面纹理破坏和结构混合；affine较少这种破碎，但熊与水纹明显模糊。dense熊姿态/头朝向在这些帧中变化较大，affine若干后续抽帧仍接近首帧姿态，支持**抽帧可见的动作差异**。
- source12的whole列水面/建筑有明显重复和破碎结构；affine保留较完整的大轮廓，但建筑和岩壁细节显著糊化。

这些是本人看过的contact观察，不是完整视频审阅。视频帧0也不是去噪step0，不能从第一行图像直接推断“第一去噪步已坏”。两来源都有明显可见差异，足以反对将本次局部SSE收益宣布为画质通过，但不构成正式人类全视频/官方评分验收。

## 15. 机制归因仍须保留实际bank数值这一强替代解释

目前可以分别确证：匹配校准下保存数据的非线性重构误差较低；实际候选减少FFN与generation时间；这两个固定原型的contact存在明显损伤。**尚不能唯一确定损伤为何发生。**

尤其是8.54%来自保存数据连续诊断；160包的shape/finite/SHA正确，不等于所有层的deployed native函数贴近该连续预测。full-width neutral只覆盖4处；此前partial数值检查仅blocks0/30，尚未独立闭合其余38层、所有所选组/FP32合成条件及自然全M上的bank/native差别。数学折叠正确、Q2调用真实，也不能自动消除这项实现/数值替代解释。

因此下一paired-prefix判别应先区分**实际部署bank同输入native结果、对应连续预测与原dense结果**，再讨论局部误差几何、层内/跨步离轨或显式轨迹修复的相对作用。它是新的有限原生观察需求，不是当前回执已包含的结果；不能直接跳到JVP或网络训练。

另一个独立记录 `independent-trajectory-error-and-correction-20261026.md` 已从真实scheduler和原论文重建这三个机制，并给出TCEC Eq7→8反例和CTEC目标的Wan特例；本文不重复把它们当作视频故障已识别的原因。

亲读新图claim `E0_GRAPH_MATCHED_NATIVE_VIDEOS_20261026.json`，实际SHA256为 `27f3c26ff72afb50cad287f1f604a57e07bfee0df0aadae7c8171daec48fbcce`，读取时为PROPOSED、review_refs为空。其匹配校准、实测时延、真实新BF16权重及“未建立可用质量—成本前沿”的范围与本审查一致；对“失败机制OPEN”还应显式包含上述all-layer bank/native数值偏差，不能仅列传播/分布偏移。本文是独立实质复核，不授予正式方法支持标签。
