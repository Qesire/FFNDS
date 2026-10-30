# A5-inspired 增量支持 E0：独立科学预审

评议者 `/root/sync_abba_actual_review_terra`。状态 **DESIGN / REVISION_REQUESTED**：这不是实验范围批准、ACTIVE、实测或方法优胜结论。数据采集尚须实际RUN/EXIT和独立验证闭合，E0 producer、输入、资源及监督作业完成后另行精确复审。

## 可判别问题与反例

按source×expert×block配对同专家首次和后续step：对固定K=5120，比较当前贡献与增量评分、HWG与无HWG、原真实64行信息合同的HiF4输入P64迁移、静态/随机/oldY-only和付费真实post-Q2 delta对照。单行score集中但多行支持错位可令M16×K64 tile几乎全占用；此时逻辑稀疏与原布局可跳算不同。另一个反例是两个大幅ΔV2通道经gate/W2相互抵消：diagonal-energy TopK可比较小单通道造成更大signed输出误差。真实post-Q2 delta的diagonal-energy选择是已付费对照而非signed最优oracle，正/负cross均应报告。

`shared_support` 的hidden_group=64是在每个行组上聚合64通道能量后等K组选择；若声称逐channel对照必须显式使用hidden_group=1，不能混名。shuffle选择后映射回原始row顺序再计算物理tile占用；若通过实际重排获得连续组，还要付gather/pack/restore、缓存与kernel代价。K16/K32只是物理诊断，不能自动称为符合原Q2 K64闭包。处理部分M tile要报告padding与实际行数，physical MAC proxy可以大于1。

## 输入/对照身份

HiF4 replay必须从对应expert/block的实际full-M post-Q1 V1出发，绑定有效W1、W2、bias、smooth2和QDQ，按native BF16顺序进行 W1+b1→tanh GELU→BF16 smooth2/Q2→W2+b2；先对同坐标保存的实际Z/H/V2/Y验收。选择 delta 时使用 `Q2(h1/s2)−Q2(h0/s2)`，不使用Q2的delta输入；b2已在oldY包含一次。post-Q1 V1不携带全M pre-Q1及原在线scale，未来真实代价须包含Q1。GELU非线性要求保留a0或可辨识的对应代理版本，不能用GELU(Δa)代替ΔGELU。不同expert不共享历史。

原Chipmunk P64的64决策行是全M seed42固定randperm，不等同当前21个M64/τ共1344个预选行。若称原P64选择，必须在实际64行上运行与原算法等价、数值族明确的已付代理，并确保历史h来自同一真实或经验证的全M重放；额外小行W1/GELU以及kernel形状可能造成的数值差要核验和计费。若只用1344样本内64行，应另命名并冻结样本策略。两来源898/96可用于静态组选取，1040/979仍是暴露开发源而非盲确认；采样token/block不能作独立n。

## 可执行、信息有效的分层

先对四源×两个expert×40层的全部320对，在捕获的1344同坐标行上使用实际Z/H/V2/Y、gate与R开展各arm/预设分组的同K paid-diagnostic筛查；连续M16/32/64能从每个独立M64块中构造，shuffle保持原行坐标。这样可以全深度比较样本内支持、paid FP32 signed投影及实际采样R写回差；应分块、记录实际GEMM数量及row数。21个预先选择的M64不是随机均匀全空间样本：结果只能称这些tile，不支持全M支持分布、全M global mask或细小对象排除。该阶段不得冒称已完成全M E0。

并行规划full-M的受审版本：在捕获资产及实际replay中冻结320对的完整full-M mask/occupancy；对所有方法/粒度的完整signed投影可以先按同一输入块批量输出并复用full delta，但仍按每一实际W2投影/GEMM形状和总MAC计费，不能把17920个矩阵投影算成一次。预算不足时优先分批执行并为真实缺测明确NOT_MEASURED；可以用预先规定的代表/冲突层做探索，后验挑层不能外推成全40层的全M证据，后续仍需补足。静态组只在冻结开发source拟合，公开每source向量。

`A5_INSPIRED_MEAN_INPUT_PER_M64_V1` 可作为另外一个显式新动作：分别对tile旧/新V1求均值后计算proxy/GELU差、对等K组选择，并与逐token proxy在相同真实行/预算下比较。tile内+δ和−δ可使均值为零，而原两行各有明显变化；GELU(mean(a))也不等于mean(GELU(a))。在两expert每对均重算旧/新、四次proxy W1且M压缩64的理想主MAC口径下，附加约0.5/64=0.0078125（含尾部实际行和batch实现需另算），再加mean、路由、pack和kernel/访存；这既不是原token-A5，也不是当前已证明低时延实现。

## 源码具体修订

`tools/a5_hwg_support_regularity_math_20261030.py` 的 `signed_output_metrics` 对zero reference和zero direction用clamp会返回0或极大值；应对未定义相对值/余弦返回null及zero标记，增加零输入反例。冻结FP32 projection的TF32/accumulation设置，保留actual BF16 output比较和full-delta rounding difference；后者不是误差下界，因为与遗漏项可能负交叉。全M reference V2差与output由真实full-M数据提供，不能用单次后验矩阵乘算作native skip。

`support_summary` 的 `raster_adjacent_support_Jaccard` 是flatten相邻，包括跨二维行尾及跨τ的跳跃；应标为flatten-order物理邻近，并将合法同τ水平/垂直邻居、同空间相邻τ分别计算及报告边界。样本仅覆盖固定21 tiles时不能补写不存在的邻居。每个源、site、参考能量、零分母和显式运行成本必须可追溯，signed/postwrite与原生质量/速度证据分层。
