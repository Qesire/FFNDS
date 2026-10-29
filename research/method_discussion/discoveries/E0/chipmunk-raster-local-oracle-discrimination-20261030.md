# 全 M 局部组与共同 mask 的等预算反事实设计

提议者 `/root/sm120_full_quality_build_terra`；OPEN，尚无独立评审、授权或执行。依据已闭合 `CHIPMUNK_FULLM_CAUSAL_PROXY_RESULTS_20261030.json`，两暴露 source×同 expert 两个相邻步的四个 case 中，P64 门后误差约 0.256–0.426，whole-M full-RMS 事后参考约 0.246–0.395；P64 仍有较大残差。它们是两个来源，非四个独立来源。这里的问题是：在相同 hidden 元素更新预算下，局部适应是否回收部分输出误差，以及此现象是否依赖真实 raster 邻近。全 Δh 为事后付费输入，不是在线选择器或稀疏 kernel 速度。

## 固定最小比较

每个 case 使用同 source/seed/expert/block20、原被动捕获的两步 BF16 h、Y 与当前 gate，及匹配的同 expert 原 BF16 W2；完整 typed header/模型和 parent RUN、actual EXIT0/独立 VERIFY 身份先复核。真实 M=31668，H=13824，D=5120，k=5120。计算 Δ=h_current−h_old 的 FP32 值并在每个组按各 hidden 通道 Δ RMS 降序、固定索引次序解平分。固定三种掩码：

* **全局**：全 31668 行计算一次 RMS，所有行共用选出的 k 通道；
* **真实 raster**：保留 row=tau×29×52+y×52+x 真实顺序，连续 128 行为一组，末组 52 行，共 247 个128行组和1个52行组；各组分别选 k；这是线性 raster 连续片段，可跨自然空间行和帧边界，不称几何规则方块或同质区域；
* **随机组控制**：固定独立 seed 生成整 M 的无放回行排列，按同样 247×128+52 切组，各组选 k；将结果散回原 row ID。相同随机排列用于该 case 的全部后验比较，记录 seed、组摘要和 mask SHA。它检验邻近排序是否比相同大小的混合行分组额外有用；一次固定排列不是对所有随机组的统计证明。

三臂均恰好选择 M×k=162,140,160 个 row×hidden 元素，且同一行/组的预算一致；这控制名义更新元素数，**不控制**实现 kernel launch、权重访问、压缩元数据、选择或总体成本。mask 本身根据当前 h 全量计算，在生成时不可免费获得。另设全量更新参考和零更新参考用于校准，前者额外付费且不与同 K 臂等价。

对每臂以当前 gate 和实际 W2 投影选中 Δ：按原 row 计算 `Yhat=Y_old + selected(Δ) @ W2.T`，并与已捕获当前 Y 比较；对相同当前 gate 分别计算门后相对 L2。也可输出遗漏项的 FP32 `omitted(Δ) @ W2.T`，及其自身范数，但要单独计费。由于真实 Y、缓存与 W2 输出均有 BF16 舍入，`Y_current−Y_old` 不保证等于 FP32 `Δ @ W2.T`；因此同时输出全量更新重建相对当前捕获 Y 的误差、捕获输出范数及可选 FP32 round-trip，对任何减差解释保留该误差底。精确固定 gate reshape、输入/输出转换、累加精度、是否在 Yhat 加和后 BF16 rounding，不能把局部差与此前 Triton BF16 delta action 数值混称；若要复现该 action 必须另计 BF16 delta 舍入及执行路径。

## 可证伪预测和边界

若局部组 RMS 在多 case 的门后误差明显低于全局，而随机组接近全局，支持当前空间邻近与行间异质性结合，使共享 whole-M mask 损失可回收。若局部与随机组均类似地改善，则有限组大小/异质性更像解释，不能把收益归因自然 raster 几何。若两者无改善，且同 K 本身仍有大误差，则该固定预算或 RMS 排序下细分行动未带来所需效果。若局部/随机 RMS 所捕获 Δh 能量增加而门后误差反增，W2 与 gate 权重及跨通道有符号交叉项提供竞争解释；RMS 是激活能量目标，不是输出误差最优 oracle。全量参考误差底较大则不能可靠归因细小臂差。报告每个 case 原始结果及两个 source 等权的探索描述，不做显著性或 held-out 推断。

实现应将局部分组操作分块，明确完整 h/Y/gate/W2 的实际 read/H2D、四组后验 score/selection、三臂相同 row×K 及真实投影 GEMM、索引 scatter、全量/零更新、误差归约、warmup/同步、分配峰值、I/O、全 wall 和失败账；模型 master 大文件重读与 typed W2 bundle 来源可用性须在 scope 草案绑定并独立审查，不能假定已验证。可以逐 source×expert 四个单 GPU 独立 scope，最多按实际 ACTIVE 并发，不能占用当前已登记作业；冻结前需预先给出最大片上/主机内存、投影次数、操作数与超时，检查 W2 源与原 captured 状态相同。此设计尚不是可运行 producer，也不声称稀疏加速、自由视频质量或真实在线动作。
