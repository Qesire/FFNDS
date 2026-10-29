# TEAL→Wan：全M保mask W2的物理可达性与最小判别

提出者 `/root/teal_full_llm_calibration_build_terra`；2026-10-30，SRC/LIT/ALG与既有OBS，未执行新CPU数据分析、GPU、模型或视频，所有新范围 REVIEW_REQUESTED。目标是原BF16 Wan2.2 I2V FFN，不扩LLM、attention或跨步cache。

## 已有现象与精确对象

固定六源校准门限的post-GELU置零，在两开发来源×160 site×256真实采样行，25/50%目标的来源等权局部门后相对SSE 0.005771/0.057950，分别优于同逐行删除数量随机0.113019/0.312668和静态0.103833/0.243413。来源为独立单位、256行/160 site是嵌套；实际行不是连续M16 tile，也不是可复制的full-M。原报告给开发实际删除率均值24.7981/49.6312%，不能将目标率当每site实际。四开发来源×seed42的完整视频使用原 dense W2，50臂source979新增彩色碎片，source456流比0.423；4来源RAFT均值>1不等于运动/语义更好；主体、背景与额外物体必须与各自Dense配对保留。

真实钩子 `tools/e0_teal_wan_all160_video_v3.py:96` 在实际BF16 W2输入上按 `abs(h)<=tau`置零、再 `_old(masked)`；W1/GELU已完整支付，后面gate和bias语义保留。H为 `[31668,13824]`、W2为 `[5120,13824]`、Y为 `[31668,5120]`，计算 `Y=H_tau @ W2.T+b2`。单H BF16 875,556,864 B；单W2 141,557,760 B；单Y 324,280,320 B；1 bit/元素mask 54,722,304 B。W2名义 2.2414e12 MAC / 4.4829e12 FLOP；原W1+W2名义MAC 4.4829e12。全省W2最多省两主投影50%的MAC，25/50%逻辑元素稀疏不等价于25/50%真实W2加速。

原 [Activation Concentration v2 §2–5](https://arxiv.org/html/2606.00567v2) 定义隐藏列整列cold需所有M行均低于阈值、保留全元素bitmask，使用16×16 systolic和GDDR6/Ramulator模拟；其cold处理属于FFN-Reuse、含先前输出复用，既非A100 BF16 Tensor Core实测，也非本TEAL零化即丢弃。文中不同workload存在浓聚/分散及阈值膨胀，不推出Wan全M空列率。此前M16×K64 100%占用事实来自旧HiF4的后验逐token delta TopK及特定原布局，不是本BF16 TEAL mask的测量。相同source/state真实256行构成full-M真实行子集时，`cold_full ⊆ cold_256`，按每site旧256行直接计算cold列给严谨full-M空列比例上界；但是采样行之间不可视为连续tile，不能得到真实M16 occupancy。

## 两种竞争机制与固定对照

**H1 分散/闭包饱和**：小幅值在token间离散，逐行25/50%零值但每个自然tile几乎覆盖所有隐藏列；保持原mask的K-column并集、块稀疏或grouped GEMM减算极少，mask/压缩/索引/重排成本大于收益；如果真实4元组多于2个非零，纯2:4无法精确承载。预测full-M列冷上界与M16/M32×K16/K64空tile率很低，低至不足支付一次完整H扫描和W2读取，特别是跨空间不相关的行。

**H2 条件集中/低代价共享**：有些source/step/block存在列长期冷或局部raster/t帧tile共享支持；实际同组并集可远小于各行支持的简单全宽并集，native BF16 Tensor Core可在分组后以少数较长K子问题回收优势。预测真实连续M16/M32行的可跳列和较小K块占比明显高于同source同site的预先固定、付费随机行分组；分组内重复行支持的余弦/Jaccard与越小K并集对应，但跨expert/跨step支持不能无成本复用。两者均可能在同一source或不同site成立；以独立图像source为统计单位，先报告site分布、尾部与极值而非把site当独立图像。

**[ALG，独立模型仅作反例而非Wan预测]** 若各元素独立、零比例p固定，16个token共享一列都为零概率是 `p^16`：p=0.25时约2.33e−10，p=0.5时约1.53e−5；M16×K64整块全零为 `p^1024`。真实token/通道高度相关，不得用这些数替代实测。按同一独立模型，每个4项中非零至多2项的概率分别为25%零目标的0.26171875和50%零目标的0.6875；它只是组资格而不是该比例的GPU速度，真实相关/tie及整个行均可编码需实测。

## 两类明确不同的物理行动

1. **与TEAL mask数学同一：共享并集或严格结构化分解。** 对真实行组G，先计算 `S_G={j: ∃r∈G, |h[r,j]|>tau}`；只打包 `H_tau[G,S_G]` 与 `W2[:,S_G]`，执行 `H_tau[G,S_G]@W2[:,S_G].T`，然后逆行置换、单次加b2及原gate。每行的零仍逐元素保留，因此是同一数学TEAL动作，数值上须验证BF16/FP32累加顺序、tie/NaN/bias/gate与原dense-masked实现的差异，不默认逐位相同。固定天然raster分组是主要路径；掩码相似度排序/分桶须在线实际付费，且输出scatter/逆置换完整计账。不能将不同group并集/packing推断为无需写入的理想MAC。若K64整块均零才跳tile，实际后端调度、输出tile、尾部和指令计数必须实测；真正内核允许更细可执行单位时另记，不将whole64满占用误当所有算子的下界。

   另一严格路线把每4组中选至多2个真实非零编码进BF16 2:4，超出的非零保留在单独残差：`H_tau=H_2:4+H_residual`，两支输出相加。这保持TEAL mask数学含义，但编码、转换、metadata、残差GEMM或稀疏乘、两输出归并、临时内存及额外读W2必须全付。若残差用普通dense GEMM，虽然元素稀疏，仍执行完整GEMM，不能声称白得半价。cuSPARSELt BF16结构化要求和操作布局需实核，[NVIDIA datatype documentation](https://docs.nvidia.com/cuda/cusparselt/types.html)只证明格式类别。2:4一组≤2个非零是必要条件，具体stride/压缩方向、0/1有效项的编码、尾行padding、backend/precision还需真实测试；不能把单token作者FP16 GEMV当大M BF16性能。
2. **改变动作的近似对照。** 每token每4组保留最大两项或对每个M16组选共享K列，并裁去仍非零TEAL元素，可直接选择硬件友好2:4/整列；这改变已验证TEAL函数，必须标为新近似，不继承旧视频质量。需以同等真实执行预算对照原TEAL mask、随机/固定/能量等量强基线，先做局部 W2/gated误差，再全160、same source/seed/recipe学生自由81帧比较主体运动、事件、新增物体、相机和背景。每组M16共享阈值/集合若只是额外保留原本TEAL已删元素，也改变mask及结果；真实输出先做neutral验证。

在线完整cost：读h/阈值、比较+掩码、per-row计数、组并集与扫描、selector/排序、prefix-sum、H/W2 gather与pack、可能的布局转换或跨组重复W2加载、grouped GEMM调度、残差、归并/scatter、bias/gate、H/输出额外流量、workspace/显存、compile、warmup、launch和全FFN wall/CUDA event。权重若按本次S_G在线重排不能算静态免费；对同一个W2在不同组的多次pack/读取也不可合并为一次。CUTLASS官方[组问题调度](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)明确形状不均造成负载不平衡；不因此假称这里有对应实测收益。

## 三个分阶段、独立授权的具体范围草案

**A / 现有行上界与2:4资格（最小真实CPU分析，非已运行）。** 复用原8源×4step×40block×256的已验证BF16操作数和冻结25/50阈值；优先开发894/256两个来源做2×160 site首批，6校准源另标历史fit/exploratory。每包读取真实`h`与行ID、确认source/seed/step/expert/sha以及当前teacher状态；一次只用其本来256行，绝不补0/复制。逐site/threshold输出实际元素删除率、cold256上界、每4连续K中非零数0–4分布、整行2:4合格率（含0/1项）、每K64组全零仅作sample-row统计、计量ties；所有比值先site内再source等权。该范围2×160×256×13824×2=2,264,924,160 B为h payload上限估算，不含原包其他字段/headers、hash/重读和阈值/报告；冻结实际asset文件大小、单次读/哈希次数、CPU线程、RSS与wall后由独立review核预算。若sample cold列上界已接近零，直接排除靠整M冷列的该机制；非零上界仍需full-M判定。独立样本数为两图，不是320site。不可将任意被采样行拼成真实M16 tile。

**B / Full-M原生采集producer（仅A不足时或同时用于组结构，非已实现/未授权）。** 新建独立producer，从真实Wan native W2输入hook读取原连续 BF16 h，记录source 1040及979、seed42、相同原Dense 4步各1视频；每source精确4个site `(step,block)=(0,0),(0,20),(2,0),(2,20)`，确保high/low专家身份对应，2来源×4=8真实H；完整原生2视频总各160 site正常计算，钩子只采8次，不额外替换forward。8份原h payload 7,004,454,912 B；含真实row排列、尺寸/stride、source图prompt、checkpoint/recipe/SHA、threshold ID、门控、运行前后GPU和producer SHA；留出适当完整文件格式/IO日志余量并实核可用磁盘。为了科学区分temporal/spatial相关，采集真实完整raster映射；两个阈值的bitmask可在线pack或由下游另审构造，不额外保存错误精度或自称原始激活。监控观测者对forward和显存的费用，预先绑定读取/写入上限；两source原视频已暴露属开发，不作为确认。若要评估candidate学生自由状态，则各source另采25/50分支各1视频、同4 site，共额外16 H≈14,008,909,824 B；必须独立增加真实预算和审查。已有4source完整视频只支撑动作/事件观察，不能把其Dense h当学生状态。

**C / 真正W2 kernel判别（B资产及独立评议后，非已运行）。** 对2 source×2 expert×2 block的8个真实full-M h和对应4个真实W2，固定两个原始门限，比较原dense W2不变、原mask+dense W2、exact-mask group-union batched稀疏及实际可用2:4+残差方案；逐source/site/threshold验证mask/tie、输出及gate，按实际BF16原生误差预冻结容差，包含zero-threshold neutral、同硬件同形状原生Dense和不同稀疏率边界。先限制1 GPU、源的普通行group与固定shuffle，warmup、交错测量、事件+同步wall，记录实际M/N/K分块/尾部、组K并集、调度、kernel调用、额外内存及selector/pack/gather/scatter/merge各成本，再报告整个W2及完整FFN/net costs；真实fullM额外多source和自由视频需要后续scope，不能从算子收益推出系统收益。任何不支持的布局、无有效省算或超资源按实失败，不自动调参重跑。

最小项目路径：A可立即交独立评议排期；B则把待观测真实full-M从开放疑问变成受审producer；C依B实际资产和kernel实现另审。原始科学/质量结果、定量seed、实际视频与原作者论文身份不改变；无新性能、质量或申请执行的声称。
