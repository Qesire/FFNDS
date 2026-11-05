# 原始 Wan2.2 逐层前馈增量与锚点缓存：实现说明

本文描述工作区内的具体实现、数据来源和数值边界。下列源码与证据链接指向仓库内直接浏览副本；其余历史源码可通过证据目录的源码包按原工作区路径查找。实验结果与评价见对应实际运行收据，本文不把配置、代码或预检视为效果证明，也不对方法新颖性作未经检索的判断。

## 1. 计算对象

对一个确定的专家、Transformer 层、CFG 分支和 token，以行向量记前馈网络输入为 $u\in\mathbb R^{5120}$。输入不是原始 latent：原生路径先将交叉注意力输出加回状态，再经过该层归一化、条件调制及相应 dtype 转换。完整前馈网络为

$$z=uW_1^\top+b_1,\qquad h=\operatorname{GELU}_{\tanh}(z),\qquad F(u)=hW_2^\top+b_2.$$

原始权重存储形状为 $W_1:[13824,5120]$、$W_2:[5120,13824]$，偏置分别为 $[13824]$ 与 $[5120]$。这里省略批次和 token 维度；实际一次输入为完整 token 网格矩阵。GELU 使用原生 `approximate='tanh'`。公式说明实数关系；实际运行中的 BF16 矩阵乘、累加、cast 与加法舍入不可互换。

$F$ 指门控之前的 FFN 输出。原生 `post_process` 随后使用当前步条件门控将 $F$ 或其替代值乘以 gate 并加回残差状态；不缓存 gate，也不把 gate 合入 $A$。支持的非融合原生路径与门控位置见 [transformer_infer.py](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/gate_source_evidence/transformer_infer.py)；r8 的 W1/W2 包装入口和实际返回见 [anchored_hook.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/anchored_hook.py)。

## 2. 完整线性增量矩阵的构造

每个专家、每层构造一个完整的 $A\in\mathbb R^{5120\times5120}$，两个 CFG 分支共用该专家/层的 $A$；$A$ 不是低秩分解。它的通道斜率从**原始 40 步模型**的 CAL 轨迹拟合，数据是 `chip10_02`、`chip10_04`，均 seed 42；每个来源各包含两条 CFG 分支，与开发视频的来源和 seed 分开。采集对每个专家/层使用 64 个固定 token 行，将非锚点与相同来源、专家、层、CFG 分支和 token 的最近锚点配对。CAL 拟合窗口采用全局 0、10、20、30 刷新及实际专家切换时强制刷新；实际切换使锚点为 0、10、15、20、30。r8 的五步视频使用同一批已构造 $A$，并未针对五步窗口重新拟合。

令 $\Delta z_j=z_j(u_t)-z_j(u_s)$，$\Delta h_j=h_j(u_t)-h_j(u_s)$，其中 $s$ 是配对锚点。这里的 $z$ 和 $h$ 分别是采集的原生 W1 输出与真实 W2 输入，并非重新计算的激活。先在每个配对的 64 token 上求 $\Delta z_j\Delta h_j$ 和 $(\Delta z_j)^2$ 的平均，再在每个来源与分支内对该专家的所有实际非锚点步等权；两个来源和两个分支各自等权。对合并后的矩 $n_j,d_j$，取

$$\alpha_j=\begin{cases}n_j/d_j,&d_j>0,\\0,&d_j=0.\end{cases}$$

即先合并矩再取比；代码不把每组斜率直接平均，不做 ridge 或裁剪。这拟合的是逐隐藏通道的增量关系，而非直接优化经过 W2 与当前 gate 的最终误差。配对及权重实现见 [runner_build.py](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/build_template/runner_build.py) 的 `moments_for_site`，采集和构建约束见 [cal_contract.py](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/build_template/cal_contract.py)。

原始 checkpoint 的 12 个分片身份由原权重 manifest 绑定。构建时按原生权重读取顺序，将存储 F32 的 W1/W2 转 BF16，再以 F64 的权重与 $\alpha$ 计算并转回 F32、BF16：

$$A=\operatorname{BF16}\!\left(\operatorname{FP32}\!\left[(\operatorname{diag}(\alpha)W_{1,\mathrm{BF16}})^\top W_{2,\mathrm{BF16}}^\top\right]\right).$$

保存布局为 `input_output`，运行时行向量计算 `u @ A`。$b_1$ 已通过真实 $z,h$ 影响拟合，两个偏置均包含在锚点完整 $F(u_s)$ 中；增量矩阵没有另外添加偏置。代码和存储格式见 [build_assets.py](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/build_template/build_assets.py)；实际构建 manifest 和溯源见 [matrix_manifest.json](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/original40_matrix_build_r1/results/matrix_manifest.json)、[alpha_manifest.json](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/original40_matrix_build_r1/results/alpha_manifest.json) 与 [build_provenance.json](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/original40_matrix_build_r1/results/build_provenance.json)。原始 40 步矩阵没有借用历史四步蒸馏模型的权重或斜率。

## 3. 锚点与命中公式

在锚点步 $s$，仍执行完整 $F(u_s)$，并以 BF16 矩阵乘得到 $P_s=\operatorname{BF16}(u_s A)$。按完整 token 网格形成

$$C_s=\operatorname{FP32}(F(u_s))-\operatorname{FP32}(P_s).$$

FP32 模式保存 $C_s$；BF16 模式先做上述 FP32 差，再将差量转换为 BF16 保存。锚点步本身向原生后续路径返回完整 $F(u_s)$。非锚点步 $t$ 命中时，以当前 $u_t$ 计算

$$\widehat F(u_t)=\operatorname{BF16}\!\left(\operatorname{FP32}(\operatorname{BF16}(u_t A))+\operatorname{FP32}(C_s^{\rm stored})\right).$$

理想实数表达为 $F(u_s)+(u_t-u_s)A$；上式才是实际计算顺序。BF16 缓存的量化和最终 cast 造成额外误差，不能把该等式当成逐位相等保证。精确实现见 [cache_codec.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/cache_codec.py)。

r8 在同一 job 中依次运行 Dense、GPU BF16 缓存、GPU FP32 缓存，使用同一图像和 seed 43。四十步中全局步 0、5、10、15、20、25、30、35 刷新；如果实际专家切换发生在其他步也必须刷新，且本 scope 对实际八个锚步作覆盖检查。每个候选臂对应 640 个锚点站点与 2560 个命中站点：40 层 × 2 CFG × 40 步，共 3200 次 FFN 站点。CFG 条件与无条件分支独立生成和读取缓存；专家也必须匹配，绝不跨专家或分支借用残差。原生采样器仍分别计算两个分支并按其 CFG 规则组合，原始视频配置为 guidance `[3.5,3.5]`、shift 5、boundary 0.9；代码根据实际 timestep 检验高/低噪专家，而不是把原来四步 dispatch 硬套到 40 步。配置与调度见 [normal_pipeline.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/normal_pipeline.py)、[anchor_schedule.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/anchor_schedule.py) 和 [runner_video.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/runner_video.py)。

## 4. 实际执行和缓存生命周期

Dense 臂调用完整原生 FFN。锚点包装 W1 和 W2，截获同一次前向的 $u_s$ 与原生 $F(u_s)$，写缓存但仍返回原生输出。命中时包装的 W1 返回输入而不执行 W1 GEMM，包装的 W2 返回 $u_tA+C_s$ 而不执行 W2 GEMM；其余原生控制流仍继续。因此该实现**仍保留**输入 `clone`、对宽度 5120 的中间输入执行但最终丢弃的原生 GELU，以及完整 block 权重预取。它不是已经实现直接前缀或选择性跳过预取的最终集成路径。后续独立草稿不属于本次运行。

r8 每个候选臂预分配一个连续 GPU slab `[80,31668,5120]`；槽位由 CFG 分支和层决定。每个槽只在所属当前专家写入后才有效，新锚点用 `copy_` 覆盖槽位；同一锚点在接下来的四个非锚点步反复读取。专家切换会使全部旧槽的逻辑有效性失效，然后复用同一物理 slab；最后一步、VAE 前释放 slab 与 A 缓存。管理实现见 [gpu_slab.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/gpu_slab.py) 和 [anchored_hook.py](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/anchored_hook.py)。

单槽有 162,140,160 个元素，BF16 为 324,280,320 字节，FP32 为 648,560,640 字节；80 槽分别为 25,942,425,600 字节（约 24.16 GiB）和 51,884,851,200 字节（约 48.32 GiB）。这只是 slab 载荷，不包含 A、两块原生 offload buffer、激活、GEMM 工作区或刷新时暂存新残差。物理 slab、已初始化逻辑槽位和 allocator 峰值是不同指标。完整单专家 40 个 BF16 A 约 1.95 GiB。每步的临时 $u$、$uA$、完整输出及编码结果在对应使用结束后可释放；未到最后一次命中的锚点不能在一次读取后丢弃。采样步之间还依赖整个上一轮双分支网络和 scheduler 更新，因此不能把某一层的五个采样步提前连续算完。

## 5. 预检、计时与历史范围

每个候选臂前分别使用已验收 E1 来源的真实完整形状 field 验证实际 $uA$、R0 的 CPU/GPU 往返和 R16 的独立 cast/decode 公式，并顺序分配该格式的完整 slab，检查分支槽不别名、覆盖、专家失效与重用；预检结束释放 slab，正式臂重新分配。预检与生成的资源/时间分别记录。真实运行是否通过，以对应收据和独立复核为准，不能用静态单测替代 GPU 结果。

每臂总时从必要清理与正式 slab 分配之前开始、到生成完成同步结束；还分别记录原始生成 wall、去噪 wall 和包含在去噪内的 A 首载 host wall。后者不是独立 GPU 搬运计时。原始生成 wall 含初态与最终 latent 诊断；去噪阶段不做每站点有限值规约、磁盘日志或 40 次中间 latent 哈希。最终 latent 和 81 帧视频在对应边界核对，Dense 的完整 decoded RGB 哈希与历史已接受 Dense 对照。单次顺序三臂只是 pilot，不能当重复试验的置信区间或整体系统加速保证；完整实现和资源合同见 [IMPLEMENTATION_CONTRACT.md](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/IMPLEMENTATION_CONTRACT.md)。

历史四步 [history_ablation.py](../../reports/20261105/g19_evidence/browse/work/g19_history_ablation_20261104/video_chip10_00_r2/history_ablation.py) 是另一实验：它在步 0、2 为各专家建立缓存，步 1、3 使用一次后从 CPU cache 弹出，并比较 reuse、static affine 与 anchored 等方案；其四步专家 dispatch、缓存键及蒸馏模型身份不能直接移植到原始 40 步双 CFG。原始 40 步的十步版本使用真实专家切换刷新和按专家/分支/层缓存，见 [anchored_hook.py](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/video_template/anchored_hook.py)。这些历史代码的结果只支持各自已记录的实验范围。


## 6. 可直接浏览的推导与证据

以下文件为原文件逐字节副本；其中历史文档里的本机绝对路径保留原样，其余历史源码位于证据目录的源码包。视频不在本目录复制。

- [ARCHITECTURE_DERIVATION.md](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/ARCHITECTURE_DERIVATION.md)
- [CACHE_COMPRESSION_DERIVATION.md](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/CACHE_COMPRESSION_DERIVATION.md)
- [INTEGRATED_CACHE_AND_EXECUTION_20261105.md](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/INTEGRATED_CACHE_AND_EXECUTION_20261105.md)
- [DISCRIMINATING_EXPERIMENT_PLAN_20261105.md](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/experiment_design/DISCRIMINATING_EXPERIMENT_PLAN_20261105.md)
- [E4A_RESULTS_AND_NEXT_DECISION_20261105.md](../../reports/20261105/g19_evidence/browse/work/g19_execution_e4a_20261105/E4A_RESULTS_AND_NEXT_DECISION_20261105.md)
- [CACHE_LIFETIME_AND_PIPELINE.md](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/CACHE_LIFETIME_AND_PIPELINE.md)
- [SOLVER_HISTORY_INTERPRETATION.md](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/SOLVER_HISTORY_INTERPRETATION.md)
- [5步窗口_GPU固定缓存_结果与判断_20261105.md](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/5步窗口_GPU固定缓存_结果与判断_20261105.md)
- [PAIRED_TABLE.md](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/quality_results/complete_r1/PAIRED_TABLE.md)
- [INDEPENDENT_NUMERICAL_RESULT_REVIEW.json](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/quality_results/complete_r1/INDEPENDENT_NUMERICAL_RESULT_REVIEW.json)
- [INDEPENDENT_PRODUCER_RESULT_REVIEW.json](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/results/INDEPENDENT_PRODUCER_RESULT_REVIEW.json)
- [INDEPENDENT_PRODUCER_RESULT_REVIEW_CORRIGENDUM.json](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/results/INDEPENDENT_PRODUCER_RESULT_REVIEW_CORRIGENDUM.json)
