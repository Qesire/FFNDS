> **2026-11-07 归档补充（优先阅读）**：本文早期方案在对旧资产复核前撰写。**Chipmunk-MLP 并非尚未复现**：FFNDS [20261030 全160 P64 视频](../../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.md)、[质量与RAFT](../../20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.md)和[真M packed](../../20261030/CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.md)已有真实收据；新任务仅在身份不等或数据不足时获批后建立匹配新scope。ICC 尚无最新版四步 Wan 复现，TARDIS 作者完整开源实现尚未核实。现行任务入口为 [README](README.md)、[资产crosswalk](G19_BASELINE_AND_ASSETS.md)、[实验合同](EXPERIMENT_CONTRACT.md)。旧指南下文任何未与新证据核对的“需要首次复现”均降为历史设计。

# FFNDS G19：FFN-only 前沿方法复现资料与实验合同

整理日期：2026-11-07。性质：既有资料归档及细化；将写入仓库，未启动 GPU 实验。

## 0. 研究问题与优先级

目标是在 **Wan2.2-I2V-A14B-LightX2V-Distill-4step、HHLL、无 CFG、BF16、A100 80GB、832×464、81帧** 上比较 FFN-only 加速方法。保持 Attention、norm、scale/shift、当前 gate、采样器、VAE、prompt、seed、token 布局以及除 FFN 以外的路径相同。

**更新的优先级：ICC-FFN 为首项新复现；Chipmunk-MLP 已有 Wan2.2 四步全160 FFN 真实实测，先复用并进行新版G19同配置对齐；TARDIS 先审论文忠实独立重实现。**若首批结果显示动态稀疏/线性折叠有潜力，再考虑 RT-Lynx、SlimDiff、ToaST/FocusDiT。

三种工作身份必须分别记录：

- **AUTHOR_ORIGINAL**：原论文、原仓库、原任务原硬件的结果，不能直接与 Wan2.2 四步数字比较。
- **WAN_PORT**：原核心算法移植到 Wan2.2 四步；模块范围及源码变动写清楚。
- **FFN_ONLY_BENCH**：Attention 及非 FFN 模块均固定；论文主表采用此身份。

### 当前 G19 基线（严格冻结）

- GitHub: https://github.com/Qesire/FFNDS ，提交 `6d7a52b7110f053b4b469c76cb0e8283620eda64`。
- 入口：`research/20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md`；证据：`reports/20261107/g19/README.md`。
- 关键实际源码：`reports/20261107/g19/source/q1_r2/direct.py`、`fused_post.py`、`selective.py`；矩阵身份：`reports/20261107/g19/Q1_matrix_manifest.json`、`Q2_matrix_manifest.json`；逐来源分数：`quality_summary.json`、`quality_per_video.json`；运动：`motion_summary.json`；工程：`engineering_summary.json`。
- G19 当前固定 `A[expert,layer] ∈ BF16^{5120×5120}` 共80张；单视频40个 BF16 残差槽，`N=31668, d=5120, h=13824`。step 0/2 为专家锚点，1/3 为命中。80次完整 FFN、160次 A GEMM。
- G0 历史配对均值 Dense 39.434518 s，G0 37.613504 s，约 1.048414×；仅为非冷启动描述性配对结果，**不可称新 Q1-r2 的实测端到端速度**。
- G0 理论四步 FFN 主 MAC 比例：`0.5 + d/(2h) = 0.685185`，节省31.4815%。不是整模型 FLOPs/实际耗时。
- 七个已暴露 DEV 来源：`raft, bicycle, guitar, tennis, spoon, eagle, canoe_static_00`；seed 43、44；CAL 四来源：`chip10_02/04/07/09`，seed 42。来源是统计单位。
- 最新 Q1-r2：I2V 主体 `0.98288357`，背景 `0.98369482`，动态阈值通过数6/14；G0相应为`0.98021090`、`0.98253355`、4/14；但 Q1-r2 的一般主体一致性、美学与运动平滑度低于G0，RAFT方向余弦的七源均值下降，故非全面胜出。
- 打包 W1/A GEMM、跳过打包 W1 预取两项 pilot 均未观察到稳定速度收益；默认继续使用 `fused_post`。当前 `Q1/Q2` 是拟合标签，**不是 HiF4/FP4 量化标签**。
- 仓库没有复制原始 MP4、完整 RAFT 大场、全部 A 权重；使用 `local_video_index.json`、`receipt_index.json` 和服务器 scope 识别原件，不得把文件路径索引当成原始媒体。

## 1. 核心来源与固定身份

### ICC：CVPR 2025

- 论文：*Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition*；https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html
- arXiv：https://arxiv.org/abs/2505.05829 。作者：Zhiyuan Chen, Keyi Li, Yifan Jia, Le Ye, Yufei Ma。
- 作者代码：https://github.com/ccccczzy/icc ；固定 commit `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`，分支 `master`。
- 核心文件：`DiT/models/models_sp_step_decomp.py` 的 `SpMlp.forward_upd` / `forward_inc`（锚点及增量），`DiT/adapters/sched_sp_step.py`（固定周期），`DiT/gen_decomp.py` 与 `DiT/func/func_decomp.py`（分解/CA-SVD/CD-SVD），`DiT/models/cache.py`（状态）。
- 原 README 命令分别为 `python gen_decomp.py --use-wtv False --rank 128`、`--use-wtv True --wtv-src mag`、`--use-wtv True --wtv-src delta_mag`。这些是**DiT 图像生成原任务**的复现命令，不能用它们直接生成 Wan2.2 的参数。
- 源码机制：锚点缓存 `fc1_x, fc1_y, fc2_x, fc2_y`；命中先用第一层低秩近似重建当前 preactivation，继续 GELU，再用第二层低秩近似校正输出。若两层低秩算子秩均为 r，hit 主 MAC ≈ `2r(d+h)`，两个调用一完整一命中的四步 FFN 比例约 `0.5 + r(d+h)/(2dh)`。ICC 还实现了 Attention 增量，但 **FFN-only 移植不得启用**。
- 适配点：两专家各自 `W1/W2` 和校准权重；step0/2执行 `forward_upd`，step1/3执行 `forward_inc`；native GELU(tanh)、偏置、gate和输出dtype保持一致；需提供 per-layer/expert/sequence 缓存身份，不能跨专家借旧状态。
- 缓存预算：直接四缓存约 `40×N×[2(d+h)]×2 bytes ≈ 89.39 GiB`（仅载荷）；实数代数可化为 `q1=z_s−B1u_s`、`q2=y_s−B2h_s`，只保留 `(h+d)` 个值/token，约 `44.70 GiB`，但 BF16 浮点顺序会变化，必须对照原源码和独立数值验收。低秩参数及临时工作区另计。
- **复现实验**：rank 64/128/256/512；分别做普通 SVD、CD-SVD，CA-SVD 视资源选；逐模型位置的 rank 与质量预算需要固定，不能在 DEV 上反复选择后称零调参。

### Chipmunk：2025 预印本、作者实现

- 论文：*Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas*；https://arxiv.org/abs/2506.03275 。作者 Austin Silveria, Soham V. Govande, Daniel Y. Fu。
- 作者代码：https://github.com/sandyresearch/chipmunk ；固定 commit `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`，分支 `master`。
- 核心文件：`src/chipmunk/modules/mlp.py`（`SparseDiffMlp`），`src/chipmunk/ops/mlp.py`（算子调度），`src/chipmunk/triton/mlp_csp_mm1_bf16.py` / `mlp_csp_mm2_bf16.py`（BF16 Triton），`src/chipmunk/util/layer_counter.py`、`util/config.py`、`util/storage/*`（调度、配置、缓存）。
- 入口：`examples/YOUR-MODEL-HERE/README.md`；`examples/wan/`是作者的Wan2.1 T2V 示例，但 `examples/wan/chipmunk-config.yml` 当前明确 `mlp.is_enabled: false`、`attn.is_enabled: true`、且启用step caching。因此**运行该目录的默认示例不是 Chipmunk-FFN 的复现**。
- 机制：完整步骤保存 GELU 后激活及 MLP 输出；稀疏步骤使用 block mean 的 W1 预测变化、动态 top-k 选择隐藏列，重算这些列的 GELU 增量并累积到输出缓存；有 mask 选择和 in-place cache 更新成本。不能把 `top_keys` 直接等同理论 FLOPs，必须计实际执行有效索引、block padding、kernel、缓存/显存读写。
- A100：官方 setup.py 中部分 CUDA MLP kernel 仅编译 Hopper (sm90)，对应有 Triton fallback；初步优先 `mlp.provider: triton`，但仍需要其 `copy_indices`、`topk_indices` 等扩展；在隔离环境中编译与 `import`，避免污染现有 wan Conda。`setup.py` 需要递归子模块（ThunderKittens）。作者依赖写明 `torch>=2.5.0`；不应仅因当前 PyTorch 更高就认定 ABI 兼容。
- G19 FFN 输入形状 `(31668,5120)`，作者 `SparseDiffMlp.forward` 要求 `(1,N,C)` 且 `N` 按128分组；31668不能被128整除，最小补齐到31744（76个dummy token）。只在FFN wrapper内做安全padding/unpadding，必须避免虚假 token 污染 block mean、top-k或有效输出；测试最后不足一个块的路径。
- 推荐四步实验配置草案（非原作者发布的 Wan2.2 参数）：`steps:4; num_model_invocations_per_inference_step:1; mlp.is_enabled:true; mlp.full_step_every:2; mlp.provider:triton; attn.is_enabled:false; step_caching.is_enabled:false; mlp.top_keys ∈ {0.15,0.30,0.50}`。`first_n_dense_layers`作为作者参数单列，先测试0和2。跨专家的step2必须完整刷新；step1/3才稀疏。
- **特别审计**：`LayerCounter.increment()`及其最终reset条件需在4步×40层、无CFG、只有MLP稀疏的组合上验证所有160站点坐标，避免单例计数器 off-by-one、重复reset或两个专家意外共用旧状态。尽量以Wan真实 `(expert,step,layer)` 驱动调度和缓存验证。
- 作者报告的数倍视频加速主要是在多步模型上联合 Attention/MLP或整步缓存，不能当作四步 FFN-only 预计速度。

### TARDIS：可实现的第三项反事实，源码身份待补

- 论文：*Accelerating Large Language Models through Partially Linear Feed-Forward Network*，arXiv:2501.10054；https://arxiv.org/abs/2501.10054 。作者 Gansen Hu 等。
- 本轮**未确认作者官方 GitHub 实现**。FFNDS 之前已进行原文审计：`research/literature/20261029/S2_WHOLE_FFN_REPRESENTATION_PRIMARY_REVIEW_20261029.md`。
- 方法：在校准分布常见区间对每个GELU隐藏通道拟合 `a_j z_j+c_j`，折叠 `A=W1ᵀ diag(a)W2ᵀ`、偏置项；在线通过低成本预测器发现区间外神经元，减去其折叠贡献并加回真实GELU贡献。这是静态线性折叠+动态回退，**不是 G19 的跨步锚点残差**。
- 可用两个不同身份的对照：`STATIC_FOLD_ONLY`（只用折叠后的A与常数，无在线回退）是廉价控制组；`TARDIS_PAPER_REIMPLEMENTED` 须包含作者的区间选择、低比特/轻量离群预测、精确修正和执行成本，才能声称算法复现。
- 先在Wan原生BF16、真实pre-W1输入上验证常见区间覆盖率、漏判率、后门控输出误差与回退成本。没有可靠预测器时，使用真W1全部检查区间的“oracle回退”只能作质量上界，不可用作实际速度表。

## 2. 次级候选及源码状态

| 工作 | 文献 | FFN机制 | 本轮核实的执行状态 |
|---|---|---|---|
| RT-Lynx | https://proceedings.mlr.press/v306/cong26a.html | N:M激活稀疏+补偿+专用CUDA | 论文核实，**未锁定经作者确认的FFN-only Wan2.2可执行仓库**；先列来源，不启动重型移植 |
| SlimDiff | https://arxiv.org/abs/2509.21498 | activation-aware功能组谱压缩，含FFN | 论文核实，**未核实完整官方代码**；若需要FFN-only应复现整体结构后的单模块消融 |
| ToaST | https://github.com/SHANNonLab-HUFS/ToaSt | per-forward FFN 通道选择；作者仓库另有Attention剪枝 | 提供实际代码，但 README 的正式论文元数据仍为 TODO、以ViT分类训练/评价为主；需要另查正式论文身份和重新设定Wan评分 |
| FocusDiT | https://arxiv.org/abs/2606.02090 | 重要query选择执行FFN | 论文核实；公开可复现实现状态待确认，属于跨架构token选择，不进入首批 |

## 3. 相同计算边界下的最小基线面板

- `D0`: 原生Attention与全部160 native FFN调用。必须保留原基线噪声/视频身份。
- `G0`: 现有G19 Q0，80原生FFN+160个A；旧速度仅描述性。
- `G19-Q1r2`: 当前输出感知质量候选，执行结构不变；新速度不可借用G0作为实测。
- `REUSE`: 每专家首次完整计算，第二步直接用缓存FFN输出；只为证明当前输入增量的贡献，不是前沿方法。
- `ICC-SVD`: 两层低秩增量，rank sweep；`ICC-CD-SVD`: 增量分布校准的rank sweep。
- `CHIPMUNK-MLP`: 动态激活列更新，top_keys sweep；只启用MLP分支。
- `TARDIS-STATIC`: 静态折叠无回退；`TARDIS-FULL` 在算法完整后加入。

确保针对所有方法：**同一四步HHLL、BF16、torch_sdpa、同一text/image/seed、同一Attention、同一当前gate及采样器**。如果内核仅支持另一种输入dtype或额外量化，必须单列为不同数值配置，不能当成公平FFN-only常规基线。

## 4. 关键数学与成本对照

原FFN单token主MAC：`2dh`。G19目前四步结构：`rho_G19 = 1/2 + d/(2h) =0.685185`。ICC假设两层低秩秩都为r，且每专家第一次全算，第二次hit：`rho_ICC = 1/2 + r(d+h)/(2dh)`。此式只比较主乘法，遗漏GELU、稀疏准备、buffer解码、CPU/GPU搬运、offload与预热。

| r | ICC理论FFN主MAC剩余 | 理论FFN主MAC节省 |
|---:|---:|---:|
| 64 | 50.857% |49.143%|
| 128| 51.713%|48.287%|
| 256|53.426%|46.574%|
| 512|56.852%|43.148%|

G19单个FFN命中只需完整d×d GEMM，但约有 `12.08GiB` BF16残差slot+`3.91GiB` BF16 A payload；ICC优化双残差约44.7GiB，原四缓存约89.4GiB（不含低秩参数）。Chipmunk保存中间激活、输出、块均值、索引等，载荷取决于实现与稀疏率，须在相同配置实测。若任何方法未能在单A100安全运行，可采用确定性分层offload并计全部访存；不可在无记录条件下删减原算法功能。

## 5. 标准执行阶段（GPU作业前须独立评议）

### S0：只读资产验收，无GPU
1. 保存本资料给复现Agent，并再次检查 `main` 和固定commit是否更新；工作目录另开 `work/ffn_baselines_20261107/...`，禁止覆盖旧 G19 资产与已成功scope。
2. 校验原Wan checkpoint、FFN参数shape、GELU近似、LightX2V接口、现有 CAL/DEV及seed身份、80处矩阵与报告的SHA；数据缺失则标识，不以GitHub归档中的路径链接代替实物。
3. 记录作者仓库commit、子模块SHA、依赖锁、编译日志、许可证约束、论文/代码版本差别；记录每项方案是AUTHOR_ORIGINAL/WAN_PORT/FFN_ONLY_BENCH中的哪一种。

### S1：原生数值等价与调度单测
1. 适配器禁用加速时，应与Dense完整FFN输出和后续BF16路径一致；专家high/low、四步、全部40层、native gate及bias不改变。
2. 计数必须恰好160站点，其中step0/2各40锚点；step1/3各40近似；对每个方案记录锚点/命中、当前模型身份、缓存失效与刷新、张量shape/dtype/stride。
3. 特别测试：Chipmunk单例layer counter的最后一层、expert切换、padding最后76位置、多个视频连续运行时缓存清空；ICC原代码的norm/drop层在Wan中应对应Identity/无drop，不能引入额外操作。

### S2：真实FFN同状态离线回放
- 对CAL与已暴露DEV的分布身份分离；对真实配对pre-W1 `U_s,U_t`、`F(U_s),F(U_t)`、gate，分别评估隐藏/输出/gated后相对平方误差、p95/p99、按专家/层/时空区域的尾部。
- 对ICC跑r64/128/256/512及SVD/CD-SVD；Chipmunk跑保留比例0.15/0.30/0.50；TARDIS区间与回退比例。选择规则与评测集隔离。低局部SSE不直接代表视频质量好。

### S3：实际 FFN 算子和缓存微基准
- 同时测anchor、hit、调度/选择、写/读缓存、GEMM、H2D/D2H、allocated/reserved显存；A100同机配对，预热、交错次序和至少数次重复。单层测量与端到端不可混称。
- Chipmunk必须测选择mask实际时间，不能只量稀疏matmul；ICC比较rank时保留中间状态的GPU/CPU保存方式与传输成本；TARDIS必须计预测器和回退。
- 若原作者kernel难编译，先记录`BUILD_BLOCKED`，不要以PyTorch dense索引实现的慢结果代表该论文成熟CUDA速度；可以报告 FUNCTIONAL_PORT 质量结果与 KERNEL_NOT_REPRODUCED 的性能状态。

### S4：少量真实完整视频筛查
- 优先选择动态明显来源（如raft/tennis/eagle）并配1个背景/静态对照；输入、seed与Dense固定。先生成完整视频再评分，不用捕获Dense中间状态替代学生闭环。
- 若候选明显降低主体保真、光流方向或经原生优化后仍不省时，停止扩展其参数组合；保留负结果。

### S5：完整开发来源和新确认集
- DEV：七来源×两seed，全部八维原有指标；追加时间闪烁（如有独立实现）、连续RAFT向量、beta/方向余弦/正交残差、CoTracker固定共同可见点、语义动作/相机/前景背景分开（这些当前尚未全部测得）。
- 统计单位为来源，seed只在来源内聚合；不能以14个视频、40个RAFT帧对、层或token当独立来源。先冻结模型/参数/阈值，再开未暴露CONFIRM来源；不在已暴露DEV上选完参数后声称confirm。
- 正式速度：对等冷/暖机、同硬件、相同non-FFN、交错测量、记录模型加载、一次性矩阵/预测器准备、完整generate、VAE时间、峰值显存。

## 6. 失败优先级和继续/退出条件

1. **数值边界错误**（pre-W1输入、GELU、gate、bias、expert/缓存复用、采样身份不符）→ 立刻停止，修正后重测S1。
2. **Chipmunk kernel/padding不正确或ICC缓存超显存** → 保留算法级/离线对照身份，先制定合法offload或分块实现，不在全视频上冒险。
3. **仅理论MAC更少，真实FFN或全模型延迟不降** → 作为负结果，不称速度基线胜出；判断是mask/搬运/小GEMM/内存瓶颈。
4. **局部误差下降而RAFT余弦或视频主体行为恶化** → 不按局部误差排序，回到同预算视频Pareto。
5. **某方法只能在50步、不同任务或H100上运行** → 归为相关方法，不与四步Wan FFN-only并排排名。

## 7. 原作安装命令（**不要直接在现有wan环境运行**）

```bash
mkdir -p third_party

git clone https://github.com/ccccczzy/icc.git third_party/icc
git -C third_party/icc checkout dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4

git clone --recurse-submodules https://github.com/sandyresearch/chipmunk.git third_party/chipmunk
git -C third_party/chipmunk checkout b5e8d1c3da6bddc7ca9007c48e91b82087c11490
git -C third_party/chipmunk submodule update --init --recursive

# 阅读和静态检查阶段：
git -C third_party/icc rev-parse HEAD
git -C third_party/chipmunk rev-parse HEAD
git -C third_party/chipmunk submodule status
```

ICC的作者自带 `conda env create -f DiT/environment.yml` 用于原作者 DiT 复现；Wan移植优先提取算法而不覆盖现有环境。Chipmunk README 原安装包含独立Python3.11/CUDA12.8环境、`pip install -e . --no-build-isolation`，必须在允许编译的计算节点、隔离环境和正式范围内操作；其包含`flash_attn`等完整模型依赖，不意味着FFN-only适配都必需。先在隔离测试中验证Triton与扩展可用，再考虑生产环境。

## 8. 交付物及核验格式

每项作者方法至少交付：

```
method_id/
  PAPER_SOURCE.md              # paper, source commit,许可证/差异及引文
  PORTING_DIFF.md              # Native module mapping, algorithm fidelity
  CONFIG.json                  # model, experts, steps, rank/top_keys, seed, dtype
  SHA256_MANIFEST.json         # source and data identity
  PREFLIGHT.json               # shapes, stage counter, expert/cache isolation
  LOCAL_FFN_METRICS.json       # delta/gated/quant, costs, timing
  VIDEO_MANIFEST.json          # MP4 SHA, seed, inputs, configs
  QUALITY_BY_SOURCE.json       # source-level paired metrics
  MOTION_BY_SOURCE.json        # RAFT/trajectory including fixed support
  TIMING_AND_MEMORY.json       # real latency, cold/warm, cache, allocated/reserved
  INDEPENDENT_REVIEW.md        # complete/not run/failure, no fabricated PASS
```

**完成判据**：必须同时有方法忠实度与数值验收、全视频配对质量、实际 FFN 与全模型延迟、显存、可核对源码/模型/数据身份。还未运行或无权访问的实验均填写 `NOT_RUN`，不得写预期分数或虚构成功。图文质量和运动行为在不同类别均有意义，不能用单一VBench总分抵消运动衰减。

## 9. 外部文献索引

- ICC（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html
- Chipmunk（arXiv）：https://arxiv.org/abs/2506.03275 ; 作者实现 https://github.com/sandyresearch/chipmunk
- TARDIS（arXiv）：https://arxiv.org/abs/2501.10054
- RT-Lynx（ICML 2026）：https://proceedings.mlr.press/v306/cong26a.html
- SlimDiff：https://arxiv.org/abs/2509.21498
- FocusDiT：https://arxiv.org/abs/2606.02090
- ToaST 仓库：https://github.com/SHANNonLab-HUFS/ToaSt

如论文与作者最新代码有冲突，应在 `PAPER_SOURCE.md` 绑定论文版本及此次检索的commit，并保留不一致项；不能默默替换论文中宣称的方法。