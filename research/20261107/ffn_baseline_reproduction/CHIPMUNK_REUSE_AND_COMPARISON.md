# Chipmunk-MLP：历史复现复用与新版 G19 匹配协议

**状态**：HISTORICAL_WAN_REPRO_COMPLETED / CURRENT_G19_MATCHING_NOT_RUN。最重要的新任务不是再从零重跑 Chipmunk，而是审定其已存结果能否与 `2026-11-07` G19 四步结果构成同条件比较。

## 1. 上游作者身份及代码边界

论文：Silveria, Govande, Fu, *Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas*, 2025，https://arxiv.org/abs/2506.03275 。

官方仓库：https://github.com/sandyresearch/chipmunk，当前材料固定 `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`（历史 FFNDS 文档曾读取另一作者提交，须记两版差异）。
- 核心 `src/chipmunk/modules/mlp.py` → `SparseDiffMlp.forward`；
- `src/chipmunk/ops/mlp.py` → bf16 sparse GEMM/缓存累加；
- `src/chipmunk/triton/mlp_csp_mm1_bf16.py`、`mlp_csp_mm2_bf16.py`；
- `src/chipmunk/util/layer_counter.py`、`config.py`、`storage`；
- `examples/YOUR-MODEL-HERE/README.md` 说明通用移植。
- **作者Wan2.1示例 `examples/wan/chipmunk-config.yml`：`mlp.is_enabled=false`、`attn.is_enabled=true`、`step_caching.is_enabled=true`**，所以默认作者Wan Demo 的几倍系统加速不是 MLP-only 成绩；仅改config布尔值不能自动把 Wan `self.ffn` 接到 `SparseDiffMlp`（见FFNDS 20261029源码审查）。

源码核心：第一个完整调用保存 `h_s=GELU(W1u_s+b1)`、`y_s=W2h_s+b2`、group/block均值状态；命中时根据块均值的当前 `W1` 代理估计隐藏列变化并选择mask，真实更新选中 `h_S`，由自定义列稀疏GEMM修正输出缓存。稀疏列索引、缓存读写、mask预测/转换均为已付费计算。此机制与 G19 的全矩阵 `Au+C_s` 不同。

## 2. 已执行资产：以实际 RUN/EXIT 为准

1. [真M=31668 packed BF16 与Triton同mask比较](../../20261030/CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.md)：两旧暴露来源×高低专家，多个mask，packed 与旧Triton该mask输出差=0；P64 local单次均约23–24ms，Dense约35–39ms。该延迟是单FFN、有观测和一定诊断，非完整生成wall。
2. [全160 P64 完整自由视频](../../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.md)：898、96两源；每源四视频(原Dense/hook Dense/cache Dense/P64学生)、全160作用范围、high/low各自缓存有效性、独立VERIFY/EXIT0。P64 wall 44.851 / 46.191秒，但固定phase并有observer，不可替代同机最新G19ABBA。
3. [8维/RAFT质量](../../20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.md)：两源等权，subject -0.013883、I2V subject -0.007873、MUSIQ -1.058402；RAFT连续top5 ratio 0.9503/0.8257，binary dynamic 各源未变。此质量不是现行七源Q1-r2集合。
4. [High/Low分阶段消融](../../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)：已有动态运动与低噪细节非加性的证据；不得把两个来源当总体普适规律。
5. [同机ABBA有观测效率](../../20261030/CHIPMUNK_SYNC_ABBA_RESULTS_20261030.md)：898/979来源FFN事件均值7.453→6.919秒、phase wall48.060→47.736秒；方法身份和来源与当前G19不同。
6. [以前失败的显存实验](../../20261030/CHIPMUNK_GLOBAL_P64_V7_FAILURE_20261030.md)：原reserved门限触发，保留失败状态/原因证据；后续V2成功不等于旧失败的真实峰值已知。
7. [20261029作者代码复核](../../20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)：作者原Wan`self.ffn`未挂SparseDiffMlp、默认只attention，以及TEAL真M回退问题。

**关键身份判定**：现有 Chipmunk P64 是 FFNDS 已实现并验证的 Wan 移植候选，可能包含当时的 mask代理/packed 选择改进和观测路径；不能在未经源码diff的情况下称完全逐字作者原算法。真正追加的 `author-code fidelity` 可做源码行为对照，但不是重新生成同样的旧source视频。

## 3. 用于新的7 DEV比较的前置审查

必须先比：旧 Chipmunk P64 的 checkpoint SHA、模型具体蒸馏版本、HHLL/无CFG、采样器、video geometry、预W1 GELU/gate、attention后端、offload、输入图片/prompt/seed、GPU/编译版本、是否有observer、是否需要历史输出重复。映射表应逐字段列 `MATCH / DIFFER / UNKNOWN`；不能“同为Wan2.2四步”就直接使用旧视频作新G19论文主表。

新比较的最小集合（必要时批准新source scope）：
- `Dense neutral`：全160原生前向并核video/latent校验；
- `Chipmunk-MLP`：Attention及整步缓存关闭，只启用FFN MLP缓存/列稀疏；High和Low各自初始Dense+第二步增量；
- `G19 G0/Q1r2`：固定相同 checkpoint、prompt/image/seed/数值族；
- `REUSE_Y`：第二步直接历史FFN Y复用，验证当前增量计算贡献；
- 每臂候选自行自由生成，不能使用捕获教师U冒充闭环。

对 Chipmunk，作者配置在四步的适配草案：
`steps=4`，`num_model_invocations_per_inference_step=1`，`mlp.is_enabled=true`，`attn.is_enabled=false`，`step_caching.is_enabled=false`，`mlp.full_step_every=2`，`mlp.provider=triton`，`top_keys∈{0.15,0.30,0.50}`；
`first_n_dense_layers` 作为另一变量(0/2)，明确实际被近似的站点数量。**这只是需审定的配置，不是既已运行的作者Wan2.2实测**。

### 输入形状与尾块

官方 `SparseDiffMlp.forward` 要求 `(1,N,d)`，其MLP内核 `BM=MBM=128`。`N=31668=247×128+52`，尾块有52个真实token，若使用最简单pad则补76到31744。对于mask/块均值，不能只给zero pad就当合法：最后块均值应只用有效52行，并保证被填充token不被选中统计、缓存读回无污染。独立保存未填充tail输出，对真实52行逐元素比较；必要时独立执行remainder kernel。不要把pad的幻影token纳入任何光流、原始位置编码或真实token指标。

### 调度计数器

作者 `LayerCounter` 维护推理步、层号、一次step内模型调用次数和稀疏子模块数量。必须按无CFG 1调用、只MLP sparse 1子模块、40层×4step验证 160个严格有序site。step2 high→low需要旧cache逻辑失效后新全量更新，两个视频之间也清空。不靠 `cur_inference_step` 推断expert，读取真实scheduler实际expert后再审。

### A100 内核和性能

官方Setup 的部分MLP CUDA核仅在Hopper `sm90` 编译；A100需优先独立测试 bf16 Triton MLP fallback，但 `copy_indices`、`topk_indices` 等扩展仍需要可用。原仓库默认依赖、子模块、ABI、BF16加法和GELU近似需记录编译版本。在允许的计算节点/隔离环境做预检，不污染现有 `wan` Conda；不能在登录节点编译运行科学GPU模型。原作者CUDA/H100结果与本地A100结果分表。

## 4. 参数选择与证据拆分

- CAL上扫 `top_keys 0.15/0.30/0.50`，`first_n_dense_layers=0/2`；先只看真M FFN同态质量与实际cost，最多选择少数进入视频，不用已暴露DEV试出最优再称确认。
- 已有 P64/Raster/Random/Global/后验Oracle 结果分清：同mask算术、合法付费选择、后验oracle和真实闭环，不混写为已部署同一配置。
- 不重复已有 20261030 PACKED/全160/RAFT/ABBA job；它们证明过去的实现与机制。新的匹配实验仅在正式差异说明与独立复审后提出。
- 若无法实现合法可靠tail和GPU内核，停止性能声称并标记 `KERNEL_BLOCKED`，可只报告算法式同态诊断（不能用torch dense indexing造成的慢计时惩罚原作者）。

## 5. 真正决定对照价值的问题

- G19的 `Au` 在哪些source上比Chipmunk有符号稀疏`Δh_S`更能保持主体/运动，尤其看同样的FFN局部延迟/实际显存？
- Chipmunk的mask计算、cache与FFN load是否吞掉少步推理下的MAC节省？
- G19是否有更低缓存/更简单执行的系统优势，尽管理论hit算术未必最低？
- 是否只有在source高运动或low expert下差异显著？需要分源/层/step报告，不按全token SSE决定最终视频。

数值和完整视频指标规范参见 [统一合同](EXPERIMENT_CONTRACT.md)。当前新七源匹配 `NOT_RUN`。
