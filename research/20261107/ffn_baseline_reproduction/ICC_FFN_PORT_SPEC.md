# ICC-FFN：Wan2.2 四步移植实施合同（首项新增基线）

**身份**：DESIGN_ONLY / NO_RUN / PREEXEC_REVIEW_REQUIRED。目标是忠实移植 CVPR 2025 ICC 的 FFN 增量机制，**不包含作者的 Attention 增量分支**。参见 [总协议](EXPERIMENT_CONTRACT.md) 与 [G19资产](G19_BASELINE_AND_ASSETS.md)。

## 1. 固定作者来源

- 论文：Chen, Li, Jia, Ye, Ma, *Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition*, CVPR 2025, pp.18011–18020。论文 https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html
- GitHub: https://github.com/ccccczzy/icc ，固定 `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。
- 原核心：`DiT/models/models_sp_step_decomp.py` 中 `SpMlp.forward_upd/forward_inc`（行69–109）与 `linear_decomp`；`DiT/models/cache.py`；`DiT/gen_decomp.py`、`DiT/func/func_decomp.py`；`DiT/adapters/sched_sp_step.py`。
- 原作者README暴露 `SVD`、`CA-SVD`（mag）、`CD-SVD`（delta_mag）入口。原 `gen_decomp.py` 针对DiT-XL图像生成/原数据，不可直接输入Wan14B就当原论文复现。
- 许可证、依赖、模型/权重使用与具体论文版本绑定：执行前实地核验 LICENSE 是否存在及允许派生；这里只保留代码链接，**不复制作者实现**。

## 2. 原算子与执行公式（统一行向量）

Wan原生 `u∈R^{N×d}`, `W1∈R^{m×d}`, `W2∈R^{d×m}`, `d=5120,m=13824,N=31668`。`z=uW1ᵀ+b1`；`h=GELU_tanh(z)`；`y=hW2ᵀ+b2`。每层按 expert 绑定2个低秩校正矩阵 `B1∈R^{m×d}`, `B2∈R^{d×m}`，实存两个因子。

锚点：`u_s,z_s,h_s,y_s` 全原生计算并缓存，返回 `y_s`。

命中：
`ẑ_t=z_s+(u_t-u_s)B1ᵀ`，
`ĥ_t=GELU_tanh(ẑ_t)`，
`ŷ_t=y_s+(ĥ_t-h_s)B2ᵀ`。

两级校正与中间GELU在 ICC 中是核心；将其直接合并成 G19 的固定 `A` 就不是 ICC。后续仍由Wan当前step的 `gate` 执行一次乘法加残差；此刻不能读取未来输入，不能把当前gate折入固定校正矩阵。

### 可验证的代数缓存压缩（应单独标状态）

定义 `q1=z_s-u_sB1ᵀ`、`q2=y_s-h_sB2ᵀ`，则实数下:
`ẑ_t=u_tB1ᵀ+q1`，
`ŷ_t=ĥ_tB2ᵀ+q2`。

只保存 `q1(N×m),q2(N×d)`；相比原四缓存减少一半。**这是FFNDS可行性变体，不可默认算作作者原码逐位等价**。在 BF16 两个大数相消、addmm bias、GELU舍入、权重因子乘法顺序下结果可能改变。先验证原四缓存 reference，然后对同态输入做逐层/逐token和全视频数值对照；引入 FP32 cache 或精度改动分别定义新臂，不与作者默认混淆。

## 3. 参数分解与扫描

候选：
- ICC-SVD：对FFN W1/W2分别做标准rank-r因子分解。
- ICC-CD-SVD：依据 *真实 high:0→1 / low:2→3 pre-W1 原生增量* 构建 channel-aware delta_mag 统计。要核作者 `func_decomp.py` 中左右权重、零值平滑/截断、奇异值重构和参数因子顺序。
- ICC-CA-SVD：若资源允许，以绝对激活 mag 对应权重重构；这是第三臂，不替代 CD-SVD。

`r∈{64,128,256,512}` 为独立控制变量。初轮离线回放可全扫；真实视频只让可行的2–3组进入。校准来源与G19原CAL一致，但必须保留 source/seed/配对坐标/专家/原生activation的SHA，避免同视频重采当前评测数据。不得在评估视频上反向定rank并称blind。

**预期主MAC算术（不含GELU/cache/offload）：**
- Dense 每 FFN: `2Ndm`；
- ICC hit rank-r (两层皆rank-r): `2Nr(d+m)`；
- 四步每专家先anchor再hit：`ρ=0.5+r(d+m)/(2dm)`。
- r=64/128/256/512时 FFN主MAC节省约49.14/48.29/46.57/43.15%。G19现值31.48%。
- 原作者核属于低秩两级增量，**计算量较低并不证明系统更快**。

## 4. 必须预审的缓存与内存

同样只保留当前专家40层、无CFG、BF16、N31668：
- ICC 原四状态 `40×N×2(d+m)×2bytes ≈ 89.39GiB`，单卡 A100 无法全部 GPU 常驻。
- 两残差 `40×N×(d+m)×2bytes ≈ 44.70GiB`，另有模型offload buffer、低秩因子、当前激活、GELU/转置临时区；**不等于能安全运行**。
- G19 `C` ≈12.08GiB，另 `80A`3.91GiB（二专家权重加载/驻留边界需另审）。
- 若原四状态采用 CPU/pinned-memory 或逐层预取：给出全量物理数据流、时间/峰值，预取与计算 overlap 的关键路径；不能将“完全免费常驻”当基线。

先以原真实 `N×d`、`N×m` 和一个层的按BF16版做 preflight，按实际 reserved/allocated 决定是否能执行 40 层全视频。OOM 时保留 `BUILD/MEMORY_BLOCKED` 客观记录，可降为单层同态诊断或正式审查确定的主机流水实现；**不任意删减 ICC 原机制**。

## 5. 与 G19 对齐的代码接口

建议新模块形状（不是已创建文件或已运行代码）：

`ICCFFNAdapter(expert,layer,B1_factors,B2_factors,cache_mode,device)`
- `begin_video()`：使所有缓存失效；
- `begin_step(0,high)`：该专家/层 `anchor`；`begin_step(1,high)`：`hit`；
- `begin_step(2,low)`：清high，low `anchor`；`begin_step(3,low)`：`hit`；`end_video()`释放；
- `forward_ffn(u, native_ffn, dtype=BF16) -> y`：只返回pre-gate输出；**不在ICC adapter内部二次乘gate**；
- `record_counters()`：full/hit 80/80，分层/专家，初态、cache解码/传输和各GEMM。

对于LightX2V，hook真实执行边界为 `phase.ffn_0.apply`、`GELU_tanh`、`phase.ffn_2.apply`，对应 `DirectFFN.infer_ffn` 的原生pre-W1输入；keep 原始`norm2`、shift/scale、gate、两专家调度、当前 BF16 写回。第0/2锚点必须在当前学生状态精确计算；即使先前hit偏离Dense，也不可把新的真实状态偷换成Dense教师输入。

同态仅用于数值检查的Dense/neutral应有输出同SHA；正式学生必须自由运行四步+40层。全部 `N=31668` token保持原位（无需Chipmunk的补丁padding）。

## 6. 具体判别与失败处置

| 问题 | 必备对照 | 停止/继续 |
|---|---|---|
| 低rank是否能复原FFN增量？ | rank64..512; SVD vs CD-SVD（CA-SVD可选），同增量、同gate | 误差p99、source差异和BF16对照，局部能改善但不保证视频 |
| 中间GELU能否帮助保真？ | ICC与G19在相同当前输入/相近实测FFN延迟比较 | 不能用只看MAC的rank作为获胜依据 |
| 内存是否抵消算术优势？ | 原缓存4态→代数2态→分层offload，分别计各值 | 如无可执行且忠实的缓存策略，报告FUNCTIONAL_PORT，不做伪速度 |
| 数值重排与作者语义一致？ | fp32实数式/原BF16原执行/变体BF16逐项对照 | 原路径存在不合期望大偏差则修正S1 |
| 可否实际进入论文主表？ | 完整视频7源×2seed + source-cluster配对，实际wall和峰值 | 未通过评审和RUN/EXIT则NOT_RUN |

新 scientific CPU/GPU 运行先按 [独立评审](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md) 冻结 scope 和资源；本文件没有审批。首个可执行范围宜只做代码+单层预检，然后只在可行 rank 上进入完整视频。
