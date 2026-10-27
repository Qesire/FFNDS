# 逐token 2:4 activation：稀疏左操作数可表达，在线压缩是关键分歧

2026-10-27 · 独立 SRC/LIT/ALG 复核 · 无GPU、模型或实验。结论：对于 \(A\in\mathbb R^{M\times K}\) 每行沿K分组2:4，\(Y=A W^T\) 可将**原行方向的A直接作为稀疏左操作数**，不必将A转置后重新按列稀疏化。PyTorch 2.10源码提供该调用路径；原生M=31668不满足其稀疏张量行对齐，需补零到31680。是否实际运行、数值正确或产生净速度收益仍须同形A100实测。

## 1. 真实本地实现和方向

本workspace `.venv` 实际导入 `torch 2.10.0+cu128`、CUDA构建版本12.8；CPU只读查询 `torch.backends.cusparselt.is_available()` 为True，`version()`为701，`SparseSemiStructuredTensor._FORCE_CUTLASS=False`。这是编译/动态库能力和默认选择，**不是A100运行通过证明**；worker环境需另行核对。

- `torch/sparse/semi_structured.py:321–385` 的 `to_sparse_semi_structured` 选默认 `SparseSemiStructuredTensorCUSPARSELT.from_dense`；该路径调用 `torch._cslt_compress`。该版本默认分支仅按 `_FORCE_CUTLASS` 选择，不能根据类注释中“cuSPARSELt不可用时”自动推断有运行时fallback。
- `_semi_structured_ops.py:113–131` 的 `semi_sparse_mm` 在左侧是稀疏对象时调用 `A._mm(B)`；`semi_structured.py:645–690` 将 packed A 和 dense B交给 `torch._cslt_sparse_mm`。因此测试应是 `A24_sparse @ W.T`，输出是[M,N]。权重的逻辑转置 `W.T` 只是dense右操作数的布局，不改变A的每行沿K四元组。右侧是稀疏对象时才走转置等价分支。
- 对例如 `[1,1,0,0]` 每行独立满足K轴2:4，若将整幅A转置并要求每个新行仍2:4，某列可以来自任意多行而在转置后四元组含4非零。这一反例解释为何不应“先转置激活再压缩”并宣称等价；不排除其它明确定义的底层layout重写和其成本。
- `semi_structured.py:250–280,555–561`：默认cuSPARSELt BF16 sparse输入2D、CUDA、连续，两个维度分别是16的正整数倍；dense右侧BF16需同dtype。`M=31668`须补12行到31680，K=5120、13824满足16倍数。CUTLASS类BF16则要求稀疏维度分别32和64的倍数；31680、两种K也满足。`W.T`的stride、alignment、底层各GEMM形状和实际可调用性须测，不能从Python检查推断所有内部限制。
- 同源码 `prune_dense_static_sort` 是对4×4 tile维护行列双向稀疏的训练辅助，不等于RT-Lynx逐行top2。标准 `to_sparse_semi_structured` 接收已满足规则的稀疏矩阵，不能当成top2 pruning。CUTLASS conversion源码对于多于2个非零会选取子集而非保证top2；测试必须先验证每4组至多2非零和所选数值，尤其zero/tie。

本地文件SHA：`semi_structured.py` `adbd51da067c149fa7833383a72910a014bbb0f5e1f816ab0f76b09264a1c3dd`；`_semi_structured_ops.py` `6a614407b422b630cffff76ba6a686979104d62bb1bc2ec0cc060f1b354fddb3`；`_semi_structured_conversions.py` `1759d16ca5dbdcf0a5cef676eafbd4f1477af5e961b18a8fc8dd83a1c1c6728e`。上游[NVIDIA cuSPARSELt文档](https://docs.nvidia.com/cuda/cusparselt/index.html)列SM8.0及矩阵压缩和稀疏矩阵乘；[PyTorch semi-structured文档](https://docs.pytorch.org/docs/2.14/sparse.html)说明元数据公式。该在线文档为较新版本，shape判断以上面本地源码为准。

## 2. 在线物理成本

每次输入/每个不同step或block，激活数值与mask可能变化；预先压缩上次activation无法无条件复用。常规路径需付完整Q1/Q2（取决于插入点）、top2选择/生成零/必要norm、M padding/contiguous、压缩与metadata构建、SpMM、输出裁剪/布局/加bias；如带LoRA，再计原dense X读与两次小GEMM及相加。仅计预压缩SpMM可测kernel能力，不是在线加速。

BF16理论压缩为每原4元素存2个BF16（8原字节→4数据字节）及4bit metadata，即dense字节的56.25%；这是格式量不含workspace/allocator/多个临时拷贝。补到M31680后，K5120时dense 309.375 MiB、压缩values 154.6875 MiB、metadata约19.336 MiB；K13824时dense 835.313 MiB、values 417.656 MiB、metadata约52.207 MiB。实际cuSPARSELt padding/layout和临时workspace以运行时为准。权重不被稀疏化；SpMM内部即使更快，巨量在线读写/压缩或W转置实现可吞没差额。[RT-Lynx论文](https://arxiv.org/html/2605.26632)在图像DiT/H20上联合在线选择、范数补偿、LoRA和专用kernel；该调查未找到作者可核验的kernel源码，不能把其fusion或速度当作本地已有能力。

**竞争预测**：若预压缩SpMM能快、完整在线路径慢，障碍主要是准备/数据移动，值得研发融合选择+压缩；若两者都慢，形状、stride/硬件或库kernel是首要瓶颈，应先优化backend/比较更强dense，而非仅优化mask；若速度通过但模型误差或视频不通过，则转向norm/LoRA或选择性层，不能从执行收益推出质量。S2仅在真实Q2完成后对W2输入稀疏，省W2而不省W1，dense两投影同成本时理想主GEMM MAC只省25%；S1与S2同时需要重新执行真实W1→GELU→Q2，不能复用原dense Q2充当新状态。

## 3. 下一最小A100固定判别草案（OPEN，非运行授权）

冻结一张空闲A100、torch/CUDA/cuSPARSELt版本、输入构造种子及原生M/K/N：至少W2 `[31668,13824]×[13824,5120]`；预算允许再W1 `[31668,5120]×[5120,13824]`。先做小维度非对称矩阵，构造各行不同top2、少于两个非零、重复绝对值tie及全零/padded行；验证 \(\text{SpMM}(A_{24},W^T)\) 与 `A24 @ W.T` 的绝对/相对误差、bias、无输入转置的方向和输出前31668行，保留真实BF16 accumulation容差及nonfinite结果；`A24` vs 原dense A 的误差另列，不能将算法剪枝误差当作SpMM算子误差。对原shape比较三个计时边界：1 dense BF16 GEMM（相同W布局和bias）；2 预构造/预压缩A24的SpMM；3 原A→top2/可选norm→padding/压缩→SpMM→裁剪/bias，实际CUDA同步/事件、热身/多次及峰值内存与pack分开。必要时比较W转置view和一次性预排布并记录其额外存储；精确计数与时长及硬件锁在运行前登记。条件如发现2快但3慢，立即转向压缩融合/不同单位，避免冒称现有stack已获真实省时。

这个S2算子试验不运行Wan整模型、不提供视频质量；其输入可使用冻结数值测试以先判执行，后续再用获准真实Q2/gate及完整自由视频检验。此前原M16×K64所有tile占用的结论只限定whole64或普通dense tile策略，与本SpTC物理编码不冲突。

## 可审查的合成探针草案

新增 `tools/e0_activation24_spmm_probe.py`、`tests/test_e0_activation24_spmm_probe.py` 及 `config/e0_activation24_spmm_probe_20261027_v1.json`。草案默认未授权且asset binding为空；必须由协调者核对实际执行环境源码/库、冻结配置和ACTIVE后才可执行。仅synthetic BF16、每行连续4取幅值前2（稳定tie）；不加载Wan权重、视频或训练数据。两种实际shape各有一次dense、masked-dense和实际cuSPARSELt sparse数值比较；三个时延边界各2热身+5次测量：原dense、预pack SpMM、在线select→padding→compress→SpMM→crop。CUDA事件和同步后wall、每次GPU peak/基准分配均记录；默认不加norm。稀疏对象必须是当前PyTorch默认cuSPARSELt类型，稀疏对照为同mask的dense乘而非原dense。

预算总计固定随机张量初始化4次、top2 16次、compress 16次、dense GEMM 18次、sparse GEMM 30次，0模型/拟合/视频/官方打分；操作前扣账，单attempt失败即停。限单A100、2 CPU、300秒、输出≤32MiB。CPU测试覆盖沿K选择、tie、padding、预算拒超/失败扣账、授权范围；不会执行GPU。默认数值容忍为相对L2≤0.02，冻结前可由协调者结合BF16计算精度审阅并固定。

该原型使用通用稳定argsort生成mask，而非论文融合kernel；它测量实际通用软件栈端到端代价，若prepacked快而online慢，是优化准备/融合的动机，而非否定所有专用kernel。当前没有新增本地RT-Lynx作者kernel。结果不能冒充真实视频质量或模型内加速。

### Independent review of the synthetic probe

Source review covered tool SHA `20f41ec7f50d76dd380520b10a21b9c3835db21b477c6a68c27cd90233134982` and test SHA `b50ae418fcf83bdfa703df10f32985ce6910bce717ccf5567c2a55d14c16d3c4`. The selector applies stable absolute-value top two to each consecutive four entries on the activation K axis; M=31668 pads to 31680. For both shapes, the requested two warmups and five measured repetitions give 4 random initializations, 16 selections, 16 compressions, 18 dense and 30 sparse calls including reference checks. Each operation is charged before invocation, and failure retains the counts. The online interval includes selection, padding, compression, sparse multiplication and cropping; numerical comparison is against masked dense output.

The source verifies SHA of listed dynamic-library files but does not establish that those specific files are the libraries loaded by the process. If the result claims runtime dynamic-library identity, compare loaded mappings against the bound files; the file hashes alone support only a file-binding claim. This is the default PyTorch cuSPARSELt path, not a fused RT-Lynx implementation, so its measured latency cannot rule out all 2:4 implementations. This review ran no GPU operation.

Follow-up: the revised probe checks `/proc/self/maps` after the first compression and requires the mapped `libtorch_cuda` and `libcusparseLt` paths to equal the previously SHA-verified bound files, rejecting deleted or multiple mappings. This closes the library-identity finding above for the checked process.

## 真实A100首测后的算法/布局判别

**SYS观察**：synthetic RUN SHA256 `1a5831cf00f84bcb668a238489005027a1f74c526b6dab60b84f76fa4aec43a6`，状态COMPLETE、预定调用计数闭合、加载库身份和采样隔离通过；A100 80GB、torch2.10.0+cu128、cuSPARSELt 0.7.1。W1形状5次中位 dense/prepacked/online为19.400/29.190/148.299 ms；W2为18.910/27.528/345.787 ms。分别约1.50×/1.46× 的**预压缩SpMM时延相对dense**说明本次不能只把总损失归于在线selector；online使用通用stable argsort、未融合压缩，成本另更大。相对masked dense的稀疏输出L2约6.07e-5/1.22e-4；它只支持合成算子粗粒度一致性，BF16与统计形状不是视频质量。

**SRC默认行为**：本地`torch/sparse/semi_structured.py`设置 `_DEFAULT_ALG_ID=0`、`_FUSE_TRANSPOSE=False`；CUSPARSELT `from_dense` 把二者写到对象，`_mm`传给 `torch._cslt_sparse_mm`。本次代码给 `F.linear` 权重连续 `[N,K]`，稀疏调用传入 `W.T` 即非连续 `[K,N]` view。PyTorch 2.10 [官方C++实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp)按`dense_B.is_contiguous()`设置不同dense descriptor和B转置标志，`transpose_result`决定输出描述及layout，默认每次初始化plan并设CONFIG_ID=0；这提供合法的布局/算法竞争解释，不证明何者更快。

本地 `torch._C._cusparselt.mm_search` 暴露签名 `(compressed_A, dense_B, bias?, alpha?, out_dtype?, transpose_result) -> tuple[int,int,int,int]`，上游C++ search实现返回选定的alg、split-K与上限相关信息；具体tuple在运行前应核对绑定语义并验证可直接传给 `_cslt_sparse_mm`。`torch._cslt_sparse_mm_search` 暴露仅整数、上游已标 deprecated，不能自动保留搜索得到的split-K。`torch.backends.cusparselt.get_max_alg_id()` 在本地version701返回None（该辅助表只列旧版本），不表示没有其它算法。[NVIDIA官方函数文档](https://docs.nvidia.com/cuda/cusparselt/functions.html)说明search会执行多算法/可能调整split-K，是阻塞的调优工作；API调用1次不等于内部GEMM 0或1次。

**可冻结的最小下一API实验（OPEN，另经登记）**：保持原两个synthetic BF16形状、同seed/2:4 mask、同A100和 `M31680` pack，按W.T view 与 `W.T.contiguous()` 两种右输入、`transpose_result=False/True` 两种输出layout形成每shape四个设置。每设置先调用一次 `torch._C._cusparselt.mm_search(packed,B,None,None,None,transpose_result)` 并保留原四值、返回码、耗时/资源；再比较相同setting的显式默认 `(alg0,split1)` 与搜索选定配置。每配置先1次正确性、2次热身、5次计时：2 shape×4 setting×2 config×8 = **128次显式SpMM**，外加每shape dense 1正确性+2热身+5计时 = **16次dense GEMM**、2次top2、2次compress、4次随机初始化、**8次search API**。search内部会运行不透明数量kernel/GEMM，登记为`UNKNOWN_SEARCH_INTERNAL`、计总CUDA/wall/峰值并设统一硬wall/显存/资源退出限制；若无法从返回值可靠恢复split-K，则报告仅alg复用，不能称完整search配置，停止或将其保留为受限对照。预先规定异常/某layout不支持时记录并停止该setting，不再尝试未登记ID。

输出对照确保按其标志恢复数学 `[M,N]`、裁原31668行，与相同mask的dense值逐个比对；比较`view`和`contiguous`时，完整一次性W重排wall/内存另列及按预设复用次数摊销，不能把静态预排布隐含免费。`transpose_result=True`可能给转置/非连续视图，记录strides并分别量最终消费者所需contiguous化成本；只比较raw kernel是不完整链路。随机输入/输出、字节数、warmup、A100隔离与库SHA、event同步、每配置显存、failure都绑定。

**相反预测**：搜索或特定B/输出布局显著改善预pack但最终带布局转换仍慢，性能瓶颈主要为布局/库参数而非选择器；即使搜索+布局均不能逼近dense，优先考虑不同kernel/更强硬件基线；若预pack快但原在线实现慢，才直接量化选择+压缩fusion的收益空间。该试验不包括selector tuning和真实视频，也不能以一次默认慢否定A100其他2:4方案或RT-Lynx的H20专用融合。

**C2草案**：新增 `tools/e0_activation24_backend_layout_probe.py`、对应CPU测试及未授权draft config；不修改C1。PyTorch 2.10作者源码 [cusparselt.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/csrc/cuda/shared/cusparselt.cpp) 确认`mm_search` 四元素为alg ID、split-K、split-K mode、max alg ID；[cuSPARSELtOps.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp) 的 `_cslt_sparse_mm` 接受这三个执行参数，split1时不设置mode。源码接口语义已核对，实际布局和search是否可用仍需授权后的GPU验证。

草案以8次search API和显式128 SpMM、16 dense、2 mask/pack、4 seeded random为上限，各操作前扣账；search内部kernel计数UNKNOWN，记录同步CPU wall/CUDA event及峰值。每次SpMM同时保留raw event及转置/裁剪/contiguous后完整boundary event，无额外GEMM。稠密权重预转置contiguous一次构建单独计时/字节；search使用对应原layout、完整恢复split参数。使用冻结C1 selector/operand helper，首次compress核对实际库映射，HF磁盘/源码小快照/单attempt等沿用。信号超时不能保证阻塞的原生库调用及时返回，必须配独立进程外部300秒hard timeout并以资源回执核对；CPU测试不验证任何GPU结果。
