# SharQ → Wan2.2 I2V FFN：真实双 GEMM 与总成本的可证伪设计

2026-10-30 | **SRC/LIT，设计未评审、未构建、未执行。** 作者 [arXiv §3–4](https://arxiv.org/html/2606.26587) 和 [固定提交 3cc584f](https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11) 的8个重点源文件逐字节留存及 SHA、Git blob、CUTLASS 子模块身份见[源码清单](../../../literature/20261029/P0_SHARQ_WAN_FFN_SOURCE_MANIFEST_20261030.json)。该清单不是完整编译闭包。未下载大权重、读取原始 tensor 或运行 GPU。

## 方法与本仓的真实边界

作者不训练、不需要离线数据校准，**但在线必须读取当前完整激活**。输入 `X[M,K]`、权重 `W[N,K]`：每8个相邻元素分4对，对每对取 `max(abs(x0),abs(x1))`，选最大2对形成4:8 in-pairs mask；ties由固定循环严格 `>` 逻辑决定。将选中值按每32原元素计算FP8 E4M3 scale并转FP4 E2M1，产生压缩A与metadata；计算同一数值域的 `R = X - dequant(Q_sparse(mask*X))`，残差每16元素重新计算scale并FP4打包。残差同时恢复未选值与主路量化误差；再对残差量化会产生新误差。

作者共享一个 **group32** 权重FP4 payload；稀疏scale与重复映射到group16的稠密scale是两种布局，不是两份不同权重。线性输出是 `SparseGEMM(A_sp,W) + DenseGEMM(A_res,W)`，并按原顺序累加与加bias；两个GEMM均真实执行。对本仓 `(M,K,N)`：W1 `(31668,5120,13824)`，bias后GELU，再W2 `(31668,13824,5120)`，每次FFN共 **2 sparse + 2 dense GEMM、两次在线准备、一次 GELU 和各自bias**。W1前 Q1 完整依赖当前V1；Q2依赖同一臂当前W1+bias+GELU，不能提前用其他臂或Dense的h。W1/W2 K均是128的倍数，满足 `bindings.cpp` 的准备入口必要条件；M=31668的实际布局/尾部与CUTLASS `can_implement` 必须运行时判定，不能静态宣称成功。

`model/quantize.py` 对每次输入先算全局 `max(abs(X))/(448*6)`、下限1e-9，再除尺度并转BF16，才执行稀疏/残差准备；因此全局归约、除法、转换、临时分配和metadata都必须计时。W的全局scale、group32 payload和两scale布局属于一次性初始化，不得混成在线动作或隐去成本。本仓 [NVFP4原生FFN探针](../../../20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.md) 是固定 checkpoint global input scale加动态group16 A scale；其 source894 M256两专家资产可作首个已暴露开发源，但权重来自另一NVFP4 checkpoint，原M31668是重复256行的形状诊断。不能把它称为原BF16权重或真实完整M状态。作者group32 shared W与本仓已有group16 packed W不可直接相同数值配对，须显式decode/requant及记录成本/误差。

作者论文视频是 Wan2.2 **T2V** A14B、4×RTX5090；480P SharQ相对FP16 1.23×，加 SageAttention为1.58×，不是FFN-only速度。源码提供 Wan T2V形状benchmark，但本次核查未找到公开 Wan视频集成路径；I2V四步两个expert是独立迁移。作者 [CMake](https://github.com/actypedef/SharQ/blob/3cc584fe48e32e59c8b4e37323470dcdd606bb11/kernels/CMakeLists.txt) 默认 `sm120a`、另支持 `sm100a`，要求 CUDA≥12.8；RTX PRO6000已知SM120与该目标方向一致，实际编译、驱动、指令及tail兼容仍待审后验证。RMSNorm融合是作者可选 LLM 路径；Wan已有调制/归一化语义不同，不得无证明融入或拿其节省冒充FFN。

## 最小有限生产者草案与强对照

先单卡有限真实FFN operator，复用已闭合 source894 真实坐标和 high step0 / low step2 block0 M256 V1、匹配两套现有 checkpoint权重及bias；所有原身份和证据SHA逐字节绑定。M31668重复256行只测shape成本，单独标诊断，不推作真实全M动态分布。后续要接原BF16 master的同source、step、block及行坐标，未绑定前 `ORIGINAL_BF16_REFERENCE=NOT_BOUND`；已有 decoded-BF16同NVFP4 checkpoint是可运行的另一对照。真实原BF16参考需逐层固定权重、bias、数据和原始全局路径，不能将两种checkpoint混为同一来源。

每个expert、每个M采用相同V1输入配对、同 bias、GELU及精度：① 原BF16-master W1/W2（真正匹配资产到位后）及另记 decoded-BF16 checkpoint；② native dense group16 NVFP4 checkpoint/作者动态scale标准实现；③ **权重匹配dense group32**：使用SharQ共享payload和其dense scale view、dense A量化/GEMM（额外强对照，使分解相对相同W可辨）；④ SharQ sparse+residual。②与④都是NVFP4但权重group和scale recipe可能不同，须如实报告；③隔离权重格式差异。每臂中W2激活来自该臂W1，不共享真实在线h。另在相同X,W对W1/W2分别比稀疏主路与稠密补偿数值，加上独立dequant后FP32/BF16线性reference，检验metadata、weight scale swizzle、tail和补偿；bias严格只加一次。读取真实 packed 权重并重新生成group32时计一次初始化成本和读字节，任何缺source/环境/SM120编译闭合都在预检失败而不是模拟通过。

独立scope freeze前固定warmup、重复、两专家顺序、源坐标及容差；每实际调用前预扣并完成后记账：①一FFN两次BF16 GEMM；②③各两次dense FP4 GEMM和两次输入归约/quant；④两次在线mask+scale+packing、两次sparse GEMM、两次dense residual GEMM；四臂各一次GELU、两bias。按原调用路径如存在独立全局amax/divide/cast、缓冲zero和workspace/dispatch也分项记录，不把 fused prepare 宣称完全消除了这些。测从原始BF16输入到最终Y的完整FFN事件墙钟，另计初始化/转码/同步/host/内存峰值、部分算子和相同路径无观测器时延；物理指令数、compiler内部/autotune等无法观察时标UNKNOWN。M256检测值与M31668 repeated-row shape结果分别报告，无外推到完整学生或整视频。

**竞争预测。** H1：稀疏主路减少足够多的实耗，抵消第二GEMM、scale、packing和缓存流量，且修复残差后误差低于同W的单路dense FP4；H2：第二GEMM与动态分解在大M吞没加速，或group32权重损失抵消补偿精度，即使孤立 sparse GEMM快仍无法胜出。作者论文的 Llama prefill SharQ为dense NVFP4的111.2%、两GEMM占比48.4%是H2的明确反例；不是对本Wan的测量。两个预测以同格式两条dense对照、全FFN端到端加paired输入数值/内存联合裁决，不以孤立sparse GEMM速度取代。

后续需真实不同Agent独立审问此scope、作者修订/recheck、精确代码/依赖、输入与预算、ACTIVE授权和硬件身份；再扩 full-M真实输入、多来源与跨expert/四步160-site自由学生，配对完整81帧和主体、事件、运动、语义、时序/相机/背景质量，同时报告kernel与真实系统总成本。当前没有native SharQ运行或质量结论。

### 配对资产选择修订（静态，未生成bundle）

已有 source894 block0 V1包不携带匹配 gate。为使 `gate×Y` 真正配对，首个实际候选改为已closed whole-FFN boundary bridge 的 source911 Dense：step1/high block20 与 step2/low block20，各32条真实行，包内同状态 `V1`、`gate`、`rows`。原BF16 master和另一NVFP4 checkpoint各须对应该expert/block；不能把这两类权重当成相同checkpoint。M31668重复32行只作尺寸/吞吐诊断。`tools/e0_sharq_wan_ffn_asset_plan.py` 仅选择已存在metadata，不读取张量；实际四组W1/W2、两bias、gate及parent full SHA/读取账须由单独评审的CPU asset producer完成后，GPU producer才能获得真实bundle身份。
