# FFNDS · Agent 执行规则：BF16 机制主线 + RTX 5090 原生 NVFP4 部署轨

2026-10-28 · **CURRENT POLICY / 用户研究决定**。本文件明确后续研究和实验的数值环境选择；不改变 [M-A/M-B/M-C方法主体](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，不修改已有 HiF4/CAL24/A0/B-H 运行收据和冻结模型范围，不自动批准GPU/视频实验。各Agent从 [任务台账](CURRENT_TASK_LEDGER_20261028.json) 恢复实际进度。

**本会话最新用户覆盖**：用户确认“没有，就用a100”。当前执行设备为获准的空闲A100，同时最多2卡，推进BF16_ORIGINAL；下面RTX5090/SM120能力与部署队列保留为将来硬件实际可用时的技术路线，不表示本会话已有该设备，也不阻止BF16科学实验继续。

## 一、必须使用的三条路径身份

| 路径标签 | 优先级 | 角色 |
| --- | --- | --- |
| **BF16_ORIGINAL** | P0 | 原始BF16模型 Dense / FFN Sparse / 学生：研究函数、运动、算术和真实BF16省算，主要高精度科学基准 |
| **NVFP4_NATIVE_W4A4** | P0 | 单张已有 **GeForce RTX 5090 SM120 32GB** 优先；复用成熟 LightX2V/CUTLASS NVFP4 W4A4 量化及计算，真实低位部署与加速主证据 |
| **HIF4_FAKE_QDQ_DIAGNOSTIC** | P2/按需 | 维持冻结的HiF4 S1P2 signed、64/8/4 scale联动、数值敏感性、历史结果归因和旧方法重放；不以BF16 F.linear推定native FP4提速 |
| **NVFP4_DECODED_BF16_GEMM** | 控制组 | *同一份NVFP4量化checkpoint* 的解码BF16矩阵，用来分离数值量化与kernel；绝不写成BF16_ORIGINAL |
| **NVFP4_W4A16_FALLBACK** | 兼容性诊断 | packed weight + BF16 activation，Marlin或其它权重专用kernel；不能计入 native W4A4 speedup |

**BF16_ORIGINAL 的权重必须是原始 BF16 checkpoint**。旧仓库离线 HiF4 PTQ 的 BF16 weight_qdq 是**被量化后再解码**的BF16，不能通过关掉activation QDQ即当作原始 BF16。

**RTX 5090的确定性能力**：Compute Capability 12.0，真实SM120 Tensor Core NVFP4支持，32GB显存。NVIDIA CUTLASS 79a 官方 SM120 packed E2M1 GEMM 要求 CUDA Toolkit >=12.8。原来FFNDS已完成SM120两组小矩阵packed FP4 primitive，但那是随机数据、非实际Wan在线激活量化和完整FFN，下一步仍必须证实模型/后端兼容和真实执行收益。vLLM某些 ModelOpt mixed W4A16层的Marlin回退是特定backend/recipe选择，不是GPU缺少原生FP4。

## 二、RTX 5090 的最短可执行工作队列

**R0：设备/环境无模型预检（READY_STATIC，本机只读）**

Agent先收集：GPU名称、VRAM、CUDA compute capability、driver、torch版本/torch CUDA版本、nvcc版本、CMAKE_CUDA_ARCHITECTURES、ninja、FlashInfer/CUTLASS版本与实际后端。建议命令：

- nvidia-smi
- python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
- nvcc --version
- python -m pip show ninja flashinfer-python torch

期望 5090 为 (12,0)；工具链对 CUTLASS SM120 编译至少 CUDA12.8。若系统为Windows、CUDA环境复杂或FlashInfer尚不匹配，优先使用隔离Linux/WSL2或成熟LightX2V可用镜像，**先完成一个原生no-op而非重建全部工程**。以上代码/命令需要在用户实际5090本机运行；GitHub只读无法检测GPU。

**R1：最小SM120 NVFP4原生W4A4基元（依赖R0）**

复用已检过的 pinned CUTLASS 79a或匹配的成熟backend。真实 packed A/W 两份scale tensor、输出与FP32/BF16参考吻合、记录GPU arch及native kernel签名和数值误差，证明使用原生block-scaled MMA。CUTLASS在架构不匹配时的示例可能返回exit0 no-op，**必须有非零的实际GPU调用及结果对比**。若安装vLLM仍出现Marlin、W4A16或软件回退，应报告精确子层、格式和kernel，不据此淘汰RTX5090。

**R2：成熟LightX2V Wan2.2-I2V-4step-NVFP4 Dense（需精确新scope）**

优先采用 [LightX2V NVFP4 I2V四步脚本](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/extreme/run_wan22_moe_i2v_extreme.sh) 与 [官方NVFP4配置](https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) 及**匹配的high/low真实checkpoint**。先确认是否为同一量化感知蒸馏模型：其公开preset为 timesteps 1000/750/500/250、dynamic_sparse_attn/sage2、cross sage_attn2、81帧、480×832；旧FFNDS HiF4先验为1000/937.5/833.333313/625。配置不是可以直接混用的因果对照。先用官方支持的 checkpoint/时间表/attention 跑native Dense FFN no-op，确认真实FP4 kernel而非weight-only fallback。显存优先保留T5 offload；OOM时可依据官方配置逐项调整offload/分辨率/图像编码器，但每个改变生成新的正式运行身份与质量比较，记录吞吐、VRAM、offload成本，不偷偷混入实验组。

**R3：BF16方法、M-B/B-H/A0与B-F（与R0-R2并行，不等待NVFP4部署）**

沿已完成的 CAL24 source-paired 标签、B-H Direct65 H128和A0全160学生资产，先把**同来源、同head、同数值路径**的比较做完，重建原BF16 Dense vs 学生的真实功能和成本。在A100或现有可用CUDA上完成合法BF16实验。B-F以及I/F/C新机制继续探索。

**R4：原生NVFP4 FFN候选与完整视频（依赖R2真实Dense通过）**

以**同一NVFP4 checkpoint和量化recipe**成对比较 dense FFN / sparse or compact FFN。W1/W2权重离线静态量化，Q1/A1与Q2/A2在线动态量化；NVFP4元素signed E2M1、16-element E4M3局部A/W scales、FP32全局scale；动态A1 scale必须赶在pre-W1决策前实际可用，A2只能影响真实W1/GELU后剩余W2动作。记录动态激活quantization、scale生成/读取、layout/swizzle、路由和物理kernel成本。同源/同seed/同调度/同attention验证全4step 160 FFN、连续运动、subject/background、camera/语义与质量尾例及整端wall。

## 三、物理与科学对照的强制身份

1. **同格式原生Dense→候选**是测量FFN增量收益的主要比较；不是“换checkpoint+换schedule+换attention+换稀疏方法”的混杂比较。
2. **BF16 original vs NVFP4 checkpoint**可以作为跨格式质量Pareto，但若蒸馏数据、四步时刻与attention不同，不能把差别归因为低位量化。
3. **NVFP4 W4A4** 与 W4A16权重专用模式分开。静态weight side / 动态activation side及实际量化kernel的scale元信息完整登记；别因看到checkpoint文件扩展名 nvfp4 就声称原生W4A4。
4. **HiF4 Fake-QDQ** 保留其数值科学功能和历史证据，不再是新低位FFN速度评测的默认后端。
5. **真实可用设备优先**：5090可做SM120主实验；32GB VRAM和编译兼容为实际工程约束，不预判失败或宣称已成功完整I2V。已有SM120 primitive只提供基础能力证明。

## 四、完整验收证据与运行权限

对 R1-R4 的新运行登记 NUMERIC_PATH、master_model/高低噪声checkpoint+SHA、scheduler/step、attention backend、quantizer/weight scale+activation dynamic scale、scale layout、GPU CC和显存、M/N/K、actual kernel name、code/protocol SHA、quantization+GEMM+overhead latency、RUN/EXIT、source image-cluster/seed、整端wall、运动及语义质量。Native只有实际kernel证据；VIDEO/SYS只有真实free rollout及测量证据。

具体原生GPU/视频作业需在 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 登记精确scope/用户授权、源与预算，历史冻结配置不改。全局 GPU 不设预先限额，但各会话资源指令与实际可调用量仍须核对。当前仓库编辑只让Agent下一次执行这些规则，不声称已远程启动用户的5090。

## 五、源级起点

- [LightX2V Wan2.2 I2V NVFP4 preset](https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) 与 [Wan22脚本总览](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README.md)。
- [NVIDIA CUTLASS SM120 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。
- [NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)。
- [当前W1/W2权重与A1/A2激活时点审计](../20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)。
- [FFNDS已有SM120 NVFP4 primitive](../20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)。
