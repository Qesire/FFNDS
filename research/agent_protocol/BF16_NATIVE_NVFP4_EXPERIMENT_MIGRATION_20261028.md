# FFNDS · Agent 执行规则：原始 BF16 研究优先，原生 NVFP4 部署暂缓

2026-10-28 · **CURRENT POLICY / 用户研究决定**。本文件明确后续研究和实验的数值环境选择；不改变 [M-A/M-B/M-C方法主体](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，不修改已有 HiF4/CAL24/A0/B-H 运行收据和冻结模型范围，不自动批准GPU/视频实验。各Agent从 [任务台账](CURRENT_TASK_LEDGER_20261028.json) 恢复实际进度。

**2026-10-28 最新用户纠正（优先于下文原计划）**：用户**没有本地 RTX 5090**，原生NVFP4部署的设备和环境准备复杂，明确要求**暂不着急推进**。现行执行优先项为**原始BF16机制、合法同条件对照、完整视频质量与实际BF16省算**，已有可用A100和研究资产继续发挥作用。SM120原生NVFP4 R0–R4 五项任务统一为 **DEFERRED / 暂不排入READY队列**；无需为其进行设备搜索、环境搭建、额外下载、checkpoint转换、CUDA编译、专项GPU占用或模型运行。未来用户明确重新启动且真实可用SM120资源与匹配checkpoint到位时，才从R0恢复。历史SM120原生primitive在 **RTX PRO 6000 Blackwell** 跑过，**不是5090**，不代表当前能够访问该机器。**本规则不修改过去冻结RUN/报告或已绑定证据。**

## 一、必须使用的三条路径身份

| 路径标签 | 优先级 | 角色 |
| --- | --- | --- |
| **BF16_ORIGINAL** | P0 | 原始BF16模型 Dense / FFN Sparse / 学生：研究函数、运动、算术和真实BF16省算，主要高精度科学基准 |
| **NVFP4_NATIVE_W4A4** | **DEFERRED（未来部署目标）** | 当有明确可访问SM120且用户重新启用部署线时，复用成熟LightX2V/CUTLASS原生NVFP4 W4A4；本次没有本地5090，也不启动配套实验 |
| **HIF4_FAKE_QDQ_DIAGNOSTIC** | P2/按需 | 维持冻结的HiF4 S1P2 signed、64/8/4 scale联动、数值敏感性、历史结果归因和旧方法重放；不以BF16 F.linear推定native FP4提速 |
| **NVFP4_DECODED_BF16_GEMM** | 控制组 | *同一份NVFP4量化checkpoint* 的解码BF16矩阵，用来分离数值量化与kernel；绝不写成BF16_ORIGINAL |
| **NVFP4_W4A16_FALLBACK** | 兼容性诊断 | packed weight + BF16 activation，Marlin或其它权重专用kernel；不能计入 native W4A4 speedup |

**BF16_ORIGINAL 的权重必须是原始 BF16 checkpoint**。旧仓库离线 HiF4 PTQ 的 BF16 weight_qdq 是**被量化后再解码**的BF16，不能通过关掉activation QDQ即当作原始 BF16。

**未来可选设备事实（非当前资源）**：RTX 5090 的 Compute Capability 为12.0（SM120）、32GB显存，硬件上可运行原生NVFP4；目前**用户没有这张本地GPU**。NVIDIA CUTLASS 79a要求对应SM120/CUDA工具链。FFNDS历史成功的两组小shape packed FP4 primitive实际运行于**RTX PRO 6000 Blackwell**，仅证明当时的原生算子数值能力，不证明当前Wan W4A4已跑通或那台机器现在可用。W4A16/Marlin与原生W4A4分别记账。

## 二、原生NVFP4的已归档后续技术路线（现已全部DEFERRED）

以下R0–R4只用于日后复用已有设计，**不是当前Agent行动项**。禁止根据文中的命令、依赖或技术可行性自动启动环境/模型工作。当前 P0 是原始 BF16 与功能实验，R3仅是 BF16 科学线的技术背景，不需要以R0为依赖。

**R0：设备/环境无模型预检（DEFERRED；未来重新启用时才做）**

Agent先收集：GPU名称、VRAM、CUDA compute capability、driver、torch版本/torch CUDA版本、nvcc版本、CMAKE_CUDA_ARCHITECTURES、ninja、FlashInfer/CUTLASS版本与实际后端。建议命令：

- nvidia-smi
- python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
- nvcc --version
- python -m pip show ninja flashinfer-python torch

期望 5090 为 (12,0)；工具链对 CUTLASS SM120 编译至少 CUDA12.8。若系统为Windows、CUDA环境复杂或FlashInfer尚不匹配，优先使用隔离Linux/WSL2或成熟LightX2V可用镜像，**先完成一个原生no-op而非重建全部工程**。上述命令仅供未来明确复启、且实际获准访问SM120主机时使用；用户当前没有本地5090，无需执行。GitHub读取不能证明任何当前GPU可用性。

**R1：最小SM120 NVFP4原生W4A4基元（依赖R0）**

复用已检过的 pinned CUTLASS 79a或匹配的成熟backend。真实 packed A/W 两份scale tensor、输出与FP32/BF16参考吻合、记录GPU arch及native kernel签名和数值误差，证明使用原生block-scaled MMA。CUTLASS在架构不匹配时的示例可能返回exit0 no-op，**必须有非零的实际GPU调用及结果对比**。若安装vLLM仍出现Marlin、W4A16或软件回退，应报告精确子层、格式和kernel，不据此淘汰RTX5090。

**R2：成熟LightX2V Wan2.2-I2V-4step-NVFP4 Dense（需精确新scope）**

优先采用 [LightX2V NVFP4 I2V四步脚本](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/extreme/run_wan22_moe_i2v_extreme.sh) 与 [官方NVFP4配置](https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) 及**匹配的high/low真实checkpoint**。先确认是否为同一量化感知蒸馏模型：其公开preset为 timesteps 1000/750/500/250、dynamic_sparse_attn/sage2、cross sage_attn2、81帧、480×832；旧FFNDS HiF4先验为1000/937.5/833.333313/625。配置不是可以直接混用的因果对照。先用官方支持的 checkpoint/时间表/attention 跑native Dense FFN no-op，确认真实FP4 kernel而非weight-only fallback。显存优先保留T5 offload；OOM时可依据官方配置逐项调整offload/分辨率/图像编码器，但每个改变生成新的正式运行身份与质量比较，记录吞吐、VRAM、offload成本，不偷偷混入实验组。

**R3：BF16方法、M-B/B-H/A0/B-F（当前P0；不依赖R0-R2）**

继续利用CAL24、B-H Direct65 H128、A0全160学生与完成的**同head A0/B-H匹配对照、B-F固定基底pilot**。最新原始BF16工作已有两来源Dense视频/八轴评分、低39跨状态四角分析及fixed-norm原生转移；下一步优先做**跨来源/跨深度的范数补全机制、Calibration V2分层、可执行低成本函数、完整自由视频运动质量和真实BF16系统wall**。已暴露开发源与新独立来源分别统计，任何新GPU作业依照ACTIVE登记。不因NVFP4暂缓而停顿BF16功能与方法实验。

**R4：原生NVFP4 FFN候选与完整视频（依赖R2真实Dense通过）**

以**同一NVFP4 checkpoint和量化recipe**成对比较 dense FFN / sparse or compact FFN。W1/W2权重离线静态量化，Q1/A1与Q2/A2在线动态量化；NVFP4元素signed E2M1、16-element E4M3局部A/W scales、FP32全局scale；动态A1 scale必须赶在pre-W1决策前实际可用，A2只能影响真实W1/GELU后剩余W2动作。记录动态激活quantization、scale生成/读取、layout/swizzle、路由和物理kernel成本。同源/同seed/同调度/同attention验证全4step 160 FFN、连续运动、subject/background、camera/语义与质量尾例及整端wall。

## 三、物理与科学对照的强制身份

1. **同格式原生Dense→候选**是测量FFN增量收益的主要比较；不是“换checkpoint+换schedule+换attention+换稀疏方法”的混杂比较。
2. **BF16 original vs NVFP4 checkpoint**可以作为跨格式质量Pareto，但若蒸馏数据、四步时刻与attention不同，不能把差别归因为低位量化。
3. **NVFP4 W4A4** 与 W4A16权重专用模式分开。静态weight side / 动态activation side及实际量化kernel的scale元信息完整登记；别因看到checkpoint文件扩展名 nvfp4 就声称原生W4A4。
4. **HiF4 Fake-QDQ** 保留其数值科学功能和历史证据，不再是新低位FFN速度评测的默认后端。
5. **设备与时序优先**：当前没有本地5090且用户明确暂缓原生FP4部署；SM120原生目标是未来可选选项，现阶段不开展专项安装或GPU调度。历史RTX PRO 6000 Blackwell primitive只提供基础能力证明。

## 四、完整验收证据与运行权限

对 R1-R4 的新运行登记 NUMERIC_PATH、master_model/高低噪声checkpoint+SHA、scheduler/step、attention backend、quantizer/weight scale+activation dynamic scale、scale layout、GPU CC和显存、M/N/K、actual kernel name、code/protocol SHA、quantization+GEMM+overhead latency、RUN/EXIT、source image-cluster/seed、整端wall、运动及语义质量。Native只有实际kernel证据；VIDEO/SYS只有真实free rollout及测量证据。

具体原生GPU/视频作业需在 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 登记精确scope/用户授权、源与预算，历史冻结配置不改。全局 GPU 不设预先限额，但各会话资源指令与实际可调用量仍须核对。当前仓库编辑仅同步行动规则；**本次不启动NVFP4相关工作，也不声称存在可访问的本地5090**。历史实验按既有source hash原样保存。

## 五、源级起点

- [LightX2V Wan2.2 I2V NVFP4 preset](https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) 与 [Wan22脚本总览](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README.md)。
- [NVIDIA CUTLASS SM120 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。
- [NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)。
- [当前W1/W2权重与A1/A2激活时点审计](../20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)。
- [FFNDS已有SM120 NVFP4 primitive](../20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)。
