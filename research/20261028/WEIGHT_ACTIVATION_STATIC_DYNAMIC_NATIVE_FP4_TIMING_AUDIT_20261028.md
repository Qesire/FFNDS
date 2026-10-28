# FFNDS · 权重静态量化、激活动态量化与原生FP4元信息时点审计

2026-10-28 · SRC / LIT / ALG。此处完成真实实现与成熟后端的时序核查，不代表新的FFNDS Native模型或视频运行。统一机器合同：[weight/activation timing contract](../agent_protocol/WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)。原先的[Q-Meta时点推导](QMETA_OBSERVATION_TIMING_AND_EFFICIENT_USE_20261028.md)和[HiF4 signed存储](HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)保留为具体研究证据。

## 1. 回顾之前遗漏的轴：operand role 与 quantization schedule 必须独立于格式

以前QCFB把Q1/Q2元信息笼统称为scale/code，侧重Q1在W1之前、Q2在W1和GELU之后，**尚未把W权重的offline静态量化、A激活的当前调用动态量化及其各自持久载荷正式拆开**。因此可能错误地将静态W尺度解释为source-conditioned在线信号，或把native FP4当成没有尺度、或者默认尺度对router免费。补齐以下信息类型：

| 操作数 | 量化发生时点 | 首次可用于计算/路由 | 当前Wan实际载荷 | 角色 |
| --- | --- | --- | --- | --- |
| W1 weight [13824,5120] | PTQ构建/模型加载前的**静态**量化 | 离线就可做weight-profile及预编译 | cache里是 BF16 weight_qdq，不是packed S1P2 | 固定权重方向/静态候选空间 |
| A1/Q1 activation [M,5120] | 这一次真实FFN的AdaLN/Smooth1后**动态**量化 | Q1生产者完成或内核内可合法共享后；W1之前 | 当前kernel只返回带符号BF16 QDQ，code/scale在内核临时 | 当前x/step/source条件下W1/替代函数路由 |
| W2 weight [5120,13824] | PTQ/模型加载前**静态**量化 | 离线就可准备216个64组贡献权重/预测器参数 | BF16 weight_qdq | 固定W2方向、布局与功能投影 |
| A2/Q2 activation [M,13824] | 当前W1+GELU+Smooth2后的**动态**量化 | 当前Q2生产者完成或尺度归约之后的合法内核分支；W2之前 | 当前kernel只返回带符号BF16 QDQ | W2剩余K-panel决策或以后step，不用于本次pre-W1 |

其中M≈31668，仅指一次真实模型状态的FFN行数，非独立训练来源。

### 锁定源码证明

[ptq/algorithms/rtn.py](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/algorithms/rtn.py) 的 quantize_rtn(weight) 返回保存的 QuantizedWeightState.dequantized_weight。

[ptq/runtime/hifx4_linear.py](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/runtime/hifx4_linear.py) 构造时把 weight_qdq 冻结在 self.weight；forward 的输入在Smooth1后经 quantize_activation，随后 F.linear(work,self.weight,bias)。

**真正的LightX2V蒸馏适配** [ptq/adapters/lightx2v_wan_distill.py](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill.py) ：_weight（约L626–649）从缓存载入静态 weight_qdq，apply（约L654–692）对当前x应用 smooth_scale、在L674调用 native_hif4_activation_qdq、最后F.linear。重量参数不是每个step重新求scale；Q1/Q2的 activation CUDA量化随当前真实状态重算。

源树的 HIFX4TensorMetadata/Python signed int8 codes 可供**研究、构建**，不能断言原runtime缓存已持久保存所有原始packed W码或量化时得到的scale。也不能把对原 weight_qdq 再次编码得到的元信息当作原PTQ第一次精确的码本。

## 2. 原生FP4：既有metadata，且权重/激活各有一份

| Format与算子 | W静态量化及可存元信息 | A激活动态量化及可存元信息 | 对合法决策的改变 |
| --- | --- | --- | --- |
| **HiF4当前fake-QDQ** | 原权重离线QDQ BF16，紧凑4bit不持久驻留在此runtime | Q1/Q2当前调用产生HiF4层级幅值scale，写出BF16量化值 | 任何观察码/scale需要额外导出或从BF16统计；原BF16 GEMM而非FP4 TC |
| **原生MXFP4 W4A4** | offline packed signed E2M1 W + 每K32元素E8M0 8-bit scale | online当前A进行K32幅值归约、scale E8M0与FP4打包；可能跟GEMM融合 | 固定W scale可预处理，A scale在每次调用临时产生；两者都被MMA消费 |
| **原生NVFP4 W4A4** | offline packed signed E2M1 W + 每16元素FP8 E4M3 scale + FP32全局scale，权重可有16×16的2D scale | online每16元素E4M3 block scale + FP4 codes；global FP32可按recipe是calibration静态或当前动态 | 局部A scale仍依赖当前输入；静态global并不使激活全部静态 |
| **NVFP4 W4A16** | offline W codes+scales | BF16/FP16激活，无激活FP4量化 | 根本不存在本次A的FP4 block scale；仍可按普通BF16特征决定动作 |

OCP MXFP4 Table1给出K32、4-bit E2M1和8-bit E8M0 scale，理想32元素每块4.25 bit/element（不计实际对齐和swizzle）。NVFP4则是K16 E4M3 local +全局FP32；其本地4bit值的符号是原本就有的。**元信息同时属于两个operands**，kernel物理上消耗W scale和A scale，而不是仅乘一个公用scale。<参见末节官方出处>

### 成熟源级实现如何分开静态和动态

- **torchao MXDynamicActivationMXWeightConfig** 使用MXFP4时是W4A4配置，指定 activation_dtype/weight_dtype 都为 float4_e2m1fn_x2 与block_size=32；模型quantize_时将权重转换并存储，Linear.forward的activation按当前输入动态quant。对应 torchao inference_workflow.py（链接在末节）。其API定义不保证当前Wan已运行这种kernel。
- **torchao NVFP4DynamicActivationNVFP4WeightConfig** 特别有两个global-scale模式：动态模式可在当前调用中计算activation全局amax；通过prepare/convert与校准可预存静态activation global-scale。但是**activation local block-16 scale仍在在线求值**，weight全局scale和quantized weight持久保存。
- **vLLM ModelOpt NVFP4** 直接把 kNvfp4Static（weight）与 kNvfp4Dynamic（activation）作为两种量化scheme。其某种模型的activation global input_scale是checkpoint静态参数，per-group activation-scale在runtime kernel内部动态生成。W4A16_NVFP4从此流程移除activation FP4，不能说它也有A FP4 scales。
- **CUTLASS Blackwell Block-scaled MMA** 显式是 A×SFA 与 B×SFB 两套 scale 操作数，scale另有专门内存布局。CUTLASS ScaledOperand 的 Swizzle32x4x4 模式是Blackwell支持的缩放张量布局，量化producer须输出该形态，不能把自然[M,K/B]视图直接当免费Tensor Core可消费的HBM形式。

**关键澄清：有metadata ≠ Python API能免费给router读取。** 已生成的packed A和scale可以是独立tensor、仅在某个kernel内部可见、swizzled之后供Tensor Core取用，或被融合量化→GEMM隐藏。想提取row-tile决定W1支持，就要评估是否有合法的producer→跨CTA归约→shared路由→kernel dispatch窗口，以及sidecar/scan/同步对关键路径的影响。

## 3. 重新定义动态决策的信息过滤

把固定权重状态记为 Wq=(codeW1,scaleW1,codeW2,scaleW2)，当前输入为 x，合法history为H，当前A1在线编码为Z1(x)，当前A2在线编码为Z2(x)。

- **OFFLINE：** σ(Wq,静态recipe/预编译weight布局)；只能构造固定或专家/step条件结构，不含当前source动作信息。
- **PRE-Q1：** 加上真实x、step/expert及此前已存在的gate/H；可以用便宜普通BF16特征决定或预判。
- **POST-Q1 PRE-W1：** 再加本次实际产生、且及时暴露的Z1(x)，可以决定W1 whole-support/紧凑函数等；Q1量化工作已付费。
- **POST-W1 GELU PRE-Q2：** W1/GELU已付费，可研究新A2的预测或观察时机。
- **POST-Q2 PRE-W2：** 再加本次实际Z2(x)，可决定剩余W2或未来状态，不追溯省去W1/GELU。

研究价值不是“weight-scale单看能否预测动态”，而是 **static W 几何先验 × dynamic A 编码侧信道** 是否在同截止时间和同成本下显著改变可执行动作的条件风险差。一般表达为 risk_delta(a|Wq,Z1,H)；static-only行动、step-only、cheap BF16-feature、scale-only、ordered signed-code sketched 和 full-information oracle 为不同层次的竞争解释。

**格式改变算子的合法单位**：HiF4 Q2有64→8→4嵌套scale连带，MXFP4是K32，NVFP4 K16、W侧可2D scale。原HiF4整64 hidden group的保持原Q2语义可交换性**不能直接迁移成**NVFP4、MXFP4的相同mask closure。支持集、scale精度、group跨row耦合、原生tile布局和reference必须随format重新推导。

## 4. 可执行最小对照与创新自由度

第一阶段 SRC/保存数据审计：
1. 对每个W1/A1/W2/A2建立 operand kind、producer、量化时序、scale axis、全局scale来源、可读时点、当前缓存身份和硬件backend。
2. 利用已有CAL24 signed BF16 V1/V2和固定 W1/W2 实体形成**离线teacher**条件风险与分组方向，不将再编码的代码误称原native导出。
3. 测试 **W-static-only / step-only / cheap-A1-BF16 / Q1-scale / Q1-sign+position / W×A1 / Q2-scale+sign** 统一物理可执行行动集合下的排序、source-heldout regret与NetVOI；合法时间和获取成本分别冻结。

第二阶段真实native操作（需新scope）：
1. 当前代码A100 BF16 fake-QDQ与已有SM120小型随机packed NVFP4原生primitive已具不同证据身份。前者没有原生packed算子；后者不是同Wan分布的动态A4 producer。先挑真实可用Blackwell backend完整实现BF16→FP4 online Q1、静态W packed、scale layout、no-op native GEMM，记录到动作截止的真实成本。
2. 比较 **original native no-op、observer-only no-op、producer-fused summary、sidecar scale reader、额外BF16 scan、实算prepacked动作**。正收益必须包含producer、读码、跨CTA聚合、layout/swizzle、dispatch、内存与kernel真实wall。
3. 之后再入真实full-M 160 FFN/81帧独立视频、motion与semantic质量/系统Pareto。M-B/H128、A0全160当前证据仍只覆盖局部恢复，另设原生视频gate。

## 5. 原始规范 / 成熟源码链接

- OCP MXFP4官方§5.2 Table1 p.10：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- NVIDIA Transformer Engine NVFP4：https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
- torchao MXFP4/NVFP4：https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/inference_workflow.py
- torchao MXFP4 API：https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html
- vLLM ModelOpt NVFP4：https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/modelopt/
- CUTLASS原生blockscale：https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
- CUTLASS ScaledOperand布局：https://docs.nvidia.com/cutlass/latest/media/docs/operators/api_reference/arguments.html
