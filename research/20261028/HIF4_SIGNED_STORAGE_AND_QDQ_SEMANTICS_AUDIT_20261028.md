# HiF4 带符号存储与 FFNDS 原生 QDQ：源码及格式语义审计

2026-10-28 · **LIT + SRC + ALG；无新增 NATIVE / GPU / VIDEO**。本审计是对已有格式事实和 FFNDS 方法约束的正式明确，不提出新的量化格式，也不改既有冻结数值路径、RUN、SHA 或 M-B 优先级。

## 1. 确定事实：HiF4 不是 unsigned-4 数值格式

HiFloat4 论文 `arXiv:2602.11287`，**Format Definition → Element Encoding / Table I / Eq.(2)**：
- 每个 **4-bit S1P2 元素明确包含 1 bit sign（符号位）、1 bit 整数位、2 bit 小数位**。带符号实值集合为 `{0, ±1/4, ±2/4, ..., ±7/4}`；论文格式允许 `+0` 与 `-0` 的不同码字。例：`0001 = +0.25`、`1001 = -0.25`、`0111 = +1.75`、`1111 = -1.75`（最高位为符号位，数值部分为无符号的三位定点幅值）。
- 每个 64 元素单元的 **scale metadata 为非负幅值信息**：8-bit **unsigned E6M2** 基础尺度 + 8 个 1-bit micro-exponent + 16 个 1-bit micro-exponent = 32 bit；每个组 `64×4 + 32 = 288 bit`，平均 `4.5 bit/element`。**幅值元信息不携带元素的符号，原元素码字携带符号。**
- 论文所定义的解码（非 NaN 时）为 `xhat_i = s_base * 2**(e8[i//8] + e4[i//4]) * c_i`，`c_i = (-1)**sign_i * (integer_i + fractional_i / 4)`。这是有符号数值表示，不是「所有量化值非负、负号必须由额外侧信道引入」。

**两条独立源码证据**（FFNDS 锁定的上游快照，源文件不在此处改动）：
1. [CUDA `hifxg_quant_cuda.cu`](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu#L187-L206)：`round_mant` 读取 `x_int & 0x80000000`，用 `x_sign` 将带符号值写回 `out = x_sign * e * mant3 * lv0`；量化阶梯对幅值求整，**符号未丢弃**。同文件 [`hifx_quant_cuda_inner`](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu#L229-L287) 的归约对输入使用 `abs` 构造尺度，然后对**原带符号 `lv3`**执行 `round_mant`，并写回带符号重建值。
2. [研究用 `ptq/formats/hifx4.py`](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/formats/hifx4.py#L303-L334)：`codes = round(normalized * 4).clamp(-7,7).to(torch.int8)`，`dequantized = codes.float()/4*full_scale`；其 `HIFX4EncodedGroup.codes` 是**带符号的 int8 软件容器**，数值在 -7..7，而不是说论文的每个 S1P2 元素要占 8 bit 进行 packed 存储。

注意零值：论文 S1P2 的 `-0` 与 `+0` 可有独立编码；软件 `int8` 的 0 没有符号，且 CUDA QDQ 后的 BF16 零符号需要单独按实际输出验证。不得将软件 `int8 codes` 和论文逐位 packed sign-magnitude 当作逐字节同一表示，更不能假设 CUDA 的 BF16 舍入完全等于该 Python 辅助编码。

## 2. 现行 Wan 实施必须区分三层「存储」

| 对象 | 此对象的符号信息 | 实际驻留与获取时点 |
| --- | --- | --- |
| **HiF4 论文 4bit packed 格式** | 每元素 S1P2 内 1 bit 符号，可解码负值 | 论文规定的 64×4bit + 32bit metadata；**并非现在 Wan QDQ 的持久 GPU 输出表示** |
| **Python HIFX4EncodedGroup.codes** | `int8` **signed -7..7**，研究/辅助表示 | 代码中可构造的显式 signed code；**不能自动认作原生 CUDA kernel 导出的 tensor** |
| **FFNDS 原生 Q1/Q2 fake-QDQ** | 输入正负号参与 CUDA 量化；输出重建 BF16 保留实际带符号数值（边界舍入须实测） | CUDA `hifx_quant_cuda_bf16(x,result,N)` 写回 BF16 `result`；内核只暂存尺度，**没有常驻 S1P2 packed code 或独立 scale/code 输出指针**。原 W1/W2 仍为 BF16 GEMM |

由此，**「格式带符号」已由格式与源码确定；「当前推理可免费读取 packed signed codes」尚不成立**。如果要在 Q1 同步取得 signed sketch，需在 producer 中低成本融合提取，或额外读取现有 BF16 QDQ 输出，两者都要计算指令/访存/CTA 跨块归约/临时存储的实价。仅从非负 `lv0/lv1/lv2` 还原不了各元素的正负及有序位置。

## 3. 对 I/F/C 研究的直接约束与新自由度

- **I·观察时点**：将原码的符号位视作 **format-native existing information**，不是「额外附加 sign bit」。主问题是：在哪个已付费 producer 阶段、以何种资源/顺序获取某些符号及其**列位置**，能在 W1 前实现有净收益的决策？
- **F·函数风险**：有符号 Q1 输入与 Q2 激活可驱动 W1/GELU、W2、gate/读出函数产生不同方向；尺度仅反映幅值。允许 scale-only 作为低信息基线，但必须同时对比 **signed-position-aware**、signed histogram、同成本普通 BF16 输入摘要。原 signed code 的存在不证明任意 signed histogram 足以预测 `Gamma` 或函数组束。
- **C·工程合同**：比较 packed 编码域真实读取路径、融合 signed sketch 和当前 BF16 QDQ 输出扫描。前者需要真的实现 packed producer/reader；不能将论文 4.5bit/value 的理论存储直接记为当前内核实际已省的带宽。W1 前的预测不得使用真实 Q2 后验码，合法 whole64/row-tile 动作与原 kernel 成本仍需核查。
- **反例的身份**：`x` 与 `-x` 的量化尺度可能相同，但元素码字不同；这证明的是 **scale-only 的信息不足**，**不是 HiF4 没有符号**。同一 signed histogram 也可能因列位置不同而引起不同 W1/功能贡献。
- **验证顺序**：先观察原 Q1/Q2 的 signed BF16 QDQ 数据，或独立导出临时 sign/magnitude metadata 的源绑定，记录 `source/step/expert/block/row` 与增量代价；其次在同样合法动作和同成本下检验 NetVOI；最后真实原生运行和自由视频评测。当前这些有效性实验仍 **OPEN**。

## 4. 不变范围

原 HiF4 上游 CUDA 代码、版本 SHA、冻结模型权重、QDQ 数值路径、已完成 B0/B1、A0 与视频 RUN 均保持原身份。此处是**符号语义纠错和研究边界明确**，不是已完成的新 sign-sketch 工程、GPU 测量或质量声称。

**出处**：论文 [arXiv:2602.11287](https://arxiv.org/abs/2602.11287)；本仓库原作者源码快照 `upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/...`；[观察时点原研究](QMETA_OBSERVATION_TIMING_AND_EFFICIENT_USE_20261028.md)。
