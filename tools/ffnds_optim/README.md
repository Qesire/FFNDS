# wide8192：分别启用的计算图与缓存优化

2026-11-28。该包将[计算图与状态推导](../../research/20261128/graph_state_theory/THEORY_AND_GRAPH_REFORMULATION.md)落实为独立组件、Triton 内核、资产编译器和现有四步生成器的适配入口。默认参照是已冻结的 **centered attention + BF16 wide8192 FFN**；优化由 JSON 配置显式选择。

**当前执行证据：50 项 CPU 测试通过；单独运行的 Triton FP32 解释器测试通过；两个 BF16 内核完成 A100 sm80 离线编译。** 当前环境没有 CUDA 设备，本轮没有模型前向、视频生成、评分或 GPU 计时。完整记录与核验文件见[实现记录](../../research/20261128/graph_state_implementation/IMPLEMENTATION_RECORD.md)。

## 1. 三种数值身份

| 身份 | 独立选项／组件 | 实际改变 | 证据边界 |
|---|---|---|---|
| 保留原操作与舍入位置的执行优化 | `ffn_staged`、`anchor_direct` | FFN 的 P64 拟合与逐块读出分离；原生 W2 在条件成立时直接写锚点槽；同一行布局跨层复用 | 分阶段 CPU BF16 与独立原公式逐值相同；CUDA 上的逐位结果待核验。直接写入受原 MM 源身份、布局和别名条件约束，回退次数单独记录 |
| 实数目标相同的图重写 | `anchor_partition`、`attention_streaming`、`block_streaming`、`projection_precompose`、`high_output_fold`、`ffn_gauge_dir` | 合并锚点分区扫描、P64 前置、完整命中块分块执行、固定线性映射预合成、high 输出投影折叠、16 模式规范变换 | 改变 GEMM 形状、结合顺序、累加次序或中间量化；不据实数恒等宣称 BF16 逐位相同 |
| 显式近似表示 | `ffn_factor_dir`、`attention_basis_dir` | FFN 底座因子化及旧输入投影缓存；low attention 联合历史编码与动态读出 | 秩、基、坐标 dtype 和资产哈希显式记录；P64 教师保持原生；压缩质量需对应模型结果 |

`config.py` 为每个运行回执产生上述身份。单独选项可以组合；高噪声输出投影折叠通过整块流式接口连接，因此要求 `block_streaming=true`。原数值路径的结论只属于第一类中相应操作；整组配置包含第二／三类时，按这些更强的变化解释结果。

## 2. 已连接的实际路径

### 低噪声锚点的一次分区扫描

`partition_triton.py` 在每个 key tile 上计算一次 QK 和 PV，并将结果累计到 selected 或 complement 的在线 softmax 统计。随后产生 full、selected、遗漏质量 rho、两种 LSE，并可直接写 centered C。它替代 low anchor 原来的 full 与 selected 重复扫描。high anchor 继续沿原生路径计算及建立路由。

`selected_attention` 接受原全局 query ID；每个 query 继续使用原位置所属的 block 路由。selected kernel 只扫描该 query block 的选中 key blocks，支持 P64、不连续非 P 行和末尾不足块。

### P64 先行与完整命中块流式执行

`AttentionReadout.fit/read` 和 `FFNStaged.fit/apply_rows` 将观测与读出分开。`WanBlockOperators` 连接当前模型实际 phase 的 norm、投影、RoPE、cross-attention、调制、门控与 FFN：

1. 在当前层先分块生成全位置 self K/V，建立 text K/V。
2. 原 P64 依次经过原生 attention、W_o、残差／cross-attention、归一化和原生 FFN，确定两套恢复读出。
3. 非 P 行按 8192 行 tile 贯穿整条命中块，用当前层完整 K/V；只在块输出处汇合。
4. 下一层在取得完整输出后才建立自己的 K/V，保持层间依赖。

RoPE 按原全局 token ID 选择 cos/sin；原 P 顺序保留。`RowPlan` 和 selected 查询布局跨层复用。命中时中间 hidden 的生命周期限制在 tile 内；输入、完整 K/V 和下一层输出仍按真实依赖保留。为省去全场 norm1，Q 阶段重新计算 norm1／调制；默认成对 RoPE 适配有一份未使用的向量结果，这些是明确的执行成本。

该绑定支持冻结 BF16 图。序列并行、额外 adapter、图像编码器分支和 MXFP8 fused 分支需要各自的图适配，当前会明确报出不支持。

### high 输出投影折叠

拟合完成后，将 high 的仿射恢复与 W_o 合成为一次主输出投影。生产路径把合成权重存为查询 dtype（BF16），避免把主输出 GEMM 隐式变成 FP32。P64 仍经原生 W_o；非 P 返回 `ProjectedAttentionRows`，后继明确跳过第二次 W_o。中间舍入的移除、合成权重的量化及每层合成成本均属于该选项。

### FFN 因子与真正减少的历史状态

`OptimizationFFNPort` 不构造旧 `ResidentJoint`。显式因子存在时，准备阶段跳过完整 GPU A0；槽只分配 `y0/z0k/a0/j0/u0p`，不会同时保留全场 u0。`a0=u0 L`、`j0=u0 J` 与当前投影用于读出；默认精确 `u0p` sidecar 保留原 P64 的差分特征。80 个专家／层站点的常量各自保存，40 个历史槽在 high／low 之间分时复用。

`ffn_gauge_dir` 提供 16 模式配对规范变换；近似分支的底座和恢复项成对修改，原生教师权重保持原对象。`compile_ffn` 是明确标注的权重 Frobenius SVD **初始化器**。`recovery_aware_factor_loss` 另提供包含真实 GLS、P64 核及 0.5 系数的可微恢复后损失；本轮没有用生产轨迹执行恢复感知拟合。

### attention 联合缓存

low 的非 P 历史由 `Z=[C,A]` 的 159 维字段改为 `U=ZE`。得到当前读出后，直接计算 `U(E^T G_t)`；也保留 `joint_readout=source_order` 的解码对照。编码后，完整 C/A 引用和旧 cache tuple 释放，只保留代码、rho、基和原生 P64 历史 sidecar。回执统计真实 tensor storage 字节，去除别名重复计数。

`compile_attention` 使用同一轨迹的配对 `(Z,G)` 优化固定秩基，目标为 `||ZG-(ZE)(E^T G)||²`。可提供 W_o，在合并各头后计算误差，保留头间交叉项。PCA 仅作初始化，后续决策使用配对目标、自动微分梯度、Stiefel 切向投影和 QR 回缩。存储基的量化损失单独报告；在线代码量化尚未纳入拟合目标。

### 独立的 LayerNorm 旁路组件

`block.DeferredLayerNormLinear` 实现 `LN(B+FU)` 后接仿射线性消费者，借助行矩、交叉矩与小 Gram 矩阵计算其作用，避免先物化 `FU` 或 `B+FU`。它已完成组件核验。完整 Wan 运行器尚未携带该延迟表示跨过所有非线性节点，因此该组件当前通过显式 API 使用，没有自动启用开关，也没有将其节省计入整图。

## 3. 独立配置

| 配置 | 启用内容 |
|---|---|
| `profiles/00_frozen.json` | 原 centered + wide8192 参照 |
| `profiles/01_staged_ffn.json` | FFN 分阶段接口与受约束锚点直写 |
| `profiles/02_partition_anchor.json` | low anchor 单遍双分区 Triton 内核 |
| `profiles/03_attention_stream.json` | attention 内部 P64 前置与非 P 分块 |
| `profiles/04_block_stream.json` | 完整命中块 P64 前置与非 P 分块 |
| `profiles/05_high_output_fold.json` | 整块分块 + high 恢复／W_o 折叠 |
| `profiles/06_projection_precompose.json` | 固定特征映射预合成 |
| `profiles/07_ffn_factor.template.json` | 显式 FFN 因子和投影缓存 |
| `profiles/08_attention_joint.template.json` | 显式 attention 联合历史基 |
| `profiles/09_combined_graph.json` | 分阶段 FFN + 单遍锚点 + 整块流式 |
| `profiles/10_combined_state.template.json` | 上一组合 + 显式 FFN／attention 压缩资产 |
| `profiles/11_ffn_gauge.template.json` | 显式配对 FFN 规范变换 |

模板中的 `/REPLACE_WITH_...` 必须替换为真实资产目录。压缩秩由资产决定，没有自动选取质量阈值。FFN 常驻槽要求所有站点使用相同 rank。`anchor_direct=false` 可单独关闭直接写入；`ffn_projection_dtype`、`attention_code_dtype` 可以显式选 BF16 或 FP32。默认保留精确 FFN probe sidecar。

## 4. 运行入口

以下命令从仓库根目录执行。仅展示配置不加载 torch、模型或资产：

```bash
PYTHONPATH="$PWD/tools" python -m ffnds_optim.run \
  --config tools/ffnds_optim/profiles/09_combined_graph.json --describe
```

实际生成入口复用现有学校 runtime、权重与冻结 CAL。下面路径和 SOURCE_ID 为占位符；本次交付没有执行该命令：

```bash
PYTHONPATH="$PWD/tools" python -m ffnds_optim.run \
  --config tools/ffnds_optim/profiles/09_combined_graph.json \
  --runtime-dir /ABS/FROZEN_RUNTIME \
  --fit-dir /ABS/FROZEN_ATTENTION_CAL \
  --manifest /ABS/EXISTING_SOURCE_MANIFEST.json \
  --source SOURCE_ID --group graph_state_component_check
```

`--manifest` 对 runtime 已知来源可省略；可重复提供 `--source`。默认 `--binding` 为仓库的 `research/20261127/centered_medium_eval/FROZEN_METHOD_BINDING.json`。运行器核验源代码、原生 SDPA 后端、CAL manifest、输入图像／提示词和 seed；输出组须为新的相对目录。

每个生成结果有原始收据和 `SOURCE_ID.optimization.json`：后者记录配置指纹、数值身份、全部优化源文件哈希、实际加载资产哈希、四步计数、历史状态与原始结果。候选 ID 为 `wide8192_optim_<配置指纹>`；完整身份还包括源代码和资产哈希。收据不会把旧方法评分转记到新配置。

## 5. 离线资产编译

FFN 从已有 PASS CAL manifest，或单个已有站点 pack，生成明确指定秩的初始化因子。输出目录须为新目录。下列 rank 只是命令示例，不构成生产选择：

```bash
PYTHONPATH="$PWD/tools" python -m ffnds_optim.compile_ffn \
  --cal-manifest /ABS/EXISTING_CAL_MANIFEST.json \
  --rank 512 --output-dir /ABS/NEW_FFN_FACTOR_DIR

PYTHONPATH="$PWD/tools" python -m ffnds_optim.compile_ffn \
  --pack /ABS/EXISTING/high00.pt --site high00 \
  --rank 512 --output-dir /ABS/NEW_SINGLE_SITE_DIR
```

可选 `--gauge` 或 `--gauge-dir` 显式提供规范变换。输出 `high00.pt`…`low39.pt` 及 `COMPILE_RECEIPT.json`，记录 A0/J/V 来源、因子身份、SVD 丢弃能量和包含建锚投影的完整周期预算。

attention 编译输入为已采集的配对 CAL `.pt`：

```python
{
    "schema": "ffnds_attention_joint_cal_v1",
    "expert": "low", "layer": 0,
    "calibration": {"split": "CAL", "source_ids": ["..."], "rows": "nonprobe"},
    "samples": [{"Z": tensor_H_M_159, "G": tensor_H_159_128}, ...],
    "output_weight": optional_tensor_Htimes128_Dout,
}
```

同一个 sample 的 Z/G 必须来自同一 anchor→hit 轨迹；元数据是资产提供方的明确声明，编译器不从评测视频推断或伪造 CAL。

```bash
PYTHONPATH="$PWD/tools" python -m ffnds_optim.compile_attention \
  --input /ABS/PAIRED_CAL/low00.pt --rank 64 \
  --output /ABS/NEW_ATTENTION_BASIS_DIR/low00.pt
```

输出 `{basis, metadata}` 与同名 JSON，运行器按 `lowXX.pt` 加载。没有现成真实配对状态时，本入口只交付算法及组件验证，本轮没有生成生产基。

## 6. 复核入口与实际范围

```bash
PYTHONPATH="$PWD/tools" python -m pytest tools/ffnds_optim/tests -q
TRITON_INTERPRET=1 PYTHONPATH="$PWD/tools" python -m pytest \
  tools/ffnds_optim/tests/test_attention.py -q -k interpreted

PYTHONPATH="$PWD/tools" python -m ffnds_optim.check_kernels \
  --mode compile --output /ABS/NEW_OUTPUT/kernel_compile.json
PYTHONPATH="$PWD/tools" python -m ffnds_optim.check_kernels \
  --mode cuda --output /ABS/NEW_OUTPUT/kernel_cuda.json
```

本地记录使用 Python 3.12.14、PyTorch 2.15.0+cpu、Triton 3.9.0。普通测试为 50 passed／1 skipped；被跳过的解释器用例另行运行后为 1 passed。CPU 测试使用明确标注的合成张量和已有 CAL 投影，包括实际新增模块的 high／low／high-fold 完整块集成、生命周期、多请求复用、身份检查和可微目标核验。

两个 BF16 内核按 N=31668、H=40、D=128、TOP=31、BM=32、BK=64 编译到 sm80 PTX/cubin。FP32 解释器用于验证实际内核索引与归约；Triton 3.9 BF16 解释器的 dot 存在 uint16 底层数组行为，本轮保留该限制，未将 FP32 解释器结果表述成 BF16 GPU 验证。CUDA 检查在本地返回 `PENDING_CUDA`、退出码 2。

全场 Torch 参考内核只供组件数值核验；生成入口要求 native／Triton 后端。完整 GPU 数值、峰值、冷／暖时间和质量属于各配置未来实际收据，当前记录为待验证。
