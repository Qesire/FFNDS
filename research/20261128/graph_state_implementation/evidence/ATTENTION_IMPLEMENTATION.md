# Attention 优化实现与验证

2026-11-28。对应冻结 centered low / high OUT-ABS 的独立、可组合实现。

## 已实现

| 实现 | 数值身份 | 具体执行 |
|---|---|---|
| `partition_attention` / `_partition_kernel` | 相同实数目标，归约次序改变 | 每个 key tile 一次 QK/PV；selected/complement 分别累计；同时输出 full、selected、rho、LSE，可直接写 centered C |
| `selected_attention` / `_selected_kernel` | 相同 selected 支持，后端归约改变 | 按原全局 query block 分组；只遍历 TOP key blocks；保留任意子集 token ID、顺序及尾块掩码 |
| `AttentionReadout.fit/read` | 默认保持冻结两级投影、BF16 feature cast、normalized coefficient 和 `(x@coef)*scale_r` 顺序 | 只用 P64 拟合，任意 query tile 读出；精确覆盖探针；动态 GEMM 行数仍属于数值执行条件 |
| `LowHistory.encode_joint` | 显式近似 | 调用方提供正交基；只保留非 P 编码、原 rho 和精确 C_P/A_P；释放完整 C/A；支持 source-order 解码及直接作用读出 |
| `fold_high_output` | 仿射实数重排；省略 feature/output cast，生产动态权重再量化 | 构造 W_eff 和偏置；生产传 BF16 storage_dtype 保留 Tensor Core 主 GEMM；非 P tile 不计算原 W_o |
| `projection_precompose` | 相同实数线性映射，乘法次序变化 | 合成固定 P_v T；默认关闭 |
| `compile_attention.py` | 离线、固定秩、配对作用目标 | 真实配对 Z/G 输入；可保留跨 head W_o 作用；自动微分、Stiefel 切向梯度和 QR 回缩；不自动选择质量可用秩 |

Torch partition/selected 是可执行的流式数值参照；生产选项使用独立 Triton kernel。`SelectedLayout` 可在多层复用，从原全局 ID 映射到当前子集行，避免把 query 子集重新编号。

## API 与轴

Q/K/V：`[token, head, channel]`。attention 输出、特征与历史：`[head, token, channel]`。route：`[head, global_query_block, top_blocks]`。

- `partition_attention(q,k,v,route, backend='triton', mean=...,inverse=...)` 返回 `PartitionOutput`；full/selected/centered 保持输入 dtype，质量和 LSE 为 FP32。
- `prepare_selected_layout(ids,N)` 预生成 token 到原 route 块的映射。`selected_attention(...,ids,layout=...)` 输出 head-first selected。
- `AttentionReadout(...).fit(selected_p,truth_p,cache=...,mean=...,scale=...)` 只读取 P64。
- `reader.read(selected,ids,prepared,...)` 接受任意顺序的原全局 ID，精确写回 P 行。
- `LowHistory.from_fields(C,A,rho,probes).encode_joint(E)` 保存调用方给出的基与非 P 编码；基的量化也纳入重建状态误差。`storage_bytes()` 计实际张量底层存储，去除别名重复。
- `fold_high_output(...,storage_dtype=torch.bfloat16)` 在生产保持 BF16 主输出 GEMM；None 供 FP64 代数核验。

## 配对 CAL 编译器

输入文件必须明确：

```python
{
  'schema': 'ffnds_attention_joint_cal_v1',
  'expert': 'low', 'layer': 0,
  'calibration': {'split':'CAL', 'source_ids': [...], 'rows':'nonprobe'},
  'samples': [{'Z': tensor[H,M,159], 'G': tensor[H,159,128]}, ...],
  'output_weight': tensor[H*128,Dout]  # 可选；保留 head 交叉项
}
```

每个样本的 Z 与 G 必须来自同一 anchor→hit 轨迹。实际优化目标是每个配对的 `ZG - (ZE)(EᵀG)`，经 W_o 后合并 head，再汇总样本。状态/读出统计不作独立性分解。PCA 只作为初始正交基，每个接受的更新以完整配对目标判断。

```bash
PYTHONPATH=tools python -m ffnds_optim.compile_attention --input low00_cal.pt --output basis/low00.pt --rank 96 --steps 40
```

输出 `{'basis': tensor, 'metadata': ...}`，与 `TensorDirectory` 按 lowXX.pt 的加载方式匹配。并写 JSON 元数据、输入哈希、CAL 来源、每步损失与基存储精度后的损失。在线编码 BF16 误差不包含在该离线基目标中，记录字段明确区分。

本轮没有真实 Z/G 配对归档，因此只通过合成输入验证编译器；未生成可被误认为已通过质量的生产基。

## 本轮运行证据

环境：Python 3.12，PyTorch 2.15.0+cpu，Triton 3.9.0。没有 CUDA 设备。

1. Torch CPU：独立 dense/selected softmax 参照、互补 C 恒等式、全选中空补集、尾块、任意全局 ID、冻结顺序读出、联合缓存误差、精确 P64、high W_o 折叠与 BF16 GEMM dtype、真实冻结 CAL 投影、配对目标梯度与编译资产往返。
2. Triton CPU interpreter：FP32 的两个真实 kernel 均执行通过；覆盖尾块、original ID 子集、full/selected/rho/C。
3. A100 sm80 离线编译：生产 N=31668/H=40/D=128/TOP=31、BF16 指针、BM32/BK64，两个 kernel 均生成 PTX 和 cubin，静态 shared memory 均 40,960 B。记录见 `ATTENTION_KERNEL_COMPILE.json`。运行端固定 `num_warps=4,num_stages=2,enable_fp_fusion=True`。
4. BF16 interpreter 的已定位限制：Triton 3.9 的 `create_dot` 直接把 BF16 背后的 uint16 数组传入 NumPy matmul，未转成浮点，因此该模式无法作为 BF16 算术 oracle。本轮保留其失败定位，不改写解释器；BF16 证据为真实 sm80 编译，GPU 输出与速度仍未执行。

复算 CPU：

```bash
PYTHONPATH=tools python -m pytest tools/ffnds_optim/tests/test_attention.py -q
TRITON_INTERPRET=1 PYTHONPATH=tools python -m pytest tools/ffnds_optim/tests/test_attention.py -q -k interpreted
```

模块数值核验、离线 kernel 编译、真实模型成片和硬件计时是分别记录的证据。本实现没有替换冻结默认方法。

离线编译和 CUDA 算子验收入口：

```bash
PYTHONPATH=tools python -m ffnds_optim.check_kernels --mode compile --output kernel_compile.json
PYTHONPATH=tools python -m ffnds_optim.check_kernels --mode cuda --output kernel_cuda.json
```

CUDA 模式显式启动两个 BF16 算子，比较 FP64 dense/selected 独立参照以及原生 SDPA，覆盖 N=259 的部分尾块和任意原 query ID。当前无 CUDA，入口返回 PENDING_CUDA（退出码 2）。测试中的数值容差仅用于该固定合成算子例，未作为模型质量界。
