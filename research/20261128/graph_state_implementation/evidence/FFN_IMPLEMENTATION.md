# FFN 实现：分阶段执行、规范变换与投影缓存

2026-11-28。新增 `tools/ffnds_optim/ffn.py`、`ffn_runtime.py`、`compile_ffn.py` 与 `tests/test_ffn.py`。原冻结 `wide8192` 源文件不改。

## 分开的数值身份

| 路线 | 已实现行为 | 数值身份 |
|---|---|---|
| FFNStaged | P64 fit 与任意原全局 token 子集的 readout 分开；full 保留8192块和最后原生P覆盖 | native 后端保留原difference/LUT/base及舍入位置；CPU BF16独立公式逐值相同；CUDA尚未运行 |
| 16模式 gauge | 成对改变近似分支 A/B，原生phase教师不变 | 实数GLS恒等；BF16权重重编码是显式数值变化 |
| A0因子化 | 离线CPU SVD初始化，或显式提供L/R；runtime不传输完整GPU A0 | 受控近似；SVD只优化权重Frobenius目标，质量未测 |
| 投影缓存 | 仅分配y0/z0k/a0/j0/u0p，旧u0没有全场槽 | 近似坐标状态；u0p保住原Q_P路径 |
| 锚点直写 | 在既有MMWeight源指纹与原生addmm布局成立时直接写y0槽，其他情况原生apply+copy | 保留原算子/舍入；CUDA执行未测，fallback计数可见 |

分阶段API本身提供跨模块调度切口，不将独立使用它声称为已经提速。full()仍使用原全场分块，历史slice使用view；任意非连续原token IDs才gather。原native difference、LUT、base与BF16 selected GEMM继续从冻结运行环境读取。

## 接口

```python
staged = FFNStaged(phase, prepared, cache, probes, inv_metric, backend="native")
fit = staged.fit(u_p)  # u_p按原probes顺序，原生教师在这里取得
y_tile = staged.apply_rows(global_token_ids, u_tile, fit)
# fit.observed可直接沿原生P64教师链送往后继。
```

`apply_rows(..., overwrite_probes=False)`用于全是非P行的整块流式调度。独立子集含P时可以启用覆写；`full(u,tile_rows=8192)`始终在所有块完成后覆写原生P。

`OptimizationFFNPort`提供现有生成器所需的`bind/trajectory/__call__`，及整块调度所需的`take_hit(model,phase)`。后者消费正确专家/层的锚点并只登记一次命中。所有缓存槽在两专家间分时复用，失败退出清空owner，第二请求复用已分配槽与已准备常量。回执计数还包含实际factor rank、坐标dtype、精确probe sidecar、因子方法和配对gauge状态。

该port从未构造旧ResidentJoint；因此因子模式真实删除全u0槽。prepare_runtime_tensors在factor资产存在时跳过A0的GPU传输，保留CPU冻结pack供来源绑定。读取当前phase权重仅生成拥有自己存储的selected子矩阵，避免prefetch指针跨层复用。

## 因子资产

`compile_factor_asset(pack, rank, gauge=None, dtype=torch.bfloat16)`在CPU上生成：

- `left:[d,r]`、`right:[r,d]`、可选`gauge:[16]`。
- 原A0/J/V的张量SHA-256，rank和`format_version=1`。
- `method="weight_frobenius_svd_initializer"`，`task_quality="unmeasured"`。
- 截断奇异值能量，供理解初始化误差；它不是恢复后任务误差。

因子中的gauge与runtime显式gauge必须一致，配对Bc由原selected W2生成；教师W2不变。提供两个loaders时，factor_loader(expert,layer)返回完整资产，gauge_loader(expert,layer)返回同一个c。没有factor时，gauge也能单独使用。生产prepare要求A0/J/V源身份字段均存在且匹配。

离线编译入口使用既有冻结CAL manifest，也可以仅编译一个站点：

```bash
PYTHONPATH=tools python -m ffnds_optim.compile_ffn --cal-manifest /existing/CAL_MANIFEST.json --rank 512 --output-dir /new/ffn_factors_r512
PYTHONPATH=tools python -m ffnds_optim.compile_ffn --pack /existing/high00.pt --site high00 --rank 512 --output-dir /new/high00_factors
```

可选`--gauge`读取一个固定.pt/JSON向量，`--gauge-dir`读取每站点的high00.pt等。输出目录必须全新；输出`high00.pt`等与根运行器TensorDirectory匹配，`COMPILE_RECEIPT.json`记录来源、rank、SVD丢弃能量和完整周期资源账本。manifest模式核对源pack的SHA/字节；单pack模式输出实际SHA。中途失败保留明确FAILED回执与已完成资产。

`recovery_filter`实现可微的`M_H(I-R_F P)E`，包含真实GLS增益、P64核及0.5系数。`recovery_aware_factor_loss`接受真实供应的DeltaU/Q/X/V/metric/P，计算`DeltaU L R - DeltaU A0`经过恢复后的误差；可选gauge目标与输出线性消费者。该工具支持对L/R/c求导，SVD编译入口仍清楚标为初始化，没有声称完成恢复感知拟合。

运行中不会在timed anchor里偷偷执行SVD。rank和坐标dtype均为显式选择。主要状态含精确u0P时，r512为19.42853 GiB，r1024为20.63657 GiB；包含全部建锚乘法后的盈亏点r<1701.33。具体预算可由factor_cycle_budget复算。

## 本轮验证

CPU Torch 2.15.0+cpu，12项测试通过（2.36秒）：

1. 分阶段完整输出与独立冻结公式在BF16和FP64下逐值相同。
2. 非连续、乱序全局token IDs及原probe顺序保持。
3. 实数gauge的gain平移、kernel系数与输出恒等。
4. 因子输出误差等于原恢复器过滤后的底座误差。
5. 精确u0P保持原Q_P/X_P，最终P输出仍是当前原生教师。
6. 实际slots不存在全u0，prepared不存在GPU A0。
7. 两次HHLL请求的缓存所有权、静态常量复用与错误清理。
8. source/gauge身份不匹配及时报出，完整周期预算含建锚及probe sidecar。
9. 可微恢复后目标等于实际FFN差分，其梯度通过中心差分核对。
10. 离线CLI导出完整因子及来源回执，运行器可以加载；已有输出拒绝覆盖。

另外，整块代理使用真实FFNStaged接口与attention fit/readout，通过high/low两路CPU完整块集成验证（由block实现记录保存）。

当前workspace没有CUDA设备或Wan运行资产。本轮验证说明实现与数学/状态契约一致；原生CUDA逐位结果、完整生成质量与速度维持UNVERIFIED。原kernel通过lazy import保留，未用CPU参考冒充生产CUDA实现。
