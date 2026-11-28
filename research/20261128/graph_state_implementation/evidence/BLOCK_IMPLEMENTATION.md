# P64 前置的完整命中块流式实现

2026-11-28。实现文件为 `tools/ffnds_optim/block.py`、`wan_block.py`，针对当前 Wan 四步的命中步1/3；锚点继续由外层分派器调用既有路径。

## 已实现

`stream_hit_block(x, probes, operators, tile_rows=8192)`先固定全位置K/V，再沿P64原生attention→输出投影→门控残差→cross-attention→归一化→原生FFN完成教师路径，确定两套读出。非P位置随后逐tile贯通完整块。P64按原生产顺序保存，非P请求携带原全局token ID。

`WanBlockOperators`把调度实际连接到原生Wan phase的 `.apply`、`pre_process`、`modulate_func`、`apply_rope_func` 和cross-attention接口。全场K/V分块构建；text K/V每层构建一次；Q只在P64和各非P tile生成。下一层接收完整输出后再创建其K/V，保持层间依赖。

原生RoPE缓存的token轴为0，依据已读取的`joint_graph/pre_infer.py`。子查询显式按全局ID选择cos/sin行。默认复用原后端的成对RoPE API，并为两输入提供独立副本，以支持会就地写回的实现。这里多计算了一份未使用的RoPE结果，没有多计算Q投影；可注入经过独立核验的单输入RoPE消费者替换这一默认实现。

残差在tile私有副本上按原乘法、加法和dtype顺序更新；输出不允许别名覆盖仍待读取的整块输入。`ffn.fit(uP).observed`直接写入P位置，非P调用`ffn.apply_rows(global_ids,u,fit)`。全块门控计数由外层分派器一次性登记，避免每个tile被误记成一次模型站点。

`DeferredLayerNormLinear`另实现了`LN(B+FU)`之后的仿射线性消费者，用行矩、交叉矩和小Gram计算，避免物化`FU`或`B+FU`。固定basis/consumer的统计与投影在构造时预计算。该工具通过显式API使用，尚未自动插入Wan默认图；它额外的交叉矩成本需随选定消费者计入。

高噪声输出投影折叠通过显式`ProjectedAttentionRows(value)`类型接入，`value`必须是`[rows,model_width]`的原推理dtype。attention会话可用`bind_output_projection(self_phase)`取得原生W_o。P64教师仍返回未投影的原生attention，由原W_o处理；仅非P折叠读出返回投影后的类型，后继链据此跳过第二次W_o。

`make_row_plan(N, probes, tile_rows)`一次构建并验证非P位置的`RowPlan`。`stream_hit_block`及`streamed_wan_hit_block`均接受`row_plan=`；后续层直接复用相同tile张量对象。计划绑定原P张量身份、几何和设备，正常版本化张量还检查修改版本；复用检查只读host元数据，不重复做GPU成员关系筛选或读取设备标量。inference-mode位置张量由计划持有并按只读约定复用。外层可用这些稳定tile对象缓存selected attention布局。

## 数值与资源身份

这属于**实数等价的图重排**。Q/K/V、输出和cross/FFN GEMM的行形状变化，可能改变有限精度归约；现有GPU后端是否逐位相等须在实际设备上核验。

为避免保存全场norm1，Q阶段重新计算norm1/modulation。默认单输入RoPE适配产生额外向量工作。实现没有将这些代价算作免费，也未报告GPU加速或显存峰值。

持久的历史缓存由attention/FFN状态模块管理。本模块减少的是中间物化及其生命周期。完整self K/V、当前块输入和下一层输出仍按真实依赖保留。

序列并行、图像编码器分支、post/before/after adapter和MXFP8 fused路径在此绑定中显式拒绝；这些分支改变本轮所绑定的图或操作数接口。当前已冻结BF16图对应受支持路径。

## 本轮执行证据

命令：

```bash
PYTHONPATH=/tmp/ffnds_r21_python:round21_impl/repo/tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q round21_impl/repo/tools/ffnds_optim/tests/test_block.py
```

环境：Python3.12、PyTorch2.15.0+cpu；本地无GPU。初轮14项通过；加入高噪声折叠后16项通过。加入RowPlan后主集 **18 passed，2.40秒**，随后新增的全P/inference-mode边界测试单独 **1 passed，1.61秒**。最终root会话/绑定接口的定向集成测试另为 **3 passed，2.52秒**，当前共22项测试。跨模块测试实际发现并修复了folded reader对不含P位置的tile尝试`reshape(0,-1)`的问题；修复后这类tile也省去原W_o的空投影分支。测试内容包括：

- 实际新`AttentionReadout`、`LowHistory`与`FFNStaged`组成的high/centered-low两条完整块路径，与独立全场参考对比；137行、64个乱序原P位置、31行tile，Float64容差2e−12。
- 原全局RoPE位置与逐token调制，原生RoPE就地写回时输入别名保护。
- self KV一次准备、text KV一次准备、Q只按P64和非P tile生成。
- 只有1行的末尾tile、输入/输出存储别名拒绝及不支持分支的明确报错。
- 秩1/5/13的归一化后线性作用，和直接物化参考对比。
- 高噪声折叠输出的完整块对照，以及W_o仅对原生P64调用一次、非P无重复投影的计数。
- RowPlan跨块复用相同tile对象、冻结P覆盖、修改身份/版本和缺失覆盖的检测，以及全P与inference-mode位置边界。
- root的`AttentionSession`/`HitAttention`→`bind_streamed_blocks`→Wan adapter→两种实际reader的完整接口；分别检查high、low和high折叠，确认anchor委托、hook恢复、slot消费以及每完整块一次的计数。

操作数为明确合成的张量；这些是实现与依赖关系的CPU核验，没有Wan权重推理、成片、评分或GPU计时。
