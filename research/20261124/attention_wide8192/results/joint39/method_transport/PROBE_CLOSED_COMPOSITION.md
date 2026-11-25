# 共享真实探针如何保护 attention 与 wide8192 的组合

日期：2026-11-25。依据 `Qesire/FFNDS@7c8196fa5380dbe67dd9772f942553085fde55b3` 的当前 wide8192 入口与执行代码。本文证明的是固定当前块输入、条件和 FFN 缓存时的算法依赖性质；实际 GPU 不同 attention kernel 的舍入误差另计。

## 1. 已核对的真实数据流

`tools/ffnds_fourstep/fourstep_backend.py` 将 `wide8192` 绑定到 `fourstep_hit.make_hit(8192,reuse_base=True)`。当前 `fourstep_hit.py` 的顺序是：

1. `operands(probes)` 产生 P64 上的 base、Q 和核特征 X。
2. `native FFN(u[probes])` 产生当前完整 FFN 教师。
3. 用上述 P 数据、固定 CAL、缓存 P 数据和固定度量求 gain、coeff 与 xp。
4. 对其余位置计算各自 operands，再使用同一个 readout。
5. 把 P 行写回原生 FFN 响应。

attention 的实际 `OnlineAttention.__init__` 断言 attention probe 与 FFN `observer_ids` 相同；`approximate` 最后覆盖这些 P 行。二者之间的 output projection、门控、residual、cross-attention 和 LayerNorm 都在视频 query 轴上逐行；cross 的文本/image条件固定。

## 2. 组合命题

固定同一个当前块输入 X、时间/文本条件、固定参数和 FFN 缓存。令原生 self-attention 响应为 A，候选响应为 A+e。假设二者在共用 P 行相同，即 Pe=0。

记 self-attention 后至 FFN 输入的逐行算子为 H，则

\[
P H(A+e)=P H(A).
\]

当前 FFN 全局读出参数 θ=(gain,coeff,xp) 是当前 P 输入、历史 P 缓存与固定参数的确定函数。因此

\[
\boxed{\theta_{A+e}=\theta_A.}
\]

非P attention误差在这个块内通过逐行的非线性尾部传播，同时FFN的当前共享拟合保持同一个解。P行的块出口也相同。

该结论说明了共用P覆盖承担的第二个作用：除使attention的P输出准确外，它还使后续自适应FFN的当前拟合输入与参照相同。新的标准化缺陷attention保持这一接口，因而保留这种组合关系。

本命题覆盖一块内的给定当前状态。下一层 self-attention 会重新混合位置；下一去噪步还会改变状态。对整条候选轨迹的作用仍需使用实际轨迹。原生full-query与P-query kernel若有数值差异，Pe会包含相应舍入项，不能把实数命题替换为未经实测的BF16逐位一致。

## 3. 有限后续作用

在固定读出 θ 后，记当前尾部为 T_θ。非P位置误差可按有限关系研究：

\[
T_\theta(A_i+e_i)-T_\theta(A_i)
=\int_0^1 DT_\theta(A_i+\tau e_i)[e_i]\,d\tau,
\]

在分段光滑实现上使用相应的几乎处处导数和跳跃项。这保留了LN、GELU、gate与residual的作用。它和上轮历史四臂交互场相容：共享读出保持不变，尾部非线性仍会产生有限交互方向。

这也区分了两种联合策略：模块边界保留真实P并继续执行当前FFN，可共享教师并保护当前拟合；完整块缺陷恢复则需要额外保留未覆盖cheap尾部，才能观测它所定义的块缺陷。后者的shadow成本来自改变了恢复对象。

## 4. 本轮数值核验

`verify_probe_composition.py` 使用明确合成的非线性块，保留P64和16模式以及当前wide8192的求解依赖。256行、32输出通道、80隐藏通道、24选中通道，seed=250911；不是Wan权重或激活。

- 对非P施加attention误差并覆盖P后：FFN输入P、gain、coeff、xp、qp、教师和块出口P的最大差均为0。
- 非P块出口仍产生范数2.2909975的差，说明此核验保留了实际非线性传播。
- 对同一扰动省去P覆盖后：gain差范数0.0254229、coeff差范数0.4133806。

这些数值验证源码依赖所导出的条件命题，完整结果在 `probe_composition_verified.json`。

## 来源

- [当前wide8192入口](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/tools/ffnds_fourstep/fourstep_backend.py)，Git blob `4b371b43a036397236ef614f6404aaafda0bcca4`。
- [当前wide8192 hit](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/tools/ffnds_fourstep/fourstep_hit.py)，Git blob `40635bf25feb03dc3f48a98f2254d1c5b0ece7a7`。`source/`保留这两个小型源码快照以复核指纹。
- [attention P绑定和覆盖](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)。
- [实际Wan联合计算图审查](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/research/20261124/attention_wide8192/results/JOINT_GRAPH_AUDIT.md)。
