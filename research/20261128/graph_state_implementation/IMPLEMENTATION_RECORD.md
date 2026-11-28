# 从理论表示到可分别执行的优化

日期：2026-11-28。实现 ID：`ffnds_graph_state_optim_v1`。

用户本轮要求：“明确区分并分别实现尽可能的优化”。本轮依据 `91732be3af6b7645f8e3f19f4286b613fb1235b9` 中的计算图推导与冻结运行来源，将相应构造落实为 `tools/ffnds_optim/`。实际模型结果的证据仍为此前 `a1eb64d63910b4d90b94ea4c641ef461c83d0d92` 的 job 206224；本轮新证据为实现、CPU 张量计算与目标内核编译。

## 1. 已交付的区分

### A. 保留原操作与舍入位置的执行优化

FFN 读出分为 `fit(P64)` 和 `apply_rows(global_ids)`，独立全场路径维持 wide8192 的原 difference／LUT／base kernel、BF16 cast 与最后 P64 覆写。锚点 W2 在原 MM 源码、权重访问器、dtype、布局和别名条件成立时直接写入 y0 槽；其余情况调用原生接口并复制，回执保留回退原因。

固定原 token 布局与稀疏 query 映射跨层复用；连续 FFN 行走 view，非连续行才 gather。prepared selected 权重拥有自己的存储，避免预取缓冲在下一层覆写。这些是执行与生命周期调整。分阶段 CPU BF16 对独立原公式已验证逐值相同；CUDA 逐位一致性仍需实际硬件记录。

### B. 实数目标相同、浮点路径变化的图重写

| 构造 | 已落实的过程 | 变化范围 |
|---|---|---|
| low anchor 单遍双分区 | 一个 key tile 的 QK/PV 只计算一次；selected／complement 统计共同得到 full、rho、LSE、C | 归约顺序与原 Flash 路径不同 |
| attention P64 前置 | 先真实 P64 拟合，再对非 P query tile 生成 selected 与恢复输出 | query GEMM／归约形状变化 |
| 完整命中块流式 | 全 K/V → P64 原生教师链 → 非 P tile 贯穿 self/cross/FFN → 完整块出口 | 删除中间全场物化，保留下一层屏障；norm1 重算与 RoPE 适配有成本 |
| 固定投影预合成 | 将固定两级特征矩阵先相乘 | 中间结合与舍入变化 |
| high 恢复与 W_o 折叠 | 拟合后生成 BF16 有效权重，非 P 直接产生投影后行 | 省去恢复结果再过 W_o 的独立中间表示；P64 仍调用原 W_o |
| FFN 16 模式规范变换 | 对近似底座及恢复项成对变换 | 实数恢复恒等；BF16 权重重编码可产生尾差 |

完整块调度实际使用原 Wan phase 的 norm、Q/K/V/O、cross-attention、调制、RoPE 和原生 FFN 教师。所有子集使用原全局 token ID，P64 顺序也保持原定义。完整块输出可写后，下一层才建立 K/V。因此这一旁路对应真实图的依赖重排，并不假定跨位置 self-attention 可以仅由同一 tile 输入决定。

`DeferredLayerNormLinear` 另提供延迟状态 `B+FU` 的归一化后线性作用：用小矩和已投影的基计算消费者输出。该工具已实现并验证，完整运行器尚未在所有后继非线性之间携带这一表示；当前交付为组件 API。

### C. 改变表示、需要独立质量证据的压缩

**FFN** 使用显式 `A0≈LR`，将历史 u0 替换为 `u0L` 和 `u0J`，默认另留精确 u0P。新 port 从分配阶段就省去全场 u0；准备阶段省去完整 GPU A0。y0 和选中 z0 仍按原空间保存。因子源 A0/J/V 哈希必须匹配，推理过程中没有 SVD。

离线 SVD 编译器已实现，它优化权重 Frobenius 初始化目标。恢复感知可微损失另已实现：将底座因子误差经过真实 GLS 与核修正后计算，支持对因子及规范参数求导。它通过了差分及梯度核验；本轮没有足够生产轨迹执行恢复感知优化，因此没有把 SVD 初始化标为最优因子或确定质量的 rank。

**low attention** 联合编码 `Z=[C,A]`：

\[
U=ZE,\qquad G_t=\begin{bmatrix}\operatorname{diag}(\sigma_t)\\-\Gamma_{t,1}\end{bmatrix},\qquad
ZG_t\approx U(E^\top G_t).
\]

C 为 128 通道，A 为 31 维读出特征；按头历史宽度为 159。运行器保留原 rho、精确 P64 C/A、代码和基，并释放非 P 的完整 C/A 与旧 tuple。当前读出形成后，只把小型动态解码器作用到代码；`source_order` 模式提供先解码再读出的数值对照。

配对基编译器使用真实输入的 `(Z_s,G_s)` 而非独立边缘统计：

\[
\mathcal L(E)=\frac{\sum_s\|Z_sG_s-(Z_sE)(E^\top G_s)\|_F^2}
{\sum_s\|Z_sG_s\|_F^2},\qquad E^\top E=I.
\]

可选输出投影后目标先合并 head 再作用 W_o，以保留跨头交叉项。实现包括自动微分、正交约束下的切向梯度、QR 回缩与回溯；合成组件测试验证目标与优化过程。真实压缩秩和生产基由显式 CAL 资产决定，本轮没有从合成测试选定它们。基存储误差与代码量化误差在记录中分开。

## 2. 开关与组合已接入现有生成器

交付 12 个 JSON：原参照、分阶段 FFN、单遍锚点、attention 分块、整块分块、high 折叠、投影预合成、FFN 因子模板、attention 联合基模板、图重写组合、图与状态组合、FFN 规范变换模板。

`09_combined_graph.json` 组合分阶段 FFN、单遍 low anchor 和整块流式；high 折叠、投影预合成与压缩保持独立选择，便于将浮点变化与表示误差分别归属。`10_combined_state.template.json` 在图组合上接入显式外部因子与基。默认 tile_rows=8192，FFN 选中通道仍为 2560。

入口 `ffnds_optim.run` 复用实际归档 `Experiment.render`。初始化核对冻结源文件、SDPA 后端、原 transformer 和 CAL manifest；运行时沿用四步 HHLL、40 层和固定 P64。无优化／仅 FFN 优化时使用原 attention adapter 本身。各次请求输出配置指纹、代码与资产哈希、站点计数、状态大小及原生收据；新的组名避免覆盖旧结果。

运行命令、资产 schema 与选项依赖见[完整接口说明](../../../tools/ffnds_optim/README.md)。本轮没有提交或排队学校 GPU 作业。

## 3. 实际完成的验证

| 验证层次 | 本轮执行结果 | 支持的判断 |
|---|---|---|
| CPU 自动测试 | 50 passed，1 个解释器用例单独运行 | 张量轴、公式、probe、原 token ID、源码身份、资产编译、状态生命周期及模块集成 |
| Triton FP32 解释器 | 1 passed，覆盖两个实际 kernel | kernel 的索引、分区、尾块及 selected 子查询与独立 oracle 一致 |
| BF16 A100 sm80 编译 | 2 个 kernel 成功生成 PTX/cubin | 生产形状可编译；每个 kernel 静态共享内存 40,960 B |
| CUDA 环境检查 | `PENDING_CUDA`，0 次 GPU 执行 | 本地没有 CUDA 设备 |
| Wan 完整前向／视频／评分／计时 | 本轮均为 0 | 新配置尚无端到端质量、速度或峰值结果 |

CPU 环境为 Python 3.12.14、PyTorch 2.15.0+cpu、Triton 3.9.0。正式 kernel 编译形状为 N=31668、H=40、D=128、TOP=31；tile 为 BM32/BK64，4 warps、2 stages。Triton BF16 解释器 dot 在当前版本会把 uint16 底层数组传入 NumPy matmul，因此解释器核验使用 FP32；BF16 当前证据是实际 sm80 编译，真实设备数值另待核验。

完整块测试实际连接 root `AttentionSession/HitAttention`、`bind_streamed_blocks`、Wan adapter、`AttentionReadout` 和 `FFNStaged`；high、low、high-fold 均对照独立完整图。测试发现并修复了非 P tile 在输出折叠中重排空 probe 的问题，并验证非 P 不重复调用 W_o。

其他定向核验包括：乱序／非连续位置、RoPE 就地写入保护、单行尾块、全 P 边界、多请求缓存复用、异常时恢复 hooks、无隐藏完整历史引用、因子身份不匹配、规范变换恒等、恢复后误差等式，以及配对损失与可微梯度。

## 4. 本轮形成的具体进展

核心实现已经从“模块输出完成后再优化”推进到“先取得恢复所需的小型当前观测，再按后继消费者执行”：P64 为拟合提供原生信息，非 P 才走分块／折叠／压缩读出。精确当前观测、历史表示和后继作用在接口上可分别替换。

缓存收益也有实际分配依据：FFN 因子模式直接不分配全 u0、不上传完整 A0；attention 联合模式直接释放完整非 P C/A。整块流式主要缩短临时张量生命周期，压缩模式主要减少持久历史；两者通过独立配置组合，回执各记其身份。

目前可肯定的是：这些构造已成为可调用、可组合、有数值核验的实现。它们在 A100 上的实际节省仍取决于新归约内核效率、重算与 gather 成本、合成权重成本以及所选 rank。此前 50 源质量与 1.1672× 完整请求加速属于原冻结方法；本记录没有把它们迁移为新优化的结果。

## 5. 证据索引

- [机器可读身份与范围](IMPLEMENTATION_RECORD.json)
- [CPU JUnit](evidence/CPU_TEST_RESULTS.xml)／[解释器 JUnit](evidence/TRITON_INTERPRETER_RESULTS.xml)
- [attention 实现说明](evidence/ATTENTION_IMPLEMENTATION.md)／[验证汇总](evidence/ATTENTION_VALIDATION.json)
- [编译结果](evidence/ATTENTION_KERNEL_COMPILE.json)／[CUDA 状态](evidence/ATTENTION_CUDA_STATUS.json)
- [FFN 实现说明](evidence/FFN_IMPLEMENTATION.md)／[FFN 原始验证](evidence/FFN_IMPLEMENTATION_VERIFIED.json)
- [完整块实现说明](evidence/BLOCK_IMPLEMENTATION.md)
- [冻结源码身份](evidence/SOURCE_IDENTITIES.json)／[本次文件清单](MANIFEST.json)
