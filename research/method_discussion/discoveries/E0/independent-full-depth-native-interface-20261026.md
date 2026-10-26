# 独立原生接口审计：双轨可按 block 交错，不能只复制 scheduler 指针

作者：`full_depth_source_audit`；日期：2026-10-26；仓库基线：`f5622ed4fd2bce7d2b964ba51f3ce2b3dd62d7db`；阶段：`E0_OPEN_DISCOVERY`。

**新结果（源码推导，未执行）。** 当前离线 Wan 的 `infer_block(block, x, pre_infer_out)` 确实提供单 pipeline、双 scheduler、逐 block teacher/student 交错的可实现切口。冻结配置有 block offload，故两轨必须在**同一当前 GPU block buffer**内执行，然后只进行一次 `swap_blocks()`；直接用 CPU `weights.blocks[l]` 或每轨各 swap 一次都不成立。四种原生 FFN 求值和完整 M 的有符号统计可当场归约，不必保存全部 160 位的两轨输入，也不必重跑 160 条后缀。但当前没有该双轨实现，也没有可信的峰值显存上界，不能由本记录宣称可运行。

另有一个需更正的实际数值边界：`prepare(dtype=float32)` **不保证随后四步 latent 始终 FP32**。继承的 `BaseScheduler.step_pre` 在当前公开可核对的 BF16/None 配置下会 cast latent 为 BF16；`step_post` 虽在 FP32 中运算，末尾 cast 回输入 latent dtype。旧审查中“此路径不是每步压回 BF16”的无条件叙述遗漏了 `step_pre`。本记录给出静态预测，并未把它伪称为已采集的逐步 dtype 迹线。

本工作只读源码、现有 JSON 标量、官方文档，执行 AST/整数元数据核对，仅新增本文件。没有 SSH、模型加载、GPU、拟合、科学张量计算、新视频或实验授权。实时研究维持暂停。

## 1. 先明确本次实际算子和四个不同问题

teacher 指原 dense D-Q/HiF4 参考，student 指固定 affine54 部署 bank，不指未量化 FP32 teacher。以原 caller 产生的调制输入 `U` 为起点，其实际函数为

```text
q1 = BF16(HiF4_QDQ(BF16(FP32(U) / s1)))
a  = BF16_linear(q1, selected_W1, b1)
h  = caller_GELU_tanh(a)
q2 = BF16(HiF4_QDQ(BF16(FP32(h) / s2)))
yS = BF16_linear(q2, V, bias) + BF16_linear(q1, A)
```

54 个原始 group64 给出宽度 3456；`A` 是固定 5120×5120 BF16 矩阵。最后的两次 linear 与加法顺序按 `tools/e0_reconstructed_runtime.py:27–41` 执行，不能擅自融合为实数式。原 norm、调制、caller GELU、gate/residual 不在该 runtime 内。dense 使用全部 13824 hidden 的原 Q1/W1/GELU/Q2/W2。bank 中的新 V/bias/A 是新 BF16 参数，不能叫原 HiF4 原权重。

本次要测而尚未测的四件事是：

1. **部署数值**：同一实际输入上，native partial 与预先明确的连续诊断如何不同。四处 full-width neutral 不能证明全部 160 位 partial 数值。
2. **前缀失配**：native `fD(U_D), fS(U_D), fD(U_S), fS(U_S)`，其中 `U_S` 来自真实 student 前缀，不能由 teacher trace 加局部噪声代替。
3. **有符号传递**：FFN 输出差怎样经过真实 gate/writeback 和后续 attention/block；双轨本身只观察累积关联，不能隔离单一 site 的后缀因果贡献。
4. **四步状态**：各轨自己的 head flow 和 scheduler latent 演进，包含真实 dtype cast 与更新顺序。

图锚点为 `G02`（专家）、`F03/F04/F09`（实际 Q1/W1/W2）、`F12/G20`（原生门控残差）、`F13/G21`（下游）、`P22`（已完成失败视频）、`P24`（未测全深度传递）、`D09`（视频/成本）。新增双轨观测边是 `conditional`，输出归因关联是 `posthoc`；不画成原 dense 已执行事实。源中 head/scheduler 是下述准确 API，不把文档索引当 runnable graph。

## 2. 原生 API 与时序

| 层级 | 实际调用与可变行为 |
| --- | --- |
| runner | `DefaultRunner.run_segment` 每步依次 `model.scheduler.step_pre(k)`、`model.infer(inputs)`、`model.scheduler.step_post()`；现有 API 只有一条 scheduler 轨。末尾还会删除 `runner.inputs`。 |
| expert wrapper | `MultiDistillModelStruct.get_current_model_index()` 按 `step_index<2` 选 high，随后选 low；写 `cur_model_index`。`MultiModelStruct.set_scheduler(s)` 广播至两个专家。 |
| scheduler binding | `BaseTransformerModel.set_scheduler(s)` 同时设置 model、pre、transformer、post；transformer 的 setter 还写 `s.transformer_infer=self`。只赋 `runner.scheduler` 或 `model.scheduler` 会留下旧子模块引用。 |
| pre | `WanPreInfer.infer(weights, inputs)` 从 `self.scheduler.latents/timestep_input/infer_condition` 读取；patch embedding 后产生自己的 `pre.x`。时间与文本 embedding、grid、RoPE、`adapter_args` 随 pre 对象传入。 |
| transformer | `infer()` 设置 `self.cos_sin`，重建 cu_seqlens，reset attention，跑 blocks，再 `infer_non_blocks(weights,x,pre.embed)` 做 norm/head。 |
| block | `pre_process` → self-attention → cross-attention → `infer_ffn` → `post_process` → 可选 adapter，见 `transformer_infer.py:171–201`。 |
| FFN 入口 | `infer_ffn(...):367` 先 `x.add_(attn_out)`，然后 norm2/modulation，再 `.ffn_0.apply(U)`、caller GELU、`.ffn_2.apply(h)`。因此它的函数入口 x **还不是**最终 FFN residual 输入。 |
| FFN 写回 | `post_process(x,y,gate,pre):418–424` 才做 `x.add_(y*gate.squeeze())`；不同 sensitive dtype 分支用显式 cast 和新 tensor。此前 cross-attention 也可能原地加 self-attention residual。 |
| head/post | `infer_non_blocks` 对 norm 输出原地调制，然后 head；`WanPostInfer.infer` unpatchify 后返回 `[u.float()]`。调用者必须取 `[0]` 并写各轨 noise 字段。 |
| scheduler post | Wan22 `step_post:80–89` 为 FP32 先减 `flow*sigma`、非末步再加 `flow*sigma_next`，最后 `.to(self.latents.dtype)`；不能用单个 `-h*flow` 替换并声称逐位相同。 |

**当前没有 seq/CFG 广播。** 合同是 `enable_cfg=false, seq_parallel=false, parallel=false, use_image_encoder=false, torch_sdpa, rope_type=torch, cpu_offload=true, offload_granularity=block`。native `TorchSDPAWeight.apply` 不保存跨调用 KV；dropout 默认 0。不能把这个判断外推到 caching/稀疏 attention、CFG 或 sequence-parallel 分支；后者有 all_gather、padding 和 rank 顺序，需重新设计。

**offload 是真正必须保留的状态机。** `infer_with_blocks_offload:39–62` 执行 init 当前 buffer、预取下一 block、在 `offload_manager.compute_stream` 使用 `cuda_buffers[0]`、最后 swap。`WeightAsyncStreamManager.swap_blocks:91–100` 同步 load/compute stream 后交换两个 buffer。teacher 和 student 共享该 buffer 是权重共享，不是隐藏状态共享。下一 block 的预取可以保留；当前 buffer 在两轨及 observer 完成前不能被复用。

## 3. 一个具体可实现的接口（新设计，尚无实现）

接口可以分成两个新对象，避免用递归 `pipeline.generate` 偷渡第二轨：

```python
PairedWanStepController.run_step(inputs, dense_scheduler, student_scheduler)
PairedBlockObserver.run_block(infer, current_gpu_block, x, pre, branch)
```

前者需要替换/包装原 `run_segment` 的单轨循环；后者围绕已有 `infer_block` 与 `runtime.apply` 安装观测。两者都应在独立 driver 中存在，不能只对现有 `generate` 多传一个 scheduler。建议时序如下：

```text
编码用户条件一次，冻结 inputs 身份和 actual grid
分别构建两个 Wan22StepDistillScheduler(full effective config)
prepare 两轨；检验初始 latent 同 dtype/shape/值且不共享 storage
for k in 0..3:
    分别调用 D.step_pre(k), S.step_pre(k)，记录调用前/后 dtype
    选中唯一专家 high/high/low/low；完成原 pre/head 权重上卡生命周期
    bind_scheduler(D); preD = native_pre(inputs)
    bind_scheduler(S); preS = native_pre(inputs)
    校验 grid、conditioning、embed0/gate 身份与形状
    初始化原 attention 元数据/状态（当前无状态 SDPA 限定）
    for l in 0..39:
        原 manager init/prefetch；infer.block_idx = l
        with 原 compute_stream，显式 expert/step context:
            bind_scheduler(D), bind_pre_metadata(preD)
            xD = observed_native_infer_block(buffer0, xD, preD, branch=D)
            bind_scheduler(S), bind_pre_metadata(preS)
            xS = observed_native_infer_block(buffer0, xS, preS, branch=S)
            全M/预冻结空间层归约，保存少量向量后释放当前site临时量
        原 manager.swap_blocks() 一次
    各轨原 infer_non_blocks + post_infer；写各自 noise_pred 系列字段
    各轨原 step_post()，记录 flow/latent 差及 dtype/shape/hash
    完成原 pre/head 权重下卡生命周期
```

`bind_scheduler` 必须调用相应 native setter，不能把两个独立 scheduler 变成一个浅复制对象。初始随机数用两个独立、相同 seed 的 generator，或者以同一已生成初始 latent 的 clone 建立两轨；不得把同一个已前进的 generator 连续交给两次 prepare 而假设噪声相同。timesteps/sigmas 可按只读约定相同，latents/noise_pred/generator/caching lists 是独立对象。prepare 一次后跑完整四步；不能每步 reset 随机状态。

### 3.1 在 caller 内取得四种函数，保持真实主轨

`runtime.apply(mm,U,expert,step_index,block_index,module_name)` 是比重算 `infer_ffn(x,attn)` 更安全的求值切口，因为进入 `.ffn.0` 时 norm/modulation 已经实际发生。建议 dispatcher 保存进入候选 patch **之前**的 `dense_apply` bound method，并为 shadow 提供纯函数 `eval_approx_pair(U,phase,pack)`，严格复写现有候选 Q1/up/GELU/Q2/down/affine 顺序。

- branch D 的 caller 消费 native dense 输出；另做 shadow approximate，产出 `yDD,ySD`。
- branch S 的 caller 消费 native approximate 输出；另做 shadow dense，产出 `yDS,ySS`。
- shadow 直接使用显式 mm/pack 和已捕获的 `dense_apply`，不可递归经过 dispatcher，也不可复用候选唯一 `pending` 槽。原 `ReconstructedRuntime.pending=(key,step,q1)` 是单对 up/down 的状态，不是可重入四分支容器。
- 主轨 caller 的 GELU、gate 和 writeback 仍执行原方法；shadow 不能修改主轨 U/x/attn、返回值或 scheduler。`fS` 的主轨和纯 shadow 实现之间，还需未来同输入 bitwise 控制，不能把本记录的伪代码当该控制已通过。
- `.ffn.0` 所见 U 在 Smooth1/Q1 **之前**。四个函数可以精确定义为包含 Q1 的 `f(U)`；若报告以 `F03=q1` 为共同输入，须另核对原 record 与 pack 的 s1、native Q1 输出同位一致，保存该边界的真实 dtype/shape/digest。不能把 pre-Q1 U 直接标为 post-Q1。

hook 的上下文有真实陷阱：adapter 仅在 `MultiDistillModelStruct.infer` wrapper 设置 `_expert_ctx/_step_ctx`，`infer_block` wrapper 设置 `_block_ctx`。新的 controller 绕过 `MultiDistillModelStruct.infer` 后，必须自行用 token/reset 方式明确设置前两个 context，并设置 `infer.block_idx=l` 让原 block wrapper 标 block。缺任何一个，`MMWeight.apply` 会直接走原实现，甚至**静默绕过 HiF4 runtime**；并不会自动报 missing context。

### 3.2 原生 writeback 必须以 clone 做 shadow

`infer_ffn` 执行完后 x 已是 `r=x+attn_out`。在 `post_process` wrapper 调原主写回前捕获该 r；针对同轨 shadow 使用 `native_post_process(r.clone(),y_shadow,gate,pre)`。绝不把同一 x 连续传给 dense 与 approximate post，因为默认路径是原地 `add_`。两轨的初始 pre.x/storage 与 `adapter_args` 也须独立。

可当场得到四个有限写回 `RDD, RSD, RDS, RSS`。对 FP32/更高精度**观测差**定义

`e_op = ySS-yDS; e_shift = yDS-yDD; e_total = ySS-yDD`。

记录 `||e_op||², ||e_shift||², <e_op,e_shift>`，以 `||e_total||²` 检验有符号分解；统计按全 M、frame/spatial strata 归约，不能先对行平均再取范数。writeback 对同样四个**实际有限输出**重新做差；例如

`RSS-RDD = (RSS-RDS) + (RDS-RDD)`。

这里第二项已经含 r 的前缀漂移以及必要时 gate 的变化，不能再叫“纯 FFN input shift”。保留 raw/gated FFN 与 residual 三种边界。此观察式在数学实数层面是望远镜式，在有限归约中应记录 closure tolerance；不主张 BF16 差向量可逐位重加闭合。随后两条 x 继续各自下一 block，得到实际累积 hidden 差；没有额外后缀因果作用已测的含义。

四种 native 求值仍不自动完成“continuous/native mismatch”对照：后者需冻结**同一 U/q1、同一 bank、明确连续公式与数值精度**的辅助 evaluator。可保存全 160 位预冻结行的 U/q1/输出供后续获准分析；该项不可用旧 teacher panels 不同输入替代。

## 4. 必须隔离的状态与不能假装纯函数的部分

| 状态 | 单 pipeline 双轨要求 |
| --- | --- |
| scheduler | `latents, noise_pred, noise_pred_cond/uncond/guided, step_index, timestep_input, infer_condition` 独立；generator 状态独立。当前 step-distill 不用父类 UniPC `model_outputs/last_sample` 更新，但不能复用它们冒称支持任意 scheduler。 |
| infer 对象 | `scheduler, block_idx, cos_sin, *_cu_seqlens_*` 随当前轨绑定或校验严格相同。当前同 grid 时可共享只读 RoPE/cu_seqlens 值，不能依赖未验证的 shape 广播。 |
| pre 输出 | 各自 x/storage、各自 adapter_args/conditional_dict；同时间条件可验证相同，但不能让浅复制 x 使 teacher 被 student 原地修改。 |
| offload manager | 当前/预取 GPU buffer、need_init_first_buffer、stream、swap 次数由一份串行 controller 独占；phase offload 的 `phase_params` 不支持按此方案直接交错。 |
| adapter | `_ACTIVE_RUNTIME/_ACTIVE_OBSERVER` 是进程全局；contextvars 只处理 expert/step/block 标签，不隔离全局 runtime。单进程只安装一个 dispatcher，不启动并发线程互相切 hook。 |
| runtime/cache | `_resident_key/_resident_weights`、counters、候选 `pending` 都可变。四个求值要可归因计数；shadow/trace 的成本不得进入部署速度收益。 |
| attention/adapter/compile | 仅对已核对无历史 SDPA/NoCaching、无额外 post adapter、非 compile 分支作设计。若实际 config 有缓存、mxfp8 fused FFN、LoRA动态修改或后处理 adapter，先 fail closed 后重审，不能假定是当前路径。 |

**dtype 的可复核更正。** `config/worker.env:12–13` 只读取到 `DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=None` 两字段；`envs.py:39–43` 的 None 表示返回 GET_DTYPE。`BaseScheduler.__init__:22` 默认 `keep_latents_dtype_in_scheduler=False`，`step_pre:24–27` 满足相等条件时 cast。当前 Wan22StepDistillScheduler 没有覆盖该 flag；全仓搜索出现 keep=true 的其它 scheduler 不属于当前模型。故当前源码/该配置预测：prepare FP32 → step0 pre BF16 → post FP32算术后 BF16 → 后续同样。旧 driver 的 hash helper 和 post hook 都先 `.float().cpu()`，hash 本身不能证实原 storage dtype。实际实验需分别记录 prepare、每次 step_pre 前/后、head flow、step_post 前/后的 dtype 和 flag；本记录不回写旧 receipt。

## 5. 显存：40.70 GiB 是已知 payload 小计，峰值上界仍未知

下列是现有 JSON 标量与形状的普通整数核算，没有加载 cache/model tensor。当前两来源历史 latent 是 16×21×58×104，对应 patch grid 21×29×52，即 M=31668；新来源实际 grid 必须从 pre 获得。

| 已知 payload | 字节 / GiB |
| --- | ---: |
| 一份 M×5120 BF16 | 324,280,320 / 0.30201 |
| 两轨×160 位全部该输入 | 103,769,702,400 / 96.64307 |
| 一份 M×13824 BF16 | 875,556,864 / 0.81543 |
| 一份 M×3456 BF16 | 218,889,216 / 0.20386 |
| 原 non-FFN decoded resident cache，两专家 | 33,554,432,000 / 31.25 |
| affine54 全80 pack（实际已有记录） | 9,860,730,880 / 9.18352 |
| 当前 block 原 W1+W2 BF16 矩阵 | 283,115,520 / 0.26367 |
| 上述后三项相加 | 43,698,278,400 / 40.69719 |

40.70 GiB **不包含** bias/scale 的额外 device copy、原 block offload 两个 buffer 与下一 block 预取、pre/head/sensitive 原权重、原始/decoded 权重的同时存活、VAE/编码器、attention QKV/SDPA workspace、HiF4 FP32临时、GELU/q2、两轨活跃 hidden、四求值暂存、归约/输出拷贝、allocator 与非 PyTorch CUDA allocations。影子求值串行能降低同时存活量，但这些未知项没有可用的严格总上界。不能用单轨 60,796 MiB 采样峰加减几个 payload 当新方案峰值证明。

现成 `AllWeightResidency` **不是**这个策略：它按 `(expert,module_name,device,dtype)` 永久 retain 访问到的所有矩阵。诊断一旦调用 shadow dense，它会把原 FFN 全部缓存；两专家原 FFN 累计 22,649,241,600 bytes =21.09375 GiB，known weight subtotal 会变为 61.52727 GiB，尚未加其它项。正确新策略至少把 `_weight` 分成 non-FFN 长驻字典与仅当前 `(expert,block)` 的 FFN 字典，在两轨/影子和 stream 消费完成后才淘汰。原 `HIF4CacheRuntime._weight` 每 block 清全缓存也不能单独满足 non-FFN 长驻。

因此判定是：**接口可写，资源可行性未证实**。未来协议必须先冻结内存生命周期/字节账，再在另行授权下记录 allocated/reserved/device physical peak 与无截断失败；现在既不预占 GPU，也不声称 80GB 卡必能容纳。

## 6. 两 GPU lockstep callback：可保持因果，不能靠标量免掉跨轨信息

替代方案是两个独立进程，各用原单轨 runner/自己的 scheduler/runtime；每个 native block callback 以 `(source,seed,step,expert,block,grid,dtype)` handshake，对齐后交换观测。每个进程在本地同输入做其 dense/approx pair，主轨始终返回原分支输出，不把对方观测喂入下一 block。这样仍是两条自由轨，减轻共享 scheduler/global patch/offload 状态风险，代价是两份 pipeline、host/checkpoint/cache、两个 GPU 的资源与同步。

全 M 的 `e_shift`、输入/hidden 差和跨轨 signed inner product 不是两端 norm 的函数；相同 norm 的同向与反向向量已经构成反例。简单精确方案必须传某些完整向量或按固定行块传输并当场归约。只传每端 scalar norm 得不到这些对象。以一份 BF16 M×D 字段、每位传一次为例，160 位合计 **48.32153 GiB/字段**；多个边界会倍增，转 FP32又加倍。分块能限制接收 buffer 峰值，不能消除总传输。抽样/随机 sketch 可减少流量，但必须标为 sampled/estimated，不能冒称 exact full-M signed geometry。

接收 shadow dense 的一端还需要当前原 FFN 权重；不能因它是 student GPU 就忽略该复制。若选择把 U_S 发往 teacher GPU 求 dense，则引入额外 U/输出传输、显式调用和同步；仍须计账。

可使用固定次序的匹配 send/recv 或 batch_isend_irecv，并为异常退出/超时设计双方终止。NCCL 不支持 message tag，故 identity 是应用层 header/顺序检查，不能只把 `(k,l)` 当 NCCL tag。异步发送完成前不可修改源 tensor；compute/offload stream 与通信 stream 要用 events/waits 和 lifetime 管理。此处没有建立通信组、测带宽或运行 callback。

| 方案 | 不同预测 / 决策价值 |
| --- | --- |
| 单 pipeline、逐 block 交错 | 两轨共享原权重/当前 buffer；跨轨差在本地精确归约，避免多字段跨 GPU 流量。新风险集中在 scheduler binding、全局 dispatcher、buffer 时序和更高临时显存。 |
| 两独立进程 lockstep | native单轨流程改动较小，状态隔离更强；新增跨轨full-M信息传输、重复模型/host内存、同步等待。不能由“两个GPU并行”推断更快。 |
| teacher全量存盘后student重放 | 无lockstep依赖，保存/读回约96.6GiB双轨输入量级且还缺其它边界；容易把保存teacher输入替代student真实prefix。只留预冻结向量行可做采样佐证，不能满足全M归约的全部需求。 |
| 每site重跑后缀 | 可做选定site有限因果干预，费用显著；160位主扫描不需要它。双轨扫描后再按独立冻结strata做少量后缀，才回答传递机制。 |

### 6.1 工作量不能写成免费 observer

每site，两轨主路径已有一份 dense（2 GEMM）与一份 affine54（up/down/affine共3 GEMM）；补两个 shadow 又加2+3 GEMM。故 native FFN诊断共 **10 GEMM、8 QDQ/site**，不含连续 surrogate、归约、写回副本或额外 neutral。两个前缀的 non-FFN 还照常执行。按当前每轨8个non-FFN target linear/site，若全部分支均为既有路径，一来源160site的完整双轨账本应为4160 GEMM、3840 QDQ；这是源码计数预测，未实测，新增实现必须按各角色实际计数核对。trace wall只能叫采集成本，不当候选部署 wall。

## 7. 本次实际静态验证与最小下一步

已完成：用 Python3 标准库 AST 读取/解析上述真实文件，无任何模型模块 import；核实 `infer_block` 的参数和调用序、`infer_ffn/post_process/infer_non_blocks`、block-offload方法及 scheduler setter；全文源码搜索 scheduler dtype flag；只读现有 JSON 中 resident/pack bytes；整数复算 M×D/H 和160位字节；SHA256绑定如下。系统 `python` 是Python2，原图工具用它会 SyntaxError；改 `python3 tools/graph_agent_bridge.py inspect F08` 与 `trace G20 G21` 成功。本错误不涉及模型或科研实验。

**最小下一步是写一个不导入 torch 的静态接口 validator 与待授权实现草稿**：拒绝非冻结 NoCaching/SDPA/block-offload/无CFG/无seq配置，检查需要的API、四角色/160site身份、dtype前后字段、bounded residency规范及状态恢复；验证器只证明协议/接线结构，不证明数值中性、显存可行或视频质量。本文本身给出了可审查接口，尚未交付/运行该新driver。

未来真正授权时，先由dense/dense identity控制检查两轨每步latent及原dense reference身份，确认 observer不会改轨；再检验actual affine主轨与原候选实现一致、shadow原生接线中性、所有160位/四角色计数。上述都属于未来模型 scope，当前未执行。每条实际新生成视频还必须调用全帧 evaluator 和 SHA bundle gate；official/human未跑保持 NOT_RUN、quality_accepted=false，不能以trace完整宣称画质通过。

## 8. 实际 Web 检索与来源绑定

2026-10-26，经 web search 检索 `PyTorch CUDA streams synchronization record_stream memory_stats` 与 `torch.distributed send recv`，随后亲读固定 PyTorch2.10 官方 CUDA/Distributed/Memory 文档相关段落。该任务是原生接口核对，未扩张为新FFN算子/论文创新主张。最接近的实现竞争者是本仓库原生offload循环与原双轨分进程方案，而非一种新的FFN近似论文。

- [PyTorch2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html)：跨stream需要明确依赖与tensor存活；CUDA event/同步用于计时。它改变了设计：观测必须在原compute stream归约，或显式加入stream等待，不能在default stream读一个“Python调用已返回”的tensor。
- [PyTorch2.10 distributed](https://docs.pytorch.org/docs/2.10/distributed.html)：send/recv的匹配与NCCL tag限制、异步发送源tensor不得过早修改。它使“双GPU callback只交换标量即可”降级为不满足signed全M目标的方案。
- [PyTorch2.10 CUDA memory](https://docs.pytorch.org/docs/2.10/torch_cuda_memory.html)：PyTorch profiler看不到全部外部分配，NCCL是例子。因此payload/allocated/reserved与physical peak分开，当前总上界标未知。

下列SHA为本次亲读文件的实际字节身份；`.runtime/code` 未跟踪文件也按实际内容绑定，不把Git HEAD当其替代。表中 L 前缀代表 `.runtime/code/LightX2V/lightx2v/`，H前缀代表 `.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/`。

| 源码 | SHA256 |
| --- | --- |
| L `models/networks/wan/model.py` | `f0b0e9904d9708ff67412d8c98cb3f384b86426bb96ebf84d17e439e7a8580ff` |
| L `models/networks/wan/distill_model.py` | `d4c9e76b9046ec9fc8d83bc5569e44b69efbcf7d97b30d5152d85c16051f72cc` |
| L `models/networks/wan/infer/transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| L `models/networks/wan/infer/offload/transformer_infer.py` | `c66f594739c4737f4d77e27db31ac990a82c539ada1d2ec267e0203b90a5fe25` |
| L `models/networks/wan/infer/pre_infer.py` | `f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5` |
| L `models/networks/wan/infer/post_infer.py` | `6567c439e2f16f322a9f10cd2ca326d3f924e9da75b3ae2ae442ec1327a327a2` |
| L `models/networks/wan/infer/module_io.py` | `3ed1a7016f5f6f8c1a0d1c25e38bcacf82004cafcb55dd6480c3cccef6ba876a` |
| L `models/networks/base_model.py` | `e171498c41484a9e2ec86d0e3259f4c7cd3e21a62ea87d75fa05ec75ba6db5e9` |
| L `models/runners/wan/wan_distill_runner.py` | `26a2a503137b0125cf4a2f8b2c9fc5dae716121c6e600b7ce85517af5ee62dc3` |
| L `models/runners/wan/wan_runner.py` | `e12bf49dfa76a3c5bde0d284952d7f8c2dcee2043737b310d51d46e7ff3d0f69` |
| L `models/runners/default_runner.py` | `4f8e4295e5301b1465c4e320641c98855f2f7e6e5734e714d17af1acb2fd4f18` |
| L `models/schedulers/scheduler.py` | `0b43c8452ee734fffbd0a753f8176a5b530980723aa54981b34a8070c68db7f7` |
| L `models/schedulers/wan/scheduler.py` | `c86af28a3b8d709692fb3a88d67ec441a77624929d397c216e0106f1c0142f4a` |
| L `models/schedulers/wan/step_distill/scheduler.py` | `2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db` |
| L `common/offload/manager.py` | `dbd68ff7af9626f128bf74baed97885f7610853067dd184010d8f3325d8aa5c0` |
| L `common/transformer_infer/transformer_infer.py` | `b266c1e1b08495614f6a8c7eb34af2a91d1a4a99992581143c89de8db2fdd2e0` |
| L `common/ops/attn/torch_sdpa.py` | `ae87ba65009c454f62e0e558cf821ee2bada1420928f408e5a7e26760984cf7d` |
| L `utils/envs.py` | `68c2f49686cb58559e8f1beaa0641b1141b5639083fe4ec3e8df771395c3189c` |
| H `lightx2v_wan_distill.py` | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |
| H `lightx2v_wan_distill_contract.json` | `f6104c256b9fb881fe33fda2707095cd8591c4010f6fff1be2cffd2b648d56aa` |
| `tools/e0_reconstructed_runtime.py` | `957588f5dc3b32334982b6247064ddece1d21ab627267eb717bc72fb31003872` |
| `tools/ffnds_all_weight_residency.py` | `d19c14f57b17426f6f3497c2af1fd726c1fb9756a46cb559d60108e8103147d4` |
| `tools/run_e0_reconstructed_videos.py` | `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005` |

本记录不修改 ACTIVE、registry、QUEUE、旧独立记录或冻结运行收据；不登记新模型作业。当前可提交审查的是上述接口、精确边界与阻断条件。
