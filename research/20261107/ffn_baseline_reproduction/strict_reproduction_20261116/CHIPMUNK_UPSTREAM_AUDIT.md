# Chipmunk 上游 Wan 执行链与最小复现边界审计

审阅时间：2026-11-15 16:47 UTC（北京时间 2026-11-16）。本次阅读原论文、上游仓库、release和issue，并复核一项纯Python计数接口检查；没有安装、导入GPU模块、模型生成或指标重算。

## 1. 对用户问题的直接回答

**Chipmunk 确实在 Wan2.1 上做过原始视频实验，也有官方 Wan2.1 T2V 执行入口。此前说“原论文的视频配置没有启用 FFN 稀疏”不等于“作者没有在 Wan 上做过”。**

但是，原论文附录 D.1 明确其 Wan2.1/HunyuanVideo 最终视频配置只使用 sparse attention；MLP 稀疏用于 FLUX.1-dev。最新默认分支的 Wan 集成也只挂接 SparseDiffAttn，FFN 仍是普通 Sequential。这与我们想比较的“Wan2.2、I2V、双专家、蒸馏四步、FFN-only”不同。

**原作者已有通用 SparseDiffMlp 和 A100 可用的官方 Triton 路径，应该首先直接复用这些模块建立数值 reference；没有依据把“在 A100 上运行”作为必须自行重写 FFN kernel 的理由。** 但这些模块不是一个对当前 Wan2.2 宿主即插即用的完整 FFN integration，需要边界明确的小适配，并对有选择余地的改动单独消融。我们目前没有在该机器上执行它们，不能把源码支持等同于已经跑通。

旧 Global-P64 改了 proxy、支持粒度、选择器与数值路径，不能当作作者原算法的复现基线。已有 AUTHOR_QUANTILE_RANDOM 历史汇总也不能替代尚未找回的实际 producer、编译 selector、neutral 和逐视频身份。若这些记录不能闭合，就应重新建立作者 reference 与最小 Wan FFN 端口；旧结果保留为有明确身份的历史变体结果。

## 2. 版本重新核验：最新默认分支正是此前固定版本

本轮重新读取 GitHub repository metadata、branches、release、issues，结果如下：

| 对象 | 本轮查得的身份 |
|---|---|
| 当前默认分支 | master，**不是 main** |
| 当前默认 HEAD | b5e8d1c3da6bddc7ca9007c48e91b82087c11490 |
| HEAD commit 时间 | 2025-09-08T23:27:23Z，message: Update README.md |
| 与此前审计 pin 的关系 | 完全相同，不是依赖过期副本的推测 |
| 名为 main 的分支 | 177fdefc4803e621fd8a02aad491812e9d1ecc94，2025-05-28，较旧 |
| 官方 release | v1.0，2025-06-16；列出 FLUX.1-dev、Mochi、HunyuanVideo、WAN2.1 |
| 论文版本历史 | arXiv 当前仍只列 v1，2025-06-03 |

v1.0 release 明确列出 Hopper ThunderKittens 与 Ampere/4090 Triton attention+MLP kernels。README 更新记录也说明跨架构支持。不能只引用 README 早期“kernels for Hopper”的旧段落，忽略同一仓库的新记录和实际 setup.py 分派。

另外直接读了较旧 main 和 evals-wan-chipmunk 分支的 Wan model/config：它们虽然 import SparseDiffMlp，但同样没有把它接入 FFN，注册仍为 is_mlp_sparse=False。这个检查不等于宣称每条实验分支都经过运行验证。

审计工作目录取回39份默认HEAD源码/配置/说明，其字节逐一核对Git blob SHA，39/39一致。本发布包不重复vendor这些作者文件；[UPSTREAM_WAN_EXECUTION_SOURCES.json](UPSTREAM_WAN_EXECUTION_SOURCES.json)保留固定URL、Git blob SHA与内容SHA256，可按固定版本重建。额外分支同样按清单中的固定URL解释。

## 3. 官方 Wan T2V 的真实执行链

以下行号对应清单所指的固定上游原文；不是声称本发布目录包含upstream/源码子目录。

| 阶段 | 具体源码 | 实际行为 |
|---|---|---|
| 官方入口 | examples/wan/run.sh L1–11 | task=t2v-14B；Wan2.1-T2V-14B；832×480；seed42；指定 configs/cuda-test/chipmunk-config.yml |
| 配置加载 | examples/wan/generate.py L85–88、L242–248 | 从 --chipmunk-config 加载 YAML，并 deep-update 全局配置 |
| 构造 T2V | generate.py L333–356；wan/text2video.py L85–109 | 构造 WanT2V，加载修改过的 WanModel |
| 静态mask与计数初始化 | wan/text2video.py L231–253 | 按预设 seq_len 映射 seq_shape；设置 local_heads_num；initialize_static_mask；reset LayerCounter |
| 每步调用 | wan/text2video.py L255–268 | cond/uncond 两次调用，均传 inference_step；后组合 CFG |
| 模块注册 | wan/modules/model.py L140–143 | 只对 self-attention 建 SparseDiffAttn；LayerCounter.build_for_layer(is_attn_sparse=True,is_mlp_sparse=False) |
| Attention 调用 | model.py L153–167 | 当前 Q/K/V+RoPE，转 BF16，调用 self.attn |
| FFN 调用 | model.py L294–296、L332–336 | 原生 Linear→tanh-GELU→Linear；输入为当前 norm/shift/scale，输出后乘当前 gate |
| 模型布局 | model.py L547、L565、L626；RoPE L71–73 | voxel=(4,6,8)；token和RoPE重排；输出逆排 |
| 层间缓存预取 | model.py L619–623 | 只预取/wait block.self_attn.attn.storage；没有 MLP storage 调度 |
| Attention 内部 | src/chipmunk/modules/attn.py L221–235 | dense时/稀疏时分派，末尾 increment counter、complete_cur_layer |
| 内核分派 | src/chipmunk/ops/attn.py L13–29、L32–65、L67–82 | 根据 attn.provider 选择 torch.ops.chipmunk CUDA 或官方 Triton |

静态几何表还存在可直接核对的不一致：text2video.py L231–238把seq_len=32760映射到(21,45,80)，乘积却为75600；把21840映射到(21,30,52)，乘积却为32760。不能把该硬编码表视为任意Wan形状的正确来源。复现应先核真实patch grid、seq_len与mask形状的一致性；若修正为从真实grid导出，保留原代码与修改diff，并把它列为必要几何适配。这个源码问题不能单独证明论文表格使用了同一路径或已失效。

### 3.1 实际 run.sh 配置不能与根 YAML 混称

默认 run.sh 指向 configs/cuda-test/chipmunk-config.yml，内容为：

- num_model_invocations_per_inference_step=2；
- mlp.is_enabled=false；
- attn.is_enabled=true；
- attn.top_keys=.1、random_keys=.01、local_1d_window=.05；
- first_n_dense_layers=2、full_step_every=10；
- recompute_mask=true、should_compress_indices=true；
- provider=cuda；
- attention cache offload、step_caching 开启。

configs/triton-test/chipmunk-config.yml 具有相同配置，provider=triton。根 examples/wan/chipmunk-config.yml 的 attn.top_keys=.15，**不是上述 run.sh 实际传入的文件**。重做时须绑定实际 CLI/config，不能只记录仓库里“某个默认配置”。

patchify.is_enabled=false 也不代表 Wan 没有 voxel：Wan model 的 voxel 调用是直接写在 forward 中的。src/chipmunk/ops/patch.py 是 token patchify/unpatchify，**不是自动寻找并替换 nn.Linear/FFN 的 monkey patch 系统**。

### 3.2 Attention 与 MLP 调度并不相同

LayerCounter.should_do_full_attn_step() 对前两步 dense，随后按 full_step_every；should_do_full_mlp_step() 是 inference_step % full_step_every == 0。不能把 attention 的前两步 warmup 当作 MLP 默认。

Attention 第一整步建立 LSE，第二步获取 column sums/mask 与遗漏输出缓存；之后累加稀疏项。它有自己的 mask、bitpacking、local window 与 cache策略。MLP 的列选择是 preactivation 块均值差，经另一个 CUDA selector 完成。这两个 selector 不能混用名称或参数解释。

### 3.3 原 LayerCounter 的边界需要显式处理

另有一个不能忽略的源代码接口问题。LayerCounter.increment() L41–51先推进坐标，L53–57却在“已经推进到本请求最后一个坐标”时直接reset；该最后一次调用此时还没有发生。独立读代码可确认，40层、1个子模块、4步、每步1次调用时，step3/layer38调用结束后就reset，下一次实际layer39看到step0/layer0。

Root用原类AST原样提取做了纯Python接口轨迹检查，没有torch/CUDA/模型运行。结果保存于 UPSTREAM_COUNTER_TRACE.json，本审阅再次按源码核对了转移条件：

| 注册与实际执行 | 错配 |
|---|---|
| 40层、4步、1 invocation | 最后call159，预期(step3,inv0,layer39)，观察(step0,inv0,layer0) |
| 40层、50步、2 invocation | 最后call3999，同样提前reset |
| 注册80层，但每模型调用实际仅运行某专家40层 | 160次调用中120次坐标与真实scheduler不符 |

Wan2.2双专家不能把两套40层都登记进一个按“每次模型运行80层”推进的singleton。最小正确端口应由实际scheduler提供step/expert/local-layer context，明确初始cache和请求结束边界；first_n_dense_layers比较专家内local layer，cache key另用expert+local layer。原类保留为来源reference，必要时钟适配单独记录并用轨迹等价检查验证，不能静默修正后仍声称原代码逐位复现。这个结果只证明已读类在上述输入调用序列中的行为，不证明论文已发表运行使用相同代码/注册方式或已因此失败。

## 4. I2V 不是当前官方默认入口已完整接好的路径

作者 issue #2，官方 collaborator SohamGovande 的回复，明确说当时没有 I2V 性能数据，建议：

1. 复制 text2video.py 的 Chipmunk 初始化到 image2video.py；
2. 在采样模型调用传入 inference_step；
3. 测完提交改动和性能估计。

本轮检查最新默认 HEAD 后，上述缺口仍能直接在代码里定位：

- WanModel.forward 在 model.py L515–523 将 inference_step 作为必需参数；
- image2video.py L281–293 的 arg_c/arg_null 不含 inference_step；
- image2video.py L305–311 两次 self.model 调用也没有传该参数；
- image2video.py 未包含 T2V 中的 initialize_static_mask/reset 初始化段。

因此在普通非 USP 这条已读代码路径上，接口本身就不闭合；这是源码检查，没有伪称本轮实际运行触发了异常。I2V 文件或 CLI 选项存在，不足以证明 Chipmunk I2V 已完整集成。

issue #4 有 A100 Wan14B I2V flicker 用户报告；作者建议检查 local_1d_window。issue #6 是2026-06的 A100 GPU/CPU内存报告，读取时无回复。这些说明官方路径仍有实现/配置/硬件边界，但都是 attention 场景的线索，**不能用来证明 Chipmunk FFN 在当前任务失败**。

## 5. 官方通用 MLP 能否直接作为 reference

可以把它作为**优先复用的原作者模块**，而不应先改成我们的选择器/packed算术。

### 5.1 构建、注册和分派

1. setup.py L14–44 区分 Hopper-only GEMM/attention 内核与可跨架构的 copy_indices、topk_indices、mask_to_indices。
2. L58、L90–107 根据实际 CUDA capability 选目标；非Hopper跳过Hopper-specific kernel，编译可移植 indexed-I/O；架构参数取实际 sm。
3. csrc/chipmunk.cu L49–87 注册 torch.ops.chipmunk。selector/copy 在 Hopper 宏之外，A100也有对应注册。
4. src/chipmunk/__init__.py 导入 cuda/triton/ops/util；modules/__init__.py 导出 SparseDiffMlp。
5. src/chipmunk/modules/mlp.py 计算状态/选择器，调用 chipmunk.ops.mlp。
6. src/chipmunk/ops/mlp.py L26–31、L55–70、L100–102 在 mlp.provider=triton 时调用原作者 BF16 mm1/mm2。
7. src/chipmunk/ops/indexed_io.py L4–21 调用 compiled author topk_indices/copy_indices；只设置 triton 并不意味着可省略这些 CUDA utility 的构建。

所以 A100 可采用作者提供的 MLP Triton backend，compiled selector保留原样。是否在当前 torch/CUDA/Triton 环境实际通过，还需要后续显式执行证据；本审计没有运行。

### 5.2 可复用的是实际 API，不能机械复制指南的笔误

examples/YOUR-MODEL-HERE/README.md Step 2 给出可选 MLP integration，并要求 GELU。实际导出名字为 SparseDiffMlp；指南代码中的 SparseDiffMLP 大小写不一致。Step 1 的示例传参方式也应以实际类签名为准。

SparseDiffMlp 是 plain callable，不是 nn.Module。它持有 fc1/fc2 引用，并在初始化时把 fc2.weight.T.contiguous() 保存在普通 list（modules/mlp.py L11–26）。因此：

- 不应直接把已注册的 nn.Sequential 属性替换为此 plain object；
- 不应假定它缓存的 W2 转置会自动跟随 model.to()、权重替换或 offload；
- 应参考实际 FLUX 的 self.sparse_mlp 调用方式，保持原 nn.Module 权重所有权，明确设备与参数生命周期。

实际 FLUX layers.py L159–161、L196 直接调用 SparseDiffMlp；L289–310 对另一类 block 亦有接入。这条已有真实例子比伪代码指南更适合作为模块 integration reference。

构建/导入也并非纯CPU只读：上游 Triton mm2 文件尾部会生成函数指针并启动示例内核，offloaded_tensor 导入时会建 CUDA streams。因此后续环境准备也应按计算任务处理；本次没有 import 这些模块。

### 5.3 selector验收必须容许合法atomic差异，并覆盖一个对齐边界

topk_indices.cu 的选中和补齐写入均使用atomic。给定相同评分，未对齐之前的有效支持可以有相同语义而索引顺序不同；补齐阶段各thread竞争进入前remainder个位置，还可能改变最终补齐支持。因此不能盲目要求两次作者selector的原始索引SHA相同。应检查有效count、范围、唯一性、阈值/随机候选覆盖，并在算术核验中向两个实现注入同一份实际inds/counts，避免把合法排序差异当数值实现错误。

只需保留一个有区分力的GPU边界用例，尚未执行：m=13824、quantile=.7、multiple_of=256、random_amount=0；所有评分为1，仅将1024、2048、…、13312这13个位置置0。前1024列的阈值是1，未选13列全落在同一个thread处理的序列中。按源码控制流，补齐需要13个位置，但只有1个thread持有last_invalid_value，可用于暴露counts是否覆盖未写索引。可先将indices填为sentinel，检查counts范围内是否仍含sentinel，再核同inds数学输出。

这是一项尚待GPU验证的具体实现验收条件，不是已经实测的作者失效结论。也不能从“所有有候选thread都写入、未显式限制remainder”直接断言必然越界：候选thread数至多为原未选列数，额外写通常仍在allocated cols内。需要区分未写有效位置、有效支持非确定性、以及实际越界三个问题。

## 6. 原模块 reference 必须保留的行为

| 项目 | 固定作者行为 | 重做时的要求 |
|---|---|---|
| proxy | 每128 token组 mean(input)→W1+b1，对比 cached mean(preactivation)，取绝对差 | 不改成post-GELU RMS、全局均值、散点均值 |
| 稀疏选择 | 前1024隐藏列估计分位阈值；全列>=阈值或curand补入；向上补至256倍数 | 直接调用作者 compiled selector；不以torch.quantile/exact topK替换 |
| 参数 | top_keys=.3、random_keys=.05 | 作为未调优reference保留；任何改动单列配置/预算消融 |
| 首层 | first_n_dense_layers=2 | 原始reference保留；全40层稀疏是额外消融 |
| dense refresh | full_step_every=10；MLP在0,10,20…dense | 不静默改成2；新专家首次无cache必须dense是宿主语义要求 |
| mask复用 | block_mask_cache=2；step<10始终重算；>=10的部分奇数step可复用旧indices | 必须覆盖实际分支，不假定“每两步永远重算” |
| proxy缓存 | 仅recompute indices时更新选中列 | 不是每次hit都更新proxy |
| 激活/输出缓存 | 每sparse hit更新选中postactivation与output cache | 多hit不是固定dense-anchor；单hit等价不够证明多hit复现 |
| bias/gate | b1参与GELU；b2在旧输出里，delta不重复加；current gate属于宿主外层 | 保持pre-gate边界及当次gate |
| MLP tile | BM=MBM=128，counts_multiple_of=256 | 开源reference保留128；论文C=192另说明，不静默互换 |
| 数值 | BF16输入；官方Triton mm1用FP32acc+bias→GELU→BF16；mm2增量先BF16再加cache | 原module reference保持；native-BF16重写单独作为算术适配消融 |
| voxel | Wan token与RoPE按(4,6,8)重排并逆排 | 保留作者布局；raster/仅FFN局部重排须记录为额外适配 |
| offload | MLP各cache开关默认false；存在MlpStorage/MaybeOffloadedTensor机制 | 首个小shape模块reference可不引入offload；完整系统部署另核生命周期与成本 |

两个容易遗漏的分支：

- dense refresh 只写 h/y/proxy，没有清空旧 indices；例如10 dense后11可能复用9的mask，这是固定代码的实际行为。若怀疑它不合理，先复现并记录，再把修复作为显式变体，不能悄悄改默认。
- MLP缓存与调用计数/CFG分支相连；当前无CFG=1简化该问题，但原模块reference仍应验证一次完整多hit与刷新边界轨迹，不以0→1两个调用代替所有分支。

## 7. 当前 Wan2.2 FFN-only 真正需要的最小适配

### 7.1 宿主接口所必需

- 接到当前归一化/调制后 pre-W1，返回 pre-gate 输出；用原生 tanh-GELU及bias。
- High/low权重与cache隔离；新专家首次建立dense状态；请求之间清空。
- 计数反映实际无CFG单调用与实际稀疏子模块，避免只改YAML却没有wrapper/注册。
- 处理 N=31668=247×128+52：原 block_mean reshape不能直接接受。尾块统计必须说明有效52行，或保留尾块dense。若使用执行padding，不能把零填充行当真实token一起平均。
- **voxel本身无需重写：**直接复用作者 voxel_chunk_no_padding 与 reverse_voxel_chunk_no_padding（ops/voxel.py L11–101）。它先处理完整3D块，再依次拼接T尾、H尾、W尾，已经支持不整除的T/H/W；当前网格21×29×52也适用。不要另写新的compact voxel permutation。3D边界尾区与最终MLP的52行尾组是两个问题，真正缺少的是原block_mean对最后不满128组的支持。
- 如果使用当前LightX2V offload，需要明确引用的是哪一组实时权重，W2转置缓存如何更新/搬运。这是当前宿主的接口适配，不是改变代理的理由。
- 在A100选官方triton provider，并完成作者 utility extension。因为官方支持该backend，不必先写自己的selected-W1/W2 kernel。

### 7.2 应独立保留/消融，不能称为“必要”的选择

- first_n_dense_layers从2改0；
- refresh10改成更激进或更保守频率；
- 关闭/修改mask reuse；
- 分位+随机换成exact-K；
- Block128换global支持；
- preactivation mean换P64/post-GELU RMS；
- voxel换raster，或者只在FFN局部重排而保持attention原布局；
- 原作者Triton舍入换为本仓native BF16；
- offload、clone/gather/packing与kernel融合改写；
- 为追平某预算而调整top_keys/random_keys；
- 用50步原模型的结果直接代替四步学生的验证。

必要的“语义正确性”改动应通过相同输入的reference、inverse-permutation、cache/state trace、dense neutral等检查证明；不需要把漏bias、跨专家串cache这种本来不合法的实现当作质量消融。对有多种合法方案的适配（尾块dense vs有效尾块稀疏、layout、precision、offload）才需要成对的数值、质量或性能消融。

### 7.3 建议的证据顺序

1. **原模块 reference：**固定commit、官方SparseDiffMlp、compiled selector、官方Triton，在整除128的小shape和真实权重shape上验证dense/sparse、first2、0→1→2、refresh/mask-reuse边界、cache逐列更新；先不把宿主接入与算法变化混在一起。
2. **最小 Wan FFN 端口：**只加入宿主接口、计数/专家隔离和尾块处理。原模块与端口对同一输入/状态/indices逐项比较，再验证完整原生neutral、真实初始noise、实际wrapper覆盖范围。
3. **明确的部署变体：**逐项比较precision、layout、tail、offload以及可选性能优化；Global-P64/Block-P64放在这里。
4. **方法结论：**只有完成上述身份和语义链，才把当前视频损失归于作者FFN机制的迁移边界；未达到时优先修复或补复现，不以“已有结果与记录相容”替代完整正确复现。

以上是只读审计导出的验证顺序，本次没有执行这些实验。

## 8. 原始资料入口

- 论文：[arXiv abstract/version history](https://arxiv.org/abs/2506.03275)、[原文D.1与算法附录](https://arxiv.org/html/2506.03275v1)。
- 当前默认：[repository metadata](https://api.github.com/repos/sandyresearch/chipmunk)、[master branch](https://api.github.com/repos/sandyresearch/chipmunk/branches/master)、[branches](https://api.github.com/repos/sandyresearch/chipmunk/branches)。
- release：[v1.0](https://github.com/sandyresearch/chipmunk/releases/tag/v1.0)。
- 官方Wan入口：[run.sh](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/run.sh)、[text2video.py](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/wan/text2video.py)、[model.py](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/wan/modules/model.py)。
- I2V：[官方作者回复](https://github.com/sandyresearch/chipmunk/issues/2#issuecomment-2977261223)、[当前image2video.py](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/wan/wan/image2video.py)。
- MLP集成：[作者指南](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/YOUR-MODEL-HERE/README.md)、[实际SparseDiffMlp](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/modules/mlp.py)、[FLUX真实接入](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/examples/flux/src/flux/modules/layers.py)。
- A100路径：[setup.py](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/setup.py)、[MLP ops](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/ops/mlp.py)、[selector](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/csrc/indexed_io/topk_indices.cu)。
- 实际使用边界：[I2V flicker issue](https://github.com/sandyresearch/chipmunk/issues/4)、[2026 A100内存issue](https://github.com/sandyresearch/chipmunk/issues/6)。这些是用户报告，不是独立完成的FFN复现证据。

