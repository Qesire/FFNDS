# 独立 producer 审阅 v3：单 pipeline 双轨的主控制器生命周期已接通，尚无原生执行证明

作者：`full_depth_source_audit`；2026-10-26；仓库HEAD：`3fbc868fc402f87475f7afac7af3fd84605fbd78`。

**结论（逐行源码审阅）。** 下列最终snapshot的主控制器在正常路径上没有发现新的阻断性接线错误：真实prepared teacher和独立prepared student逐block使用同一当前GPU buffer；metadata/scheduler/context按轨设置；四arm归约在下一块修改hidden前完成；每site仅swap一次；两个head/post结束后先下卡，再调用各轨原scheduler。原生GPU数值中性、峰值显存、视频身份和质量仍未实测，本记录不把软件接线通过写成可执行实验授权。

审阅初始snapshot时定位到两处具体问题：静态设计别名误作真实runtime config键、两次上卡在清理try之外。最终snapshot已分别改为实际scheduler/attention校验、上卡进入try并同步/逐项清理；本审阅亲读了修订后的实际代码，不以修复意向代替复核。

全程只读源码、AST与SHA/整数元数据，没有运行模型、GPU、SSH、软件测试或新增科学张量计算；只新增本文件。前219行审计、consumer审查与v2设计保持不变。实时研究维持暂停。

## 1. 被审阅的精确版本

| 文件 | SHA256 |
| --- | --- |
| `tools/e0_paired_wan_controller.py`，最终snapshot | `bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd` |
| `tools/e0_paired_ffn_observer.py` | `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` |
| `tools/e0_paired_weight_residency.py` | `75ecc7a5cc8d338c2e0afee8c301d3eefaabb3ddb6daa48814eb0b9853052bc1` |
| `tests/test_e0_paired_wan_controller.py` | `eb7c2404b22dbee7e42a30f6ea231eb3d7552d5c2ad17fec513b7bb0864ca21f` |
| `tests/test_e0_paired_ffn_observer.py` | `86aa0cc11e6f9844ec07a0736d0af6926e873312c7616533480f17d475d313a0` |
| `tests/test_e0_paired_weight_residency.py` | `93834b426f6e118b703b750c849e788569a87c406ce7a34066f8c756b37ea550` |
| 原 `LightX2V/lightx2v/pipeline.py` | `6b3eb5aa5a9ad9c3046df3baf8d2bd7feeda12adb327954a5d7298842b265062` |

上表LightX2V路径前缀为 `.runtime/code/`。其余真实runner/pre/transformer/post/offload源码与v2记录所列SHA相同；原六视频driver为 `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005`。本次没有阅读其它agent的结论来替代独立源码推理。

图锚点仍为 `G02→F03/F09→F12/G20→F13/G21→P24/D09`。实现提供 `conditional` 原生观测接口；尚未产生新的native观察事实，双轨累积关系也不是单site后缀因果作用。

## 2. 两项具体问题的最终状态

**真实配置校验。** 原adapter contract的 `scheduler`、`attention` 是描述字段；`build_pipeline_runtime_overlay_from_contract`没有把这两项写入effective config。原 `LightX2VPipeline.set_infer_config`给Wan写的是 `self_attn_1_type/cross_attn_1_type/cross_attn_2_type`，实际scheduler由runner构造。AST核对constructor及set_infer_config均未赋值self.scheduler/self.attention。

初始controller将整个 `REQUIRED_CONFIG` 逐键用于runner.config，可能在真实原生入口因缺这两个别名拒绝执行。最终 `__enter__:55–64`跳过别名，检查实际 `runner.scheduler` 类名及三个attention实现配置；同时检查BF16 infer/sensitive dtype。这项修正解决了静态设计schema与真实runtime schema的接线差异，不要求修改冻结原配置来迎合新工具。

**部分上卡失败的清理。** 初始controller两次to_cuda在try外；若第二次失败，第一次上卡没有进入finally下卡。最终 `_step:115–190`从第一次上卡前进入try，finally逐项尝试 `cuda.synchronize`、pre_weight.to_cpu、non_block_weights_to_cpu；即使一个清理操作失败也尝试另外两个。无原始异常时抛出清理失败，有原始异常时附加说明。CUDA fatal error仍必须终止该scope；这个finally不是可恢复GPU硬件错误或自动重试的承诺。

## 3. 正常路径生命周期逐项核对

| 接线 | 实际代码及判断 |
| --- | --- |
| 原生准备入口 | `run_segment:210`取wrapper.scheduler，确实复用原runner.init_run已prepare的D；S按同类、同effective config独立prepare。验证storage指针不同、dtype/latent值/sigmas相同，没有再次消耗D.generator。 |
| step cast | `_step`保存before_pre克隆与raw identity，调用各轨原step_pre，再保存after_pre克隆；分别调用entry cast和step_post几何consumer。没有把prepare FP32误作四步一直FP32。 |
| expert | bind D后调用get_current_model_index，再读取cur_model_index；没有使用该无return方法的None作索引，且核验high/high/low/low。 |
| pre/head上卡 | 当前专家pre_weight与non_block_weights上卡一次，两次pre都在该try内。没有用m.to_cuda把整个专家搬上卡，也没有误认为HiF4 attention cache包含head。 |
| 两份pre | 每轨先经wrapper原setter广播scheduler，再调用原pre_infer；断言grid和时间/文本/RoPE一致。第一step还核对初始pre.x相同。两次真实patch embedding分别产生各轨hidden。 |
| metadata | reset_attention_states在step前执行；每轨当前block调用前设置cos_sin和reset_infer_states，未遗漏原transformer.infer隐藏的cu_seqlens初始化。当前无状态SDPA允许这种重复初始化，额外CPU开销属于采集成本。 |
| block context | `target_context`设置三个ContextVar并逆序reset；`infer.block_idx=block`与实际GPUbuffer同步。调用patched infer_block可以保留adapter自身block作用域；重复相同block context不会更改其值。 |
| offload | 每block依次init-if-needed、prefetch-next、policy.begin_block、当前buffer内D/S、finish_site、一次swap，finally通过compute-stream fence释放当前原FFN cache。当前/下一buffer用途没有颠倒。 |
| head/post | 两轨各自setter后调用真实infer_non_blocks与post_infer，取list的[0]；得到完整4D FP32 flow，而非M×patch_dim head面板。四个noise字段按原无CFG路径写入。 |
| 下卡/step_post | 两轨head/post都结束并同步后下卡，随后两轨调用原step_post。flow已在各自scheduler字段中，不依赖已下卡head权重。 |
| 返回原生成接口 | run_segment返回S latent给原run_main的VAE decoder；D最终latent另存teacher_latents。finally恢复原scheduler绑定不会把已返回的S tensor变回D。原decode/视频保存函数消费显式返回latent，不靠wrapper.scheduler.latents重新取值。 |

原Wan22 step_post的先减后加/末步不加零、BF16回写仍由原方法执行；observer没有以实数 `z−h*v`重写主轨。完整flow和latent观察另以float64有限差记录defect，属于观测层。

## 4. stream与tensor存活核对

当前pre及after_pre复制在调用者current stream；`manager.compute_stream.wait_stream(cuda.current_stream())`在进入40块前显式建立依赖。每个当前block的D、S、shadow、四arm归约及采样callback均在同一compute-stream作用域，完成后才swap。swap同步load/compute两个原stream；policy.end_block又fence后才clear FFN字典。

两个head在current stream执行前也有wait，最终cuda.synchronize覆盖head/post与观察工作，随后下卡。成功路径没有发现跨stream读写缺少顺序的阻断点。`tensor_identity`会把真实dtype/shape与raw uint8 bytes一起绑定，保留BF16位；legacy float32 hash独立保留用于旧dense参照，没有拿它代替native identity。

关键alias的寿命也闭合：teacher真实writeback可与teacher当前hidden同storage，但S有自己的hidden；`finish_site`发生在下一block原地修改teacher hidden之前。shadow writeback使用x.clone，不在同一residual上加两遍。原GPUbuffer phase/mm/bias被shadow立即消费，未把其引用拖到swap之后求值。

**callback责任。** `save_vectors`若启用，应当在返回前完成对应copy/保存，或者显式clone并管理其stream/lifetime；不能把当前hidden的view留给异步后台writer后假设controller替它保活。progress回调也须保持被冻结的只读观察含义。当前source没有额外communication stream，未来driver不能偷偷添加未join的消费者。

## 5. 可变state、恢复与bounded residency

`_bind`使用wrapper的原setter，因而model/pre/transformer/post及scheduler.transformer_infer反向引用一并更新。高/低两个manager各自持续存在，块39预取并swap回块0的状态跨step保留，没有每轨复制manager或重置need_init_first_buffer。

observer的active/pending对每轨独占，native caller只消费该轨所选main结果；另一个operator是shadow。q1被实际核对同U时dense/candidate相同；四writeback来自各轨真实residual和gate，前缀项含residual/gate变化。原始调用的FFN/post方法在branch finally恢复，runtime.apply在observer退出恢复。

新residency按module suffix把non-FFN长期cache和当前(expert,block) FFN cache分开，拒绝跨block FFN取权重，在显式fence后释放。最终policy退出也在已初始化CUDA上同步再恢复_weight并清cache。这解决了直接使用AllWeightResidency会把shadow访问的全部原FFN长期retain的问题。

这仍只限制该policy自己的decoded matrix payload：原runtime `_resident_weights`若在进入前已保有旧neutral权重、GPUbuffer、bank、pre/head、VAE及临时量不自动纳入这个字节上限。独立driver须清楚处理旧cache并统一记账。不能从当前FFN cache峰值推导总物理峰值，原40.6972GiB known subtotal也没有变成可运行证明。

控制器退出恢复runner.run_segment、runner.scheduler、wrapper.set_scheduler和原cur_model_index；block分支finally恢复adapter context与infer函数。若某个block中途失败，manager可能已预取或交换，当前代码没有回滚全部GPUbuffer内容；driver应保存失败并结束attempt，不将“Python指针已恢复”当fresh pipeline重试依据。

## 6. 生成driver仍须明确的边界

本组件不是生成launcher。独立driver仍须绑定原生源码/checkpoint/cache/bank SHA，先装nativeHiF4 hooks、bounded policy、observer，再让controller替代run_segment；结束时退出controller/observer/policy并deactivate原hook。典型退出顺序是controller→observer→policy，最后关闭adapter，保证policy释放前有明确fence。

当前 `trajectory`、`step_wall_seconds`在controller构造时清空。建议每source使用一个新controller；若复用同controller多次pipe.generate，这些list会累加，trace_factory在step0则创建新trace。外层必须按source截取/保存，不能把八个step记录归给一条四步视频。observer/policy counters若跨source复用也必须取明确delta。

原generate经本controller只会自动解码/保存S视频；teacher_latents只是保留的D终态，不是已经生成的D视频文件。未来driver要单独进行匹配D decode/保存、dense/dense与candidate/candidate中性控制、历史dense SHA核对、真实FFN角色调用账、实际资源采样。每条真正生成的新视频仍须独立全帧evaluator和SHA bundle gate，official/human未跑就保持NOT_RUN/quality_accepted=false。

`step_wall_seconds`以及observer的primary_plus_shadow时间包含诊断工作；当前cost_scope已标为paired/shadow/numerical/stratum observation，不是candidate部署latency。四arm/trace closure也不识别单site后缀因果作用，不能直接解释全部画质失败机制。

## 7. 实际验证范围与剩余未知

本次亲读三个组件全部源码，及controller、observer、residency各3项CPU synthetic测试源码。controller fixture覆盖完整四步/160site、320分支、每专家80次swap、异常恢复和dtype-bound identity；它不使用原生Wan/HiF4、CUDA stream或真实权重。**本独立审查没有执行这些测试**；不把root的测试运行报告改写为自己的原生验证。

已用标准库AST核对实际pipeline字段赋值及新Python文件可解析、重算SHA；没有导入原生model模块。当前未覆盖：真实GPU数值中性、底层CUDA错误路径的可恢复性、实际attention对象与配置绑定、完整调用/内存峰值、视频解码身份及质量。这些由未来另冻scope验证，不阻止当前接口继续作为待授权producer草稿接受审查。

本次是窄source review，未增加Web/论文检索；使用前份已亲读的官方 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html) 判断stream等待及lifetime条件，未据此声称本项目已有GPU测量。
