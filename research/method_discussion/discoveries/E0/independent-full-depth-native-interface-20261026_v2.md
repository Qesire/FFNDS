# 独立 native producer 接线审阅 v2：逐 step 的权重、metadata 与 buffer 生命周期

作者：`full_depth_source_audit`；2026-10-26；本次仓库HEAD：`3fbc868fc402f87475f7afac7af3fd84605fbd78`。

**新结论（源码设计，未运行）。** 主控制器可以复用一个实际 Wan 专家的 pre/head 和一套 block-offload manager，在每个当前 GPU block 内依次消费 teacher/student 两条 hidden。但必须补回原 `WanModel.infer` 和 `WanTransformerInfer.infer` 隐含的生命周期：pre/head 权重上卡、各轨 scheduler setter、infer metadata 初始化、adapter 的 expert/step/block context、原生四个 flow 字段，以及两轨/observer完成后的单次 buffer swap。它不是把 `infer_block` 放进两层for循环便自动成立。

本轮两个值得直接影响实现的隐含状态：

- **manager跨step有状态。** block39原本会预取block0并swap；每专家manager的 `need_init_first_buffer=False` 与两个buffer的当前位置继续用于下一次该专家推理，不随step或source自动重置。
- **pre/head不受HiF4 target hook代管。** decoded non-FFN cache常驻只覆盖block attention线性层；原patch/time/text embedding、head/norm/modulation仍须用原生命周期上卡。`m.to_cuda()`会搬整个专家，不能代替局部调用。

这是一份独立只读接线设计，不是root新producer实现的验证。未启动模型、GPU、SSH或测试；未改实现、ACTIVE、旧两份审计及任何运行收据。实时方向维持暂停。

## 1. 本次审查对象及适用条件

实际参考是 `tools/run_e0_reconstructed_videos.py` 的原生全视频路径：构建原pipeline、激活HiF4 hook、预载实际使用的decoded权重，再调用 `pipe.generate`。teacher仍是原dense D-Q/HiF4函数；student仍是当前fixed affine54 bank，不将“局部8.54%”当视频质量或全160位native数值结论。四类问题保持分开：same-input部署数值、student prefix输入改变、gate/后续block有符号传递、四步flow/latent漂移。

本设计限定 `wan2.2_moe_distill / i2v / NoCaching / cpu_offload=true / block offload / torch_sdpa / seq_parallel=false / enable_cfg=false`。还应在实际对象拒绝lazy/unload、changing_resolution、额外before/after/post adapter、动态LoRA/diff、phase/model offload、compiled block和fused MXFP8 FFN。当前原函数若在这些分支可运行，不意味着本双轨接线已覆盖它们。

锚点是 `G02→F03/F04/F09→F12/G20→F13/G21→P24/D09`。源码已有的生命周期标SRC；拟实现双轨关系是 `conditional`；trace与画质关联是 `posthoc`，不是单site后缀因果实验。

## 2. source 入口：编码和 prepare 不应被偷偷重复

原local路径为 `WanRunner.run_pipeline` → `_run_pipeline_local` → `run_input_encoder()` → `run_main()` → `init_run()` → `run_segment()`。编码结果会设置 `runner.input_info.latent_shape`；真实shape不能从CLI的480×832强行推算。历史来源的latent是16×21×58×104，patch grid21×29×52。

控制器有两个明确可选入口，冻结一个即可：

1. **替换run_segment。** 保留原pipeline的编码与 `init_run`，原scheduler在进入替代segment前已经prepare了一次。把它作为D；另构建S并prepare一次或以D初始latent clone初始化。不要无意再调用D.prepare消耗下一份随机噪声。
2. **独立paired driver。** 完整调用原输入编码/初始化语义，显式管理两次prepare；最后把两轨latent交给另有协议的decode/保存层。不得调用两次完整 `pipe.generate` 冒称一次逐block配对。

旧video driver在每来源前显式 `scheduler.generator=None`（line147）。`prepare_latents`遇到已有generator会继续用其状态，不会根据新seed自动重置。两轨必须使用各自独立、同seed的generator，或同一初始latent的独立clone；不能连续借同一generator生成两份所谓相同噪声。初始raw dtype/shape/hash与storage不共享须有接线控制。

`WanStepDistillScheduler.prepare`从FP32初始化，`step_pre`可能按环境和flag转BF16；四步过程保留原prepare→四次pre/post，不在每step reset。`wan2.2_moe_distill`不满足父Wan `model_cls=='wan2.2'` 的conditioned-prefix分支；不要从另一Wan variant照搬mask clamping或清空 `inputs.image_encoder_output.vae_encoder_out`。本模型pre仍会把该I2V条件与当前latent concatenate。

原 `run_segment` 在最后segment会 `del self.inputs`；手工配对控制器应只在两轨所需处理完成后结束inputs生命周期。若未来还需decode两轨视频，不要让第一次原 `end_run()`先清掉第二轨所需状态。

## 3. 可直接实现的逐step调用顺序

每一步k都先断言两轨 `step_index/timesteps/sigmas`对应同一个冻结k，high/high/low/low，用户条件身份相同；各自latent继续自由演进。

| 顺序 | 必须执行的实际动作 | 关键约束 |
| --- | --- | --- |
| 1 | 保存两轨before_step_pre状态/原dtype/shape/hash；分别执行 `D.step_pre(k)`、`S.step_pre(k)` | 保存真实状态，不先float32规范化；after_pre state是本step模型真正使用的latent。 |
| 2 | `wrapper.set_scheduler(D)`；`wrapper.get_current_model_index()`；取 `m=wrapper.model[wrapper.cur_model_index]` | wrapper的setter广播两个已加载专家；get_current_model_index会写cur_model_index。当前block offload下不会搬整个专家。 |
| 3 | `m.pre_weight.to_cuda()`；`m.transformer_weights.non_block_weights_to_cuda()` | 原 `WanModel.infer:161–163`。本次两轨共享选中专家，调一次，保持到两个head/post完成。 |
| 4 | `m.set_scheduler(D)`；`D.infer_condition=True`；`preD=m.pre_infer.infer(m.pre_weight, inputs)` | setter绑定model/pre/transformer/post；不能只赋wrapper或runner.scheduler。 |
| 5 | `m.set_scheduler(S)`；`S.infer_condition=True`；同原API得到preS | preD/preS的x/storage与adapter_args独立；pre读取的latent必须来自当前分支scheduler。 |
| 6 | 取 `infer=m.transformer_infer`、`weights=m.transformer_weights`；初始化metadata和attention state | 见下一节；该步骤原本藏在transformer.infer中。 |
| 7 | 明确打开expert/step context，按原manager顺序跑40个block；每个buffer内D→S；归约后swap一次 | 原block wrapper还要block_idx/context。当前block phase对象必须来自GPU buffer。 |
| 8 | `m.set_scheduler(D)`；`headD=infer.infer_non_blocks(weights,xD,preD.embed)`；`flowD=m.post_infer.infer(headD,preD)[0]` | post实际返回list；flow是完整4D、float32输出，不能把head的M×patch_dim输出直接交scheduler。 |
| 9 | 对S重复相同head/post；写各轨noise字段 | 保留两个完整flow，同时不混用preD/preS的grid/embed。 |
| 10 | 同步所有使用pre/head权重的工作，执行 `m.pre_weight.to_cpu()` 与 `weights.non_block_weights_to_cpu()` | 原 `WanModel.infer:202–204` 的下卡位置在返回runner之前；故应在step_post前完成。两轨head结束前不可下卡。 |
| 11 | 两轨分别执行原 `step_post()`，保存各自after_post dtype/shape/hash、flow和signed geometry | 使用原sigma顺序，末步sigma_next在观察公式中为0，但原方法不执行加零kernel。 |
| 12 | 结束context，释放本step临时量，保留两个scheduler的下一步latent及各专家manager | 不销毁manager、不把一轨latent赋给另一轨。 |

**无CFG下四字段完全按原源设置：** 对每轨s与其完整flow，执行 `s.noise_pred_cond=flow`、`s.noise_pred_uncond=None`、`s.noise_pred_guided=flow`、`s.noise_pred=flow`。三个flow字段原本是同一对象别名，不要求额外clone。仅给 `noise_pred`赋值虽足够当前step_post计算，却没有复原原model.infer状态合同。

`m.set_scheduler(s)`调用 `BaseTransformerModel.set_scheduler`，会同时绑定四个对象；`BaseTransformerInfer.set_scheduler`还令 `s.transformer_infer=infer`。两轨可以先后指向同一infer对象，因为串行执行；不得把“两个scheduler都存在”误当成infer同时有两个scheduler指针。

wrapper.scheduler可在整个配对step保持D，当前专家m在分支间切setter。这样wrapper的专家选择和runner外层主scheduler明确，S只在自己的调用范围绑定。退出/异常时应恢复进入控制器时保存的wrapper/runner/model绑定，避免给随后原生dense控制留下S指针。

## 4. metadata 初始化、可共享状态和不可共享hidden

原 `WanTransformerInfer.infer:114–119` 在blocks之前执行：

```text
infer.cos_sin = pre.cos_sin
infer.reset_infer_states(pre.x, pre.context)
infer.reset_attention_states(weights.blocks)
```

`reset_infer_states`写 `self_attn_cu_seqlens_qkv`、`cross_attn_cu_seqlens_q`、`cross_attn_cu_seqlens_kv`，并在image-context分支另写img长度。当前use_image_encoder=false，不能误扣257个context token。这些cu_seqlens按原源码在CPU构造；当前SDPA不消费其值，控制器不应自行改device后声称完全复原原流程。

当前两轨有相同grid/query长度、context长度，且同一个 `WanPreInfer` 对同shape复用cos_sin cache。因此可每step以preD初始化一次，严格校验preS的长度/grid/cos_sin身份后共享这组只读metadata。更一般的做法是分别生成metadata snapshot、每次分支调用前恢复对应值；不能仅先后reset两次再让D误用形状不同的S状态。

`reset_attention_states`遍历 `weights.blocks`中的self-attention对象，若存在reset_state则调用；当前TorchSDPA没有历史KV或reset_state。这一现状允许在同GPUbuffer上顺序D/S；若换成有内部历史的attention，单次reset不保证GPU buffer或两轨独立，须拒绝而非假定支持。

`WanPreInfer`自己的 `freqs,cos_sin,grid_sizes`是每专家持久缓存；grid变化时原infer会重建cos_sin。它们不应在每branch被误重置成None，也不能把来源A的grid绑定到来源B。preD.x与preS.x不能共享storage：cross-attention、FFN入口和post_process均可能原地修改x；原 `pre.x`在普通无adapter路径也可能与当前hidden别名。

## 5. 每个block的严格offload顺序

对选中专家唯一的manager以及 `weights.blocks`，l=0..39按以下顺序：

1. `infer.block_idx=l`。
2. 若 `manager.need_init_first_buffer`为true，调用 `manager.init_first_buffer(weights.blocks)`；仅fresh manager需要。当前设计拒绝lazy，所以不进入CPU lazy buffer swap。
3. `manager.prefetch_weights((l+1)%40,weights.blocks)`，写的是 `cuda_buffers[1]`，不改变当前buffer0。
4. 使用 `manager.compute_stream`，绑定D scheduler/pre metadata，调用当前patched `infer.infer_block(manager.cuda_buffers[0],xD,preD)`；S亦同。
5. 当前site的四native FFN/四writeback、block输出、full-M归约及需要保存的向量必须完成，或者明确把其异步依赖加入正确stream。尤其dense shadow的mm对象/bias来自该GPU buffer；不能缓存phase引用等到后续buffer被重新装载时再求值。
6. `manager.swap_blocks()` **一次**。它会同步load/compute两个stream后交换buffer0/1；同步范围不自动包括额外observer/communication stream。

原最后一次预取是block0而不是停止预取；swap后manager.buffer0已经是block0，`need_init_first_buffer`保持false。步0→1复用high的manager；步1→2换low，low若从未使用则首次初始化；步2→3复用low。若相同process有下一个source，两个manager也仍是各自block0；权重未变时沿用符合原pipeline行为。

不要每轨swap，不要逐branch创建manager，不要用 `weights.blocks[l]` CPU对象替代buffer0，也不要调用 `m.to_cuda()`“简化”生命周期。后者包含整个transformer/post容器，会破坏有界显存设计。

**stream边界需要明确。** 手工pre/head通常发到default stream，原block发到manager.compute_stream。源码审计不能证明Python返回即GPU完成；producer应在首block前使compute_stream等待pre所在stream，head/step_post前使消费者stream等待block工作，shadow归约/保存同理。当前正确性草稿可选择显式同步，费用进入采集账本；以后才在另冻的时序实现中细化events/waits。不能把trace同步成本算成部署速度收益。

## 6. adapter context与runtime的真实接线

adapter的原补丁仅在 `MultiDistillModelStruct.infer`入口设置 `_expert_ctx/_step_ctx`，在 `WanTransformerInfer.infer_block`入口根据 `self.block_idx`设置 `_block_ctx`。手工控制器绕过前者后，要用token/reset作用域显式设置expert/step；仍调用**已patch**的infer_block便能复用block作用域。若绕过block wrapper，则也须明确设block token。

`MMWeight.apply`只要三个context有一个None，就静默回原MM实现。故“没抛异常、输出有限”不能证明HiF4路径生效；producer必须验证所走module/record和计数。buffer对象的weight_name可带其最初slot编号，adapter用 `normalize_module_name(...,actual block context)`重写编号；不要从buffer名字推断当前l。

target regex仅覆盖block `self_attn/cross_attn q,k,v,o`与`ffn.0/ffn.2`。pre与head均不在其中；所以即使已预载decoded non-FFN权重，也不替代第3/10步原pre/head生命周期。head所在 `transformer_weights.head` 与 `head_modulation`由 `non_block_weights_to_cuda/cpu`负责；`WanDistillModel.post_weight`当前是空 `WanPostWeights`容器，不能把head误认为在那里。

`_ACTIVE_RUNTIME/_ACTIVE_OBSERVER`是进程全局，不能用并发线程切两份hook；本方案只安装一份角色dispatcher。四FFN求值直接使用明确的dense base apply或纯candidate pair，不递归重入候选单pending槽，也不重跑 `infer_ffn` 而重复加attention residual。FFN四对象与shadowwriteback的clone要求已在前两份记录展开，此处不改变。

所有角色上下文、runtime.apply覆盖、类method patch与scheduler绑定都要在finally恢复；若current block中途失败，manager可能已经预取/交换，不能只恢复Python指针便自动重试，保留失败并结束该scope。不能擅自把失败后的manager状态当fresh起点继续下一source。

## 7. 最小接线验证与本次证据范围

本次完成的是source/API、AST与SHA核对，未运行新producer。实现后的最小软件验证可用状态机fixture核对：每step pre/head各上/下卡一次、每site D/S各一次、40次prefetch/swap、每branch全setter、context恢复和两轨flow字段；它只能证明控制顺序。真正未来native范围仍须另冻后才做dense/dense身份、affine主轨旧实现身份、160site覆盖、实际资源与完整视频metrics bundle。

前份consumer审查保留其限制：它接收观测并归约，不证明张量来源。40.6972GiB已知权重payload不是新producer峰值上界；本生命周期审计也不承诺80GB卡可跑。全轨观察不识别单site后缀因果，后者仍是另外冻结的少量hybrid干预。

本次是窄静态接线复核，未增加Web/论文新颖性搜索；stream结论沿用前份亲读的 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html)。该文档要求跨stream显式依赖和tensor lifetime；不是项目已实测的同步/显存结果。

## 8. 实际源码SHA

L前缀为 `.runtime/code/LightX2V/lightx2v/`，H前缀为 `.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/`。

| 实际亲读源码 | SHA256 |
| --- | --- |
| `tools/run_e0_reconstructed_videos.py` | `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005` |
| L `models/runners/default_runner.py` | `4f8e4295e5301b1465c4e320641c98855f2f7e6e5734e714d17af1acb2fd4f18` |
| L `models/runners/wan/wan_runner.py` | `e12bf49dfa76a3c5bde0d284952d7f8c2dcee2043737b310d51d46e7ff3d0f69` |
| L `models/runners/wan/wan_distill_runner.py` | `26a2a503137b0125cf4a2f8b2c9fc5dae716121c6e600b7ce85517af5ee62dc3` |
| L `models/networks/wan/model.py` | `f0b0e9904d9708ff67412d8c98cb3f384b86426bb96ebf84d17e439e7a8580ff` |
| L `models/networks/base_model.py` | `e171498c41484a9e2ec86d0e3259f4c7cd3e21a62ea87d75fa05ec75ba6db5e9` |
| L `models/networks/wan/infer/pre_infer.py` | `f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5` |
| L `models/networks/wan/infer/transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| L `models/networks/wan/infer/offload/transformer_infer.py` | `c66f594739c4737f4d77e27db31ac990a82c539ada1d2ec267e0203b90a5fe25` |
| L `models/networks/wan/infer/post_infer.py` | `6567c439e2f16f322a9f10cd2ca326d3f924e9da75b3ae2ae442ec1327a327a2` |
| L `models/networks/wan/weights/transformer_weights.py` | `c14e7eeaf085762cf4148a6745d9208098693dca5dbc5971737bbc4c344452a7` |
| L `models/networks/wan/weights/post_weights.py` | `a55002fbcf296599b74500f9b712de47828b9d5398ddb2a9628dd52275fac8b8` |
| L `common/offload/manager.py` | `dbd68ff7af9626f128bf74baed97885f7610853067dd184010d8f3325d8aa5c0` |
| L `common/modules/weight_module.py` | `264927800f22068b48e74d521de3b795fbc2b49a6fcf5f76e22b04d38dc2c6a1` |
| L `common/ops/mm/mm_weight.py` | `5987645f1883e92c08a7e416fbe25be3c0fadec2355ae4732fa1182d9094b64c` |
| L `common/transformer_infer/transformer_infer.py` | `b266c1e1b08495614f6a8c7eb34af2a91d1a4a99992581143c89de8db2fdd2e0` |
| L `utils/custom_compiler.py` | `8abb3df81e0d1bc3ba5b947b1ba9fa22d930a4b44076b31a23019f243e9e6c2d` |
| H `lightx2v_wan_distill.py` | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |

旧独立记录SHA维持：原219行审计 `9ed02bc4e3299b95e3881d3c0a5dcf161afce106c7a079991ac2ca6c317e0faf`；consumer审查 `4fbee4cb9f8f25c48cf84bae184a0a1998846452faac23352c63d35cabf77a7d`。本轮只新增本v2文件，不回写旧文件。
