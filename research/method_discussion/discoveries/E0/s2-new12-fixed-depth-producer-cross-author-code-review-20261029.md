# New12 固定深度 producer：跨作者代码评议

2026-10-29 · `/root/s2_theory_design_independent`。root编写 `tools/e0_s2_new12_fixed_depth_videos.py`；我编写其依赖的 `tools/raw_bf16_fixed_layer_w2_runtime.py` 初版，因此对producer是跨作者只读评议，对runtime不是独立作者审签。非方法支持票、非ACTIVE授权、非真实模型运行证明；未改producer、旧代码、协议或ACTIVE。

受审文件在评议过程中由root并行修订。初读producer SHA `b8168ae9f76afcf9bf310e5fbbf7abd829dd8a09b2c6324cd0b041782b90063d`；后读SHA `e0e44f27bfe893547f353bfa7f81a25367a0fdd7b24ece1c4b0f40f1e15d8658`。runtime后读SHA `8415731c4bfc4780b7be8c1e55a2910dd9d54686e157d1f8e54db30dd3f49012`；代码仍可能继续变化，以下不对任何后续SHA自动生效。

## 具体问题

1. **原初版两处确定阻断，后读代码显示已更正，仍需回归与实际CPU proof。** `draft` 原将 spatial capture 的 `completed_video_protocol` 指向的质量汇总协议作为 `old_factor_parent`；质量RUN实际状态为 `COMPLETE_TEMPORAL40_EIGHT_AXES_REUSED_DENSE20_QUALITY_NOT_ACCEPTED`，不满足 `neutral_reference` 要求的生成RUN状态。原 `draft` 又缺 `source_cohort`，而 `vm.registration` 必须读取该字段，因而在模型setup/因子preload后会KeyError。后读代码改成按part的 temporal generation parent并加 `source_cohort=source_bank`；已只读查旧generation part0包含898、part1包含96。必须实际走两个source的parent闭环及manifest合成测试，不能因静态改名声明已经执行通过。
2. **新worker proof V2 的路径层级冲突，可能在任何模型/GPU工作前阻断。** [cache_entries](../../../../tools/e0_s2_causal_worker_proof_v2.py:61) 将四个模型文件及一个 `model_view_root` 作为5个“均不存在”的目标，先依次为4个文件 `destination.parent.mkdir(parents=True)` 再链接view目录；经只读检查，4个模型文件全都位于该view根内，因而前几步创建了view目录，最后链接view将触发 `Only absent local cache entries may be linked`。即便目标源都存在也不能通过该顺序；如果源或目标预检查失败则更早退出。此判断针对后读proof V2 SHA `21e06b3a4a1889ca7b1c16529b8e97fe25f26562c433b4b1526a04b7aabc616b`，我未启动或更改该scope。其外层EXIT与实际文件状态决定究竟走到哪一步；若已冻结执行，不得原位修补或自动重跑。
3. **Dense参考在CPU proof后可能漂移而不阻止完成。** [require_cpu](../../../../tools/e0_s2_new12_fixed_depth_videos.py:253) 在模型worker调用 `references(..., payload=False)`，重验GEN JSON及其identity hash，却不重验已通过CPU proof的Dense视频和状态文件SHA/stat；后续 `legacy.core_report` 的 `evaluate` 会记录实际参考视频digest，但只调用结构`validate`，没有将实际digest与预绑定Dense视频SHA比较。`collect` 验证文件存在及噪声byte配对，也不核视频hash。若预检后视频被替换，能在旧GEN身份下评价新视频并达到COMPLETE。应在GPU运行前或每次比较前绑定/核查视频及实际使用的状态文件，且断言CORE参考digest等于预期，同时将输入IO预算及稳定性检查记录清楚。
4. **父RUN/GEN的细节校验不足以单独证明注册元数据。** `references` 检查GEN的source、seed、prompt及160位置标签，但未显式核 `gen.states == parent phase.states`、GEN protocol SHA == 实际parent protocol、GEN视频 == parent phase视频、十状态键及真实 `frames/fps/size`。`legacy.registration_record` 将81、尺寸和fps填为常量；`collect`据注册行检查尺寸而不是自己重新解码对照既有GEN。现有父producer曾真实执行验证，不代表此读取器已经完整验证了字段。应将GEN、父RUN及视频几何的关键实际字段相互绑定，避免把常量转成观测证据；补造假的反例软件控制。
5. **完成事件检验缺逐位置kind对照。** [phase](../../../../tools/e0_s2_new12_fixed_depth_videos.py:366) 检查160个step/block唯一覆盖、Dense总数、factor GEMM总数、预取总数和bytes，却未对每个`fixed_W2_dispatch_returned`显式验证 `kind == ('dense' if block in BLOCKS[mode] else 'factor')`、expert对应step。当前runtime自身按policy派发并校验marker，所以这是可独立审计和防回归的不足，不是已发现实测错误；如果实现发生位点错置但总数相同，现有producer终验无法单独排除。
6. **失败和CPU输出边界。** preflight在创建RUN前进行authorize、输出存在与环境检查，run也在输出目录创建前核验父与CPU proof；这些早期失败只能依赖外层EXIT/stderr，不能由内部RUN说明。preflight 的 `args.out` 未见相对 `hf_root` 的显式目录限定，也未以增量计账强制 `CPU_RESOURCES.new_HF_bytes`；将受控worker输出目录及退出失败凭据作为协议和软件控制的一部分，不应把名义8MiB写成已核实际输出上限。进入`try`前的 `CheckedPackageFinder`/file guard初始化异常亦需由外层收据处理。

## 已核对的执行及计数边界

- `legacy` 私有加载 temporal producer，再私有加载 `vm` 与 `native/io`；`vm` 的mode、phase_order、run_id和transfer_geometry只在其私有树中重绑定。runtime类由我编写，当前root增加 `fixed_W2_dispatch_returned` 完成事件；单次marker `(expert,block,kind)` 防止旧W2或错误buffer复用，原W2分支调用已由OriginalGuard包装的old_apply，factor分支经active.mm_call与meter执行两次真实factor GEMM。异步预取按其自己的block通过`is_factor`查询固定全phase policy，未在compute边界切换global模式。真实双缓冲、跨step block0和stream事件仍需实际neutral与合成回归，不能用此静态审阅代替。
- 每part控制1次全factor旧neutral，6新图各三臂：19完整视频，3040 FFN位置；全factor共7次×160与早/晚各6次×120，共2560 factor位置、5120 factor GEMM，480原W2计算位置。对应skip `7×162 + 6×120 + 6×122 = 2586`，原W2 prefetch `6×42 + 6×40 = 492`，总预取3078。19×7=133 stage pair、190状态写入；18新候选各比Dense两噪声，36匹配检查，旧neutral十状态控制在六个新图之前。这些为代码和源码推算，不是运行实测。
- Dense及factor参考分别需要actual EXIT0、同checkpoint、runtime overlay、source/seed与factor bank identity；新图来自开发source，不是盲确认。全图自由轨迹 `flow/latent` 已保存供后续分析；后读DESIGN已改为独立注册的signed state/motion/visual分析，当前producer计算的是18个paired CORE，而不是已经完成flow方向或人工/官方质量评估。
- A100新worker及package/model proof尚需**该worker**的真实EXIT0、同路径、全部payload SHA/stat及本producer CPU proof；一张卡顺序完成两个part可保持每次1GPU、pair最多2卡；绝不能把空闲Blackwell结果当A100同recipe。资源和成功状态应以实际冻结、实际RUN/EXIT和监测记录为准。

建议在根因修复后，以新的实际source SHA重新做两part草案/CPU preflight和针对注册及parent身份的合成控制；由于当前所有修改仍在进行，本文不声称最终实现已可启动。

## 后续只读修订快照

root反馈已接受上述问题。随后只读 producer SHA `c53e5d115c2ce10962d461db6a5f82a287de2a4cd03db119a4f070b0fc10554f`：已见`old_factor_parent`改成按part的真实generation协议、`source_cohort`绑定并验证、Dense GEN与父phase的states/video/protocol及81帧几何和actual counters交叉校验、old factor GEN与父phase交叉校验、逐dispatch的kind/expert约束、DESIGN将flow/latent明确为仅保存待后续分析。以上是**源码静态复核**，不是实际CPU/GPU执行或最终冻结SHA。

root告知正在补CORE实际reference/candidate SHA与所需噪声/视频重验；在它完成并再次审阅前，第3项的预检后漂移问题仍按待修处理。worker proof V2的父子路径冲突需以其实际EXIT核定；root拟在保留失败后用新版本验证已有master链接并补metadata leaf，不修改已冻结失败scope。输出限定和启动早期失败留证也需按最终程序/外层实际凭据核查。上述更新不构成对新worker、新producer后续SHA的批准。
