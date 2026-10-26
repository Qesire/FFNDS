# 独立 v4：paired FFN observer 与有界驻留的源码语义审查

作者 `/root/independent_e0_audit`，2026-10-26。只读实现/数学，不运行张量、CPU科学实验、模型/GPU/SSH/Git，不修改实现或旧独立文件。软件测试的PASS来自协调者报告；本 Agent只阅读其定义，未重跑。此为独立范围审查，不是native neutrality验证、方法认可或新增执行授权。

## 1. 精确版本与科学对象

| 文件 | 本次实际读取SHA256 |
| --- | --- |
| `tools/e0_paired_ffn_observer.py` | `d22ad4751293c051b1edd75473c24f12c9236212eb45e0951a25ed822ae8015d` |
| `tools/e0_paired_weight_residency.py` | `fdd693603757030ddec5326c1195804b2a7992dcb38c7c86d44362b41bfe29fb` |
| `tests/test_e0_paired_ffn_observer.py` | `bafb665a52cef849c4f32bf4da9d562db684a43afb493147701b544df680ec4f` |
| 先前独立v3，只读保留 | `2cc89de0cd6633371e61f3dd74272c1eb3ce76b50e3e55b1e410e6047919a46c` |

对应原adapter、native backend、caller与full-depth contract的精确SHA已在v3第2节列出。v3的计划性“随后在原文件追加”由协调者的新指令改为本独立v4文件；v3原字节不动。

被解释对象仍是离线 `affine54`：实际modulation后U，经Smooth1/native Q1、保留54×64 hidden的W1/tanh-GELU/Smooth2/native Q2、新BF16读出及输入affine支路，再通过原gate/residual写回。终端失败仍分为同输入native执行/参数问题、student prefix改变输入、signed下游传递和四步latent/flow漂移。当前只是实现阶段，`P23/P24` native paired sites仍为0；`P22/P25`终端视频/补评不能关闭这四种原因。

## 2. 已实现的四臂语义：与原路径相符的部分

`branch`明确指定teacher/student前缀及primary kind；原caller正常执行FFN，observer在actual `.ffn.0.apply`捕获U/q1，`.ffn.2`后闭合pending。shadow在primary pair完成后单独从同U重算，不递归进入MMWeight hook，因此避开v3指出的单pending重入问题。

Dense `_project`使用原record的Smooth scale、native QDQ、对应decoded weight、原mm bias并做F.linear；二维BF16边界下，其scale reshape、除法FP32→BF16、QDQ→BF16和bias/device/dtype语义与原adapter一致。`_record`会拒绝缺记录、缺Smooth、非要求QDQ/force_fp32或activation override=False；这是一个受限来源实现，不能泛称支持任意adapter分支。

Candidate使用原已部署pack的w1/b1、s1/s2、v/bias/matrix：没有额外加原b2，也没有对v/matrix再HiF4量化。shadow的tanh-GELU和caller选择一致。它额外重算Q1，而非复用primary q1，并以`torch.equal`检查同U的dense/candidate q1完全相同。这样多算但直接验证当前输入的Q1一致；不能据此证明未观测输入上两套scale参数全局相同。

各轨得到D/S两输出后，finish_site的映射确实为 `fD(U_T),fS(U_T),fD(U_S),fS(U_S)`，native端取实际full-M输出切片做数值比较，没有用小M重跑native冒充原输出。这修复了v3提出的重要定义风险。

## 3. QDQ/GEMM实际账本

每site两prefix，各算一个dense pair和一个candidate pair。对带matrix的affine54：

| 计数 | 每site | 160site（尚未实际执行） |
| --- | ---: | ---: |
| dense main GEMM | 2 | 320 |
| dense shadow GEMM | 2 | 320 |
| candidate main GEMM | 3 | 480 |
| candidate shadow GEMM | 3 | 480 |
| native QDQ，含重复Q1 | 8 | 1280 |
| 两prefix的FP32 continuous诊断GEMM | 6 | 960 |

这是FFN observer子范围，不含attention/head、任何额外neutrality control或完整视频渲染成本。理论账本只对应当前teacher dense/student candidate模式；D/D或S/S control会重新分配main/shadow类别，但四函数总算量仍是同一个定义。

必须注意：`runtime.counters`在新observer里只手动累计dense linear/QDQ；candidate另计入observer自己的counts，而adapter/backend还有全局native QDQ计数。全RUN总账不能漏掉candidate，也不能把这些重叠计数直接相加。连续诊断的系数FP32 cast bytes也有显式字段，但不是真实峰值显存/传输时间的代替。

## 4. 真实postgate四arm与一个有限精度反例

原caller在infer_ffn内对x原地加attention。wrapper只调用一次原infer_ffn，故不会为了shadow重复加入attention。post wrapper先对`x.clone()`用shadow输出调用原post，再对实际x/actual y调用原post；在当前plain Wan调用顺序下，四writeback的计算为：

`W(rT,yDT,gT), W(rT,yST,gT), W(rS,yDS,gS), W(rS,ySS,gS)`。

它没有以FP32 `g*(yS-yD)`代替真实有限乘法/写回，方向正确。FFN-level prefix term是dense函数的输入响应，writeback-level prefix term则同时含r/g变化；contract warning已明确区分。

**仍需controller保证的边界：** state中的actual writeback保存的是原x的引用，不是snapshot。如果随后post-adapter原地修改x，或在finish_site之前进入下一block再次原地加attention，record就不再指向刚刚的postgate边界。应clone实际writeback，或严格限定无额外mutator的plain Wan路径并在任何后续修改之前完成reduce/保存。当前模块自身不验证这些条件，不能只凭四arm closure认为边界已被验证。

一个纯数学的BF16反例说明postgate不能由raw FFN差代替：取g=1、yD=0、yS=2^-9。rT=1时两个BF16 writeback都舍入为1，operator差为0；rS=0时差为2^-9。若归一化隐藏全通道常数平移，可有相同q而不同r。于是即使raw FFN差和q不变，student侧postgate operator误差也能增大；不能单独归为FFN输入分布失配。这里没有运行合成张量实验。

## 5. 源码适配与恢复：controller的潜在阻断点

Adapter的patched MMWeight.apply要求expert/step/block三个context都非None，才进入runtime/observer。通常expert/step由patched MultiDistillModelStruct.infer设置，block由patched infer_block设置。若新controller绕过MultiDistillModelStruct而手动推进pre/block/head，仅设置observer.active元组不够：缺context时MMWeight会静默回退原apply，可能使teacher/非FFN路径改变，observer也收不到FFN。

因此需要在controller里显式建立和finally恢复原adapter context，并保留原block context的实际生命周期。不能简单令head也始终处在某个block context下，因为原head在block外的fallback行为也属于来源路径。此项已在controller完成前立即报告协调者；是否被正确实现须审新controller，而非在此宣布已经闭合。

模块的branch finally确实恢复infer_ffn/post_process，清空active/pending；observer context退出恢复runtime.apply。synthetic异常测试覆盖了一种中断恢复。这依赖独占hook访问和规范嵌套with结构；它不是线程安全或任意嵌套重入的证明。primary/shadow输出状态不能在异常后继续当完整site使用。

finish_site本身不验证teacher/student两branch属于同一step/block/expert，也不记录branch主路径kind。active虽在各自apply时验证tuple，却未跨branch绑定site身份。因此controller必须在每次pair固定同site、在进入下一site前finish，并记录D/D、S/S或D/S控制mode；否则可能将不同site的两branch凑成一条几何自洽但语义错误的记录。建议在observer state显式保存并交叉验证该tuple，而非把这层保证完全留给调用者。

## 6. 有界驻留实现支持什么

`PairedWeightResidency`将non-FFN长期保留、dense FFN只在active expert/block内保留；FFN fetch越界会失败。key含expert、module名、device/dtype，module名自身含block索引。按dtype实际element_size计payload，超过各自limit失败；end_block要求显式fence后才clear，因此比只在切block时随意清缓存更适合双轨/shadow。

不过end_block只知道fence callable，不知道它是否真实等待所有相关CUDA/offload streams；__exit__也依赖driver finally先同步再关闭。清Python字典不证明其他CUDA stream已不引用原buffer。统计只覆盖本wrapper retained字节，不含pack、两轨activation、原native offload buffers、FP32 cast副本或workspace，不能把这两个limit相加作为峰值显存保证。

当前模块本身没有模型/GPU启动入口，但被driver使用后当然会执行真实CUDA转移/线性计算。其静态可导入与CPU synthetic test不授予模型执行权，也不证明真实显存足够。

## 7. continuous和strata解释准确，仍需范围记录

`_continuous`从actual q1样本和已部署BF16 pack升FP32计算W1/tanh-GELU/Smooth2、v/bias及matrix，没有Q2/中间BF16。check文字明确它不是原pre-BF16拟合系数。其native侧是actual full-M candidate输出行，符合v3定义。它综合Q2与有限执行差，不能单独分辨拟合、系数舍入、Q2和GEMM各原因。

若给whole94 pack，当前continuous会访问不存在的matrix而失败；当前工具应显式限定affine54，不能称通用D/S任意operator dispatcher。actual输入二维和BF16已guard，但5120维、两expert选择、有效config/NoCaching/非mxfp8-fused、post-adapter支持范围及pack形状/hash等仍由driver管。

`raster_strata`的9格确为frame-third×vertical-third，覆盖全width、行号f-major/h-major/w-major，纯几何区域不是语义ROI。将它用于真实token需要实际pre_infer grid，不能从81视频帧或输出图片大小猜grid。shape乘积相同也不能证明teacher/student token顺序相同，controller需对两轨layout作一致性检查。

## 8. CPU synthetic tests的证据边界与最低下一检查

已读的三项测试覆盖：9格partition、synthetic dense/candidate wiring与计数、异常恢复。它们采用4维/8hidden、自定义round-to-eighth QDQ和简化caller；不能验证Wan实际norm/modulation、HiF4 group64、全部40×4、block-offload streams、head或scheduler生命周期。测试PASS是合理的软件进展，不是native neutrality证据。

源码层下一项决定性检查是controller：是否同site交错两轨、真实postgate及时freeze/reduce、原adapter context完整建立/恢复、同block所有consumer完成后再swap，以及两独立scheduler的before-pre/after-pre/after-post实际状态。无需追加科学计算即可先否定结构错误。

未来如另获授权，可用零额外模型调用的sanity：同seed/condition两轨在step0 block0首次替换前，U/q1/residual/gate应一致；此前尚无approx动作，若已不同，优先怀疑controller条件/状态路径。D/D和S/S控制与shadow开关中性也应和真正四函数科学记录分开标记。无论代数closure和synthetic计数多严格，都不能用它们替代真实native边界对齐。

## 9. 检索与独立贡献

按pinned AutoResearcher/WARD继续亲读新observer/residency/tests，并对原adapter/caller和backend作逐操作比较；没有新增文献宽搜。本次沿用v3亲读的PyTorch2.10官方数值准确性/GELU/linear定义，避免将full-M native输出与小M重算混为一谈。独立贡献是明确的alias/context/site身份反例、总账重叠界限和BF16残差基底反例；不是对新producer提前背书。

## 10. Controller与修订后代码的实际复核

随后收到controller，并亲读其完整实现，对照原 `WanModel._infer_cond_uncond`、`WanTransformerInfer.infer`、`WanOffloadTransformerInfer.infer_with_blocks_offload`、`WeightAsyncStreamManager`、`MultiDistillModelStruct`及runner `init_run/run_segment/run_main` 调用。以下限定对应实际再次读取的版本：

| 文件 | 再次读取SHA256 |
| --- | --- |
| `tools/e0_paired_wan_controller.py` | `bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd` |
| `tools/e0_paired_ffn_observer.py`，含首site/kind/event修订 | `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` |
| `tools/e0_paired_weight_residency.py`，含try/finally同步恢复 | `75ecc7a5cc8d338c2e0afee8c301d3eefaabb3ddb6daa48814eb0b9853052bc1` |

在受限的plain Wan、NoCaching、单段、非CFG/seq/compile/lazy/LoRA、原torch_sdpa注意力范围内，没有发现当前组合必然改变primary数值的源码阻断点。这个判断是逐操作静态比较，不是native中性测试PASS。具体已闭合的前述风险：

- controller为block内每轨建立expert/step/block context并finally恢复；head仍在block context之外调用。observer.apply再核显式实参tuple，避免手动controller遗漏context导致MMWeight静默fallback。
- 对同一实际buffer先跑teacher，再跑student及两影子，随后立即finish_site，才执行一次原manager.swap_blocks。原manager的swap同步load/compute两stream；current-block decoded FFN cache随后fence/释放。不会在两轨之间提前换原block buffer。
- controller禁止has_post_adapter、before_proj/after_proj及fusedFFN；因此当前受限路径中，post_process后到finish_site没有另一个原地mutator。第4节actual writeback引用风险在这个组合下被控制器关闭，不能再把它报成该具体路径未解决的错误。若以后扩展source或延迟reduce，需重新审查。
- 两轨scheduler分别prepare，检查非同storage、初值/dtype/sigma相等；同step pre→pre_infer→实际grid/time/condition/RoPE一致性检查后，按原high/high/low/low推进。每block重新绑定scheduler及infer metadata，保留各自hidden。head、unpatchify与noise_pred设置之后才分别step_post，没有把teacher flow用于student更新。
- tensor_identity绑定真实dtype、shape、BF16原字节，而不是先转FP32再把hash当dtype证据；before-step_pre、after-step_pre、after-step_post和signed全state geometry被分开，符合此前dtype勘误。legacy FP32值hash单独保留，只用于可比的旧指纹。
- 新observer在site存 `actual_trajectory_operators`，trace还存D/D、S/S、D/S pair mode；第5节控制模式身份缺项得到修订。首site还检查两轨actual q1、加attention后的residual及gate一致；controller先检查初始pre hidden一致。它们是尚未运行的实际sanity guards，不是已发生的中性测量。

我在中间版本发现residency `__exit__`先CUDA同步、再恢复hook，导致同步抛错时可能跳过恢复；已立即报告。上述最终读取版本用try/finally保证runtime._weight恢复和字典/active清理，关闭了此具体异常恢复缺陷。controller也把上卡操作移入try，对同步/pre.cpu/head.cpu逐项清理，保留原始异常并加cleanup note；没有因为第一个清理操作失败而跳过后两项。未运行CUDA异常测试。

此外，当前controller的跨branch/site tuple一致性由明确嵌套的step/block循环保证，generic observer仍未在state内独立校验这一组身份。第5节对模块独立复用的建议保留；它不是说当前controller真的把两个不同site混在一起。

## 11. 计时、内存与控制尚未证明的部分

新event从原infer_ffn入口开始，经过primary norm/modulation/FFN、shadow pair及q1 equal检查，结束于postgate之前；字段名为 `primary_plus_shadow_ms`。它不含postgate、finish_site的FP32诊断/FP64 reduce/9格统计/保存，但可能含q1检查同步导致的GPU排队或idle间隔。step_wall另含大量观测/同步/hash/数据转移。二者应以实际范围报告，不能与旧单轨部署FFN latency直接作算子速度比较。

non-FFN长期驻留、单block dense FFN驻留只界定payload。双轨pre/hidden、四函数输出和writeback、actual q1、原offload双buffer、FP32系数cast、数值诊断和FP64分块中间量都在此外。当前没有原生显存峰值，不能从payload limit宣称一定适配某卡，更不能把额外同步与CPU-offload成本算成部署加速。

D/D或S/S每block相等能发现两轨不对称错误，却可能共同受到同一个错误，所以仍需将instrumented dense与原未插桩reference的真实latent/video指纹对齐、候选与原prototype指纹对齐，以及启用/停用shadow的中性对照。原软件fixture只走简化模型，尚未证明这些。新controller返回student latent，由原runner执行VAE；teacher latent另存。未来driver要为每个实际解码/保存的clip逐条绑定metric bundle，不能将只存在latent的轨迹写成已经生成并评价的teacher视频。

full-depth trace内的差分/内积可定位误差出现在哪个边界，仍不能唯一归因单层后缀敏感性。native-versus-continuous即使很小，也只对已部署BF16系数及冻结行说明执行差，不能排除bank系数/训练覆盖本身错误。更不能因160site完整覆盖就把终端视频质量或因果机制标为通过。

截至本节，没有本Agent运行任何新科学张量或模型。controller/residency/observer的CPU synthetic测试定义已阅读；PASS数量和结果仍来自协调者，未冒充亲自执行。此次审查可支持“具体接口与操作序列在受限来源内静态相符，已指出并修复若干恢复/身份缺陷”，不能支持“全160 native已验证”或任何实验授权。
