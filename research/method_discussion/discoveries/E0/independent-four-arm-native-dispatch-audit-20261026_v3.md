# 独立 v3：四函数 native dispatch 的同输入、状态中性与真实写回边界

作者：`/root/independent_e0_audit`。2026-10-26。只读源码/数学审查；未运行张量、模型、GPU、SSH、Git或新拟合。只新增本文件，不改实现、共享状态或旧记录；不作方法 endorsement。root 新 producer 的精确版本审阅将在本文件另列，不把接口设计当作原生测量。

## 1. 被解释的离线算子与图锚点

`affine54` 是固定离线 bank：保留54×64 hidden，原 Smooth1/Q1→选取 W1/b1→tanh-GELU→对应 Smooth2/Q2→新 BF16读出 `v/bias`，再加新 BF16输入矩阵支路 `matrix`。原 caller 的 norm/modulation、gate和residual仍存在。它不是旧权重原样、不是训练授权，也不是实时算法。视频失败仍须区分：(a) bank/原生执行同输入失配；(b) student前缀改变实际FFN输入；(c) signed后续block/head传递；(d) 四步flow-state变化。

图锚点为 `F03–F09→F12→F13/G21→D09`，设计诊断对应 `P23/P24` 的 OPEN/conditional 边；终端 `P22/P25` 不能代替这些原生数值/轨迹记录。当前 ACTIVE 为 E0，full-depth tooling 的 native measured sites仍为0、model execution authority为NONE。此文件提供实现审查，不改变权限。

## 2. 亲读源码与版本

| 文件 | SHA256 |
| --- | --- |
| `tools/e0_reconstructed_runtime.py` | `957588f5dc3b32334982b6247064ddece1d21ab627267eb717bc72fb31003872` |
| `HiFloat4/ptq/adapters/lightx2v_wan_distill.py`，位于本地HiF4源码树 | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |
| `HiFloat4/ptq/adapters/wan_i2v.py` | `d5a9be8ddb777fe360e66ba8e4ccde033b4a2969d2a0b853c7aefa0f45c44293` |
| `quant/hif4_backend.py`，位于本地Wan权重项目源码树 | `8d7589f6b65acb8e96ce14cf18adbce44f729110536725a825da47f521befbb6` |
| `lightx2v/models/networks/wan/infer/transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| `tools/e0_full_depth_trace_contract.py` | `28154f7d340cf164f07bbb908e31b37f04414b3fb7ef4f9f03e45fbc377ba98e` |

另亲读 bank builder 的 scale、选组、偏置构造；全驻留 wrapper 的 `_weight`替换；静态design/interface checker及继承scheduler的pre/post边界。旧 v2独立审查SHA仍为 `b08080bc3812a24b06b5aacd618ebf94b3a0e3e4574ee4ed0eb09519575256a4`，未改其字节。

## 3. 四函数的精确定义：U、q与整个block状态不同

令 `U_T/U_S` 为两轨 **真正传给原 `.ffn.0.apply` 的 modulation后BF16输入**，不是pre-norm residual，也不是observer自造的均值。令

`q_b = BF16(native_QDQ(BF16(FP32(U_b)/s1)))`, `b∈{T,S}`，

其中 scale、QDQ开关、force_fp32、dtype必须取原运行的实际设置。四输出应为 `y_DT=fD(U_T)`、`y_ST=fS(U_T)`、`y_DS=fD(U_S)`、`y_SS=fS(U_S)`；若两operator真实Q1映射一致，可把FFN部分重新记为 `fD(q_T)` 等。必须先证明/断言同scale与Q1条件，不能默认不同bank的Q1天然相同。

FFN向量恒等式是

`y_SS-y_DT = (y_SS-y_DS) + (y_DS-y_DT)`。

第一项为student输入上的operator差，第二项为原dense函数对实际prefix输入改变的响应。它只在这个函数边界成立；不能直接推广成经BF16残差写回之后相同分量的线性相加。

原adapter observer在调用runtime前接收的是 **U，即Q1之前**。将它命名为post-Q1会错置本次最重要的输入边界。真实q应来自该次原生Smooth1/QDQ/cast的产物；观察它不需要另跑W1。只能保留本次pair内的局部引用/副本，避免依赖可复用storage地址作跨层/跨轨缓存key。

一个机制反例说明为何必须同时看q和residual：两个不同U可以落在相同Q1量化cell，因而有相同q和相同dense FFN输出，而两轨residual基底仍不同。更理想地，未量化LayerNorm满足对全通道常数平移的不变性，raw residual差可以被norm隐藏。故“q相同/FFN误差为零”既不能证明block状态相同，也不能否定prefix漂移。

## 4. 重入和缓存：单一pending不支持递归shadow

当前 `ReconstructedRuntime.__enter__` 替换同一个runtime.apply，`.ffn.0`后保存单一 `pending=(expert/block,step,q1)`，`.ffn.2`才消费并清空。若在primary pair未闭合时再次调同一runtime的shadow `.ffn.0`，会触发pending guard；若绕过guard覆盖pending，可能把别的q用于affine支路。应在primary闭合后使用不共享pending的纯求值入口/隔离状态，而不是嵌套进入同一个hook。

直接用 `MMWeight.apply`作shadow还会重入 `_ACTIVE_OBSERVER`、runtime及expert/step/block context。它可能把shadow当成actual模型调用，污染采样、计数和待处理状态。显式传expert/step/block的底层求值可避开observer递归，但如果调用base adapter apply，仍会改变 `_resident_key`、`_resident_weights`、weight_transfers、linear/QDQ counters；它不是无状态数学函数。

最安全的状态范围是“当前source/step/block、当前已驻留的权重、当前T/S前缀、当前四输出”，完成减法/样本摘录后释放。不要为了少一次Q1而跨step或跨block复用q。异常退出也必须恢复原method/global hooks，清空或验证pending，无未完成pair。

resident缓存的数值值可相同而执行计数/准备成本不同。固定的non-FFN驻留与当前block FFN驻留应在双轨/四臂使用完后才换块；不能让base `_weight`的换key清空逻辑驱逐另轨仍需要的权重，或因shadow触发额外完整权重转移。CPU-offload原buffer的生命周期同样必须覆盖两轨与shadow；“只算shadow不返回网络”不保证buffer未提早交换。

## 5. 必须保留的参数条件

原dense adapter根据record/override判断 `activation_qdq`、`force_fp32`、Smooth scale与keep_fp；候选hook固定执行native QDQ与force_fp32=True。未来dispatcher应对有效cache记录断言这些条件相符，并在record缺失/返回None时失败，不能悄悄fallback到MMWeight原实现。

bank builder实际使 `pack.w1` 为decoded BF16 W1选行，`s1`为原全5120输入scale，`s2`为原hidden scale按相同ids选取；`v/matrix/bias`是最终新BF16参数。`pack.bias`已经含原b2、拟合截距以及affine消项。candidate shadow再加一次原`.ffn.2` bias，或把Smooth2补偿再乘/除一次，都会改变被审查的算子。hidden indices/完整64-group顺序必须在w1、b1、s2、v之间一致。

完整64组的保留维护原组内依赖，不授权逐列裁切后重算原Q2。输入和所有stride/shape/dtype都应明确，尤其不要把packed参数升FP32后用于“native”shadow。TF32、BF16 reduction flag与原native GEMM行为应保持并记录；FP32数学诊断另设命名，不混入native arm。

## 6. 四个writeback必须真的执行原post_process

实际caller `infer_ffn`先执行 `x.add_(attn_out)`，然后norm/modulation、两个linear与GELU；block随后调用 `post_process`。在当前infer_dtype=sensitive_dtype分支，后者执行 `x.add_(y * c_gate_msa.squeeze())`。它包含真实乘法及原地加法的有限精度，不能用FP32 `x+g*y`计算再声称是原writeback。

因此既不能对同一x重跑infer_ffn造成attention重复加，也不能让四arm的post_process共用一个被原地修改的residual tensor。应在正确的加attn后基底 `r_T/r_S` 处捕获，分别对独立clone执行原post_process：

| arm | 原生writeback |
| --- | --- |
| dense_teacher | `W(r_T,y_DT,g_T)` |
| approx_teacher | `W(r_T,y_ST,g_T)` |
| dense_student | `W(r_S,y_DS,g_S)` |
| approx_student | `W(r_S,y_SS,g_S)` |

主轨必须返回与所选arm相同的真实结果，shadow clone不应写入任何后续状态。即使同一step的gate理论上只由固定time/condition决定，也应使用各轨实际g并记录形状/dtype，不能为了简化默认为相同。

writeback的prefix项 `W(r_S,y_DS,g_S)-W(r_T,y_DT,g_T)` 包含residual基底及gate改变，不再只是dense FFN的输入响应。现contract的warning已正确说明这一点。四writeback的向量加法分解仍是差分恒等式，但将其归因为gate-weighted局部SSE需要另一个条件论证。

## 7. 少算且同输入的可行边界

若primary原生路径已经给出 `q_T/q_S`，且D/S的Q1条件逐项一致，可在每一prefix内复用这两个 **实际q**，无需为shadow再做Q1。两个实际输出 `y_DT/y_SS`也直接复用；只额外求 `y_ST/y_DS`。它节省的是诊断冗余，不是部署稀疏收益。

按affine54、一次160-site双轨pair的理论账本：每site四FFN函数共有4个W1/W2 pair，加两个affine GEMM，即10个linear；共享Q1时为2次Q1+4次Q2=6次QDQ，不共享时是8次。对应FFN子范围分别1600个linear和960/1280次QDQ。这不含attention/head/额外中性控制/FP32样本诊断，不能拿它当全网计数上限。最终应以实际dispatcher路径冻结独立计数。

可最小化保存量：full-M四函数输出用于流式FP64标量reduce，只有冻结row_ids保存向量；完成该site后释放。必须遍历完整hidden维求native输出，不能先抽hidden列。native-continuous比较时，**native样本应从实际full-M output切片**；不要用小M重新native GEMM替代，因为改变M/批处理形状不保证逐位相同。PyTorch2.10官方明确区分数学等价和batch/slice数值等价；旧运行环境记录为2.10.0+cu128。[官方数值准确性说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)

## 8. native–continuous诊断的对象与混杂

最明确的连续诊断是：从当前actual q1取冻结行，使用 **同一个已部署BF16 pack的数值升FP32**，以tanh-GELU和原s2做FP32求值，不再Q2，添加同一v/bias/matrix。这样没有重新训练或新参数，但比较仍综合了Q2、中间BF16、GEMM和支路加法路径。

它不识别“原先FP32拟合系数→保存BF16参数”的损失：把已舍入的pack升FP32不会找回丢失的系数。若原FP32系数没有另行冻结保存，不能从这项诊断宣称那个误差已排除。

若需要分拆而不重跑native，可复用actual hidden/preQ2/postQ2样本建立小型桥接：

1. actual q2升FP32 + 同pack读出，用于观察最后GEMM/affine加法/回写精度差；
2. actual preQ2值不QDQ + 同FP32读出，与1对比量化引起的输出差；
3. actual q1出发的FP32 W1/tanh-GELU/s2 + 同读出，与2对比早期BF16路径差。

这些是不同定义的诊断函数，必须记录每一步dtype、输入行、系数hash和finite误差。向量差可以按明确定义相加，SSE不能忽略交叉项直接相加。原生Q2后系数与GELU hidden依然不同。FP32诊断本身也是有限运算，并非数学实数真值。[PyTorch2.10 GELU定义](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.gelu.html)、[linear定义](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.linear.html)

## 9. 最低中性控制与尚不能得到的结论

最终producer的最有意义控制是实际primary边界值与原路径对齐：D/D、S/S以及启用/停用shadow的同轨对比；要分别覆盖true U/q1、FFN输出、post_process、block输出、step_pre/post dtype/latent/flow。原full-width neutral只证明很窄入口与旧两层条件下的hook等价，不保证新双轨控制器/全部层/partial pack等价。

中性控制应保留真实调用顺序、非FFN cache/residency与计数，记录shadow额外计算，而不是用synthetic accounting通过代替native neutrality。不会因neutral值相同就宣布instrumented时延可与旧普通生成直接比较。两轨相关也不能给出单site下游因果贡献；混合prefix/suffix干预仍为另外的科学问题与授权范围。

目前最强实现反例为原地alias：若四arm共用r，后面的“student”输出会包含前面shadow已经加上的残差；即使最终四函数差分恒等式数值闭合，也只是对错误构造的数据闭合。**geometry closure检查不能验证四arm来自正确的原生调用。** provenance、严格状态分离和native中性控制不可由代数恒等式替代。

## 10. 实际检索与后续精确producer审阅

本次按 pinned AutoResearcher/WARD读真实adapter→native backend→Wan infer_ffn/post_process→bank builder/cache wrapper，并结合当前full-depth contract追踪输入与消费者。官方PyTorch stable页面指向2.14；确认原RUN环境为2.10.0+cu128后，实际另开并阅读2.10 numerical accuracy、GELU和linear原文，未把新版本行为冒充旧运行证据。这里的closest mechanisms是原生fake-QDQ/有限精度函数、同输入shadow求值及残差差分，而非一个新近似方法。

新认识是：低额外成本的四函数诊断可以复用实际q/actual输出，但“重入”和“中性”不是数学函数重算的自然属性。shared pending、原地残差与resident cache本身就是需要审查的实验对象。此处尚未收到新producer冻结身份，以下保留给后续源码精确复核；当前未测任何native paired site。
