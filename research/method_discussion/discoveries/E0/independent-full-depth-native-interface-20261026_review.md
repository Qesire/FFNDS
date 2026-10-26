# 独立 consumer 审查：4D 调度状态与四种有限写回的代数可用，生产证据仍须另证

作者：`full_depth_source_audit`；日期：2026-10-26；对象：新静态 consumer，非 native producer。

**结论（只读源码/数学）。** 最终 `step_pre_geometry`、`scheduler_geometry` 与四 arm 归约没有发现阻断性的数学或原生维度错误，可以作为未来 native producer 的代数 consumer 接口。它们可记录真实 4D flow/latent，同时要求 FFN/site 输出为 2D full-M；scheduler 已把入口 cast 与 step_post 更新分成两个有限边界。最终collector还强制完整state-shape声明、六个scheduler边界SHA及跨step连接；writeback记录已明确prefix包含residual/gate变化。这个结论支持准备性接口，不证明已有native迹线、数值中性、资源可行或视频质量。

没有执行模型、GPU、科学张量、测试或新的数值实验；本审查读取测试源码，仅核对 AST/函数签名、原生源码及整数形状。本文件保留前一份219行审计原样，未修改其字节或旧结论。

| 本次审查对象 | SHA256 |
| --- | --- |
| `tools/e0_full_depth_trace_contract.py` | `28154f7d340cf164f07bbb908e31b37f04414b3fb7ef4f9f03e45fbc377ba98e` |
| `tests/test_e0_full_depth_trace_contract.py` | `f9a10e89a39fed80228fcb1ee0c7aefc7b2d8ecf6eb41ee67010313d92053b1b` |
| 前份 `independent-full-depth-native-interface-20261026.md` | `9ed02bc4e3299b95e3881d3c0a5dcf161afce106c7a079991ac2ca6c317e0faf` |

图锚点仍为 `F03/F09→F12/G20→F13/G21→P24`。这里的 consumer 接口属于 `conditional` 观测设计；不是新增执行边，也不把局部 closure 解释成下游因果归因。

## 1. 4D latent/flow 没有被当成 full-M FFN

原生历史 latent 为 `[16,21,58,104]`，包含2,026,752个标量；FFN hidden/residual 对象为 `[31668,5120]`，其中 `31668=21×29×52`。两者不等宽、不等元素数，不能把同一2D schema强加到全部对象。

本次源码正好分开了这两个领域：

- `pair_geometry:38`、`four_arm_geometry:52` 要求二维同 shape；`FullDepthTrace.add_site:139` 要求 site 的 q1、四FFN输出、四writeback和block输出均匹配声明的二维 full-M shape。
- `step_pre_geometry:79`、`scheduler_geometry:100` 允许至少一维，并要求本次求值的全部 state/flow shape 严格相同。它们对 `v[start:start+chunk_rows]` 转 float64，再 `.sum()` **全部剩余轴**，故直接输入4D张量会覆盖其全部元素，不会只归约第一个frame或漏掉空间轴。
- `scheduler_geometry` 的 dtype 字段取传入的原 tensor dtype，而非内部 float64 观察dtype。由 BF16 转 float64观测差不会恢复已丢精度，但保留输入有限值。

默认 `chunk_rows=256` 对 C=16 的4D latent只产生一个channel chunk；这是完整归约，不是256个token抽样。它也意味着该默认值不会把这些4D对象再细分为很小的临时 buffer，未来 producer需把真实临时/同步成本计入采集账本。它不影响数学正确性。

## 2. scheduler 的符号、sigma 和 cast 边界已分清

令 `δ⁻=zS_before_pre−zD_before_pre`，两轨入口操作的有限改变量分别为 `cD,cS`，令 `c=cS−cD`。`step_pre_geometry` 实际记录

`δ⁺=δ⁻+c`，及 `||δ⁻||², ||δ⁺||², ||cD||², ||cS||², ||c||², <δ⁻,c>`。

故入口cast的能量变化可由 `||c||²+2<δ⁻,c>` 重建，而不会把 prepare FP32、step_pre BF16和step_post内部FP32混写成“一直FP32”。

随后 `scheduler_geometry` 明确 before 为 **after_step_pre**，flow取该轨真实模型输出，after为 **after_step_post**。以 `d=vS−vD`、`h=sigma−sigma_next>0` 定义

`ideal=δ⁺−h*d; defect=δ_after−ideal`。

其 `ideal_energy_change=h²||d||²−2h<δ⁺,d>` 符号正确；`ideal_update_defect_SSE` 包含原生 `step_post` 的两次乘法、先减后加、FP32舍入与输出dtype cast的**两轨差**。入口cast已另记，未重复归给update defect。原Wan最后一步没有加零项，这个区别也会进入 defect，而不是被理想实数式抹掉。

`sigma/sigma_next` 是显式keyword，源码拒绝非有限、负sigma_next和非正gap。`add_step` 检查入口/更新boundary、与声明state_shape相同、dtype衔接及 `entry_cast.delta_after_SSE≈update.delta_before_SSE`。这些修订解决前份审计指出的step_pre边界歧义。

**仍有限定：** 一项差能量相同不证明两个完整向量是同一状态；相同norm的不同方向就是反例。最终consumer已新增六个边界SHA，`finish`核对上一step的after_post与下一step的before_pre相同。SHA的实际生成、与传入tensor的绑定及同step两次引用同一after_pre对象，仍是native producer责任，不能由caller填写的字符串自行证明。

## 3. 四种 writeback 必须按两种真实 residual 输入定义

以 `W(r,y,g)` 表示实际native gate/residual writeback，包含dtype、乘法与加法顺序。producer传入的四个writeback arm应严格为：

| arm | 必须来自的对象 |
| --- | --- |
| `dense_teacher` | `W(rD,yDD,gD)` |
| `approx_teacher` | `W(rD,ySD,gD)` |
| `dense_student` | `W(rS,yDS,gS)` |
| `approx_student` | `W(rS,ySS,gS)` |

`rD,rS` 是各轨完成 `infer_ffn` 中 attention residual相加后的实际x；不是尚未相加的infer_ffn参数初值。每组同输入 dense/approx shadow都从对应r的clone开始，不能对一个原地x连续写回两次。当前consumer接收四个**实际输出矩阵**，没有使用 `gate*(FFN_error)` 代替它们，方向正确。

套用通用四arm归约后：

```text
operator = W(rS,ySS,gS) - W(rS,yDS,gS)
prefix   = W(rS,yDS,gS) - W(rD,yDD,gD)
total    = W(rS,ySS,gS) - W(rD,yDD,gD)
```

`total=operator+prefix` 及有符号能量式成立，且不要求gate/residual或QDQ可微。**writeback的prefix含rS−rD，以及必要时gate变化**；它不是仅 `fD(qS)−fD(qD)` 的FFN输入响应。字段名 `prefix_response_SSE` 在这里是广义前缀响应；最终源码的warning已经逐项给出四种W定义并明确该限定。

consumer并未传入r/g本体，所以不能凭四个输出自动验证上述四种输入关系；native producer仍须提供相应调用身份/预声明观测协议。相同逻辑也适用于post-Q1标签：形状合规不能把pre-Q1调制U自动证明为真实q1。

## 4. 最终snapshot的证据校验与调用者责任

**完整shape绑定已补强。** 最终 `FullDepthTrace.__init__:128` 强制声明 `state_shape`，非synthetic要求rank4；`add_step:180–184`把entry/update shape同时绑定该声明，因而不能再用二维state panel进入非synthetic完成路径。`finish:208–212`核对两轨每个相邻step的post→pre入口SHA连续，并在结果保存 `full_state_shape`。

声明shape必须来自该来源的实际scheduler/patch grid；任意缩小的4D panel不会因rank4自动成为完整latent。SHA需冻结dtype/shape/字节序及真实tensor字节的canonical规则，不能重复旧driver中先 `.float()` 再hash而丢失原storage dtype的信息。sigma四步序列和来源/模型身份也由producer绑定到冻结协议；consumer只验证本step的有限sigma与正gap，不执行模型。这些是调用者责任，不要求当前代数consumer变成模型加载器。

其余限定可保留在当前准备性用途：

- native-versus-continuous记录要求status/coefficients provenance/row IDs，但consumer不执行该比较；字段合规仍不是native测量已经发生的证明。
- `synthetic=True` 的完成状态明确保留synthetic，quality始终false，并声明双轨不识别单site后缀因果作用。这些状态边界正确。
- 已读取10个测试函数；除原3×2的NumPy软件fixture外，最终测试新增rank4入口cast吞掉小差异、非synthetic二维state拒绝及完成后改断跨step哈希的拒绝。rank4 cast示例使用NumPy float16，明确只验证cast边界的记账，不是BF16 Wan原生复现。源码另覆盖抵消、修复/增长、非零rounding defect、非有限/重复/覆盖拒绝和synthetic标识。**本独立审查未复跑测试**；不把根代理报告的10测通过扩写成独立原生验证。

因此，此最终snapshot可以作为**准备中的代数consumer**审查/集成，当前无阻断性数学/API缺陷；它本身不生成“已完成全M/全state原生迹线”。原生producer、身份中性与真实资源/视频证据仍待以后单独授权，不能由这份源审阅授予执行权。

## 5. 检索与最小下一步

本次任务是已有source上的窄静态复核，未另做论文/网络检索；沿用前份亲读的PyTorch官方stream/memory语义，不提出新算法或新颖性主张。实际读取新consumer、10项测试源码，并重新对照前份审计已绑定的真实 Wan `infer_ffn/post_process` 与 scheduler调用边界。

最小下一动作是由serialized integrator将最终consumer作为待授权native producer的明确输出接口，继续准备接线/状态恢复草稿。shape绑定、writeback角色说明及rank4软件fixture已具备，无需为本审查自动扩大接口或运行模型。当前没有新GPU、模型或视频作业。
