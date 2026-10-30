# Student observer：原生 FFN residual、gate 与 stream 的独立源码审计

审计者 `/root/chipmunk_student_discrimination_terra`，状态 **SRC / DESIGN_REVIEW**，没有执行模型、GPU或数值实验，不是实验批准。范围是现有原 BF16、block offload、Wan2.2 I2V 路径及未完成的 `chipmunk_student_observation_*` 草案；对不同后端、dtype、模型及未来代码不外推。

## 实际顺序与对象身份

[SRC] 先前 input-video owner 设置 `DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=BF16`，把 frozen local package 与 `.runtime/code/LightX2V` 加入路径；其 [protocol](../../../config/e0_chipmunk_input_video_20261030_part0_pending_review_v1.json) 配置四步、`cpu_offload=true`、`offload_granularity=block`、5120/13824。四个关键 runtime 源文件的当前 SHA 与该 protocol 对应绑定逐一相同；新运行必须重新验证实际 import 路径、hash、dtype 和分支。

1. [block offload](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py#L39) 预取下个 block，在 `compute_stream` 内调用 `infer_block`，结束后才 swap。[`infer_block`](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py#L171) 调 `infer_self_attn` → `infer_cross_attn` → `infer_ffn(phase,x,attn_out,...,c_gate_msa)` → `post_process(x,y,c_gate_msa)`。
2. [`infer_cross_attn`](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py#L307) 同 dtype 分支先在 `x` 上原地加 `self-attention y_out * gate_msa.squeeze()`，再对其做 norm3/cross-attention，返回同一已更新 `x` 和 cross-attention 输出 `attn_out`。随后 [`infer_ffn`](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py#L366) 首行 `x.add_(attn_out)`，这一步再次原地更新该对象。
3. 因此，在草案 [`tap` 的 W1 callback](../../../tools/chipmunk_student_observation_tap.py#L68) 执行 `observer.observe` 时，`p['residual_input']` 对象就是 **已加入 cross-attention 的 FFN write 前基底** R，`p['attention_input']` 仍是 cross-attention 结果，不能再次加它；也不能用 W1 callback 的 `x` 替代 R，因为后者是已规范化/调制的 V1。若必须保存 attention 前 x，须在 `_old(phase,...)` 调用前 clone，否则引用会被 `add_` 更新。保存既有 R 应在 post_process 前立即 clone 或在同 stream 安全读取；不得对真实 R 原地做 shadow 计算。
4. FFN 在 R 上 `phase.norm2.apply`、有条件的 dtype 转换，再 `modulate_func(norm2_out,scale,shift).squeeze()`；当前配置匹配的具体 norm/modulate 实例与无 MXFP8 融合应在新 producer 运行时锁定，不能据 `norm_modulate_backend` 名称独自推断 `modulate_type`。同 dtype 的 [torch LayerNorm](../../../.runtime/code/LightX2V/lightx2v/common/ops/norm/layer_norm_weight.py#L222) 直接传原 dtype 输入，具体内部累积精度不能由 Python 代码断言。W1 → `gelu(approximate='tanh')` → W2，返回的是 **pre-gate Y**；gate 未在 `infer_ffn` 中乘入。
5. [`post_process`](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py#L418) 对 BF16/BF16 同 dtype 精确调用 `x.add_(y * c_gate_msa.squeeze())`：先形成乘法结果，再原地加到 R，两个操作各按真实张量类型与 kernel 舍入；不应替换为 FP32 融合乘加后一次 BF16 cast。若 sensitive dtype 不同，分支则是 `x.to(sensitive) + y.to(sensitive) * gate.squeeze()` 并返回新张量，此审计不将它混入 BF16 结果。

既有 block20 四步采集记录 gate 实际为 BF16 `[1,1,5120]`、stride `[30720,5120,1]`，见 [真实 header](../../../research/20261030/CHIPMUNK_WAN_FULLM_RASTER_RESULTS_20261030.json#L515)；`squeeze()` 后 `[5120]` 对 `[31668,5120]` 所有行广播。该记录限定既有源与层；新观测必须在每个 site 实际核查 gate dtype、shape、stride 和 broadcast，拒绝不支持的形状，不能无条件把其他 shape 当逐行切片。R、candidate Y、Dense shadow Y 与 V1 均需校验 `[31668,5120]` BF16；old/current h `[31668,13824]` BF16；mask `[5120]` 为列索引。被测 `gate`、R、candidate Y、V1、old cache 及 W1/W2 均为只读，禁止 sink、write_formula 或评估代码修改。

## 最小忠实写回误差及其验证

对同一 student 输入和同一预写回 `R`，用两个独立 clone 逐一执行：`C = R.clone(); C.add_(candidate_Y * gate.squeeze())`，`D = R.clone(); D.add_(dense_Y * gate.squeeze())`。再用 `C.float()-D.float()` 度量写回差；逐 chunk 同样有效，前提是 gate 广播严格对应行、clone 的时点与实际 `post_process` 一致、op 的 dtype/shape 不变。省内存可每个 chunk 只持有 R 的两个 clone 和两次乘法结果；**绝不可直接对 R 调 `add_`**。这个量仍是同状态本地反事实，不是两条自由轨迹视频差的归因。

最强自校验是在 wrapper 捕获 R 前，构造独立 `C`；调用一次真实原始 `post_process` 后，捕获其返回值，逐字节比较原生输出和 `C`，并核对 `y`、gate、R 预存快照不变以及 native 输出与原 R 是否同对象。校验不可调用原 `post_process` 两次作用于真实 R：该操作原地修改。对 Dense 参考的 `D` 只能是同状态影子写回，且其 reference GEMM 需有单独真实性验收。此校验需要另审注册实际模型运行；软件 fixture 可先用相同函数骨架、非零 attention、异号 gate、BF16 rounding 边界、非零大残差、重叠别名和 chunk/squeeze 对照，证明公式及错误实现不同。CPU fixture通过不代表 GPU kernel、完整轨迹或 shadow neutral 已验证。

## Shadow位置、stream、缓存和风险

草案 observer 在实际 `readonly_action` 的 `out` 完成之后、构造并返回 sentinel 之前调用；此时原始 FFN 正等待 W1 callback，R 已经过 cross-attention、原生 post_process 尚未执行，old h/Y 仍在 CacheBook，phase 当前 W1/W2 及 bias 仍属于活跃 block。这是源码上可行的截点，但不是安全性已验证结论。`validate_active_compute_buffer` 验证 compute stream 及活跃 `cuda_buffers[0].compute_phases[2]`，[manager](../../../.runtime/code/LightX2V/lightx2v/common/offload/manager.py#L77) 在独立 load stream 预取另一 buffer，`swap_blocks` 在 compute/load stream 同步后交换；shadow 和 D2H 必须在同 stream 完成或以事件管理生命周期，sink 不得延迟读取未来将被覆盖的 buffer。当前 `charged` device-wide synchronize 可保护已提交工作完成，但引入大量同步、阻塞并改变资源/时间，必须计账且不可拿作公平 benchmark。

[`MMWeight.apply`](../../../.runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py#L344) 使用 `torch.addmm(bias,input,weight,out=BF16)`；observer 草案用转置/contiguous 的 `F.linear`。数学同构但权重布局、GEMM调度、累计和舍入不保证位相同；应校验同一 V1、权重和 bias 下 shadow 与真实 native Dense 路径的 h/Y、以及在真实 full-M shape 下可接受的误差，再描述 reference 身份。`w1,w2` materialization 如果 alias 原 live buffer，必须视为不可变；如果复制则支付读、分配和可能的同步。全M shadow 在 sentinel shim 启用期间调用 F.gelu，只有 sentinel 对象才被短路；验证 current shadow 的输入绝不别名 sentinel。

observer 草案的 [`paired_errors`](../../../tools/chipmunk_student_observation_math.py#L2) 使用 FP32 的 `ideal_gated_error`，这可作为理想指标，但不等于实际写回差。`write_formula` 尚未提供；此前只能称 pre-gate 与理想 gate 误差，不能声称 actual write。`sink.gate_slice`、数值精度、跨chunk累加、gate 形状等仍需实现及冻结。`omission_energy` 的全隐层 W2 FP32 扩展/高级索引与每 chunk 重复权重及布尔mask构造可能占显存、带宽和额外算子；固定 4,960 chunks只是调用次数，不能忽略临时分配、`torch.unique`、mask 检查、同步和实际硬件数学模式。FP32 投影的 cross 是其数值实现下的诊断，不是数学无限精度精确值。

## 反例与停止条件

选 BF16 很大的 R 和很小的 `y*gate`，两个不同 pre-gate 输出可有非零 FP32 理想差，但两次实际原地加都舍入到同一 R；反之中间乘法舍入和 residual 量化边界可使 FP32 一次cast 的近似显示不同值。再用非零 `attn_out` 构造输入，如果把 `residual_input` 在 W1 时误认为 attention 前 x、再加一次 attention，便会在每个重放中注入错误基底。测试必须对比原生源代码式调用，采用不同 clone 并断言被测输入保持不变；记录这些只是软件回归，不替代 GPU neutral。

正式使用前另行复审 observer 的 source closure、实际 W1/W2 backend/weights/stream、BF16/write replay 比较、全状态与完整视频 observer on/off byte gate、成本和预算；未满足时停止实际写回结论。既有视频结果和其他运行 scope 不为本采集提供授权。
