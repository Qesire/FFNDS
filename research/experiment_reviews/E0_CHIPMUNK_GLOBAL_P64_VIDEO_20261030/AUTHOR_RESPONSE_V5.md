# 作者回应 independent review V1

FFN0候选动作在首次读取当前block的W2前，新增 `validate_live_w2`：逐对象核active tap mapping的同expert ffn.2、原始MMWeight精确类/模块及无adapter、BF16 CUDA tensor、权重[h,d]及(1,h) stride、BF16 CUDA bias[d]及stride(1)，再进入packed W2。合成负例覆盖错误mapping、非BF16、错误stride；真实 `mm_call` 路径对错mapping在任何ledger/packed action之前拒绝。无此guard的V4未执行协议已归档。

实际CUDA初始化后持久化Torch/CUDA版本、设备类型、allow_tf32、allow_bf16_reduced_precision_reduction和float32 matmul precision；不导出设备UUID。保留root指出的旧Triton路径REVISE，以及首high/low复用同mask原packed全M对照；后者先写RUN再判，诊断资源预扣/同步/计费。10项CPU合成测试通过；未运行GPU/model或新实验。两part、CPU/GPU/VERIFY私有job需对修订版重新精确独立复核。

root另要求核提前使用W2的ready：已读绑定原源码 `wan/infer/offload/transformer_infer.py`、`common/offload/manager.py` 和 `common/ops/{utils.py,mm/mm_weight.py}`。首buffer在init stream加载后同步，后续buffer在load stream预取并在swap前load/compute双流同步；当前block由buffer0在compute stream执行，其FFN phase为compute_phases[2]。原MMWeight在无diff时直接取self.weight/bias，load_state_dict已将二者copy到buffer，没有apply时的额外W2 prepare。新运行guard在candidate真实FFN开始检查lazy_load关闭、phase确为active buffer0且当前流即compute stream，并在W2操作前核mapping/dtype/stride与weight、bias的自有CUDA prefetch buffer data_ptr一致。manager持有缓冲且本动作同步完成后才可能swap，未引入隐式额外拷贝或免费省掉原预取；异常或未满足即失败。11项CPU合成测试含错phase/stream和错W2拒绝，未执行GPU。

root随后核算：两次旧helper observer还各含一次全1 gate乘法；V7增 `observer_gate_product=2` 和两次M×D BF16逻辑输出字节，独立于原post_process。P64分数内部的输入/旧h gather、proxy W1、GELU和RMS也逐项同时预扣；11项CPU合成通过。
