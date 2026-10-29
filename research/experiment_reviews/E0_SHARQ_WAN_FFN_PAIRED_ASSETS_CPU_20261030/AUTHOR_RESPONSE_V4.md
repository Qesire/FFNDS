# 作者回应：V4 同状态原生边界补齐

V3 对评审V1两项逐条修订与合成测试保留；V4 将同一捕获包中原生 `a`、`h`、`Y` 一并按typed header/raw SHA/元数据严格校验并打包，与 `V1`、`gate`、rows共同绑定source911、step1/high和step2/low、block20。该补充为后续原BF16 W1+bias/GELU/W2+bias 的同状态 neutral 提供真实参照；GPU仍须预先冻结可解释容差和完成真实调用，不能据此声称已通过neutral。

两包完整SHA预算6,182,512B不变，typed range改为40次共730,505,760B，仍在原64次/768MiB限额内；每个新tensor实际读前预扣、原始与暂存byte SHA比对、bundle完整SHA单次另计。tiny双expert真实 tensor_rows→save_file→readback测试新增a/h/Y，并通过gate/shape错误与四个inflight中断边界；全部10项合成测试通过。V4 protocol/private job/source SHA已重建；尚未登记或实际提取。
