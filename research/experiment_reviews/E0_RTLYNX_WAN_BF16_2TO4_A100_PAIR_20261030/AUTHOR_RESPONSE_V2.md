# 作者回应独立初审

接受 GPU 异步反例。每项 extra（reference mask、converter、dense W2、比较、fullM repeat 与输入有限检查）在同一个预扣调用内完成 `torch.cuda.synchronize()` 后才记 Python returned 和清除 inflight；异常时在途完成 UNKNOWN。每次额外调用的 CPU wall（包括同步）单独记录；bundle hash也走此同步边界。合成注入测试覆盖已发调用后同步失败。完整 FFN 的 event 包含同次同步，先整体预扣，失败保留 UNKNOWN；CPU 对账不把硬中断后的 CUDA 完成当作已知。

额外增加 M32 四个真实稀疏case各一次独立稳定 argsort 选择参考，逐字节与实际 materialized mask / 行范数后的 BF16值比较；M31668 沿用实测过的选择代码和本次 converter packed/meta、同mask dense W2参照，不声称独立逐位验证所有大形状输入。归约规则仍为公开声明的每行 Wan 适配，论文作者代码轴未证实。source closure 额外绑定 parent 与 paired 所有已导入源码；真实闭合 bundle 与 C5 绑定不变。

预定预算的 4 次 independent_select_M32、8 次 reference mask/converter/dense W2、60 次完整FFN与 event 等均记录入新协议。三项数值候选的实际输出与公平系统速度尚未测量；固定顺序短时间采样保持诊断身份。8项合成测试通过，未运行科学数据、GPU或模型。
