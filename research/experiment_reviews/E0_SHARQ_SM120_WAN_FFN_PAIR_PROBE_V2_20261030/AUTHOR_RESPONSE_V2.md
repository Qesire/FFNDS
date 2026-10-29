# 作者对 V2 初审的回应

真实初审结论 REVISE。1. 作者 `reorder_quantize_x` 用 `torch::empty` 创建 padded SFA；其中未写的其他行不构成可比的值。修订保留两种独立严格分配长度检查，将 scale bytes 按作者写入的 128-row / 4-group tile 的坐标逆置换，并只逐字节比较实际 M 行、K/16 组。实际 q payload、全局 scale 仍逐字节比较；SharQ 的压缩、metadata、sparse scale 和 residual 参数均为作者 zero 初始化路径，继续比真实全部缓冲。合成反例将 M2 的未用第2行 scale 改变，完整线性参考仍通过；改变有效 row0/group0 则拒绝。

2. 新增独立坐标公式合成 fixture：按 row=0/31/32/127/128 与 group=0/3/4/7 填入非均匀的不同 E4M3 字节；按 `row//128` 外 tile、`group//4`、`row%32`、`row//32%4` 与 `group%4` 的源码坐标分别计算物理位置，并核逻辑字节和 Q payload 实际解码数值。覆盖 author 128 对齐额外尾块、129 非对齐、LightX 严格尺寸和真实 dense32 FP32 参考的合成路径。目标 worker isolated CPU-only 11 测试实际 EXIT0，无GPU或真实数据；fixture只证明实现与编码映射的单测一致，不声称真实设备数值通过。

初次 scope 的实际失败与消耗仍按原 RUN、EXIT、stderr 绑定，新的 240 FFN 预算保持独立。原数值阈值、full-K residual、作者 selector 共用 debug 限制、setup/observer成本及 M31668 可能失败均保持，scope 不含视频或完整系统验收。请求独立评议者核对修订后的源码、协议、私有作业和 fingerprint；APPROVE 前不注册、不运行。
