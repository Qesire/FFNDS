# SharQ Wan FFN V2 独立范围：作者自审

原 scope 真实 EXIT1，RUN/EXIT/stderr 由本 scope 直接绑定并校验；高 expert、M32 的 BF16 和 group16 两完整 case 已执行，dense32 的全部 15 次 FFN 也已完成，但其首个 W1 同编码参考的权重 scale 解码失败，SharQ 和 M31668 均未执行。失败和消耗保持原身份，V2 是重新登记的独立 240 FFN 范围，先前执行的两臂重跑并再次扣账，不重写失败产物，不自动重试。

源码证据：作者 `kernels/src/bindings.cpp` 的 get_sfa/get_sfb allocation 均为 `(rows / 128 + 1) * 128 * K / 16`。rows 是 128 倍数时比 ceil-tile 多一完整 128 行尾部；W1 rows=13824、K=5120，作者实际分配 4,464,640 字节，有效 tile 4,423,680 字节。原 decoder 将两种不同 producer 的长度都严格等同有效长度而拒绝；author 版本现要求完整作者长度、截取完整有效 tile 前缀后按原 LightX 已验证的 128×4 物理 swizzle 解码；LightX 继续只接受自身 ceil-tile 长度。author K 对齐 64 是这次两个 K 的真实条件，额外未对齐 K 一律拒绝。作者 bindings、reorder.cu、reorder.cuh 已加入精确作者源码绑定。该修复不改量化、两 GEMM、scale 值、阈值或 FFN 数学；失败对应的是参考中的权重 scale，不代表 dense32 数值正确性已通过。

目标 worker 的独立 CPU-only 合成测试覆盖 N=128 的 author 额外尾块、LightX 精确长度及错误长度拒绝、N=129 无额外尾块、实际 dense32 参考管线，以及稀疏+残差和故意错 payload。它们不执行真实模型、GPU kernel 或新科学实验。请复核 swizzle 有效前缀与作者 CUTE layout 对应关系，而非仅检查长度；同编码参考保留 0.03 固定阈值，数值协议未放宽。N=31668 仍可能触发作者 CUTLASS can_implement 失败，须按真实失败保留。

保留原 scope 的四臂、两 expert、M32/重复32行到 M31668 的吞吐诊断、15次/每组合、原 W/同状态 gate 和 Q2、实际 payload 对照、两种计时、720+实际逐动作预扣、full-K residual；整个 730509120 字节 bundle 单次预扣重新 hash，mmap 实际物理读 UNKNOWN，H2D 同步记账。只评估单开发源 operator，不推视频质量或完整系统；observer debug 与作者 selector 共用源码，非 selector 数学独立证明。输出、protocol、ID、私有 job 和 attempt 均为新身份；单卡、4 CPU、3600 s、无自动重试，准确资产和旧失败先绑定后独立审查和 ACTIVE 注册。
