# 作者对初审的修订，待精确复核

初审真实结论 REVISE。1. 采用直接导入冻结的 author `model/quantize.py`，不执行 `model/__init__.py`；实际检查 quantize、hif4、sharq_loader 和扩展的加载路径，三份作者源码与构建 ELF 均进入协议。一次 CPU-only 直接导入工程检查真实退出 0，核对三模块来源且 CUDA 未初始化，记录见 research/20261029/SHARQ_CPU_AUTHOR_SOURCE_IMPORT_ENGINEERING_20261030.json；GPU 目标 worker 仍须再次核对确切加载来源。

2. M32 每 expert、每线性层、每量化臂重做同一输入的量化准备，独立按已闭合 NVFP4 探针的 E2M1 nibble 与 E4M3 swizzle 解码实际 W/A，TF32 关闭、以 FP32 GEMM 建立参考。group16 使用实际 alpha 和 bias；dense32 使用实际共享 W、动态 A 和全局比例；SharQ 用作者 debug 的逻辑 sparse payload 和 residual 做两次 FP32 GEMM，显式模拟中间 BF16 舍入，再比较真实 E 驱动的 sparse+residual 输出；实际 correctness 首次量化的完整操作数保留，重准备的 packed、scale、稀疏 metadata 和全局比例先逐字节比对；正常与 debug 的 sparse scale、residual payload 和 scale 还须逐字节相同且稀疏逻辑值非退化。逐线性 relative L2 ≤0.03 是预声明的实现一致性界，依照 BF16 两次累积容许舍入，未使用新实验调参。debug 与实际 prepare 共用 selector，因而不是 selector 数学正确性的独立证明。每个参考量化、debug、scale 转换、解码、FP32 GEMM 分别预扣，上界 12/4/4/28/16；组内返回与 GPU 同步分开记，失败前中断项保持 UNKNOWN。实际 alpha/global scale 有限正值检查纳入每个参考组。

3. 增加每臂每 M 的另一组 2 warmup+5 measured，只保留整体 CUDA event；原组保留逐操作 event。每组合共 15 次 FFN，4 臂×2 expert×2 M 共 240 次 FFN；两组使用同输入且计数均预扣。两者都保留 Python 调用与账本观察成本；分别标记整体事件和逐操作事件，不能推成公平系统速度。静态共享权重量化、bundle hash、行重复、H2D 与参考诊断单独记录。

M31668 的 can_implement 是否支持尚无静态证明；不可假报适配，实际失败保留并按未成功报告。真实 CPU paired bundle 闭合仍为前置，完成后再生成含完整软件身份、真实 bundle SHA 的协议及私有 spec，并请求不同作者 exact recheck；此回应不构成批准。
