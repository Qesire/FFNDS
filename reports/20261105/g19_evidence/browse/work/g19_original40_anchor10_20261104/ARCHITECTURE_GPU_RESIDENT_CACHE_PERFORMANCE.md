# 原模型40步锚点：驻留位置与性能边界（源码/形状分析）

状态：设计分析，不是候选或系统加速实测。当前已审视频范围继续使用CPU FP32 C，不改变冻结配置。以下 T=31668（对应实际网格的一个已知形状）、d=5120；T随图片裁切变化时应重新计算。CAL日志报告的GPU峰值 allocated=10,536,513,024 bytes=9.813 GiB，是原Dense CAL路径的实测峰值，并非候选运行的allocated/reserved峰值；因此下列峰值叠加只是容量估计。

单个expert、单个CFG分支、单层的FP32 C为T*d*4=0.6040 GiB。一个当前expert的40层×两分支为48.3215 GiB；40个BF16 A为1.9531 GiB。两者与CAL峰值相加约60.088 GiB，距离候选76 GiB PyTorch缓存分配器预算约15.91 GiB（以allocated与缓存预算作粗略对照，二者不相同）。仅按这些已知主体估算，GPU常驻C值得后续验证，CPU保存不是由形状必然决定；不能以CAL峰值直接保证候选能运行：峰值不一定同一时间，CAL未计算A或存完整C，PyTorch reserved、显存碎片、其他进程和实际shape也未量到。

锚点现场还有 BF16输入克隆、当前F和GEMM结果（每个0.3020 GiB）、FP32预测与新C（每个0.6040 GiB），更新同key时旧C在新C建立前仍存活。当前质量hook的重构数值检查将整网格error、ref转为FP64，每个1.2080 GiB，平方归约还会产生全网格临时；与CUDA GEMM workspace、Attention/预取双buffer、实际分配顺序可能叠加。命中步还会有FP32预测、BF16输出和转换中间值；驻留时应逐操作记录allocated/reserved峰值及NVML可用量，不能只凭最后余额判断。切换expert时先释放旧expert两分支全部C并确认stream完成；否则第二份48.32 GiB就使主体超过108 GiB。预加载下一expert的40份A会额外占1.95 GiB。避免全网格FP64误差副本或改为分块累计，属于新数值/诊断实现，需独立审查并验证与原检查一致，不能回写当前视频scope。

当前实现全网格C位于CPU；按推定切换step15、40层×2 CFG，40步共400个anchor、2800次hit。每次anchor写约0.604 GiB到CPU，累计约241.61 GiB D2H；每次hit将同形C复制回GPU，累计约1691.25 GiB H2D（1.652 TiB），不含A、原始权重、临时张量或诊断。实际计数依运行收据而定；传输时长、是否重叠及带宽须实测。视频scope属于带shadow/哈希/IO的质量运行，不能用其墙钟推出干净速度。

同一T下单FFN在标准2次矩阵乘FLOP口径（乘加记2）下约8.966 TFLOP；一次u@A约1.660 TFLOP，约前者18.52%，不含GELU、norm、gate、缓存、attention及调度。当前候选hit通过替代返回值实际跳过全网格W1/W2 GEMM，但代码仍执行一次d宽GELU及64 token shadow；不应将前者算作零开销或把小M shadow与原全M计算视为相同舍入。

源码显示 block CPU offload 的 WanOffloadTransformerInfer.infer_with_blocks_offload 在当前block计算之前调用 WeightAsyncStreamManager.prefetch_weights(next block)，后者 load_state_dict 整块；当前 infer_ffn hook在整块加载之后才决策，因而跳算W1/W2并不自动跳过它们的权重加载。每层W1与W2 BF16总计 2*5120*13824*2=283,115,520 bytes（约0.2637 GiB）；若真实同精度整块每hit都传入，2800 hit的两矩阵合计上限约738 GiB，具体传输与重叠仍要用profiler核实。最小可部署边界：在调度/预取前依据实际step、expert、CFG branch、layer及已存在C判断anchor/hit；anchor和缺cache时完整加载W1/W2，hit时只排除这两权重而保留norm2及其参数、attention、gate和其余block内容。双buffer和跨stream同步必须保持正确，尤其末层预取循环到下一分支/下一step时，下一次branch/专家与缓存可用性若未确定应保守完整预取，禁止引用另一分支/专家缓存。移除多余GELU、精简权重复制、GPU驻留C及异步重叠各自改变执行路径，应在现有质量结论后另建scope，以同输入噪声和dense/neutral比对、逐动作trace、真实H2D/D2H字节、FFN与整系统时间、allocated/reserved峰值验证；不可把算术比例或理论传输量称为实测速度。
