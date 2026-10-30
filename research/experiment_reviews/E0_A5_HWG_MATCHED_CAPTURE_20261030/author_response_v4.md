# Bootstrap 与观察成本的最后修订

作者 `/root`，不预填批准。

执行入口现在先启用wall与文件上限；仅做≤64KiB job peek来定位supervisor创建的fresh HF目录，然后立刻建立 `STARTUP_READ_LEDGER.json`。protocol、千余源码、registry、review、private locator、原模型/cache/包的实际读取都在这个durable ledger可持久预扣后开始。source/protocol读取失败也能留下actual返回字节、inflight UNKNOWN及失败startup记录；成功进入VERIFY后由固定RUN继续持久预扣。新fixture在bootstrap预扣10字节、返回3字节后抛错，独立检查失败记录仍保留3/10和UNKNOWN。M1+工程共20项小fixture通过。

观察器不只执行runtime的1600 linear/QDQ。合同新增每site额外sample GELU=1、sample index_select≤24、R clone≤2/add≤3、Y×gate≤1、GPU equal≤6、CPU/CUDA RNG-state读取≤6、CPU finite tensor检查15、fullV1 D2H1、sample field D2H15及W2 hash/D2H边界；实际sample GELU和index_select另记计数，完整160位置必须恰有160个额外sample GELU。这些算术与CPU/D2H/同步均属被动观察额外成本，采集wall绝不称为FFN性能。

存储张量raw SHA的内存读取、最多320份W2 raw SHA内存读取以及状态/metadata余量分别列上界；actual Writer计数只覆盖其自身负责的raw读取，继承W2校验按SRC边界另列，不能把该计数称为总GPU工作或总物理I/O。原runtime linear/QDQ计数仍只是模型800模块×两次expert调用的原数值路径覆盖。

三个worker的独立M0包核验现均真实通过：每个2302文件、42,053,803字节，与相同旧inventory逐文件SHA一致。56GB现存cache与新HF镜像也均真实全SHA通过。仍需最终精确代码/protocol/jobs复核后才登记启动；没有任何native新范围自动重跑。
