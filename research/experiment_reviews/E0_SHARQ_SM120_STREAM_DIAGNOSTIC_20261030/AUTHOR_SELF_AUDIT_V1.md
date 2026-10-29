# 作者自审：SharQ SM120 流契约与只读完整性，待独立评议

问题：V6 BF16 cache neutral 在两来源第0步flow nonfinite，打包发生在LightX非默认流，而作者裸kernel使用默认流。竞争机制：生产/读取跨流无序使编码异常；kernel越界或临时生命周期污染BF16相关内存；另可能有尚未观测的非流原因。二者都有可证伪预测，当前不能确定根因。使用实际已闭合paired bundle source911、两expert block20真实W1/W2、对应M32真实V1/h及bias；这不是完整模型、视频或性能评测。

两完全独立fresh GPU scope，SAFE先执行且必须真实EXIT0与closed，UNBRIDGED之后才可单独登记/启动，避免未定义行为污染安全对照；每卡严格worker/UUID flock、单GPU、会话最多4卡，失败保留且不自动重试。SAFE对4矩阵各1次默认流reference及1次非默认流event桥接，合计8次权重打包，8个Dense32物理路径+8个SharQ物理路径（每条SharQ含实际sparse和full-K dense residual）。作者量化准备、激活重排、CUTLASS计算、bias均在默认流；非默认调用先compute→default事件，输入递归record_stream(default)，然后default→compute事件、输出record_stream(compute)。同bundle真实权重与M32输入，原GPU buffer视图为[K,N]、W.T.contiguous()物理同储[N,K]，记录/验证实际stride和alias。各case生成时在原stream入队copy和BF16 addmm前测，不先强制生产流与默认流同步；量化后和结果使用前显式device同步且计账。

对W与输入分别逐位检查未改变，真实BF16 addmm pack前后逐位相等并检查finite。SAFE对桥接packed Q、稀疏32与Dense16的实际逻辑scale坐标及global scale相对默认流逐位比较，跳过分配padding，Dense32/SharQ输出分别对默认流比较并预冻结relative L2 ≤0.001和finite。单矩阵两线性比较不是完整FFN/GELU/gate；编码/metadata比较为同作者路径，不声称独立证明算法。UNBRIDGED使用独立进程4次打包，每个权重一次；原stream与默认stream间不人为加入先行依赖，当前ledger文件持久化可能给生产流更多时间，故单次无异常不能证安全。其packed digest与前述实际SAFE reference比较，只记录差异，不自动宣称因果。BF16 pre/post不等或weight/input改变直接失败并保留失败账；SAFE结果完成也不能宣称全模型neutral修复。

每个scope 1GPU/4 CPU/1800s（outer1920s）/显存软上限32GiB/host RSS16GiB、新HF256MiB，bundle全文哈希一次730,509,120B预扣、mmap物理读UNKNOWN，H2D逻辑字节预扣/同步后完成；每次packed digest D2H总额≤1GiB。预扣并记录实际Python入口/返回：SAFE W pack8、逻辑scale提取16、原BF16 addmm24、Dense32及SharQ各8、对应prepare/dense/sparse/residual物理入口各8、output_scale和bias各16、bitwise及输出比较、finite检查、case准备和49次device sync；UNBRIDGED W pack4、逻辑scale提取8、BF16 addmm8、无候选GEMM、对应只读完整性检查及17次sync。CUDA kernel内部次序和同步前中断状态UNKNOWN；所有输出源码身份、本scope worker/monitor、原V2 operator actual EXIT0与两V6失败RUN/EXIT绑定。bundle中非选中模型不运入GPU。仅使用已构建作者ELF，无新下载。

请独立审查：作者两组GEMM默认流依据、事件以及tensor异步分配生存期、复制/alias与父真实buffer一致性、unbridged是否实际保留竞争及独立进程、包装对象内tensor record_stream、真实物理调用和digest计数、SAFE未完成时unsafe禁止、输入/身份/输出预算、强反例和exact private spec。草案不授予ACTIVE，尚未运行实际数据、数值、GPU或模型。

同步异常策略：桥内函数抛错时同步drain默认流与caller，保留原异常；同步失败附注异常并终止scope，CUDA完成仍按UNKNOWN，不继续后续case。新增CPU合成测试覆盖中途异常drain，以及scale按源码CUTE坐标跨row/group边界、padding变更不影响与逻辑元素变更被检测。精确V4协议与私有spec由评议者审核；旧V1是未执行废稿。

作者回应独立初审与原生BF16负零反例见 AUTHOR_RESPONSE_V2.md。源/ELF、4个真实失败收据、bundle分别预扣，授权前小身份读取由supervisor负责记录进程退出。

每case先持久化BF16 byte/finite通过gate，才能开始任何候选GEMM；候选结束后再次逐位检查W/输入/原BF16输出。此顺序与额外操作均计预算。
