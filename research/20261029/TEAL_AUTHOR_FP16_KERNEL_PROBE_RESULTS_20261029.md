# TEAL 作者FP16单token内核：真实有限验证（2026-10-29）

**V2实际EXIT0，闭环验证PASS；两个门限均进入真实作者稀疏核并通过masked Dense数值检查。这是单shape作者内核probe，尚未复现完整LLM或Wan方法。**

固定作者commit `fb7373c93ac3594817c9ee64d4e08b47430a1822`，A100 SM80，Torch2.4.0+cu121 / Triton3.0.0，FP16，输入1×1×4096、列主序权重14336×4096，seed42。先实际初审APPROVE、作者回应、独立recheck APPROVE，再登记精确ACTIVE、启动规格和版本指纹。

| 门限 | 实际置零比例 | 对masked Dense相对L2 | sparse事件调用中位数 | Dense事件调用中位数 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0244% | 0.001257 | 0.202752 ms | 0.139264 ms |
| 0.25 | 51.1230% | 0.001249 | 0.192000 ms | 0.146944 ms |

两bin实际都选BLOCK_M64/BLOCK_N128/2warps。62次作者forward、dispatch和Triton入口计数闭合；自动调优内部物理launch数UNKNOWN，不能把62当总GPU核次数。60次Dense计时调用、两次数值参考、一次seq2 fallback及两个对照也均闭合；GPU峰值494,927,872B、内部wall32.110s，编译/调优/数值检查24.082s。全部配对20次原始事件与同步wall区间保留在[JSON](TEAL_AUTHOR_FP16_KERNEL_PROBE_RESULTS_20261029.json)。

本测量是带Python入口观测、逐调用同步的平衡交错eager调用区间；本条件下没有速度优势。作者原脚本使用Triton do_bench、不同cache处理和重复安排，不能以本表判定论文性能失败，或把差额直接归为Python开销/权重流量。下一项须独立复审更忠实的作者harness；保留原样核与观察成本的区别。

seq_len2的作者输出与**未掩码**Dense逐值相同；相对masked参考差异0.380145。它确实回退普通Dense，并不应用门限。Wan大token迁移需要独立的真实物理算子，而非直接套用这个单token核。

V1实际EXIT1、25.186s，在Triton导入时缺setuptools，所有直接矩阵/forward计数0；原RUN/EXIT/环境保留。V2新环境独立增加setuptools/wheel，无CUDA导入软件检查EXIT0，硬链接去重新增约13.3MB；V2是另经复审的一次有限范围，无自动重试。

本结果提供实现与数值锚点，未包含LLM质量、原版解码吞吐、Wan BF16/GELU全160视频或系统Pareto。下一项并行准备是6cal/2dev已有全160位点的真实门限局部基线，执行前另审，不用本内核的批准覆盖。
