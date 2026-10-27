# 固定 UB1 支持与 R256 的全160处消费者重构

**已测现象（NATIVE-OBS，保存状态、训练内）**：24个原校准来源、160处FFN的3840个嵌套source×site上，真实BF16主分支与原R256相加后的平均gate输出相对SSE从 **0.055510降至0.009434（下降83.01%）**；3840/3840训练来源位置更低。重新拟合输出消费者在原支持下恢复了相当一部分输出信息。第一步仍承担新残余误差均值的约90.1%，支持“同一固定支持的可恢复程度随step不同”这一竞争解释，也可能受到固定ridge正则、有限来源及教师状态限制；这尚非容量下界或视频因果。

| Step | 原 UB1 平均相对SSE | 修正后 | 训练内下降 |
| --- | ---: | ---: | ---: |
| 0 | 0.112796 | 0.034013 | 69.85% |
| 1 | 0.047771 | 0.0017355 | 96.37% |
| 2 | 0.044287 | 0.0017470 | 96.06% |
| 3 | 0.017185 | 0.0002392 | 98.61% |

每个step恰有24来源×40位置，先按source×site等权汇总；独立来源数为24，3840不是独立视频数。所有24来源都用于拟合。本结果区别于此前4位置、12/12来源交换的[匹配CV](UB1_MATCHED_RECOVERY_RESULTS_20261027.md)：该CV消费者之后另拟合R256，本次固定原R256并拟合其剩余误差，不能继承其39%持出局部改善。

原支持54/108/108/162、alpha、折叠bias与R256均以旧参数身份载入。以每源M256真实保存post-Q1/post-Q2，先执行旧BF16 main+RR，再对 `float(Y)-float(BF16(main+RR))` 解固定0.01 trace/N的中心化FP64 dual ridge，折回新BF16 W2/bias。评价使用新BF16线性、原RR一次相加及实际gate乘积；非实数求解残差，也没有残差写回、学生前缀或全M/自由视频。首次完整视频前不据本分数换λ、来源或支持。

四个预先固定10-block分区实际各使用一张空闲A100，同时峰值4张；各40 fit与3840次BF16预测，实际全部退出0。总计160 ridge fit（含FP64 Gram/Cholesky/solve等）、15360次BF16 forward linear、160个新consumer pack，0新R256 fit、W1、QDQ、全模型、视频或官方评分。四分区含设置计算751.95–782.80秒，outer launcher766.95–795.01秒；这是并行分区时间，不能加作单任务wall。所有已执行源码/协议快照及160文件SHA/大小、原W2/b2/gate绑定核对通过，1408次资源采样无违规。

单bank权重/偏置 **11,326,283,840 bytes（约10.55GiB）**，HF全部新增约11.332GB，小于12GiB冻结限额。它替换所选W2及bias，线上主GEMM宽度/数量不增加；真实加载、驻留、setup、全160 FFN与视频成本仍由新的运行器独立计量。原模型分片未搬迁，第二执行环境只补齐1.60GiB系数/元数据。历史recipe原SHA与当前公开文件不同的事实保留，新的参数链直接绑定实际payload，未修改旧实验身份。

冻结[协议](../../config/ub1_fixed_rr_consumer_bank_20261027_v1.json) SHA `d35b0d1233b762236eb75d98a3f93484b639766266280b139008d4fef26d88fe`；[机器结果](UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.json)含四RUN/launcher与逐step统计；完整160 pack manifest位于HF，SHA `c0a29a8c1adace7e7223aa64c9136ff7dba49c44deef034973d2fed7c1bf1ff5`。独立[数学/预算审阅](../method_discussion/discoveries/E0/ub1-matched-recovery-design-20261027.md)和[原生运行器审阅](../method_discussion/discoveries/E0/ub1-native-consumer-independent-review-20261027.md)保留确切版本。

**下一判别**：同一当前图片、source/seed，两个已暴露formal20例各新跑Dense→原UB1 neutral→candidate。neutral必须复现旧UB1全部四个latent与完整视频SHA后才允许candidate。完整81帧core报告、真实全160执行计时及后续独立官方多维/人工评价，分别检验局部恢复能否传到视频功能和系统成本；训练内SSE不作验收门槛。
