# TEAL 原作者 Llama3 全32层生成：实际失败范围中的部分结果

9次各64 token 的生成与5个 neutral 检查已完成，但后置 observer 在首个观测前向时因 `Attention` 没有 `w1` 属性触发 `AttributeError`，整个范围实际 **EXIT1、未 closed**。失败保留原始身份：没有采到任何 mask/原零统计，第二次 checkpoint SHA 也未执行；不能将本范围标为整体成功。

Dense、truezero 和 uniform50 每臂各1次 warmup及2次测量，实际完成9次 prefill、567次 decode和10次 teacher 调用；observer只完成1次前向，mask/零值归约完成0次。三模型均实际FP16加载，32层共享相同KV容量布局。5个 teacher neutral 的 top1 均相同，最大 logits 相对 L2 为 **0.00306447**；该检查为 eager，不能代表 compiled decode 数值全程相同。该固定 prompt 下 Dense 与 truezero 生成的64个 token相同，uniform50输出不同，不能从单个 prompt 判断模型质量。

| 测量臂 | 两次测量生成墙钟（秒） | 平均（秒） |
| --- | --- | ---: |
| Dense | 0.676240、0.697454 | 0.686847 |
| truezero | 0.743556、0.747661 | 0.745609 |
| uniform50 | 0.491716、0.487861 | 0.489789 |

这两次测量的 Dense / uniform50 平均墙钟之比约 **1.402×**，是单 prompt、三模型同时驻卡、逐 token CUDA event 插桩下的局部诊断；只用两次测量，不是置信区间，也不是作者单模型独占速度、完整系统或论文 benchmark 复现。warmup 分别为 75.703、58.552、30.788 秒，包含首次编译和调优等成本。缺少实际 mask 统计，不能推断观察到的稀疏率；未测 PPL、广泛文本质量或物理 kernel 次数。

子进程墙钟 259.190 秒、外层 263.457 秒；全卡显存采样峰值 48,358 MiB，120次采样未见外部 compute；完成一次完整 checkpoint SHA、256次 histogram 加载，第二次 checkpoint SHA未完成。9次生成的所有 token IDs、分项计时、neutral和逐项预扣/完成账见[白名单结果 JSON](TEAL_LLAMA3_AUTHOR_FULL32_GENERATION_PARTIAL_RESULTS_20261030.json)。实际 RUN SHA `d02ee849b92bb3e895ff61ec693be7d0d15d76406e16d5cb971fb020c0ae8bad`，EXIT SHA `685a762867f14b6c71fad4b852f383b452ea60fa10ec920e5f6de6e68f645f05`。
