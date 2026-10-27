# UB1 disjoint20 保存状态增量：独立冻结前审阅

**结论（SRC/ALG/CPU 合成测试）**：数值定义与 scheduler 源码审计一致；当前草案尚不能冻结运行。其输入清单仍为空，且 attempt 收据在环境与全部输入预检之后才创建，因此预检失败不会在本 scope 留下失败扣账，需调整后重新审阅。真实 recovery part 0 与原 part 1–3 的身份和 244 项输入尚无法从空草案核验。

审阅版本：工具 SHA256 `8295082f88c13e46bd074dfc71d1f78bfe45e9d6834642c6d9e4e50175aaa630`，测试 `84cceecd814c351584db3cec0ebf80b75e4c40ab094a589327a015524479d01f`，草案 `6f4b858c46ed534ee0ff94e3c4238de593c64a821984cba8e51b277e43419d30`。项目虚拟环境下合成 unittest 5 项通过，exit 0；未读取真实 tensor，也未执行 GPU、模型或真实数值分析。

初始 BF16 从原始16位字解码；post FP32 检查形状、finite 和低16位零，并核对原始字节 SHA。每 source 的 Dense 与 candidate 初始文件完整字节比较。将四个状态先升格 FP64 后计算增量差、先前误差及其 signed cross，并验证能量闭合；零范数 cosine 为 null，非正步长拒绝。`axis=1` 的差分是 21 latent 时间槽之间的20个差值，不是四个 denoise step，也不代表光流或主体动作。有效速度只是保存状态差除以实际 sigma 差，不能恢复原 native noise prediction 或舍入项。汇总按 source 等权，四步嵌套于每个来源；这不是80个独立 source。

预检逻辑要求20个来源、四个 producer、40份 generation 与200个 tensor 的互不重复路径及 SHA；校验来源、seed、模式、生产者状态、轨迹及初始噪声身份，强制 part 0 使用不同于原协议的生产者协议、其余三 part 使用原协议。最终绑定仍须核实 part 0 确为完成的 recovery RUN，而不是仅凭不同协议散列；还须核验实际 source 分组、40生成收据、200 tensor 和配对 sigma 与冻结清单。原失败 part 0 应继续隔离，不与 recovery 拼接。

计划列出 2 CPU、1 GiB RSS、300 秒内部及360秒外部、单次、零 GPU/模型调用。输入在路径约束和 SHA 核验下只读；内部计时器和 RSS 检查在 preflight 后才开始，整个过程的时限与内存上限需由外层 launcher 覆盖。保存结果可描述状态响应及竞争机制，不能直接推断视频质量或原生 flow。

## 最终资产复核

修订工具 `6bcc09889856fe58cde5272e65cc64aeff87ad7f6f4d994f0805a12992eefd06`、测试 `c381361ed809d1f955d23a39b4c08d7eca592d141eede18e711b7ed3c5866de5`、冻结配置 `1caba11c1ac1921ce881038b34012cd4715460b909f7c933ab64d9609b36cae5`：7项合成测试 exit 0。先前预检前未创建attempt的缺口已修复；只读实际资产 `preflight` 返回 True，核对244项绑定及恢复part 0和原part 1–3完整生产者收据。实际运行依赖外层RLIMIT_AS和360秒进程边界；此次未解码或分析真实tensor。
