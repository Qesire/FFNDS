# 作者对独立初审 V1 的回应

1. 三 arm 加载后同样预扣、分配 `T+68`，作者向8取整，逐个检查32层 K/V shape `[1,8,capacity,128]`、cache batch/capacity 和 causal-mask shape；记录 prefill/decode 位置和 source-defined 下三角语义，运行前后再比形状。记下三次 cache allocation 及墙钟，三模型同时驻留。布局核对本身并不声称逐元素验证KV写入。
2. neutral 的 prefill 延续作者默认 SDPA，4个 teacher decode 使用作者 math-only SDPA context。明确标为 eager logits，不能证明 compiled decode 中 logits/top1 相同；真实 compiled 路径由9次生成测出 tokens 与失败，物理执行和数值复现保持有限身份。这样保留有效的原 zero 门又避免插入额外模型编译提前热缓存。teacher tokens 仍是固定 prompt 片段而非自主轨迹。
3. 每 token 双 event 的开销、三模型共驻留、CPU compile wrapper 和各 arm warmup/measured 墙钟、prefill/decode event 时间、加载/布局/阈值和 cache setup 均有标签或账；多流/graph 内物理 launch 不可从 logical count 推断。EOS 继续生成到64，并记录 token 和所有 EOS 位置。不是论文性能测量。
4. 校准 V1实际失败，生成器现仅接受 `E0_TEAL_LLAMA3_AUTHOR_GRAB_ACTS_V2_20261030` 的真实成功 RUN+EXIT，另需 32个 forward pre/post、实际 token SHA 对照、observer、monitor 和64文件身份；后续本范围对64文件独立预扣完整 hash。实际 module 来源含 `compile_wrapper.__file__`；预检成功必须有独立 supervisor 实际 EXIT0。真实父代尚未完成，因此草案不可冻结，最终 private specs/父hash仍待审。
5. observation 后验证每 arm 恰有全部32×7 个唯一坐标，原零和mask各224次/arm预扣并单独墙钟；调用一次额外prefill和decode/arm。比较的是该 forced decode 的输入，不是生成轨迹的 mask 平均；这些逻辑数和算术统计不能代替物理kernel追踪。对KV目前只检查实际布局、容量和位置计划，不声称逐层KV内容实测核验；这是当前范围的限制。

八项小型合成测试及目标环境 CUDA_VISIBLE_DEVICES='' CPU模块真实导入通过；没有 GPU/模型/实际校准执行。需未来真实父闭合及最终完整版本的独立复核，本回应不请求使用旧初审批准运行。

补充环境反例核验：本 gpt-fast 生成环境没有 flash_attn（直接强制该分支会报缺包），实际 Transformers `is_flash_attn_2_available()` 因未安装而为 false；在 `CUDA_VISIBLE_DEVICES=''` 下仅模拟 `torch.cuda.is_available=True`，实际生成作者及绑定的 compile_wrapper 导入通过。该路径没有调用校准使用的 Transformers Llama/FA2。此检查不证明 A100 Triton/compile 兼容。现在十项合成/工程测试通过，增加了固定9次/567 decode 预算和该条件导入检查。这个环境仍保持冻结未修改。

`prepare` 入口现在要求真实 V2 calibration RUN 和 EXIT、private worker profile、全新 protocol/CPU/GPU spec 路径；先验证实际 parent、source/runtime，再生成 `FROZEN_PENDING_REGISTRATION` 协议及 `-I -B` 的两份私有 spec，输出 fingerprint/SHA。ACTIVE 和独立 recheck 仍由协调者另行完成。缺任何 parent 不生成协议；这不是实际启动。11项合成/工程测试通过，包括空父拒绝和两阶段 spec。固定 9 generation 的账是逻辑入口而非物理 launch 次数。

输入读取账补充：每次 CPU preflight 显式读取 tokenizer 2,183,982 B 以 SHA 验证；hist hash 256 MiB 上限只覆盖直方图，不覆盖额外 source、receipt、environment metadata/RECORD。3次模型 mmap 实际读取量与编译/调优物理 kernel 数 UNKNOWN，2次完整文件 SHA 单独预扣。模型逻辑调用上限为 590：生成576（9 prefill、567 decode）、teacher 10、observer 4；其中13 prefill、577 单 token。作者投影矩阵合法 shape 和分块边界经独立源码审阅，GPU上数值和实际执行仍待结果。
