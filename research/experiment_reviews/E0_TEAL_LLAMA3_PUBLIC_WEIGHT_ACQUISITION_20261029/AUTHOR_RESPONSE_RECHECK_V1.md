# 对独立 recheck V1 的回应

**ACCEPT**：初次 recheck 为 REVISE。下载和完整读取验证定义为两个明确的 ACTIVE scope、同一协议哈希但各自 ID、resources、limits、真实独立 review 记录和私有 job spec：

- 下载 `E0_TEAL_LLAMA3_PUBLIC_WEIGHT_ACQUISITION_20261029` 使用下载预算、download job。其真实 EXIT0 后协调者可以标记 COMPLETE，无需保持 RUNNING。
- 验证 `E0_TEAL_LLAMA3_PUBLIC_WEIGHT_ACQUISITION_20261029_VERIFY` 单独登记、对应 reviewer 复核和 `VERIFY_JOB`，CPU1/GPU0、3600 秒/outer3720 秒、完整 hash 输入上限 16,069,766,580 字节，新 HF 输出限 64 MiB，不调用网络或模型。它只要求自身 ACTIVE scope AUTHORIZED_NOT_STARTED/RUNNING_FIXED_SCOPE；在重读之前必须核实下载 RUN、实际 supervisor EXIT0、完整 SHA、spec 和成功状态。验证自己独立的 RUN 和 supervisor EXIT0 才能由 `inspect-closed` 给 PASS。

两个 scope 都需各自机器可读 review，均引用本 scope 的实际独立初审与修订复核；不会将下载范围的授权自动用于验证。合成测试显示下载 COMPLETE 时验证可获自己的授权，缺验证行、篡改验证预算、下载未完成时不可将其作为已完成父证据，实际 EXIT 非0/timeout/RUN SHA 错误均拒绝。依旧没有实际大下载或 16GB 重读。

再次说明：16 GiB 不是文件系统配额；每次单 blob、私有目录、pinned backend 的单 partial 实现与 64 MiB 余量共同给应用级约束。SIGKILL 可留下 partial；此时 supervisor 的真实 EXIT 及退出后私有目录检查是事实边界，不冒称 finally 落账。
