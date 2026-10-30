# 独立复核：PART0 与 PART3 namespace V2

评议者 `/root/sync_abba_actual_review_terra`，作者 `/root`。本次复核面向 PART0_NS_V2、PART3_NS_V2 及各自的条件 VERIFY。它是两个新范围的精确审查，不启动任务、不登记 ACTIVE、不证明实验结果；相应 VERIFY 只在其 producer 实际 EXIT0 后可登记和运行。

两个旧 producer 的原始 RUN 和 supervisor EXIT 收据逐文件核过大小及 SHA：均真实 child EXIT1、process_exited、无 timeout、phase 列表空，错误为 `native namespace module lacks bound source`。旧失败保持原身份，新输出路径检查时不存在，不能把新版本视为旧任务自动重试。PART0 的 seed42 来源为 898，PART3 为 979；两张实际输入图像均按冻结大小和 SHA 核验。两源均为历史暴露开发源。

逐键对照各自 V1 协议，除 scope/新输出、实际失败父 RUN/EXIT、三项 namespace 修正源码及其映射以外，科学设计、数值路径、来源、模型资产约束和资源不变；1021 项源码大小和 SHA 逐项核对，140 项 namespace 精确映射与此前独立审查的 PART2_NS_V2 一致。分别从原作业重构两 native 和两 VERIFY 的完整私有 job，并逐键核对入口、协议路径、输出、scope、参数及运行约束；四项 job、协议文件及 canonical fingerprint 的绑定一致。私有资产定位文件及 worker profile 的实际 SHA 也与协议一致；实际设备与私人路径不写入公共审查。

保留先前审查的边界：逐来源采集 plain 与被动 HiF4 Dense 两条轨迹，各 4×40 位置，固定十状态和视频 byte-neutral，记录实际 Q1 后 full-M 与对齐采样；每 producer 1 GPU、4 CPU、单机最多两个模型，四来源合计会话最多 4 GPU，预算及失败不自动重试均不变。对应独立 VERIFY 为 0 GPU、1 CPU、独立预算，并保持实际输入验证门。真实模型资产的完整身份、启动时资源和其余运行验收由注册后的 owner 实时校验；本审查不代报未来运行通过。既有 namespace 修复只接受绑定源码树中精确的单根 PEP420 namespace，同时保留普通文件 SHA 核验；前次独立软件 fixture 与原始范围复核证据持续适用，不能保证后续不会发生新的启动或数值失败。

本范围只授权这些源的 Dense 原始采集和条件核验，不授权 E0/E1/E2 回放或质量/稀疏收益结论。任何新错误须保留实际失败和消耗，不得热修自动重跑。PART2 的部分运行进展不构成其完整结果。

**APPROVE**：精确 PART0_NS_V2 和 PART3_NS_V2 的 native 采集及各自条件 VERIFY 可进入登记。两份冻结协议在提交 `247adc9924461c573031ddac31a89af8d0a7372b` 中的实际 blob SHA 分别与对应两份私有 job 的绑定一致；登记和启动仍须重新核验当时的 HEAD、ACTIVE 与资源占用。
