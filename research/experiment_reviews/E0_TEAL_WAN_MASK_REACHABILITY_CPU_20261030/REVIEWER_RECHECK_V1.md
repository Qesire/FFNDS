# TEAL Wan sampled-row CPU 阶段 A：V5 独立复核 APPROVE

2026-10-30；作者 `/root/teal_full_llm_calibration_build_terra`，独立评审 `/root/sm120_full_quality_build_terra`。确切版本 producer SHA `2f3f08983b24aeab25f36db4beded63559bd2160df8ae4e410f85c55d292e9f6`、tests `3674e7ef41a0892503b8ce2fbe725afbe4db38ba749d89a292d3678896ea5093`、protocol `04bd7fdc85655b1c9c907a8f8102252f90f19e878a9a23a6d5e85d9a85447105`、私有 spec `96a244f6f3f3acc8b4ce547d54637e9c777d0b259339ef9395a0eaa484a28dd2`，fingerprint `5a73d9d98db9a85682617f33e85ba56e112ccdbf48b996592554badf9ede8fba`。

初审提出的640次评估失败账已修：每次 compute 开始前验证上限并持久预扣与 in-flight UNKNOWN，返回后独立持久完成；合成中断和第641次拒绝测试实际通过。文件单次完整SHA前后及处理后检查stat，输入改动和无法观察的中断仍不假称成功。四项合成测试通过，包括与原视频V3实际函数的BF16阈值边界比较；inspect得到上述 fingerprint。

反例边界保持：256采样行全冷不说明 full-M 全冷；全行2:4资格只适用于已采样行，并不能证明任一具体稀疏存储或加速。320个包完整hash及h/rows读取的预扣预算，不包含metadata、代码包身份读取和内存扫描；RSS为采样、HF输出在退出时扫描而非连续硬配额，非合作写者与中断窗口仍有边界。失败保留真实账、不可自动重试；两来源均暴露开发源、行和160位置不是独立source。无W2、GPU、模型或视频质量测量由评审执行。

结论 **APPROVE**，仅对上述确切范围进入协调者 ACTIVE登记及启动前身份/资源检查；实际数值结论须真实 EXIT0与closed验证。
