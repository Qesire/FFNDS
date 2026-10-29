# Chipmunk global-P64 V2 全160 student 自由视频

两暴露开发来源各产生原Dense、hook Dense、构建cache Dense、global-P64 student四条完整视频；各来源20对状态及两对视频在三条Dense之间逐字节中性。GPU及独立输出VERIFY均实际EXIT0。

| 来源 | 原Dense wall (s) | hook Dense wall | cache Dense wall | P64 wall | P64 FFN事件 (s) | GPU物理峰值(MiB) | host RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 898 | 46.646 | 45.299 | 46.442 | 44.851 | 5.680 | 66858 | 108.986 |
| 96 | 47.734 | 47.146 | 47.270 | 46.191 | 5.924 | 66859 | 108.988 |

两来源的候选各完成80次缓存建立、80次 packed P64 复用和释放；两个 expert 的同 M、同 mask 旧 packed 对照相对 L2 和最大绝对差均为 0。每次安全边界清理耗时约 0.032–0.039 秒，清理前 reserved 约 56.8–64.6 GiB、清理后约 7.6–8.3 GiB；两路累计 peak reserved 均为 69,327,650,816 B，峰值统计未重置。该观测显示本次清理释放了未使用的 allocator 保留段，V7 的实际 reserved 峰值仍未知，不能据此确定其失败原因。

每个candidate在四步×40层上自由传播，首步按expert创建80份缓存、第二步使用自己的P64选择并释放80份。每源两次真实同M、同mask原packed边界对照的数值和门限、完整账本、状态与视频SHA、独立VERIFY及设备成本见同名JSON。

固定phase顺序，时间包含原权重预取、cache、选择、四次expert边界empty_cache及同步、数值observer和视频处理，不作为公平系统速度。当前视频质量尚未测量或验收；来源已暴露，不是heldout。
