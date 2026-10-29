# Global-P64 全160视频 V7 失败记录

两来源各完成 original Dense 和 hook Dense 各一条视频，10对状态与一对视频逐字节中性。后续 cache Dense 在保存 latent_step2 后触发冻结的 76 GiB GPU 累计 reserved 门限；两作业均真实 EXIT1，无自动重试。global-P64 候选没有进入，不能将此失败归因候选算法。

| 来源 | GPU wall (s) | 采样物理GPU峰值 (MiB) | host RSS (GiB) | 已完成缓存h/Y / 已释放 |
|---|---:|---:|---:|---:|
| 898 | 224.087 | 66858 | 108.985 | 80 / 40 |
| 96 | 229.443 | 66859 | 108.987 | 80 / 40 |

实际触门的累计 reserved 峰值在该版本中没有保存，保持 UNKNOWN；物理GPU峰值为采样值，不能填补这一缺口。未使用的 allocator 保留/碎片与真实活动集短时峰值仍是竞争解释。全部实际RUN/EXIT身份和部分操作账见同名JSON。
