# Chipmunk Wan 原 BF16 全 M raster 被动采集

source898 和 source96 的两对原始 Dense / 被动采集视频已真实完成并经独立 VERIFY 实际 EXIT0 闭合。每源10对状态与1对视频逐字节一致；共4条完整四步视频、20对状态及2对视频 neutral。

每源在 block20 的全部4步保存实际全 M V1、post-GELU h、W2 输出 Y 和同次 gate。共8个 typed header、32个 BF16 tensor 文件、原始 payload 12193021952 B；grid `[21,29,52]`，行号 `tau*29*52+y*52+x`，未重排或重复。资产在 HF；每个文件的路径、shape、dtype、文件 SHA 与 raw tensor SHA 见同名 JSON。

| 来源 | 采集 GPU wall / outer (s) | CPU preflight wall (s) | VERIFY wall (s) | VERIFY 实读 (B) | 样本峰值 GPU (MiB) | 主机峰值 RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| 898 | 497.795 / 507.986 | 2.509 | 41.823 | 6222310976 | 21835 | 108.978 |
| 96 | 497.714 / 512.361 | 2.510 | 37.205 | 6222387746 | 21835 | 108.978 |

每源独立验证实际逐文件哈希44个输出对象，并在同一流中复核每个 tensor 的 BF16 header、尺寸、数据范围和 raw SHA。每源原始捕获 D2H 6096510976 B、raw SHA 6096510976 B；文件写入、文件哈希预扣、20状态D2H及其读写均另行记账，精确ledger在 JSON。

两条轨迹使用相同原 BF16 checkpoint、seed 和四步 recipe。旧 A100 的128行非 raster 结果使用历史轨迹；此处不声称跨设备张量逐字节配对。计时包含被动拷贝、序列化和检查，仅描述采集成本；模型 mmap 与隐式库实际物理读取 UNKNOWN。

两来源均已暴露，block20 是本轮问题的采集位置；本结果本身不测 proxy、候选视频质量、heldout 或全160位置的可部署性。
