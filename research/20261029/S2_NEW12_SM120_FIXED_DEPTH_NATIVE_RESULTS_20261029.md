# S2 new-12 SM120 固定深度四臂：原生生成结果

2026-10-29。四个分片均实际退出 0；逐分片执行原 producer 的只读 `verify-closed` 均为 `PASS_CLOSED_SM120_FIXED_DEPTH_SHARD`。完成的是 **56 条完整生成及其注册/账目核验**，不是质量或速度验收。[机器结果](S2_NEW12_SM120_FIXED_DEPTH_NATIVE_RESULTS_20261029.json)逐条保存 56 个 clip 的 GEN、视频和全部 10 个 state 的 locator、SHA、实际计时、W2 传输、policy 与候选 CORE；其中 48 条是研究生成，8 条是重复 neutral 控制。每个视频 4 step × 40 block，合计 8,960 FFN 位点、560 个保存状态（其中 224 个 flow）、224 个原生 flow step。4 对同机 runtime neutral 均通过 10 个状态逐字节及视频逐字节比较：共 40 次状态、4 次视频比较。neutral 是重复对照，不增加独立来源。

每个研究来源使用新鲜同 SM120 Dense 配对、同源图/原 prompt、seed 42 和配对初始噪声；原 checkpoint、BF16、torch_sdpa、固定 value10 rank960，未重新拟合。12 个来源是已视觉审阅的开发来源，不是盲确认。三种候选分别为全部 160 位点压缩，或每步 blocks 0–9 / 30–39 用原 W2、其余用同一 factor；Dense 逐来源先生成，三候选顺序按预定义方案平衡。设备为 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（compute capability 12.0）。跨硬件 neutral 比较未测。

## 原生动作、计时与传输

以下为 12 个研究来源各一条视频的等权描述，时间为实际观察，不是稳定 benchmark。generation wall 包含输入编码、VAE 解码及 observer；FFN 时间是已记录的边界 CUDA 事件时间，不是整个系统的端到端速度。实际完整生成不经 warmup，且多分片并发 I/O 影响时长。

| 模式 | Dense / factor 位点每 clip | factor GEMM | FFN CUDA 秒均值 | generation wall 秒均值 | 原 W2 H2D bytes/clip | 原 W2 prefetch/clip |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 160 / 0 | 0 | 5.941 | 43.181 | 22,934,016,000 | 162 |
| value all | 0 / 160 | 320 | 3.982 | 38.160 | 0 | 0 |
| 前 10 Dense | 40 / 120 | 240 | 4.472 | 43.459 | 5,945,856,000 | 42 |
| 后 10 Dense | 40 / 120 | 240 | 4.470 | 39.601 | 5,662,720,000 | 40 |

虽然两个混合臂计算位置同为 40，前臂多两次实际预取；两者传输不能按相同计算位点数等同。每分片原模型 setup 依次为 543.454、425.375、425.354、494.088 秒，factor preload 为 18.940、0.680、0.677、0.457 秒，均为单次观察。每分片 factor CPU load 80 文件、2,910,617,600 bytes；GPU preload 1 次、240 tensor、2,910,617,600 bytes。账本每分片预扣/观测均满足冻结限制：14 视频、2,240 FFN、140 state D2H 和写入、56 flow、2,400 factor GEMM、1,212 次原 W2 skip、9 CORE；完整各字段与实际资源保存在机器结果。固定运行计数和每 clip dispatch 已由闭合验证器复核；时间均值不构成吞吐或速度声明。

## 已有 CORE 描述

36 个候选 CORE 均与同分片新鲜 Dense 的视频和 GEN SHA 配对，下面是先取每视频 CORE 的时间均值再对 12 来源等权的描述。Farneback 位移为缩小至最大边 320 后的像素，不等同主体运动；候选/参考 flow 幅值比按 CORE 的帧级定义再平均，低参考运动可造成不稳定。RGB 差异和 flow 差异并不直接证明语义质量或因果行为。

| 模式 | paired RGB RMSE | 相邻帧差 RGB RMSE | Farneback flow L2 px | 候选/参考 flow 幅值比 |
|---|---:|---:|---:|---:|
| value all | 0.11566 | 0.05189 | 0.45381 | 0.68971 |
| 前 10 Dense | 0.11244 | 0.05137 | 0.43692 | 0.65819 |
| 后 10 Dense | 0.11560 | 0.05335 | 0.44931 | 0.68127 |

这些是既存简单 CORE 输出，不将其解释为动作恢复的判定，也不据此宣称哪一臂通过质量标准。RAFT、八轴和人工完整视频审核的结果不在本报告中；各自的实际收据决定其状态。

## 闭合收据

| part | RUN SHA256 | EXIT SHA256 |
|---:|---|---|
| 0 | `78937450a2abcfca46e6299e710a448e7a652d6ec0006f3287b5b4e3649bbebc` | `e181f76a49c8a680ee8e84c3d7b63e33e2810a5942744d92be3ba14fd9d14218` |
| 1 | `14da890d26f062b64bf907dc65feda50dc3423f34d72db77e8c6f594a8d07bdb` | `78c76748e370cfce274683276b1d4e192f25b7f72e1841d557c99a52b592c3aa` |
| 2 | `60ce455d61e43835828554b5bb1b26a94bafc3de3ee66c7f5731cf9ea3455229` | `5efe038501701d65b46fbc1da957788e832fe8bba97cfb0c1617fbd09423a69a` |
| 3 | `8400d1e1845f616c244f6125dad293716946f262aba291375f335d43a454c09e` | `195c020b9c4fbca9e974bc379e048df0c9cc47ee53a6beddac26966a13f0167f` |

本报告只读取已存元数据和 CORE，调用原生只读闭合验证；未运行模型、GPU、视频解码或新实验。机器结果包括全量逐 clip locator 和哈希，HF 相对路径以 `/home/xiang/.cache/huggingface` 为根，原视频字段保留 producer 实际绝对路径。
