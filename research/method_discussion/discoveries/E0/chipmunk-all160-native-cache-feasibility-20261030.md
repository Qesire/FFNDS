# Chipmunk 风格全160 student 自由视频：缓存与接入可行性（SRC / ALG）

提议者 `/root/sm120_full_quality_build_terra`；静态设计 OPEN，尚未独立实验审查、登记或执行。当前只讨论四步 Wan BF16 FFN，是否执行取决于 full-M 算子结果及独立审查。

## 生命周期与容量

真实步序 high/high/low/low，每 expert 40 个 block。令真实 raster M=31668、隐藏宽 H=13824、输出宽 D=5120，BF16 按2字节。每 expert 保存每层 step首轮的 post-GELU h 与含 b2 的 pre-gate Y，逻辑 payload 为 40×M×(H+D)×2 = **47,993,487,360 B = 44.697 GiB**；其中 h 35,022,274,560 B，Y 12,971,212,800 B，每层共 1,199,837,184 B。该值不包含 tensor 元数据、allocator 预留、输入、临时缓冲、网络、权重或任何 observer。

step0 逐层建立 high cache，step1 同层读取后可在该层所有异步使用完成后释放；step2 才建立 low cache，step3 对应消费并释放。理想安全生命周期不同时保留两个 expert 整套缓存，但异步流事件、allocator 保留及异常清理都必须实际控制/计量。禁止拿 high 缓存用于 low，亦禁止 teacher 状态替换候选自身自由轨迹。首步即候选自身的 Dense 调用创建该 expert 的缓存；第二步用当前候选输入、该 expert 当前 W1/W2 和同一层旧缓存计算更新，因 earlier blocks 的改变，当前输入可不同于 Dense teacher。step2 首次 low 也来自当前 student 轨迹。

已完成 SM120 原 BF16 block20 passive 两源模型采集报告观测物理 GPU 峰值各 21,835 MiB，包含其特定运行和采集开销；SM120 全160 SharQ 另有 30,099 MiB 的采样峰值，包含不同 cache/操作。将前者作为粗略参考加上理想 GPU resident 44.697 GiB 是约 66.0 GiB，低于 A100 80 GiB 和 PRO 96 GiB 的标称容量，但此和既非实测联合峰值，也不是实际设备可用容量保证：模型预取双缓冲、当前 attention/GELU、W1/W2、行选择/索引、delta、克隆、CUDA 临时区、reserved/碎片、外来占用及阶段峰值重合均未明确。原始模型是 offload/prefetch 路径，不能把完整两个 28.58 GB checkpoint 文件当作同时驻 GPU；完整高低模型 master 映射/主机 RSS 也须按真实 process 计量，已有被动采集主机 RSS 108.978 GiB，非新范围上界。A100 上该完整模型路径没有本设计的实测峰值，此 SM120 样本不能直接证明 A100 可运行。必须在资源预算内测 physical peak、allocator allocated/reserved、host RSS 及 cache 生存期，空间不足要失败保留，不能隐式将缓存开销消掉。

## 接入和必要成本

现有 `e0_chipmunk_wan_fullm_raster_capture.py` 的 `RasterTap` 证明可在实际 `infer_ffn`、`MM` ffn.0/ffn.2 及 `post_process` 绑定 expert、step、block、V1/h/Y/gate 和 raster grid；`e0_sharq_wan_all160_runtime_v2.py` 已展示原 block 缓冲与 FFN dispatcher 的真实身份检查，原 `OriginalFFNPrefetchAudit` 保留并计量原始预取，已有实际视频 162 次/phase、零跳过。新 student runtime 应在相同 FFN 边界接管**完整第二步 FFN**，不能把需要当前 V1、h/Y 的动作简单挂在单一线性层而破坏先后依赖；按 expert/block 检查首步建缓存、第二步消费恰一次、原 post_process 使用当前 gate，并保留 attention、调度、噪声和视频路径。不可用预捕获 teacher V1/h/Y 替代自由输入。已存在 A100 selected-delta 算子不能因路径相似就视为 SM120 已验证；设备、编译和布局必须在新范围绑定和实测。

当前 `chipmunk_p4_sparse_action.py` 每次 clone 整个 h 和 Y，核 `selected_w1_gelu_delta` 写入 hc 并产出 BF16 delta，W2 核原地加 yc，仅返回 yc。单次每层 clone 逻辑 1,199,837,184 B；K=5120 的 delta 为 324,280,320 B。对每 expert 的**最后一次**复用，后续没有同 expert 第三步，更新后的 hc 没有算法消费者：不必为了未来缓存保留写回。但现有核确实会写 hc，不能通过省略 clone 而不核别名/并发就声称无写入成本；新的只读 old-h/no-h-store 核需单独实现和验证。若 old Y 独占且最后一次使用，可设计将其就地更新并移交 post_process，避免 Y clone；这要求验证无其他消费者或别名、流依赖和生命周期，且原地语义、异常边界和数值精度另审。不能删去首轮建立的 h/Y 或其驻留/搬运成本。CPU offload 是另一付费方案，必须包含双向传输、同步与主机内存，不预设优于 GPU resident。

按40层两 expert，最小逻辑动作有80次首轮 Dense FFN、80次后二轮候选 FFN、80层次缓存创建和80次末次消费/释放；全四步仍有160个 FFN site。必须逐项预扣 actual selector/P4/P64 W1、两次 selected kernels、动态索引和打包、BF16 delta、缓存读取/写入或 clone、GPU流同步及原权重 prefetch。Y 本身已有一次 b2，增量 W2 不应再加 b2；使用当前 gate，不能缓存旧 gate。成本按真实调用、字节和 wall 区分 setup、FFN、整视频、observer、转移和 peak，不以局部两 GEMM 速度充当完整系统速度。

## 最小 neutral 和证据

同机同 checkpoint、source、seed、noise、flow、recipe 下，先做无 hook 原 BF16 与已安装 hook 的 Dense passthrough 两完整4步×40层视频；检查所有10个已定义状态文件和视频逐字节一致，并计数160 FFN/320 linear、expert/步/层、162 原 prefetch（如实际路径相同）及失败。若开启缓存复制本身，再做 hook Dense + 缓存构建 neutral，要求同样状态/video byte identity，验证缓存读取不更改原函数；这是附加完整 phase 与真实成本。之后独立的新 student 候选完整4步自由运行，闭合 state/video 及资源账，质量对同源 Dense，来源仍开发暴露。两种 Dense neutral 不证明候选质量；full-M oracle 是诊断后验，不能用于 causal 生成。
