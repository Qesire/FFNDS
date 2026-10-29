# 作者自审：V2 质量准备：真实父已闭合，待独立评审

每个来源选择 V2 首条 BF16、Dense32 和 SharQ 三条独立真实视频；另一次 BF16 用于父范围的逐字节 neutral，不重复评分。父 GPU RUN/actual EXIT0 和其独立 VERIFY RUN/actual EXIT0 必须均闭合，四 phase、当前状态和 800 桥接调用/1600 event record/1600 wait 完整后方可产生 draft。两来源 1040/979 为既暴露开发来源，各一 seed，来源等权；不是 heldout、官方五采样或人工验收。

每视频真实调用原 VBench 八维，I2V 两轴也逐视频实际调用。官方 dynamic degree 内的 RAFT 与独立 RAFT 均实际付费，后者逐视频采帧索引 0,2,...,80 的 40 对 stride-2 邻接，3 视频共 120 forward、2400 iterations；不称作原始 80 对全相邻。视觉对每视频全部 81 帧生成 contact sheets，不替代完整播放或人审。

三类各有独立输出和范围，GPU 指标的 CPU 预检真实 EXIT0 才允许 GPU，另设独立 CPU 验证及其实际 EXIT0。父视频/状态与额外重复视频 hash、输入文件 SHA 逐项账按各 scope 单独冻结；源快照及子进程细节分别说明。父身份 child 内部的嵌套回执与协议 SHA 不经过主 HashReads，调用次数和字节 UNKNOWN；视觉授权前对应嵌套 SHA 亦在该账之外。已登记原实验的模型环境和作者函数复用，不下载权重。

目标 A100 worker 的私有配置字节 SHA 286d232b47138d3c44d7b45e0dedb5fa0e7fd6e372225ec57590da246de18ff0；父 part0/1 分别拟分配该 worker 设备 index1/2，需每次运行前实际复查空闲和全局不超过4卡。只读工程检查已见原 metric Python 哈希匹配、253/253 资产路径存在和视觉依赖 metadata；不等同正式 byte preflight。视觉实际运行 Python 使用 -I -B，metric 采用原 -I -S -B，所有 GPU job UUID、profile 及设备绑定都在私有 spec；不得公开节点或 UUID。父 GPU 和独立 VERIFY 均真实 EXIT0，冻结 scope 以各 part 的真实 RUN/EXIT、状态及视频 SHA 绑定；本质量范围仍待独立评审与 ACTIVE，不允许擅自启动评分。
