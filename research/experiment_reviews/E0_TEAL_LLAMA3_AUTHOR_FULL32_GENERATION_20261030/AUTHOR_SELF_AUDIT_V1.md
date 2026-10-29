# TEAL 作者 gpt-fast 全32层有限生成：作者自审草案

身份：原作者 fb7373c 的 gpt-fast `generate.py`、`model.py` 和 SparseGEMV，Nous Llama3 8B 转换后 BF16 checkpoint 实际闭合；加载时按作者路径转 FP16。实际资产报告、转换及独立验证 RUN/EXIT、tokenizer 原始 SHA 全部绑定；GPU 范围内对模型文件预扣两次全 SHA。64个校准 histogram 需要真实校准 RUN、其真实 EXIT0、32层/64文件账本、来源/内容 SHA；目前草案为 null，禁止冻结、登记或运行。实际 closed 后必须重新生成 draft 和 worker specs，独立评审精确版本，修订并复核后协调者才可登记 ACTIVE。

问题：相同 prompt 和模型下，原 Dense、作者全部 attention+FFN 稀疏路径的 exact threshold zero、统一50%阈值，在单次有限 seed 生成中出现怎样的 token 差异和测得成本？9次交错执行，每 arm 1 warmup+2 measured、每次 64 token；最多9次 prefill和567 decode 入口。全32层七个投影，非 FFN-only；truezero 强制所有阈值精确0，并以同 prompt 预填+4个固定 teacher token 的 logits finite、relative L2≤0.03及 top1 相同为 neutral，才执行后续范围。校准 median 不代表 exact zero。生成是开发诊断，不是论文 benchmark、质量验收或速度结论。

成本预扣覆盖 3模型加载、2次完整 checkpoint hash、1次64 histogram完整 SHA、256次 histogram load、3次 compile wrapper、9次 generation、9次 prefill、567次 decode、10次 teacher 强制调用、4次 observer model 调用及各448次 mask/原零计数。原 compile/autotune/CUDA graph 内物理 kernel 次数 UNKNOWN；计时保留编译、测量 observer 与缓存效应，不能称公平系统比较。两个稀疏 arm 的 mask 统计另在计时后支付：一次 prefill 及一次固定 decode，逐层逐投影记录阈值、输入数、原零和 masked 数；不冒称完整生成中实际稀疏分布。失效时保留预扣和已完成、实际 EXIT，不重试。

原模型按三份 FP16 权重同时驻留并进行交错对照；单 A100、CPU2、7200s、GPU reserved 约76 GiB、host address space 96 GiB、HF新增4 GiB；单独 CPU1 预检只重读 hist 不加载模型。GPU UUID 锁与进程级预占、采样 monitor、离线 flags、已绑定源/环境与新输出作用域；均需实际注册及严格工作机身份检查。监控不是完整瞬时硬限，编译峰值、A100内存不足、作者 Triton autotune 或 GPU graph 不兼容均可能导致范围失败，外层无重试。模型 fd 对照/前后 SHA 只对非对抗稳定源成立。

竞争反例：某层校准直方图的阈值精度、原 Triton 累加顺序或者 compile graph 路径，即使阈值零也可能造成 logits 顶词变化；预声明 neutral 会拒绝把它报告成受控零干预。若 uniform50 的 tokens 或延时改变，仍可能由所有矩阵、observer、编译/缓存或 RNG 自回归放大共同导致，不能归因于 FFN 或普遍模型速度。若只看到 mask 零率而没有可归因物理调用成本，也不能宣称省算。

工程核验：六项合成测试通过；目标环境设置空 CUDA 可见集的 CPU-only 真实作者模块导入通过，未加载 checkpoint、创建科学张量、编译或运行模型。后续必须对实际 calibration closed 父代、协议、精确 CPU/GPU 监督配置、完整运行闭合路径和新源码做不同 Agent 独立审查。author module 导入测试不是 GPU 兼容证明。

2026-10-30 独立初审后的修订以 `AUTHOR_RESPONSE_V1.md` 为准：统一三臂缓存、neutral eager 边界、完整投影观察、EOS位置和 compile_wrapper 实际路径已补齐；校准 V1 的实际失败不被接收，候选父范围改为单独评审的校准 V2，仍需真实 EXIT0 和 final identities。八项测试通过。
