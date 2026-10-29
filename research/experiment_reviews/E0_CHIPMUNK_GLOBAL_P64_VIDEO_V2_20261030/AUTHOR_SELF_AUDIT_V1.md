# 作者自审：global-P64 全160视频 V2 allocator 生命周期

作者 `/root/sm120_full_quality_build_terra`；待真实独立评议、登记及运行。逐part绑定 V7 原协议及已actual EXIT1的RUN/EXIT，旧scope候选未入且失败时累计reserved峰值未保存，保持UNKNOWN。观测到的采样物理峰约65 GiB不能证明reserved没有短峰。竞争解释是inactive allocator reservation/碎片与真实live-set瞬时峰；V2不是事先证实哪一种，也不是静默重试。

新ID/HF输出、owner/driver/tap/tests；源、checkpoint、2暴露source×4完整视频、P64 seed42/K5120、三Dense逐状态/视频neutral、真实student cache、packed W1/W2、两次真实同M old-helper observer、原预取、GPU reserved累计76GiB上限和不得自动重试保持原方案；A100 V5 actual CPU master proof绑定，CPU preflight和独立output VERIFY各有实际supervisor EXIT要求。冻结的V7代码、协议、失败证据不改。两项同机完整模型最多2并发，全session最多4卡，每source ≤512 MiB输出。

V2仅在cache_dense和candidate的 step1/step3 block39 的真实 `infer_ffn` 返回后，检查该expert全40层cache已经释放；此处仍在活动compute stream、当前buffer0的该block，随后先全设备同步、`torch.cuda.empty_cache()`、再同步（复用charged包装还有一次完成同步），每source预扣4次及实测wall。PyTorch API只释放未占用的allocator缓存；不删live h/Y、改变dtype、CPU卸载、回退或跳过预取。绑定源码 `wan/infer/offload/transformer_infer.py` 的block0 init sync及后续prefetch/swap双流sync；此时整个expert的cache已没有后续消费者，原post_process的输出仍活跃且不可释放。两个清理观察phase增加真实同步/allocator代价，视频wall和FFN CUDA事件都包含，不能推成公平系统速度。

记录实际 `memory_allocated/reserved/max_memory_allocated/max_memory_reserved` 与allocator backend，每phase前后、四个专家边界清理前及清理后均写RUN；清理前写入独立事件，即使清理异常亦保留。失败finally仅在CUDA已初始化时记录当前及累计peak；CUDA未初始化或读取异常则记录具体UNKNOWN，避免收尾触发新GPU初始化；本scope不调用peak reset，历史V7缺失的peak不能被补造。原check仍直接以累计max reserved<=76GiB裁决；前段已越限则即使empty_cache也不得重置/掩盖。缓存逻辑字节和采样物理峰分别记录，不等同实时reserved或硬件总可用。

15项CPU合成覆盖cache身份/容量、两项packed同mask对照与失败门、原source W2/stream guard、真实preflight拒绝无ACTIVE、真实run注入torch已导入但CUDA未初始化的失败并核finally不触发CUDA初始化、GELU shim异常恢复、边界仅空cache才允许cleanup、同步顺序/累计峰值不重置。当前没有GPU/model/科学实验运行，结果由真实独立reviewer审后再登记；V2实际失败同样不自动重试。数值、视频质量、联合峰值及清理是否足以使范围成功均待测。
