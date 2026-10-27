# C5 fused CUTLASS 选择与打包：独立冻结前审阅

**结论（SRC/CPU 合成）**：已审版本的 packed 值和 metadata 设计对齐固定 PyTorch 2.10 converter 的非零规范编码与 swizzle；真实 GPU 上两小边界形状及两个完整形状均须与原 converter 的 packed/meta 字节完全相等，完整形状还需 materialized mask 与 stable reference 字节相等，之后才进入 SpMM 和计时。CPU interpreter 不代替 GPU 代码生成或性能验证。独立10项 CPU 测试通过，exit 0；未运行 GPU 或真实 tensor。

审阅时 pack SHA256 `f8eaac49f1a7737bad8ec27b46a2d66d7a69533af7fba657af508776dc265b11`，probe `15ea7103abbf1563721cf98b8df2e139220a1bd159b412718af3ae2f09b3588e`，test `e9c3eeb86b47566ed9596c5a7a07de2e1bca4967e6cc1bcabb522fa239c89267`，config `13288cdebf3a273d704fd10465bf6f1f06c90d43fcac00696964dd01c62ad9db`。作者仍在收尾，须以最终冻结身份再次核对。

BF16 位模式绝对值排序对 finite 值按幅度、同幅度低索引优先；原 converter 依据数值非零选择规范位置，零或一个非零时补零位置不等价于直接编码 top2 support。CPU 合成检查覆盖16种非零模式、signed zero、subnormal、tie、有限极值、padding，metadata scatter 与安装版原函数对照且为双射。两固定小形状执行 Triton CPU interpreter 对照原 converter，也不初始化 CUDA。实际 GPU 门槛检查 uint8 字节而非数值 `torch.equal`，失败不 fallback；同 masked dense 对照另检查相对 L2、形状和连续输出。

账本将预扣与完成计数分列；含两个 online 全路径、2 warmup 和5次计时、实际 edge/fullshape gate，搜索/拟合/模型/视频/官方评分为0。scope 要求用户当前授权及精确协议注册、1张空闲 A100、2 CPU、480秒计算/600秒外层、存储上限与 sampled monitor；最终 monitor 错误传播非零。`verify_bindings`、worker 和旧 C4 收据核对发生在 attempt 创建及内部 480 秒计时之前；如协议要求预检失败也消耗本次尝试，需前移 attempt；外层 600 秒应覆盖预检。64 GiB 是 Torch peak allocated 检查，不是进程 GPU 实际硬内存墙，峰值由运行期与监控记录判定。

这是合成算子格式与执行成本问题，不是模型质量或视频功能验收。

最终冻结协议 SHA256 `3237d21b5fd5778edb99cfff9991849cc06d1a4245a7cd79b8e0b6d24629b7ac`；packer、probe、tests 与上列审阅 SHA 一致。协议明确 CPU 身份预检在 GPU attempt 前，外层600秒覆盖，GPU attempt 后不自动重试。实际运行结果及独立收据核验见 [C5结果](../../../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)。
