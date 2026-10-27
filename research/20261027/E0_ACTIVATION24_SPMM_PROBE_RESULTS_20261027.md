# A100 激活2:4：默认后端与完整在线成本的首次原生判别

**已测计算现象（NATIVE-C，合成输入）**：沿K每四取二的激活左操作数可以在当前A100/cuSPARSELt实际运行，且与相同mask的BF16 dense输出接近。默认预压缩SpMM在两个Wan FFN形状下均比dense慢；在线稳定Top2、padding与压缩进一步增加成本。下一步首先区分默认算法与矩阵布局，而不只优化selector。

| 合成BF16形状 M×K→N | dense 中位ms | 预压缩SpMM 中位ms | 完整在线 中位ms | sparse vs masked相对L2 |
| --- | ---: | ---: | ---: | ---: |
| 31668×5120→13824 | 19.400 | 29.190 | 148.299 | 0.00006067 |
| 31668×13824→5120 | 18.910 | 27.528 | 345.787 | 0.00012238 |

每边界2次热身、5次测量，表为CUDA事件中位数，机器结果保留全部样本及对应wall。原M为31668，稀疏输入补零到31680并裁回；预压缩对照不含当前输入的选择/压缩，完整在线包括稳定沿K Top2、padding、压缩、SpMM和裁回。当前选择实现为通用稳定argsort，**不是RT-Lynx作者融合实现**；本轮无算法搜索。预压缩本身也较慢，意味着仅消除选择器费用尚不足以在这个默认配置超过dense。

对象是事前冻结种子的高斯BF16矩阵，既没有加载Wan权重/状态，也没有运行norm compensation、恢复拟合或视频。mask相对原始合成dense输出约0.364的L2变化仅描述该高斯对象，不能当Wan质量下降。精度门槛0.02是粗BF16算术控制，实际约1e-4；不声称位等或优于原HiF4数值格式。

实际环境PyTorch2.10.0+cu128、cuSPARSELt701、A10080GB PCIe；运行前核对原始PyTorch sparse源码/动态库SHA，首次压缩后从实际进程maps核对加载的libtorch_cuda与libcusparseLt路径，非只相信build可用标记。计数为4次随机张量初始化、16次Top2、16次压缩、18次dense GEMM、30次sparse GEMM，均调用前扣账；6项CPU测试含错误/缺失实际库映射反例。单GPU实际scope24.345秒、outer launcher64.419秒，退出0，6次隔离采样无冲突。软件/协议快照和原始RUN存HF；未复制动态库进入结果目录。

冻结[协议](../../config/e0_activation24_spmm_probe_20261027_v1.json) SHA `d94036ada6b2303e3489070a84f14ee888aff6c6357b60a717d7c05b3affe11c`；[机器结果](E0_ACTIVATION24_SPMM_PROBE_RESULTS_20261027.json)包含原RUN/launcher SHA、每次计时、实际库与预算。源码、同行审阅及不同默认布局的后续辨别见[独立C线记录](../method_discussion/discoveries/E0/activation-24-operand-source-readiness-20261027.md)。

**机制意义**：原whole64大tile全占用与真正2:4 SpTC是不同计算单位；本轮真实验证了后者的操作数合法性，同时发现默认实现的性能缺口。它既不否定A1002:4硬件，也不把外部H20/图像DiT成绩直接迁移到Wan。最小下一证据是同一packed输入、匹配输出布局，比较默认与官方搜索配置，并单列搜索内部不透明kernel/setup费用及W重排/输出转换；只有存在快的完整预压缩边界后，再判断在线融合的必要收益。
