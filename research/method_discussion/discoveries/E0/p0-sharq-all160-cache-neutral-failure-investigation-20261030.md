# SharQ 全160缓存 neutral 失败：静态调查

两个有限scope真实 EXIT1，失败均发生在第二条 `bf16_cache` 第0步 `scheduler.noise_pred` 的 `flow_step0` D2H 后 finite 检查，异常 `ValueError: nonfinite state`。这不是首条BF16失败：每part首条原BF16实际完整生成一条81帧视频并保存10状态；第二条已保存raw FP32与初始BF16 noise，二者与各自首条的对应文件SHA一致，随后完成 high expert 40块、80次静态W量化，才在flow保存前失败。每part账本 `video_generations` predebit 2、observed 1；320个原BF16 FFN线性加第二次80个BF16线性；候选Dense32/SharQ前向和GEMM均未开始。`phases=[]`是owner仅在整个四阶段函数返回后写回phases，不能推断首条无结果。两个原视频仍属失败scope内部分证据，没有完整parent closed。

实际小文件身份（HF相对前缀 `ffnds-derived/e0_sharq_wan_all160_video_20261030/part{n}/attempt1/worker`）：
- part0 RUN SHA256 `b5763f111dcb4ca7f2c2d8842e66bd8bb9cb94695911c46351495b5dfb355b86`；EXIT SHA256 `1991e787a52cd9e5473b58e1d63d2b640a76e5fa10411a4d273b5feb8a6c27bd`。
- part1 RUN SHA256 `9667b57508e6015b18da8a7aa296c5c12b9c057d1bcedf936b7e06c392831adb`；EXIT SHA256 `c56c88c5f572517731b0b42f3c233dcb9feb212be159468930b706c536875c89`。

源码级可证实风险：LightX2V `infer_with_blocks_offload` 在非默认 `compute_stream` 内运行当前块，`_pack` 在该流计算转置和原作者 `global_nvfp4_scale`、BF16缩放临时；作者 `kernels/src/shared_weight_nvfp4.cu:133-134` 直接用未指定stream的 `<<<grids,blocks>>>` 发射两个默认流kernel，绑定里的`torch.zeros`在当前PyTorch流分配/初始化。现行 `self.sync()` 在量化函数返回**之后**；它等待已发射的任务，不提供kernel启动**之前**对另一流上生产者/初始化完成的先行依赖。非默认PyTorch流与作者默认流、临时tensor可用期和allocator跨流记录因此需要额外边界。已完成的单FFN算子V2是在不同调用环境中验证，不能证明本异步流组合安全。该竞争可使packed输入/scale不正确，但其本身尚不能证明为什么原BF16 flow出现非finite。

另一个独立竞争原因是本次大量缓存及两个作者quantization kernel对周边存储的写越界或CUDA异步生命周期破坏；相同形状单FFN已运行不等于全40层同一allocator布局安全。相比之下，作者quantize模块和extension早在首条BF16视频之前导入，首条BF16正常；纯导入造成跨两臂相同全局行为的解释与现象不符，但不能排除打包调用时引入的副作用。`weight.T.contiguous()`若原GPU buffer转置视图已连续，PyTorch允许返回同存储alias，账本“materialization”计的是入口/逻辑字节，不证明发生新物理拷贝；作者Python scale/division为out-of-place，源码未见故意原位写原BF16 W。首条及第二条同noise不证明中间激活或W一致，故根因仍OPEN。

最低区分证据应是另行自审、独立评议并注册的新有限诊断：一组在模型外用相同真实权重形状和stream关系，对只读W、相邻guard、原输入及packed输出进行操作前后实际位身份和finite检查，比较原未隔离stream与明确单默认流+生产者同步；分别记录量化结果和所有分配、同步、时间。若W/guard遭破坏支持越界/生命周期；若W/guard不变而仅原stream packed异常支持流依赖。再在已授权小范围逐块查看原BF16运算前后finite及相同输入/W身份，hook-only、同步后的pack两条件分别与真实原BF16 neutral比；这一步实际资源和计数另行冻结，且不可将内核诊断替代全160视频质量。同步修正若采用，应在进入作者默认流前等待当前compute流的原W与输入就绪，并在该流内执行作者所有相关张量生产和kernel、返回前建立完成依赖；只在量化函数结束后同步不修复生产前竞争。这个测试和任何V2重跑均尚未执行。
