# 完整21τ CPU几何与固定代表行复用：协调者审阅

这是ROOT协调者的代码、数据身份和软件对照审阅，不是独立Agent的同行支持票。两位Agent遇到服务额度上限，未完成本版本独立审阅；不填充SUPPORT/PEER_SUPPORTED。所有新数值仍需本次注册CPU范围实际执行。

已逐段阅读驱动、纯NumPy核心、原冻结decoder/字节预算/Meter，以及原生Wan `x.add_(y*c_gate_msa.squeeze())` 的eager BF16路径。原Y×gate先BF16舍入，再与r相加并BF16舍入；每一包必须在任何复用统计之前逐字节恢复原u，否则失败退出且不换阈值。gY的FP64几何乘积与原生舍入update分开。

13项实际合成测试exit0：正交时空分解与显式所有时间对恒等式、共同时变/空间静态/交互分离、同能量相反相位、巨大DC掩盖AC丢失、零范数与单时间缺失值、固定代表行保τ0及真实行比例、BF16 ties-even与分步乘加反例、来源等权和重复/缺失位置拒绝、实际退出False不冒充0、9tensor完整字节/header/坐标与损坏拒绝、单次stream真实stat、同buffer manifest、元数据缓存/拒绝payload、失败后预扣不回滚。测试没有读取真实operand或启动模型/GPU。

真实metadata-only预检通过：两个part实际exit0，十张新development图片与原始prompt SHA相连，各有完整GEN/CORE/160个step×block packet，继承的旧894十状态中性不计新来源。1600包完整168个行坐标由τ×xy解码一致，全部payload预算及9tensor rawSHA已绑定。检查只读provenance JSON，没有预读真实Tensor数值。

固定三种复用在W1前只依赖坐标：保留τ0，τ10代表时间复用为16/168行；横邻或远邻为88/168行。它们是Dense前缀/代表Y的保存数据有限算术对照，不包含小M GEMM、gather/scatter、学生后继或新视频。按source等权保留全160层步与每source尾部，τ0、τ1..20和全21分开；时间均值仅后验最佳常数的oracle，不宣称免费计算F(meanX)。

CPU4、wall1200/outer1500、完整payload流≤32GiB、HF新增≤64MiB；单pass全文件及9tensor SHA、u neutral、完整header/stat manifest、POSITIONS与SUMMARY的写出均受预算。0GPU/model/fit/decomposition/video/official score。本审阅对实际数值结果不作通过推定。

精确绑定：

- config/e0_raw_bf16_full21tau_new10_geometry_obs_cpu_20261029_bound_draft.json SHA256 `c780ddbeedb81b8e1641ddf0497585640bce94a799fc46d93589998e39e65894`
- tools/e0_raw_bf16_full21tau_analysis.py SHA256 `3abd6943618888e7d30ebf93463b7fc0093ada14d74e22033f72582931615ece`
- tools/raw_bf16_full21tau_geometry.py SHA256 `d5766447a35ea56101395c016a5886e6e40fd8f144edfc55168bc4672269dbd4`
- tests/test_e0_raw_bf16_full21tau_analysis.py SHA256 `ba108d36af18b6714d237fcecaa0590f44f78bf49cc049ec82c0fd51e7bc83a7`
- tests/test_raw_bf16_full21tau_geometry.py SHA256 `d17a44021f883fb9435cebef59786ce1d6cb1569a1f96ff766dc5330df52fd0f`

注册前所有1600包预算：`18494182400` 字节；绑定的真实provenance metadata文件 `1632` 个。
