# 协调者自审：A100 单组列差分算子

协调者 `/root`；实现者 `/root/sm120_full_quality_build_terra`；独立复审者 `/root/s2_theory_design_independent`。本自审确认最终待登记范围，不代替独立评议。

**问题与相反预测。** 选中列的真实W1/GELU重算、BF16激活差分及W2增量能否与同索引独立参考一致？H1预测空集严格保持旧缓存、未选h逐字节不变，完整/固定子集的h、delta、Y及当前gate后输出均满足预先固定的误差界；H2预测非线性、偏置重复加入、更新索引或BF16抵消使其中至少一项控制失败。失败在本次范围内保留，不依据数值调整阈值或自动重试。

**计算与证据对象。** 来源898、seed42、high expert block20、相邻step0/1、配对相同坐标的128行，是已暴露teacher资产且非原raster组。首次缓存分别由M128 Dense和原full-M捕获建立；当前h以实际W1+b1写回BF16后GELU，delta写回BF16，再以选中W2列的FP32累加更新旧BF16 Y。b2仅存在初始Dense Y，输出采用当前step gate。六个case覆盖empty/full/fixed_first_4096，部分选集有独立FP32参考和未选h逐字节检查。M128与full-M的GEMM差异单独报告。

**忠实度与成本。** 作者Chipmunk的均值W1代理、top/random在线选择未执行；作者Hopper内核也未复现。这里是受控A100移植的物理列差分算子。选择张量、缓存clone、gather/回散、同步和可能编译均记录；单次wall不用于加速结论。回散只写128个位置，其他位置未初始化，不能充作完整视频输出。128行的内核与缓存规模也不能外推M31668的系统节省。

**独立意见处理。** 接受初审四项修订：同选集混合缓存参考、checkpoint/payload读取前扣账、异常和host/output资源证明、作者proxy与性能范围分离。最终独立V2 recheck绑定协议 `27ccc2a74aa5efd0610f2c8113002d7dc9a8f6d4ba7a75be4482c92fba33da16`、指纹 `75bcb4302c617f4f4167a1f0374d218679225967d719a74f33dadad92e786ea2`、私有spec `40135b602ccbec044628d1c0ab2851e07990c839ec36ed952808728b82477b4a`。

**资源和启动。** 本scope单张A100、CPU0–3、900秒/outer1020秒、24GiB GPU reserved、64GiB进程RSS边界高水位检查、30GiB逻辑输入、512MiB新增HF。当前用户明确总并发最多4卡；该scope固定GPU0，视频part0须等待它退出。目标端将核对真实worker/env/monitor/source、实际空卡锁、精确argv与fresh output，再登记及运行；此文没有启动任何模型或GPU。全文件hash在范围内按真实预扣和字节进度记录；软资源边界不标成连续硬限制。
