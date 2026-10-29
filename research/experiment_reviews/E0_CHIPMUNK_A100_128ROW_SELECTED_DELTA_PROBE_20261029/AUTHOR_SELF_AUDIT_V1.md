# Chipmunk A100 单组选择性差分：作者自审，待独立评议

状态 **REVIEW_REQUESTED**。本草案尚无实际独立评议、ACTIVE、启动配置、编译、GPU 数值运行或结果。它移植选择性更新的物理动作到 A100 BF16；不是作者 Hopper 实现，也不建立 Wan2.2 端到端或视频结论。完整 LLM/视频实验均不在此 scope。固定 source898、seed42、high/block20 的 step0→1，选用已采集的首128条匹配行，是离散非 raster 混合组，不是全31668行的置换或作者原始组选择。

## 可判别问题与竞争预测

固定权重和源输入上，选择集为空是否完全保持缓存；全选时 BF16 差分链是否在事先固定的 0.03 relative-L2 限内接近相同 M128 Dense；固定前4096个 hidden channel 时该独立组的 h、Y、当前门控输出偏差有多大。H1：两个直接 gather 核能完成全选和部分选定的有界更新；H2：BF16 中间舍入、GELU 近似、归约次序、缓存差分或 tile 索引导致中性失败，或 cache/访存成本使低选择比例无法转化为速度。固定子集只是无数据选择的算子控制；不比较或冒称作者 proxy 选择价值。

## 原作者与实际操作

固定的作者 `chipmunk_mlp.py`：`bmfc1=fc1(block_mean(x, MBM))`，与存储的 `block_mean(mid, MBM)` 求绝对差，再以 `BM//MBM` 组织并求和；调用作者 `topk_indices` 和 `copy_indices`，更新历史参考。计算 `fc1` 覆盖全部 hidden 通道，存在完整代理 W1、均值、选择和参考维护成本；不能用 h 均值或真实当前 per-token 误差替代。此范围不实现或执行该代理、top/random、256对齐或自然组缓存，因此其成本未知；后续自然分组 scope 必须另行付费核对这些语义。旧移植设计中把作者代理近似描述为缓存激活均值的说法被以上源码修正。

内核 A 读取 `x[M,D]` 与原始 `W1[H,D]` 中所选 hidden 行，BF16 dot / FP32 accumulator，加 `b1` 后先取整成 BF16，再用 tanh GELU 写 BF16 h；(new h−cached h) 先按 BF16 取整保存。B 用 BF16 delta 与 `W2[D,H]` 对应列做 dot/FP32 累加，读取 pre-gate Y cache 并一次 BF16 写回；b2 只来自初始化的 Dense Y。当前 step1 gate 在输出时按 BF16 乘，不缓存 gate 后结果。唯一且越界检查发生于 CPU list；每行×选中通道及每行×输出通道只由一个 tile 写，尾行、选择和输出均 mask；空选择无两核 dispatch。选择 fixed-first、完整和 empty 的不同初始化缓存使用独立克隆。

比较保持三条关系：M128 Dense 与原 M31668 capture；选择性结果与相同 M128 Dense；选择性结果与原 fullM capture，h / Y / gate×Y 均记录 max、relative-L2、finite 及两项误差的 signed cross。full-M anchor 初始化单列，不能把 M128 重算 GEMM 差异归入压缩误差。对同 M128 初始化的全选 control，0.03 是预先提出的较宽失败阈值，尚待独立评议；它不是事先证明的正确性精度。空选 h/Y 要求 bitwise equal。两个非空选择各以两种初始化执行，一共 A/B 各4次 dispatch；两个 empty 不 dispatch。记录选择 ID SHA、选中数、逻辑程序和 padding 数。真实 GPU kernel launch、HBM 和 torch 内部 kernel 次数未知；单次同步 wall 包含账本/可能编译，不是 warm kernel latency；分开报 cache clone、选择、核和回散，回散分配全尺寸占位，其余未触及行不读取。

## 真实对象、成本与门禁

两个 metadata 和完整 safetensors 的身份、源生成 RUN/EXIT/GEN、原 high checkpoint 的已登记 SHA、作者源码、固定 producer/kernel/tests 与 helper 绑定；实际核准后读取完整 payload/权重并计算 SHA、检查 header、行号、形状和来源。源 full-M 生成已闭合、10状态与视频 neutral；此时只用已存输出，不新生成。一个 checkpoint slice load、两个 packet load、两个 Dense M128、6选择、6回散、A/B 各4 dispatch；GPU1、CPU4、24 GiB GPU reserved、900s worker、1020s outer、512 MiB HF 输出、失败不自动重试。全部GPU/CPU数据检查属于未来实验，不在本静态准备时执行。

每项显式操作在调用前预扣，失败时 inflight 表示完成数未知；实际物理 launch 不由 Python 入口次数冒充。先验证 source、数据和作用域；不同 source、expert 或 row 不继承缓存。二进制环境选项目 `.venv`，metadata 显示 torch 2.10.0+cu128、triton 3.6.0、safetensors 0.8.0；本提案记录 package RECORD 和部分入口源码 SHA，而非全部依赖完整二进制证明。Triton 3.6 对此具体核、libdevice、SM80 和 BF16 的编译/运行未验证；尤其请独立 reviewer 核查 `tl.dot`、GELU、尾块读写、原 runtime bias/门控舍入，以及单次 wall 的可解释性。环境与 supervisor 的最终身份须在 revision/recheck 之后由协调者冻结，当前草案不具运行权。

作者自审仍有需要对抗的具体反例：如果同组 token 变化方向相反，作者均值代理可能把变化抵消；即使此静态固定子集运行正确，也不能据此判断作者代理的质量。另一方面，M128 全选与捕获 M31668 不同的 GEMM rounding 足以产生偏差，不能将两种 reference 混为一谈。真计时即使表面上更快，单组权重复用与全M不同，也不能外推全系统；full-M 合成诊断和原 raster/source 真采集另需单独 scope。

待审草案 SHA `3fb04b4e9c1049c8f13840db18616c194349c982eee85d2deaeebc0b2a15f8ac`，fingerprint `b82804f5e7fa21c50dbf8dfb21c9cea620b083d559f70c6d028712145ca0aaef`；producer SHA `ebbd5cef12330b461db176b3dd3609960d79c1e575b2e42185b547f602c6b364`、kernel SHA `ec2994a7b446d31b7aa967b85a13a528791678c72eb0e93df35e356e2cbfc0d0`、tests SHA `b8d13908266c2358a2d0c57bb811b8c8b82847565afa1946178bfba5336877b5`。7项无 GPU、无模型、无数值张量的单测通过。元数据 step1 真实 SHA 末尾为 `...de99a`，旧补充文曾误写，已通知原作者。
