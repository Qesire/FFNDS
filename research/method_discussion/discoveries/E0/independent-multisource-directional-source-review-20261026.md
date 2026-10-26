# 四来源 native signed-direction 复现：独立源码审阅

日期：2026-10-26。作者：`whole_flow_native_review`。审阅的是待运行固定范围，不是新增原生结果。

**DISCOVERY — 正向研究对象。** 同一个实际 native 状态偏移同时具有方向、相位和时空频率结构。把
\(P=T_D(z_S)-T_D(z_D)\)、\(N=T_C(z_S)-T_D(z_S)\) 与 \(\delta_{next}=P+N\) 放到固定互斥的正交子空间中，可以观察全局 SSE 隐藏的相消，以及不同来源间符号能否转变。这给“两来源局部 FFN 误差多数变小，但状态继续发散”的现象增加了可复现的方向性问题，而没有预先指定新的 FFN 算子。

**竞争预测与廉价判别。** 若已有 finite dense response／局部误差共存结构具有来源稳健性，四个新增 exposed 来源会再现相应逐来源现象；若先前符号依赖特定内容，则同一 band 的 signed cross 可变化或与 global cross 不同。协议先固定来源 4／8／16／20、seed 46／50／58／62、旧 bank 和 band，再记录每个反例。完成此次八视频／640 sites 的固定观察是下一项判别；sites 是来源内重复测量，不能作为 640 个独立来源。

## EVIDENCE：冻结对象

| 文件 | SHA-256 |
| --- | --- |
| `config/e0_multisource_directional_native_20261026_v1.json` | `931b8162f8379193697a00899b617f3851f70ef0182ca7126c2ea4511a7f4e77` |
| `tools/run_e0_multisource_directional.py` | `dd23afd15abd555559b6520090bcb58980f2cd71fca509726b5b47c86749920a` |
| `tools/e0_disjoint_native_bands.py` | `1c459aa84082ec4a98ad77333cb072ce2f226780450be3141e31717757c6c28d` |
| `tests/test_e0_multisource_native_ledger.py` | `820f262ae0d08ca8a93a351770f102b8e4bf24ad9967271e9db4abd456dc6aab` |
| `tests/test_e0_disjoint_native_bands.py` | `d67f09a0e115921bdcc77ef01c3854b3730df66422656cf51b297e988a609fa6` |
| `scripts/run_e0_multisource_directional_serial.sh` | `30e2b69559c009331e95b9c640f3017680f26e7427709d12131f58ec33404fd0` |

我阅读了新 driver／band／协议／serial launcher，以及复用的 paired observer、controller、whole-flow controller、实际 native decode／media 消费路径、strict gate 和失败证据 helper，并在本机独立核对了所有协议绑定的 driver/native **源文件**哈希。root 报告已在配置 worker 完成最终只读预检，返回 `source_and_original_dense_verified=true`、`GPU_initialized=false`；我没有将本机不存在的 node-local checkpoint 路径当作已验证对象，也没有独立执行该 worker 预检。

### 原生接线、基线与完整视频

原有 paired observer 和 whole-flow controller 均未修改。实际 runner 仍执行原生输入编码、两条四步 scheduler、全部 40 blocks、head/post 与 VAE；每步在实际 Student 状态额外做一次完整 dense flow，再通过 scheduler 的原生 shadow step_post 得到第三个 after-state。这个 shadow 不写回两条主轨迹。step0 继续要求相同输入的 dense flow 和 native after-state 逐字节相同。

新 driver 将当前 dense 分支的四个 latent hash 和 MP4 SHA 与原始 passive 来源、图像、prompt、seed、checkpoint／cache／contract 绑定后逐一比较。四个新来源没有历史 candidate 轨迹，最终协议已把这一点和真正的 entrypoint／八视频目的写清。所有来源都标记为旧 Affine54 calibration 暴露来源，结果用于 exposed replication。

Student 由 `pipe.generate` 的正常 VAE／media 路径生成；Teacher 由同一 runner 的 native `run_vae_decoder(teacher_latents)` 再进入真实 `process_images_after_vae_decoder`。当前代码对后者实际消费的 `gen_video_final` 赋值，且恢复 runner input_info。最终协议显式补齐 WanVAE、media utils、包初始化／global dtype 来源与实际 Wan2.1 VAE checkpoint 身份，记录 cuDNN TF32 选择；这与使用 Wan2.2 VAE 的不同模型分支保持区分。

每个实际 clip 的 report 指向当前冻结 `NATIVE_RUN.json`；Teacher 对自身作 dense reference，Student 也引用这个已和历史 MP4 逐字节相等的 Teacher。该结构能被原 strict gate 的 paired receipt schema 直接识别：source、seed、dense/candidate model digest、bank、artifact inventory 与 reference 均来自实际 generation receipt。它既不把 passive dict-phases 假装成旧 list-phases schema，也不修改历史 gate。所有八视频仍需 core evaluator 和 core／strict bundle；这些门禁不能替代 VBench／完整人工观看或构成质量接受。

### 固定互斥 band 的数学对象

每个 channel 独立在 T／Y／X 三轴上做 `norm='ortho'` DFT，不裁剪、不 padding。四个 mask 为 TYX 全 DC、temporal-DC 中 spatial-low 非 DC、temporal-nonDC 中 spatial-low，以及所有 spatial-high；spatial-low 预先固定为两个空间频率轴绝对整数索引均不超过 8。mask 逐频点恰好覆盖一次，且在频率取负下不变。

因此它们对应实状态空间上的正交投影 \(\Pi_B\)。对每个 band，

\[
\|\Pi_B\delta_{next}\|^2
=\|\Pi_BP\|^2+\|\Pi_BN\|^2
+2\operatorname{Re}\langle\widehat P_B,\widehat N_B\rangle.
\]

求和恢复完整 native signed identity。Hermitian 成对频点的数量等于其对应实自由度，并不需要将完整 FFT 的频点再乘二。四个实维数为 16／4608／92480／1929648，合计 2,026,752，等于实际 16×21×58×104 状态维数。这里的 DC 是逐 channel 的时空均值，band 也不是主体／背景／相机的语义 mask。分解描述的是当前真实 native 状态；它本身不是 band 干预或视频影响归因。

一手 [NumPy DFT 定义、unitary normalization 与实输入 Hermitian 说明](https://numpy.org/doc/stable/reference/routines.fft.html) 以及 [fftn 的 axes／shape 语义](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html) 已实际读取，用于交叉核对这套实现约定。

### 审阅发现的实质问题及修复

1. **接近零的 signed cross 需要自然误差尺度。** 初稿将跨 band 与时域 cross 的差按 cross 自身相对误差／绝对 `1e-9` 门禁；大能量 P／N 近乎正交时，这会把正常 FFT 舍入误判为 Parseval 失败。我用固定随机种子 31、shape `(2,3,6,8)` 的正交化 P／N，放大 `1e4` 后复现了只在 signed-cross closure 失败的反例。root 改为自然尺度 `max(2||P||||N||,1)` 的预定容差，同时保存 residual／tolerance／scale；真实的逐 band signed identity 仍有单独门禁。按 root 委托，我仅在 band 测试文件补充该反例，保留原三项测试。
2. **有限账本要验证真正的来源前缀。** 初稿仅检查旧 entries 状态；损坏的 `dc_source08` 首条 PASS 可以使第二次同来源启动并让 close 命中错误旧条目。最终代码验证精确有序 prefix，close 要求唯一实际 attempt，新增回归覆盖该反例。canonical study root、恰当协议 SHA 和运行／失败阻断继续生效。
3. **视频范围的来源链与继承文字。** 最终版本已补齐实际 decoder／media／checkpoint 绑定并修正旧 latent-only／旧候选回放文字；失败路径也尽量保存 partial wall／setup／采样显存，然后调用保留的所有实际 clip forensic evaluator。缺失的边界仍标为未测量，不冒称完整 cost。

最终 3 项 ledger 与 4 项 band 合成 CPU 测试均通过。它们证明固定软件命题，不证明新来源原生中性、数值恒等、显存适配或视频质量已经通过。

### 资源与真实调用口径

每来源保留两条主轨迹、160 个四臂 paired sites、4 个额外 complete dense flow 与两次 VAE decode；四来源合计 8 视频、640 sites、16 whole-flow、16 shadow step_post、8 decode。原始 80 packs 仅加载旧字节，不拟合新权重。

每来源 runtime 计数应为 4800 linear／activation-QDQ calls；candidate 另有独立 640 QDQ 和 960 GEMM。因此总 QDQ 为 `(4800+640)*4=21760`，总原生目标线性 GEMM 为 `(4800+960)*4=23040`，32-row 连续 FP32 diagnostic GEMM 为 `960*4=3840`。这些离线 observer／shadow 成本全部保留；不是单条 candidate 部署延迟。

每步保存 5 个 BF16 state 与 3 个 FP32 flow，四来源的原始 tensor bytes 为 `2026752*(5*2+3*4)*16=713416704`，另有 headers、trace、source snapshot、视频与 metrics，统一受整个 study 的 1 GiB 输出门禁约束。serial shell 以 `set -e` 顺序调用四个新进程，每次通过原 UUID/flock 空卡 launcher；任一 source 失败即停，没有自动重试。

## EXECUTION 与结论边界

执行边界：本审阅只进行源文件／数学／一手文档核对与合成 CPU 检查；零 GPU／模型／真实保存张量数值分析，未生成视频。root 负责临时持续授权下的精确 ACTIVE 登记和这一次固定运行。最终冻结版本没有阻塞该范围的剩余实质问题。

图锚为 F08／F09→P23／P24 与视频 P25；band 边属于 posthoc、机制与部署关联保持 conditional。真正可新增的认识是逐来源／逐 band 的方向几何与完整视频分量如何共存，而不是用局部 SSE、频率标签或强制统一符号替代功能判断。此次离线诊断付出的完整模型／observer 工作不会被记成 FFN 省算；新表示带来的真实 compute／quality frontier 仍需其对应的原生与视频证据。
