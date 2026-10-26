# 已付费末步 latent 的视频桥接：VAE-only 原始路径审查

2026-10-26；独立讨论 Agent `independent_e0_audit`。对象是已有 whole-flow 记录的末步状态，scope 为源码/身份审查；本记录未运行 VAE、模型、GPU、张量或软件测试，也未修改协议、共享状态、冻结源码或旧审查。

**正向构造：** `flow_s3.safetensors:z_dense_at_student_after` 已经是原生 scheduler 从实际 student step3 输入，使用整个 dense low-noise 模型 flow 完成最后一步的状态。只解码它，就得到“前三步 candidate，最后一步全 dense”的宏干预视频，无需新增 FFN/denoiser NFE。原 `z_dense_after` 可作同一 VAE/编码链的中性控制。两来源×这两状态共4次 VAE decode，是把有限 flow 反事实接回可见视频的低成本途径。

这一区分很重要：0个**新增 denoiser NFE/FFN调用**仍包含4次 VAE 神经网络推理、载入、输出处理和视频编码；不能记作0模型计算或0新视频。

## 1. 精确的原始构造链

实际 `wan2.2_moe_distill` 注册类继承 `WanDistillRunner→WanRunner`，保留 `WanVAE` 与 `Wan2.1_VAE.pth`。只有另一个 `Wan22DenseRunner` 分支才改成 `Wan2_2_VAE`；不能由模型名 Wan2.2 推断 VAE 版本。

亲读入口：

- [Wan22MoeDistillRunner](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py:120)：继承而不替换 VAE 类。
- [WanRunner.load_vae_decoder](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py:280)：原构造 kwargs。
- [DefaultRunner.run_vae_decoder](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/runners/default_runner.py:485)：调用 `vae_decoder.decode(latents.to(GET_DTYPE()))`。
- [WanVAE.decode](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/video_encoders/hf/wan/vae.py:1531)：包装层自行 `unsqueeze(0)`，接收 `[C,T,H,W]`。

冻结 contract 显式 `parallel=false`、`vae_cpu_offload=false`、fps16。原 builder 没有 `vae_dtype` override；worker 的 `DTYPE=BF16`，所以原 VAE 参数、scale 与入口 latent 都走 BF16。其余原 kwargs 是 `use_tiling=false`、`use_lightvae=false`、`load_from_rank0=false`、`dummy_model=false`；`use_cache_vae` 保留类默认 false。模型 `.eval().requires_grad_(False)`，原整个生成调用处处于 `torch.inference_mode()`。专用 decode 路径只需直接创建这个 VAE 与调用原工具函数，无需构造完整 LightX2VPipeline、加载 DiT、T5、CLIP 或重新编码输入图像。

**expected asset identity（亲读两 manifest 的身份，不声称本轮重读了0.5GB权重二进制）：**

| 字段 | 值 |
|---|---|
| repo | `Wan-AI/Wan2.2-I2V-A14B` |
| revision | `206a9ee1b7bfaaf8f7e4d81335650533490646a3` |
| filename | `Wan2.1_VAE.pth` |
| bytes | `507609880` |
| SHA256 / LFS SHA256 | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| logical manifest SHA | `932bc14ec0cd8d73a13532191ea202b0bc22478d7da26ba5029658329f257df3` |
| local manifest SHA | `97b2d991f6312ab67ee439ea28a03989bd86b1f68151a6eca11102b949e539d1` |

执行端应从这两份 manifest 选定该文件，并核对实际字节数/全文件 SHA；不需为 VAE-only 渲染额外读取并解码全部 DiT cache。公开记录保留上述身份及相对来源，不复制私有节点、镜像路径或硬件 UUID。

## 2. 原生 latent、scale、cache 与数值环境

末步应绑定原 whole-flow RUN、其 `flow_s3.json`、`flow_s3.safetensors` 的精确 SHA 和 tensor key。对控制用 `z_dense_after`，对宏干预用 `z_dense_at_student_after`；二者都必须是原生 BF16 `[16,21,58,104]`，而非 legacy `.float()` hash 字节或某种反归一化 latent。前者与同来源旧 dense 的最终 native latent identity 一致是进入解码的先决身份检查。来源0/12实际 seeds 为42/54。

WanVAE 自身保存16个 mean/inv_std。内部 model.decode 在卷积前做 `z / inv_std + mean`，外部不要再做一次 scale。原包装入口接收四维，外部预先加 batch 维会造成多加一维。也不要换成另一个 VAE 包装器的 latent scaling 或 `decode_video` 接口。

原 `WanVAE_.decode` 在入口和出口都 `clear_decode_cache()`；正常顺序解码可复用同一个 VAE，前一次 clip 的时间卷积缓存不应泄漏。`cached_decode` 不清 cache，`decode_stream` 也有不同生命周期，均不属于这条中性等价路径。如果解码异常，记录失败并停止本固定范围，不在同一尝试中换 tiling、dtype 或缓存策略继续生成。

dtype/env getter 使用 `lru_cache`：`DTYPE`、`SENSITIVE_LAYER_DTYPE`、`USE_CHANNELS_LAST_3D` 应在导入/创建 VAE 前冻结。`GET_USE_CHANNELS_LAST_3D()` 源码默认0；开启时 `_video_vae` 会转换 Conv3d 权重 memory format。原 worker 没显式开启这个变量，但旧回执未逐项记录全部 ambient env，故“源码默认”仍需中性控制来闭合实际环境等价。

原 `pipe.generate(seed)` 调用 `seed_all`，设置 cuDNN benchmark=false、deterministic=true；仅调用 `torch.manual_seed` 不会复现这两个开关。原 paired driver 还设 `cuda.matmul.allow_tf32=false`。应分别记录 `cudnn.allow_tf32`、cuDNN版本、torch/CUDA版本和 TF32 reduction 配置：matmul 开关不等于 cuDNN 开关。不要在 standalone decode 中新增 autocast，或为追求显式“更精确”而换 FP32 权重；中性目标是复现原函数与媒体。

## 3. 媒体尺寸与编码等价

原 [wan_vae_to_comfy](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/utils/utils.py:179) **原地**执行 `add_(1).mul_(.5).clamp_`，再把 `[1,3,T,H,W]` 转为 `[T,H,W,3]`；不要对同一 decoder 输出重复调用。原 [save_to_video](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/utils/utils.py:217) 采用 `method='ffmpeg'`、fps16、默认 `lossless=false`、`output_pix_fmt='yuv420p'`；RGB经原规则截断到uint8，再转BGR送 libx264，threads4、无音轨。imageio 默认分支、四舍五入替代截断、lossless 或 libx264rgb 都会改变历史媒体身份。

旧两来源实际 teacher metric artifact 都是 **81帧、464×832×3、16fps**；contract 的 target_height480 是目标配置，不是实际输出高度。应由保存 latent 的8倍空间尺度及历史 artifact 确认 `[81,464,832,3]`，不额外 resize/pad 到480。帧数为 `1+4*(21−1)=81`。

旧 dense MP4 精确中性目标：

| source / seed | 旧 dense MP4 SHA256 |
|---|---|
| 0 /42 | `23f5063d3565f672ae82244ddaa7d18591ee84f2484ac8b5a0ab21dea75b1ee4` |
| 12 /54 | `6d757a154fc15ca79689b65a5bc95bf9d3338b3d6618d2734645a3414a00919e` |

在同一冻结软件/编码环境里，先要求中性 clip 与历史 MP4 字节一致，是比只看 latent SHA 更强的检查。如果失败，应保留实际 clip、原参数与全帧 metric，按预先约定停止；不能自动更换编码器或将“看起来一致”当作相同中性控制。decoded RGB 一致、MP4不同可以定位编码环境，但它是另一层证据，不能回填未满足的原冻结条件。

## 4. 最小运行与指标账本设计

在未来独立工具中，默认 preflight 可只读 manifest/源 SHA、完成的 whole-flow RUN及tensor文件的SHA/header、旧视频/receipt。执行入口才导入 torch 并认领已分配的单卡，在载入 VAE 前按现有 reservation/monitor 模式建立同进程记录。一个进程加载一次 VAE，固定顺序每来源先中性、再 late-dense；每次 decode 开始即扣账，生成异常也计入已消费范围，不自动重试。记录 `VAE_decodes_started/completed`、`denoiser_NFE=0`、`FFN_calls=0`、`new_videos_started/completed`，不能把4次decode只记成两次来源。

每个输出保存 latent lineage（parent whole-flow RUN/tensor SHA/key）、decoder asset/source/config digest、source/seed/prompt identity、clip SHA、decode+后处理+encode各自时间。`generation_wall_seconds` 若为适配 core schema 而填写 VAE-render wall，必须另写 `cost_scope=VAE_ONLY_RENDER_NOT_FULL_GENERATION`；`ffn_boundary_seconds=0` 是本次额外渲染费用，不是所代表宏路径原本只花0FFN。完整宏路径的旧计算费用可作为 parent lineage，避免与新decode重复扣费。

≤0.1GiB 的 HF 输出帽要求**只保存 MP4、紧凑 JSON/必要的小联系图**：单个完整 uint8 RGB movie 已有93,809,664字节，BF16 movie有187,619,328字节，后者单独就超过整个输出帽；不应再保存4份原始全帧张量。CPU/GPU内存峰值与HF磁盘帽是不同账本，必须分别记录。每clip后及最终检查实际文件总字节；异常仍保留已产生输出与失败收据。

防止SHA环的顺序为：完成实际 render并冻结 `DECODE_RUN.json`（含全部已生成 MP4 清单及费用）→所有clip分别调用冻结 `ffnds_video_metrics.evaluate`→保存4份 SHA-bound metric→core BUNDLE→原core gate→独立VAE inventory验证→最终摘要引用以上SHA。每个新clip，包括中性和失败clip，都需要全帧报告；不能因为MP4等于历史文件而省掉当前run绑定。正式VBench/human保持NOT_RUN、quality_accepted=false。

冻结的 `ffnds_paired_video_strict_gate.py` 目前只识别原 native `teacher/student` 双clip receipt，并从 `pair.modes` 推dense/candidate model digest。late-dense 是不同的**宏 hybrid**，伪装成纯dense或纯candidate会误报模型身份。保持原core gate不变，另建适合VAE-only的inventory检查：实际本次输出目录MP4集合恰为预定4项，无缺失/额外/重复；每项decode receipt唯一命名其clip与latent来源；metric中model/decoder/parent digest均从实际receipt构造，reference identity来自旧dense真实receipt，而不是抄某份report。中性与hybrid都可用同来源原dense clip作paired reference；旧student是宏干预前的额外比较来源，不会新增一个生成计数。

## 5. 这个视频反事实能回答什么

末步 shadow scheduler 复制 step_pre 后真实 student BF16 latent、使用 dense-at-student flow并按原顺序 step_post；其 `z_dense_at_student_after` 是明确的 `S,S,S,D` 路径终点。完整clip若相对原student在动作/纹理/条件一致性上改善，会支持“最后一个整网dense更新可挽回一部分已形成偏差”这一有限结论；仍要对照完整metric及人工观察。若没有改善，竞争解释包括前三步信息已丢失、当前状态在dense下也继续偏离、VAE/像素量不敏感等，需要结合whole-flow有限分解。

这次宏干预一次改变了最后step的40个FFN及其后续内部输入，不能单独识别某一层、H-NUM/H-CAP/H-STATE的主因，也不提供新160site paired轨迹。它没有部署在线teacher免费纠错机制；只是使用已付过NFE的反事实接到可见终点。保持offline四步研究对象，未引入实时/流式任务。

Graph锚点为 `F13/G21→whole flow→scheduler→terminal latent→VAE→D09/P25`，末步宏干预为conditional实验边；它与`P23/P24`逐site机制识别互补。此文件是设计审查，不宣称4个decode已完成。

## 6. 本次亲读来源身份

| 文件 | SHA256 |
|---|---|
| `tools/e0_whole_flow_counterfactual.py` | `0592cfe2a3b92b64b36ecc6e12b99ca653d5939f20305ae02b5ce35d7115306b` |
| `LightX2V/.../wan/vae.py` | `5adc56ca075ffa1e3ccc486f9f5d8ea0faf90cdaaa657cdc87c6bef3a45a16bd` |
| `LightX2V/.../runners/wan/wan_distill_runner.py` | `26a2a503137b0125cf4a2f8b2c9fc5dae716121c6e600b7ce85517af5ee62dc3` |
| `LightX2V/.../runners/wan/wan_runner.py` | `e12bf49dfa76a3c5bde0d284952d7f8c2dcee2043737b310d51d46e7ff3d0f69` |
| `LightX2V/.../runners/default_runner.py` | `4f8e4295e5301b1465c4e320641c98855f2f7e6e5734e714d17af1acb2fd4f18` |
| `LightX2V/.../pipeline.py` | `6b3eb5aa5a9ad9c3046df3baf8d2bd7feeda12adb327954a5d7298842b265062` |
| `LightX2V/.../utils/utils.py` | `82137228f5d573adb36a845e3b2e8a9e47d6e7e8f8e23c2fb8ceee149492c9ce` |
| `LightX2V/.../utils/envs.py` | `68c2f49686cb58559e8f1beaa0641b1141b5639083fe4ec3e8df771395c3189c` |
| `tools/ffnds_video_metrics.py` | `0c6ab066674adb2a04097978ced52abcd7cbf1125918b74d59cf976a56ffec66` |
| `tools/ffnds_video_metric_gate.py` | `3f1b70fa25b7c961f8141ba396ee3b9507e028d60fd168e642e0548de99a2d07` |
| `tools/ffnds_paired_video_strict_gate.py` | `59d6596cff43cc6197d4b1ab76eb90e9ca92855901b6d3394b7eabc257c9a6d3` |

均为本地原始源码及实际manifest/旧metric的只读核对；没有为此窄源审查新增文献检索、运行服务或触发任何训练/生成。
