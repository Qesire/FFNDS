# Wan2.2 I2V 四步 Dense 全局鱼鳞纹：分层定位实验与资产树

> **2026-11-17 · DESIGN_ONLY / REVIEW_REQUIRED / NOT_RUN。** 本文是新的独立视频质量诊断计划，未提交 GPU/CPU 原生实验，没有新视频、人工评分或根因结论。请按现行 [预执行独立复审](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md) 和实际 ACTIVE 限定范围执行；本文不构成运行授权。与 FFN 稀疏算法优化、G19/G20/ICC 的旧 RUN、结果和 ACTIVE 完全分离。
>
> **首要评价信号是用户观看完整视频后的直接判断。** 所有试验保留原尺寸 81 帧完整视频，无损版和 MP4 预览分别保存；自动频谱、VBench 和模型产生的描述不能代替用户的鱼鳞纹判断，也不能单独触发下一轮。

相关入口：[机器资产表](ASSET_REGISTRY.json)、[阶段预算](EXPERIMENT_MATRIX.json)、[人工观看流程](HUMAN_REVIEW_PROTOCOL.md)、[S2 原始 Dense 结果](../../20261029/S2_NEW12_DENSE_BASELINE_RESULTS_20261029.json)、[原生 VAE-only 源审查](../../method_discussion/discoveries/E0/independent-saved-final-latent-vae-bridge-audit-20261026_v8.md)。

## 0. 执行环境与资产位置：2026-11-17 补充纠正（强制先行）

**先停止依赖 S2 历史二进制文件的 A0/A1/A2/B1/B3，直到发现并验证真实文件。** 之前把 `root=hf` 的相对路径直接当作当前学校集群可读，属于错误的环境假设：

- **当前学校账户目录**：`/public/home/scc/pb23061276/`。截至本次 GitHub 文档核对，没有任何证据证明旧 S2 的完整 MP4 和 latent 已被复制到此目录。
- **历史 S2 代码**：[e0_s2_new12_dense_baseline_v2.py](../../../tools/e0_s2_new12_dense_baseline_v2.py) 与 [e0_s2_new12_dense_baseline.py](../../../tools/e0_s2_new12_dense_baseline.py) 的 `--hf-root` **默认**为 `/home/xiang/.cache/huggingface`；S2 冻结配置把模型视图写为 `/var/tmp/ffnds-xiang/huggingface/hub/.ffnds-model-views/...`。真正运行时是否通过 CLI 覆盖默认 HF root，必须读取原始 GPU RUN/执行命令才能最终确定。
- [worker.env](../../../config/worker.env) 中实际是 `HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}`，**不是写死 `/home/xiang`**；[run_on_worker.py](../../../tools/run_on_worker.py) 通过 SSH 别名 `ffnds-worker` 访问另行配置的 worker；[worker_exec.sh](../../../scripts/worker_exec.sh) 的 “home directory shared between hosts” 只描述该 worker 组内部共享 home，不能推导学校集群也挂载此路径。
- [R12_D0_worker_migration.json](../../20261022/R12_D0_worker_migration.json) 记录了用户指示由一台 worker 迁移到第二台 worker 的历史事实，但不证明 S2 在当前学校 Slurm 集群运行。S2 原结果中只有 HF 相对路径与历史文件 SHA，没有公开的学校绝对路径、物理主机身份或完整二进制。仓库本身不保存 S2 的 `.mp4` / `.safetensors` 文件。

**真正的 preflight 结果必须在实际执行节点获取，不能从 GitHub 的记录推断**：

1. 在当前学校计算节点以执行任务的同一用户检查 `/home/xiang/.cache/huggingface` 是否存在、可遍历、可读取指定的 `part0/.../new12_dense_ord898_seed42/output.mp4` 和 `latent_step3.safetensors`，以及 `part1/.../new12_dense_ord96_seed42` 两个文件；以文件内容 SHA 验证实际字节。若不可读，状态为 `ASSET_UNAVAILABLE_ON_CURRENT_EXECUTOR`，**不是 S2 GPU 生成失败或模型质量失败**。
2. 若仍持有旧 `ffnds-worker` 的访问权限，在旧 worker 通过实际 RUN/EXIT、当时的 `--hf-root`、路径挂载信息逐项定位，确认两个完整 MP4 和两个末步 latent 以及后续按需的其余状态；如果旧 worker 有文件，先做只读打包/传输和新目录 checksum 复验，再允许新集群重解码。
3. 原 worker 无法访问或资产不存在时，**不要重新生成历史 S2 全部 14 条**。先枚举当前学校集群已有的可读 Dense 完整视频与末步 latent；找到匹配者先用它，登记新来源和 parent。若没有同源成对资产，只申请**一个最小 Dense 视频及 final latent 的新捕获**，保留现有正确推理 recipe 和全部完整视频，再进入 A1。与旧 S2 的跨机器运行保持 `NEW_CLUSTER_BASELINE`，不可写为旧 SHA 的原位恢复。
4. 只要 A0 的可播放完整视频尚不可获取，就不能宣称已由人眼确认该具体历史来源的鱼鳞纹；A1、A2、B1 也不应读取猜测的 latent 路径。B3 历史原版 40 步视频同样要经过这一位置检查。

**本次补充仅更正资产归属与执行前条件，未访问学校或原 worker 文件系统，也未执行新实验。**

## 1. 对象、已有资产、需要证伪的解释

**现象**：FFNDS 当前 Wan2.2-I2V-A14B LightX2V 四步蒸馏配置输出的完整 Dense 视频也出现全画面或大范围细密鱼鳞状纹理、跨帧持续抖动。由此可知 FFN 稀疏替换并非必要条件；但 **Dense 的字节可重复性不等于输出无误**，也不意味着蒸馏权重本身没有质量问题。

**原有事实**：2026-10-29 S2 已生成 14 条真实 BF16 Dense 视频：12 个新开发输入及 2 个历史重复控制。每条 seed42、81 帧、832×464（实际输出，不是 target_height480）、16 FPS，保存初始 FP32/BF16 噪声、4 个 FP32 flow、4 个 BF16 post-latent、原始 MP4 和生成收据。两历史 neutral 的 20 个状态历史逐字节一致。所有来源为暴露的开发数据，不是独立确认集。具体 14 条绑定在 ASSET_REGISTRY，运行前必须在实际 HF worker 验证存在性、file SHA、tensor raw SHA、dtype/shape；资产清单本身不代表远端文件现在仍在。

**计算链**：
1. High/High/Low/Low 四次 Dense DiT；无 FFN 替换、BF16、无 CFG，原生 torch_sdpa。
2. 原生 flow scheduler：配置 indices 1000/750/500/250、sample_shift 5，实际模型时间点 1000/937.5/833.333…/625；末步系数约 0.625。这是数值敏感性的线索，而不是根因证明。
3. 末步 latent：原生 BF16 [16,21,58,104]，WanVAE 解码为 81×464×832 RGB。
4. 当前 LightX2V 蒸馏 runner 采用 WanVAE 和 Wan2.1_VAE.pth、BF16、use_tiling=false；再归一化、转 RGB8、保存 libx264/YUV420 MP4。

| 候选解释 | 竞争预测 | 最小区分实验 |
|---|---|---|
| E-CODEC：浮点→RGB8、YUV420、H264 或播放器导致伪影 | 同一 VAE RGB8 的无损片正常，原始 MP4 才异常 | A1 |
| E-VAE-NUM：BF16 VAE 中间计算精度不足 | 同 latent 的**真正 FP32 VAE**解码显著更好，且不损清晰度 | A2 |
| E-VAE-TEMP：VAE 时空卷积/首帧/缓存导致周期纹 | 纯净静止或可控平移片经 VAE encode→decode 也出现同类纹理 | A3 |
| E-LATENT：四步生成 latent 中已有不稳定高频结构 | A1/A2 均有伪影，A3 干净重建无伪影 | B0/B1 |
| E-RUNTIME：high/low 模型装错、Attention 或执行后端偏差 | 同权重数值参考的第一不等运算边界可定位；或运行时绑定不符 | B0/B2 |
| E-DISTILL：少步蒸馏/采样问题 | 原版多步 Dense 更少出现类似纹理，且加载数值链已核验 | B3 |

这些解释可以并存。先确定“在哪个边界首次出现可见鱼鳞纹”，再决定如何修复。外部官方 issues [Wan2.2 #143](https://github.com/Wan-Video/Wan2.2/issues/143)、[#139](https://github.com/Wan-Video/Wan2.2/issues/139)、[LightX2V #447](https://github.com/ModelTC/LightX2V/issues/447)、[#612](https://github.com/ModelTC/LightX2V/issues/612)、[#708](https://github.com/ModelTC/LightX2V/issues/708) 提供像素抖动、马赛克和模型映射错配的先例，但不等于我们的 BF16 鱼鳞纹已被解释。

## 2. 强制交付：完整视频给用户判断

所有新视频保留**两种可播放形式**：
- **无损 RGB8**：FFV1/MKV，81 帧原尺寸，记录无损解码回读的 RGB24 逐帧 SHA。注意 FFV1 只对已经量化的 RGB8 无损，并不保存 VAE 全部 FP32 中间浮点精度。
- **可播放预览**：高质量 H264 MP4、原尺寸 16fps、原长度。若实验涉及旧原生编码还额外保留原函数生成的 native_h264.mp4。原始历史 MP4 永不覆盖。

每次提交一份 REVIEW_INDEX.json，逐条提供**实际能打开的完整视频位置**、文件 SHA、来源、seed、成对关系；仅截图、联系图、ROI、频谱不能替代完整视频。用户可原尺寸循环播放、A/B 互换顺序、局部查看，建议关闭播放器插帧、锐化、降噪、HDR 自动增强，记录显示倍率和播放器。第一遍盲标 A/B，记录后才揭盲。未得到用户观看反馈时状态始终为 AWAITING_HUMAN_REVIEW，禁止 Agent 根据自动分数自行进入后续步骤。

人工每条记录 severity 0–4（无、轻微、明显局部、大面积持续、严重破坏画面）、全局/局部范围、时间窗口、ROI、是否随物体或相机运动、是否新增模糊或细节丢失。配对记录 fishscale_change=clearly_better/slightly_better/unchanged/worse/uncertain、整体偏好、可信程度、文字描述与下一动作（继续/解释后停止/再次观看/要求新配对）。机器辅助分数保留但 NON_VOTING。

## 3. A 层：先排除媒体与 VAE，新增 DiT 调用为零

### A0：历史视频目视筛查（只读，0 次模型调用）

从现有两条历史中性控制开始：
- ord898 / seed42 / i2v_subject
- ord96 / seed42 / i2v_background

先让用户完整观看各自原始视频；两条优先仅因身份与重复控制充分，并**没有证据表明它们的鱼鳞纹最明显**。若两条均不明显，在已有的 12 条 S2 开发视频中挑选用户人眼明显阳性的一个来源。冻结最多 2 个 A1/A2 诊断来源，尽量包含不同类别或一阳性一弱阴性，记录人工选择依据与症状出现的秒数。

若 14 条原视频均无法找到症状，记 PHENOTYPE_NOT_REPRODUCED，不启动后续解码；保留旧视频清单供用户指定其他来源。输出 A0/REVIEW_INDEX.json、A0/HUMAN_REVIEW.json。

### A1：同一 final latent 原生 BF16 VAE 中性重解码，拆开 RGB 与视频编码

对 A0 选中来源，各自从 S2 的 latent_step3.safetensors 读取真正 BF16 [16,21,58,104]，验证 file/tensor SHA、父 RUN、VAE 权重和软件身份。**每源只调用一次原生 BF16 VAE decode**，从同一个 RGB 结果分别导出：
1. VAE 浮点输出及归一化前后摘要、预先固定的小 ROI 浮点副本；
2. 全 81 帧 RGB8 FFV1 lossless MKV（读回逐帧 SHA）；
3. 原生产者编码流程的 native_h264.mp4；
4. 同标准的 review.mp4；旧原始 output.mp4 原封保留。

重要的原生等价约束：
- WanVAE.decode 接收四维 [C,T,H,W]，内部自己添加 batch；外部不得多加 batch。
- mean/inv_std 已在 VAE 内部应用，外部不得重复缩放。
- 正常 decode 在入口/出口清理时序缓存；不得偷换 cached_decode/decode_stream。
- use_tiling=false、use_lightvae=false、vae_cpu_offload=false、inference_mode；DTYPE、SENSITIVE_LAYER_DTYPE、USE_CHANNELS_LAST_3D 等带缓存 getter 的值要在 import 前确定。seed_all、cudnn deterministic/benchmark、torch/CUDA、TF32、编码工具版本照真实源记录。
- 原 wan_vae_to_comfy 会原地 add_(1).mul_(0.5).clamp_，对一个 buffer 只做一次；RGB8 转换使用原生 truncation，不能悄悄 round。原生编码保持 H264/YUV420、fps16、实际 464 高，不进行补到 480 的 resize。

若软件/FFmpeg 环境与原始一致，先要求新 native_h264 与原 MP4 完整 SHA 中性；若环境不同，保留差异并比较能验证的 RGB8 帧哈希、逐层后处理边界。不能仅因 H264 容器字节不同就认定 VAE 变化。若无法验证中性，停止后续归因，标 NEUTRAL_MISMATCH，保留视频和错误边界，不同条件的重试另开 scope。

**关口**：用户见原 MP4 有纹而 FFV1 无损 RGB8 没纹 → 查 RGB8 后处理、编码、播放；FFV1 已有纹 → A2；不确定 → 回看或 INCONCLUSIVE。

**成本上限**：2 VAE decode，0 DiT NFE。

### A2：相同 BF16 latent 的真正 FP32 VAE 解码

前提是 A1 的原生中性已通过且无损 RGB8 也有鱼鳞纹。固定 VAE checkpoint、latent、mean/std、空间时间 shape、RGB8 后处理及 FFV1/MP4 编码，仅把**VAE 参数、输入、Conv3D、归一化与内部运算**切换 FP32；在独立进程运行，禁 TF32、autocast、隐式 GET_DTYPE BF16 回落，并保存关键 tensor dtype/finite 证据。输入仍是由原 BF16 latent 提升为 FP32，不能恢复已经丢弃的 latent 精度。

每源仅增加 1 个 FP32 解码，保存 fp32_rgb8_lossless.mkv、fp32_review.mp4、dtype_trace.json、RUN/EXIT、所有 SHA。用户对 BF16 和 FP32 的完整视频盲审。FP32 清晰改善且细节保留 → 小范围确认 VAE 数值机制；没改善 → A3。若混入 BF16/TF32 或显存不足，标 INVALID_NUMERIC_CONTROL / RESOURCE_BLOCKED，不可把失败当负结果，更不可偷偷打开 tiling。

**成本上限**：额外 2 VAE decode，0 DiT NFE。

### A3：干净视频的原生 Wan VAE encode→decode

构造并先让用户查看两段 81 帧、832×464 清晰原片：
- STATIC：选一张含真实高频纹理的干净图片，按原生预处理后复制 81 帧，理论时间变化为零；
- TRANSLATION：基于同一静图以冻结的已知亚像素平移参数生成 81 帧，记录插值核、坐标、边缘填充及量化方式；由于插值本身会产生差异，始终以 clean 输入视频作参考。

分别用**同一 Wan 原生 VAE** encode→decode；须核像素域范围、latent mean/inv_std、时间首帧特例与 81=1+4×20 时序长度、cache 清理，不能换另一 VAE 解决接口问题。每组保留 clean_rgb8_lossless.mkv、recon_rgb8_lossless.mkv、recon_review.mp4 和 RUN。先核 clean 控制本身确实没有同类鱼鳞纹。

**关口**：干净片原来无纹、VAE 重建后出现与 Dense 同类纹理 → VAE 时序/空间机制优先；干净重建正常但生成 final latent 解码仍异常 → B0；两者不确定 → 固定更有判别力的控制，不盲目开大矩阵。

**成本上限**：额外 2 VAE encode + 2 VAE decode，0 DiT NFE。

**A0–A3 最坏基础预算：0 新 DiT、2 encode、6 decode。** 人工每过一关口才开放下一段，不能自动一次生成全套。

## 4. B 层：只有 A 层无法定位才推进

### B0：只读高低专家 / scheduler / 数值运行身份

依据真实 RUN、loader logs、checkpoint SHA、source snapshot、high/low 对象和 block 绑定检查：是否误把 high 权重加载到 low、是否 quantized_ckpt 和 BF16 original_ckpt 混用；HHLL、CFG=false、BF16 Dense/no QDQ、image/T5/CLIP、SDPA、shift5、实际 timesteps、VAE 身份是否一致。官方类似 issue 是排查线索，不意味着本仓必然有错。模型复现身份不足时先补源和实际证据，不直接重跑 14B。

### B1：在现有末步状态上做受限敏感性检查

优先复用 S2 的 latent_step2、flow_step3、latent_step3。先按真实 step_pre 和 step_post 数值运算/舍入核实 latent_step2 确实是第三次 step3 计算的 pre 状态（不满足则停止本对照）。记录准确的 z_pre、z_post，构造 z_beta=cast_BF16(z_pre+beta*(z_post-z_pre))，固定 beta=0.95/1.05，beta=1.00 已是原始 Dense。先只在**一个人眼阳性来源**做两次 VAE-only decode，保存完整 FFV1/MP4，用户判断是否鱼鳞纹变化；有价值时再扩另一来源，额外 2 decode。

这种 z_beta 是离线端点干预，可能不在合法真实采样轨迹上。鱼鳞纹强度随 beta 改变只说明末步敏感性，并**不能**单独证明末步或 scheduler 是最初起因；应注意欠去噪、细节降低、色彩变化等混杂。

### B2：只有运行时数值证据指向 DiT 时才做新四步对照

先查现有 full-flow、数值 neutral/loader 证据。必要时在同一 checkpoint、图像、prompt、初始 raw noise SHA、CFG、scheduler、VAE、geometry 下，只变一个被上游支持的 Attention 后端（原 native torch_sdpa 与匹配参考），记录**最先出现差异的中间边界**、四步 flow/post latent、完整视频和人眼评审。若不能证明同一初始 noise/同一权重，标 NON_MATCHED，不能作严格数值归因。每条完整四步新视频至少 4 个 denoising steps，新增 GPU scope/预算另批。

### B3：原版 40 步与四步蒸馏的归因——历史优先

已有 [G19 原版 40 步 BF16 Dense](../../20261105/G19_METHOD_AND_RESULTS_20261105.md) §chip10_00 / seed43§；[G19 四步蒸馏 BF16 Dense](../../../reports/20261105/g19_evidence/browse/outputs/G19_v7_真实生成与质量证据_20261104.md) 也有 §chip10_00 / seed43§。**先恢复这两条历史完整 MP4** 并核实文件、SHA、图像、prompt、裁剪、VAE、CFG/seed/输入噪声身份，给用户直接看，0 新 DiT。它们是描述性比较；原版 CFG、权重、采样配方不同，不足以单独证明蒸馏是根因。

只有旧视频无法回答区分性问题，才另定新原版多步和四步蒸馏的同源视频；使用**各自正确的官方推理配置**，而不是任意给四步权重增加步数。每条生成完整视频，原版若为 40 步将有至少 40 个 denoising-step 更新，实际 model-forward/CFG 数由运行器记账；新模型预算必须审查。

## 5. 辅助分析：为用户指出的纹理寻找机制

- **同帧/跨帧**：无损解码 RGB8 SHA、配对 RMSE/MAE、逐帧时间差、FFV1 与 MP4 回读差；注意旧 MP4 压缩差本身。
- **运动补偿高频**：r_t=H(I_t)-H(W_{t-1→t}I_{t-1})，仅统计可靠非遮挡 ROI；运动场尽量从低频图像估计，减少伪影污染。同时报告高频绝对能量和边缘/细节保真，防止模糊假改善。
- **空间/时间频谱**：2D PSD、时间谱、按 t mod 4 分组及第一帧/前五帧与后续区分；检查 x/y 8px 与时间 4 帧结构是否显著（只是 VAE 结构启发的假说，不是证明）。按 4×4 网格加人眼指定 ROI 报告。
- 这些指标是辅助诊断，不决定用户主观明显程度。VBench temporal_flickering 对动态视频解释受真实运动干扰，不应用作单独的“鱼鳞纹消失”验收。

## 6. 资产树与机器可追溯性

~~~text
历史只读输入：S2 14×{original.mp4, raw FP32 noise, BF16 noise, 4 flow, 4 latent, RUN/EXIT}
             └─ G19 原版40步 Dense + 四步 Dense 历史完整视频（B3）
新派生存储（HF root 实际路径在 worker 环境确认）
  temporal_fishscale_20261117/
  ├── A0/{REVIEW_INDEX.json,HUMAN_REVIEW.json}
  ├── A1/<source>/bf16/{rgb8_lossless.mkv,native_h264.mp4,review.mp4,frame_sha.json,RUN.json}
  ├── A2/<source>/fp32/{rgb8_lossless.mkv,review.mp4,dtype_trace.json,RUN.json}
  ├── A3/static/{clean_rgb8_lossless.mkv,recon_rgb8_lossless.mkv,review.mp4,RUN.json}
  ├── A3/translation/{clean_rgb8_lossless.mkv,recon_rgb8_lossless.mkv,review.mp4,RUN.json}
  ├── B0/identity.json
  ├── B1/<source>/beta_{095,105}/{rgb8_lossless.mkv,review.mp4,RUN.json}
  ├── B2|B3/<source>/<arm>/{full_videos,flow,latent,RUN,EXIT}
  ├── HUMAN_REVIEW/{REVIEW_INDEX.json,REVIEWS.jsonl,BLIND_MAP.json}
  ├── DIAGNOSTICS/{per_video.csv,psd/,roi/}
  └── ROOT_CAUSE_REPORT.md
~~~

所有派生文件关联 **source_id、image SHA、prompt SHA、seed、历史父 RUN/MP4/file+raw tensor SHA、high/low 权重 SHA、VAE 权重 SHA、代码 SHA、dtype/backend、decoder/encoder版本、frame_count、fps、size、decoded RGB8逐帧 SHA、实际 GPU/时长/输出字节、开始/完成数、真实 RUN/EXIT**。新 GPU worker 必须先验证 HF 相对路径映射和资产真实可读；历史 manifest 不等于实时可用。

每段 81×464×832×3 uint8 RGB8 未压缩 **93,809,664 字节**（约89.5 MiB），FFV1 大小依内容变化。建议新 A 层 scope **先预留至少2 GiB 新输出存储空间** 再核实实际预算，不能沿用另一个历史 VAE-only 桥接试验的 0.1 GiB 输出帽导致无损视频丢失。该 2 GiB 是建议容量，**不是已批准或已经分配**。完整视频、preview、RUN/人眼评审在结案前不自动清理。浮点全片可不全部存，保留数值摘要、必要预选 ROI 与哈希以节省空间。

| 工作 | 新 DiT NFE | VAE encode | VAE decode | 是否立即 |
|---|---:|---:|---:|---|
| A0 看现有 14 条（先看 2） | 0 | 0 | 0 | 首个任务 |
| A1 两个 BF16 neutral | 0 | 0 | ≤2 | A0 人眼确认阳性后 |
| A2 两个真 FP32 | 0 | 0 | ≤2 | A1 发现无损 RGB8 仍有纹 |
| A3 静止+平移 clean | 0 | ≤2 | ≤2 | A2 未解释 |
| **A 层最大总额** | **0** | **2** | **6** | 每轮人工关口 |
| B0 身份检查 | 0 | 0 | 0 | 按需 |
| B1 beta 0.95/1.05 单来源 | 0 | 0 | 2 | 按需，扩另一源 +2 |
| B3 历史40步 vs 四步 | 0 | 0 | 0 | 先恢复旧 MP4 |
| B2/B3 新完整生成 | 按实际4步/40步独立 scope | 原生链另计 | 完整视频另计 | 仅确有判别收益时 |

VAE-only 的 denoiser_NFE=0 不等于无模型计算。运行失败仍保存视频残留、错误和真实费用；不在同一冻结 scope 中更改精度、tiling、权重后静默重试。

## 7. 修复确认、停止条件与必要工作流

| 用户审看的结果 | 立即行动 |
|---|---|
| A0 找不到明确的同类鱼鳞纹 | 先定位一条人眼阳性历史来源，不启动生成 |
| A1 原生 lossless 正常、旧 MP4 异常 | 锁定编码/颜色/播放器，暂停 VAE/DiT |
| A1 lossless 已有纹 | A2 |
| A2 真 FP32 明显改善且细节不受损 | VAE 数值分支小规模确认 |
| A2 两者接近 | A3 |
| A3 干净视频也被 VAE 引入相似纹 | 分析 VAE 首帧、缓存和时空重建 |
| A3 干净重建正常而生成 RGB8 异常 | B0，再按证据选 B1/B2/B3 |
| B1 变更 beta 后纹理变化 | 仅证明局部敏感性，不归因为原始根因 |
| 原版多步好于蒸馏四步 | 保留描述性结论，先处理 CFG/轨迹/权重混杂 |
| 任何一个候选看似少纹但细节/动作退化 | 记录有代价的缓解，不认定画质修复 |
| 原生 neutral 不等、FP32 路径不真实、原片 SHA 不符 | 保存全部产物并暂停该层归因 |

确定一个有效修复后，优先在 **至少四个不同已暴露开发 image-cluster** 上做完整视频配对人工复验，覆盖静态/运动及不同类别。已有开发来源用于挑选与确认修复方向，**不能宣称盲测泛化**；若要发表普遍质量结论，另冻未接触的确认源和不同 seed。结案须报告所有完整视频索引、人眼原始评价、首个异常边界、单因素证据、模糊/动作代价、未解决的混杂，以及下一步最省钱的实验。

**实施顺序与权限**：先实现只读 manifest/HF preflight，再实现单独小型 VAE-only 诊断 worker（native-neutral、true-fp32、clean encode/decode、FFV1/H264 双输出、视频逐帧 SHA、REVIEW_INDEX 汇编）。这里约定的是**待实现接口**，不是已经存在的脚本或已完成运行。每个新 CPU/GPU scope 按 FFNDS 现有流程：作者自审 → 真正独立 reviewer → 修订与复核 → ACTIVE 登记预算/源码/资产 SHA → 运行前身份复核 → 实际 RUN/EXIT → 保存完整视频 → **用户亲自观看决定下一轮**。本提交不能自动触发 GPU 作业或修改 ACTIVE。

**最终标准**：不是降低某个公式的高频分数，而是用尽量少的新增 DiT/ VAE 运算提供可独立观看、身份可复验的配对视频，并由真实边界证据解释用户观察到的鱼鳞纹为何改变。