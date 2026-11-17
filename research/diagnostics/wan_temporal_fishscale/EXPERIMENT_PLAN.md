# Wan2.2 I2V 四步全局鱼鳞纹：USTC 学校集群独立实验方案

> **2026-11-17 · SCHOOL_ONLY / DESIGN_ONLY / REVIEW_REQUIRED / NOT_RUN。** 新实验只在学校 Slurm 计算节点运行，所有新生成与派生视频只写学校账户对应的实验目录。旧 SSH `ffnds-worker`、`/home/xiang`、旧 HF 根目录及 S2 的 14 条远端视频/latent **不属于运行依赖**。研究源头可以引用历史结果，但绝不把历史 `root=hf` 重解释成学校集群挂载。本文已经纠正执行语义，**并没有登录学校集群、验证资产或提交作业**。

## 0. 统一场地、身份与执行门槛

**学校端（来自用户目录与 [模型/服务器旧配置](../../../docs/01_模型与服务器配置.md)，当前尚未实地验证）：**

| 对象 | 学校路径或身份 |
|---|---|
| 声明的学校 home | `/public/home/scc/pb23061276` |
| 较早学校文档使用的 home | `/home/scc/pb23061276`；仅在计算节点用 `realpath` / `samefile` 证明别名相同时才互通 |
| Wan 项目根 | `$SCHOOL_HOME/projects/wan-quant` |
| 学校蒸馏 D-BF16 | `$WAN_ROOT/weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step` |
| 学校原版 O | `$WAN_ROOT/weights/Wan2.2-I2V-A14B-bf16` |
| 新诊断输出 | `$WAN_ROOT/experiments/ffnds_fishscale_20261117/` |
| GPU 管理 | 学校 Slurm；Students/qos 等旧配置只作待现场核验的 hint，不套用旧 SSH worker 的 GPU0/1 锁 |
| 仓库代码 | FFNDS 在学校端的实际 checkout 路径由现场确认，不能误认为旧 worker 的 `.runtime/code` 已挂载 |

机器配置见 [SCHOOL_CLUSTER_CONFIG.json](SCHOOL_CLUSTER_CONFIG.json)。**每个任务先用当前分配的计算节点查真实模型文件、视频与磁盘，不用历史 SSH 路径。** 预计学校模型 high/low checkpoint 的文件名和原生四步配方在配置里给出，旧冻结 manifest 的 SHA 只是待比对的参考；学校实际字节未读到之前不能把它写为 PASS。

执行分为：
- **P0：学校资产发现与完整性验证**。在学校计算节点只读执行 [预检程序](../../../tools/fishscale_school_preflight.py)，记录 HOME 真实路径、school project、D/O 权重、VAE/T5、可读候选视频/latent 的 stat、SHA、header/视频几何、存储空间和 Slurm job 身份。源 pair 必须来自学校；未绑定时状态 `SOURCE_PAIR_NOT_REGISTERED`，绝不退回旧 S2。
- **P1：复用学校已有 Dense**。只有学校上**同一次 Dense 生成**的 MP4 与末步真实 BF16 latent 都可读、来源和 SHA 绑定完成，才进入 A0（如果只有视频，先 A0 看，再决定是否需重新捕获 latent）。
- **P2：最小学校 Dense 捕获**。如果学校没有可用成对资产，但 D-BF16 模型/代码/输入准备完毕，另申请**一个** 4 步完整 Dense 视频，保存原生 BF16 末步 latent、原始 MP4、RUN/EXIT；人眼确认有目标鱼鳞纹再进行 A1。第二来源只在首来源有明确判别价值时增加。若模型或输入不完整，先补齐学校端资产，不提交模型 job。

P2 **不是重做旧 S2 的 14 条视频**，也不伪造旧 ord898/ord96 的 SHA 或覆盖原来的 checkpoint。当前仓库 [run_distill.py](../../../tools/run_distill.py) 有后处理 `scheduler.latents.detach().float()`，输出的是 **FP32 观察副本**，不能冒充原生 BF16 latent。学校 capture 实现必须在原生 scheduler.step_post 后读取未经 `.float()` 的 BF16 latent，或对原生 BF16 缓冲的保存行为做完整数值验收；四步 flow、pre/post 状态是否能用于 B1 需分别验证。若学校执行的其他 runner 与 [学校冻结方法合同](../../../docs/01_模型与服务器配置.md) 不同，记录真实代码、输入和数值差异，保留新的生成身份。

**报告中的 baseline 来源始终是** `SCHOOL_REUSED_VERIFIED` 或 `SCHOOL_NEW_CAPTURE`。旧 S2 和 G19 远端记录只保留 `HISTORICAL_REMOTE_REFERENCE_NOT_EXECUTION_INPUT`，也不要求到别人的 worker 复制数据作为先决条件。

## 1. 科学问题与不同机制的预测

目标：定位完整 Dense 视频全局细密鱼鳞纹与持续闪烁**首次出现在哪个计算边界**，在用户人眼能够确认现象之后，控制额外计算成本，得到可解释的改善手段。

```text
学校 D-BF16 high/low + 原生 4 step DiT / scheduler
            ↓
学校生成并保存的 native BF16 final latent
            ↓
学校 Wan2.1_VAE.pth / 原生 WanVAE
            ↓
浮点 RGB → RGB8
            ↓
FFV1 无损 / 原生 H264 → MP4
            ↓
用户完整视频观看、判断鱼鳞纹
```

| 候选解释 | 不同预测 | 可区分对照 |
|---|---|---|
| 媒体量化/编码/播放 | FFV1 RGB8 无纹，H264/播放后才出现 | A1 |
| VAE BF16 数值敏感 | 同一 latent 的真正 FP32 解码改善且不损失纹理 | A2 |
| VAE 时空卷积/首帧/缓存 | 干净静止/平滑运动输入经 encode→decode 形成同类纹 | A3 |
| 四步 DiT latent 不稳定 | A1/A2 解码仍有纹而 A3 干净重建正常 | B0、B1 |
| 模型绑定或 Attention 后端不一致 | 学校 high/low 权重绑定异常，或同输入 first-bad-boundary | B0、B2 |
| 四步蒸馏/采样本身 | 学校原版多步完整视频明显优于四步，且别的实现变量已核实 | B3 |

Dense 有纹已使“FFN 稀疏近似独自导致”不成立；**没有排除蒸馏权重自身或模型加载问题**。多个机制可以共同影响最终结果。

## 2. A 层：人眼先行的最小分支

### A0 / 人工确认学校 Dense

只看从 P1 或 P2 验证取得的学校原始完整 81 帧 MP4。用户以原尺寸播放判断：是否为大范围细密周期鱼鳞纹、出现时段及伴随哪些场景运动。若阳性，冻结这个 source/seed/image/prompt 的学校 pair 身份；若没有同类现象，改用**另一个学校来源**，而非回退到缺失的旧远端视频。尚未看到实际学校原视频时，必须写 `AWAITING_SOURCE_VIDEO`，不能声称某一旧视频“已由用户确认”。

### A1 / 1 次原生 BF16 VAE 解码

同一学校 final latent 和 WanVAE，在学校 Slurm GPU 上一次 decode 后**从同一个输出**分别导出：
- 主要浮点 RGB 边界摘要与选定 ROI 数值（避免复制整段 FP32 张量），
- RGB24 FFV1/MKV **81 帧无损视频**及逐帧 decoded RGB8 SHA，
- 原生 `save_to_video` H264/YUV420 版本，
- 便于观看的标准 H264 MP4 预览。

旧原始 MP4 不覆盖。原生包装接收 4D latent，VAE 内部使用 mean/inv_std，`wan_vae_to_comfy` 为原地归一化且只可执行一次；不额外 scale、不额外 unsqueeze、不改 `use_tiling=false`，`decode` 时序缓存由原函数清理。精度、TF32、backend、cudnn/环境 getter 导入时点与学校原始生成真实配置必须匹配并记录。

**中性控制**比较学校原始 MP4 与重新原生生成的 MP4 和逐帧 RGB8。若编码环境不同，区分编码元信息差、RGB8 差，不能以 MP4 字节不等替代 VAE 数值结论。中性失败立刻保留视频、日志并停止这一条归因，不能调参后继续称 neutral。

**人工关口**：无损 RGB8 正常、MP4 异常 → 编码/显示；无损仍有鱼鳞纹 → A2；不确定 → 回看。

### A2 / 同一学校 latent 的 FP32 VAE

只有 A1 已验证中性且 FFV1 有纹时执行。固定学校 checkpoint、latent、视频几何、后处理/编码，仅在**独立进程**将 VAE 参数、Conv3D、归一化及内部运算真正切为 FP32，核对中间 dtype 和 TF32/autocast。输入仍是 BF16 latent 提升类型，不能补回原始损失精度。保留 FP32 FFV1/MKV、MP4、全部帧 SHA、dtype_trace，交用户盲评。改善还要检查是否只是变模糊。失败/显存不足不能静默打开 tiling。

### A3 / 两个干净视频的原生 VAE 重建

构造学校端两条 81 帧、832×464 的清晰片：
- **STATIC**：固定高纹理图像逐帧精确复制，原视频理论时变为零；
- **TRANSLATION**：同图按明确亚像素平移公式生成 81 帧，固定重采样核、边界、量化；以干净原片作为误差基线。

两条分别经过学校 Wan 原生 VAE encode→decode，保留 clean/recon 的完整 FFV1 与 MP4，不允许换 VAE 或直接读取远端生成 latent。用户判断：干净原片没有、重建后有同类纹 → VAE 机制值得定位；干净重建正常但 DiT final latent 仍有纹 → B 层。

**基础 A 层在一个来源上**最多 1 次中性 BF16 decode + 1 次 FP32 decode + 2 次干净视频 decode、2 次 encode = **4 decode + 2 encode**，均为条件执行；不含 P2 新 Dense 生成自身的一次 VAE 解码与输入编码。所有 arm 分阶段等待人工反馈。

## 3. B 层：仅在 A 层证据指向生成侧时

**B0 学校加载链审计（只读）**：核 school high/low safetensors 的真实 SHA/模型实例、step0/1 high step2/3 low、CFG=false、BF16、无 QDQ、attention=torch_sdpa、shift=5、真实 timestep、图像/T5/CLIP、WanVAE 的版本和数值 recipe。官方曾有模型 high/low 误绑定导致马赛克的反馈，但不是我们这里已证实的错误。

**B1 同学校原生 flow 的末步敏感性**：先确认学校原始 step2 post、step3 pre、step3 flow 和 step3 post 均可用于同一条真实轨迹，确保 scheduler.step_post 的 FP32 运算/舍入可重建；若 capture 没保存这些变量，不能假装拥有历史 flow。固定 `z_pre,z_post`，构造 `z_beta=cast_BF16(z_pre+beta*(z_post-z_pre))`，只取 beta=0.95/1.05，beta=1 复用学校原视频；每 arm 保留完整 FFV1/MP4。它只测末步敏感性，不是合法新采样，更不能单独证明 step3 为根因。

**B2 学校数值后端配对（按需）**：只在 B0 发现实际不一致/怀疑后端时，冻结同一学校模型、原图、prompt、初始噪声、step、CFG、VAE，单独改变一个被上游正确支持的 Attention 后端，保存四步 flow/latent 和完整视频，找 first-bad-boundary。跨 GPU、kernel、随机性差异必须记录，不将不同运行前提宣称 bitwise 等价。每新增完整四步视频是新的模型计算费用，单独审批。

**B3 学校原版多步 vs 学校四步蒸馏**：先只检索**学校端已经存在且可验证**的原版 O Dense 完整视频和蒸馏 D Dense 视频；不再默认旧远端 G19 的 chip10_00 历史 MP4 在学校可读。若学校端没有同源成对视频，且上面实验有强烈定位价值，再使用学校现有原始 O 模型和 D 模型各自正确原生推理配置，生成新学校配对视频。40步原版与4步蒸馏具有不同权重/CFG/轨迹，结果可描述症状差异，但需后续单因素机制来支持归因。

## 4. 实验资产树（学校端唯一真实输出）

```text
/public/home/scc/pb23061276/projects/wan-quant/
├─ weights/
│  ├─ Wan2.2-I2V-A14B-LightX2V-Distill-4step/   # 实际字节待核
│  └─ Wan2.2-I2V-A14B-bf16/                     # 实际字节待核
└─ experiments/ffnds_fishscale_20261117/
   ├─ preflight/
   │  ├─ INVENTORY.json             # 只读学校文件发现
   │  ├─ FULL_SHA_PREFLIGHT.json     # 严格完整验证
   │  └─ RESOLVED_SCHOOL_PATHS.json
   ├─ school_sources/
   │  └─ <source_id>/
   │     ├─ input_manifest.json
   │     ├─ original_dense.mp4
   │     ├─ final_latent_bf16.safetensors
   │     ├─ optional_steps/
   │     └─ CAPTURE_RUN.json
   ├─ A0/<source_id>/{review_index.json,review.json}
   ├─ A1/<source_id>/{rgb8_lossless.mkv,native_h264.mp4,review.mp4,RUN.json}
   ├─ A2/<source_id>/{rgb8_lossless.mkv,review.mp4,dtype_trace.json,RUN.json}
   ├─ A3/{static,translation}/{clean_lossless.mkv,recon_lossless.mkv,review.mp4,RUN.json}
   ├─ B0/{runtime_identity.json}
   ├─ B1/<source_id>/beta_{095,105}/{rgb8_lossless.mkv,review.mp4,RUN.json}
   ├─ B2_B3/<source_id>/<arm>/{full_video.mp4,states,RUN.json}
   ├─ human_review/{REVIEW_INDEX.json,REVIEWS.jsonl,BLIND_MAP.json}
   ├─ aux_metrics/{per_video.csv,psd/,roi/}
   └─ ROOT_CAUSE_REPORT.md
```

此树为**拟定的学校输出目录**，并非目前已经在集群建立的目录。源 MP4 在人工评审关闭前不清理。所有新视频存 full 81 帧原尺寸（不可降采、插帧、自动清理），同时保留 FFV1 与标准 MP4；每个文件绑定学校真实 path、父 RUN、source/image SHA、prompt SHA、seed、high/low/VAE SHA、shape/dtype、源码 commit、GPU/CUDA/ffmpeg/encoder、完整视频和 decoded RGB8 哈希。

一个 81×464×832×3 的原始 RGB8 约 93.8 MB；FFV1 大小与内容相关，首个学校 A 层建议申请 ≥2 GiB 输出余量，正式预算按当前学校 quota 与实际授权计算。诊断方案不能擅用既有 ICC/G20 的 GPU 分配；不套用老 worker 的 0.1 GiB 输出帽。建议单次 1 GPU，学校 Slurm 分区/QOS/时长由当次 `sinfo/sacctmgr` 和批准 scope 验证。

## 5. 第一轮执行次序与有效证据

1. 在学校计算节点运行学校 preflight（见 [SCHOOL_CLUSTER_RUNBOOK](SCHOOL_CLUSTER_RUNBOOK.md)），严格区分 `INVENTORY_ONLY` 与 `VERIFIED_READABLE_SHA`，核别名、模型、视频、latent、剩余空间。**只有资产验证完毕才进入后续实验**，不能称之为 smoke。
2. 以学校 `ASSET_REGISTRY.json` 绑定已有同源 Dense/latent；没有现成 pair 就冻结一个学校来源和输入，使用学校 BF16 原生 runner 额外做 1 次 P2 dense capture。所有新任务通过真正独立预执行复审、ACTIVE 及当次 Slurm 资源授权。强制记录真实 RUN/EXIT、源 hash 和模型权重 SHA。
3. A0 的原始 Dense MP4 完整转交用户；**用户确认鱼鳞纹阳性后**，才申请 A1 的 VAE-only 重解码。此后每轮先保存完整视频并让用户看，再决定后续分支。
4. 首个明确定位的修复在至少 4 个不同学校开发 image-cluster 上做完整视频人工复验；这些仍是开发源。泛化声明须新增独立确认源和 seed，不把 81 帧或多次观看算独立样本。
5. 结案报告必须给出可观看的所有完整视频、人工原始观察、首个异常边界、单变量/数值证据、画面模糊/动作保真代价和仍开放的因果解释。没有用户审片标 `AWAITING_HUMAN_REVIEW`；没有新作业标 `NOT_RUN`。

## 6. 成本判断（不把缺失学校资产算成“0 DiT”）

| 前提 | 首个可观看原始 Dense 视频的新 DiT 费用 | A 层新增 VAE |
|---|---:|---:|
| 学校已有完整可验证 Dense MP4 与 BF16 final latent | **0** | 最多 4 decode + 2 encode |
| 学校仅有原始视频而没有合法同轨迹 latent | 用户可先 A0；要做 A1 则需新捕获 | 新捕获本身会调用原生 VAE，再计 A 层 |
| 学校完全没有可用配对，但模型与输入均验证完毕 | **一次四步完整 Dense（至少4个去噪步）** | 同上，另计新 Dense 的 VAE/输入编码 |
| 学校 D-BF16/ VAE / 输入缺失 | **禁止启动**，保持 `SCHOOL_ASSET_MISSING` | 0 |

如无 school original40 视频，B3 额外原版40步开销单独批准。所有 phase 用真实学校 RUN 的源数、step数、encode/decode次数及运行时间记账。

**最终目的**：不依赖旧工作机、不重做整个旧 S2 数据集；以学校集群最小可验证资产为起点，保留全部可播放视频交给用户人眼判断，再用分层干预定位鱼鳞纹的真实产生边界。