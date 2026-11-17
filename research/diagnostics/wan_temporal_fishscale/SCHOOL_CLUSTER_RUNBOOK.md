# USTC 学校集群执行手册：鱼鳞纹诊断

**2026-11-17；执行位置唯一为学校 Slurm。** 本文是就绪流程与样例命令，不声称已连接学校机器，也不意味着任何新实验得到了 ACTIVE 批准。原 SSH `ffnds-worker`、`/home/xiang` 资产和缓存不出现在新任务的执行路径。

## 1. 首先确定真实文件系统（在学校计算节点）

以学校账户已确认路径为准：

```bash
SCHOOL_HOME=/public/home/scc/pb23061276
WAN_ROOT="$SCHOOL_HOME/projects/wan-quant"
FFNDS_REPO=/path/to/your/FFNDS-checkout    # 填真实的学校 checkout，不能默认与 WAN_ROOT 相同
test -d "$SCHOOL_HOME" && test -d "$WAN_ROOT" && test -d "$FFNDS_REPO"
realpath "$SCHOOL_HOME" "$WAN_ROOT" "$FFNDS_REPO"
if test -d /home/scc/pb23061276; then
  test "$SCHOOL_HOME" -ef /home/scc/pb23061276 || { echo "HOME ALIAS MISMATCH"; exit 2; }
fi
sinfo -o '%P %G %l'     # 实际分区、GPU 与时间限制；文档中的 Students 仅为 hint
df -h "$WAN_ROOT"
```

请将以下命令运行于 `srun` 或 `sbatch` 真正分配的**学校计算节点**；在登录节点做的检查只能列为 `LOGIN_ONLY`，不等于执行节点文件可读性验证。不能假设学校上可以访问另一台 SSH worker 的 NFS home。

## 2. 学校资产分两步检查：inventory 不是严格 preflight

在学校端 checkout 执行：

```bash
cd "$FFNDS_REPO"
SCHOOL_OUT="$WAN_ROOT/experiments/ffnds_fishscale_20261117"
mkdir -p "$SCHOOL_OUT/preflight"
python3 tools/fishscale_school_preflight.py \
  --phase inventory \
  --config research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json \
  --registry research/diagnostics/wan_temporal_fishscale/ASSET_REGISTRY.json \
  --out "$SCHOOL_OUT/preflight/INVENTORY.json"
```

`inventory` 不对大文件算 SHA，产物只能叫 `INVENTORY_ONLY_NOT_AN_EXECUTION_PASS`，不是 smoke、不是模型运行授权。当前登记的学校 source pair 数量为零，属于必须补齐的事实。

**若已找到学校成对的 Dense MP4+final native BF16 latent：** 按 [资产登记模板](ASSET_REGISTRY.json) 在学校端输出目录中新建私有 `SCHOOL_PAIR_REGISTRY.json`，每条记录填写实际 school path、文件 SHA、源图片与 prompt、seed、父 RUN；不需要修改历史 S2 JSON。注意这里的 `relative_path` 是相对 `WAN_ROOT`，若输入绝对路径则必须位于学校项目真实目录内。

接着在**学校 Slurm CPU allocation**运行：

```bash
python3 "$FFNDS_REPO/tools/fishscale_school_preflight.py" \
  --phase review \
  --config "$FFNDS_REPO/research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json" \
  --registry "$SCHOOL_OUT/preflight/SCHOOL_PAIR_REGISTRY.json" \
  --pair-id school_dense_01 \
  --out "$SCHOOL_OUT/preflight/REVIEW_FULL_SHA_PREFLIGHT.json"

python3 "$FFNDS_REPO/tools/fishscale_school_preflight.py" \
  --phase decode \
  --config "$FFNDS_REPO/research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json" \
  --registry "$SCHOOL_OUT/preflight/SCHOOL_PAIR_REGISTRY.json" \
  --pair-id school_dense_01 \
  --out "$SCHOOL_OUT/preflight/DECODE_FULL_SHA_PREFLIGHT.json"
```

严格模式会要求 `SLURM_JOB_ID`、真实学校目录、模型/视频/latent 的 SHA、81 帧 832×464 16 FPS、latent 为 BF16 [16,21,58,104]；`review` 只需原 MP4，`decode` 还需要 VAE 和末步 latent。**报告有 VERIFIED_SCHOOL_INPUTS_READY_FOR_* 才表示输入身份与可读性验证通过**；它仍不是实验批准或诊断完成。如果没有 source pair，脚本应直接返回非零状态，不允许回退到旧 `root=hf`。

`generate` 严格检查高低噪声、VAE、T5、学校输入图像及 prompt manifest；`SCHOOL_CLUSTER_CONFIG.json` 的输入最初为 null，所以会 BLOCK。要捕获新 Dense 前，另建新 scope、绑定学校 source 输入、检查原生生成程序的确切调用路径，再重新验证。**不要对空输入或缺失配置默认运行官方示例图片。**

## 3. 没有学校 Dense 配对资产时的最小恢复

1. 先检查学校已有的视频集合（例如 WAN_ROOT 下的真实实验输出目录），按真实学校 RUN 和 SHA 寻找完整 Dense 与同一轨迹 final BF16 latent；不要在共享 FS 上无界全盘 find，也不要按文件名相似就接受。
2. 若学校仅有可播放的 Dense MP4，先交用户 A0 观看；如用户确认鱼鳞纹且确需重解码，再考虑同配置新 capture。旧视频与新 capture 是两次不同的 run，除非每个初始噪声、输入与末步状态做过配对验证，否则不能宣称同源 bitwise 中性。
3. 若没有可用学校 pair，就注册一个来源（school image、prompt、seed=42、裁剪、输入哈希），用学校现有 Wan2.2 I2V D-BF16 四步 runner **只跑一次 Dense**，同进程保存原生 BF16 final latent、81 帧完整 MP4、实际权重/代码/输入/后处理与 RUN/EXIT。第二来源条件性增加。
4. 对捕获的 native BF16 latent，不能直接使用 FFNDS 旧 `tools/run_distill.py` 产生的 `latent_step3.safetensors` 作为严格 A1 输入：该脚本保存前执行了 `.float()`。如采用它的原生模型流程，需另加**不改变 scheduler 数值的 BF16 捕获**，确认 `scheduler.latents.dtype == torch.bfloat16`，在 `.float()` 前保留原生张量；模型输出 neutral 不应被捕获钩子扰动。捕获适配代码须由学校执行环境独立测试与审查。
5. 第一次新生成还会产生一次原生 VAE 视频解码和 I2V 输入编码；不能把“后续 A 层 0 DiT”误认为从缺失资产到首条视频总共 0 DiT。完整模型生成需要 Slurm GPU、原生 encoder/VAE/DiT 真正就绪、独立评议和 ACTIVE 许可，不以此文档自动启动。

## 4. 完整视频交付与用户人工决策

对于每个 A1/A2/A3 对照，保存完整 `rgb8_lossless.mkv`（FFV1 真无损于 RGB8）和 `review.mp4`（H264 高质量），还保留 P2 `original_dense.mp4`。不重编码覆盖原片。生成 `human_review/REVIEW_INDEX.json`，记录学校绝对路径、SHA、source/seed、对应 arm。用户可以通过当前学校允许的 SCP/SFTP/SCOW 文件浏览流程把真实完整视频取回电脑观看；不能用“已写到了 GitHub”冒充已交付视频。没有实际文件时不放虚构下载链接。

用户确认某一组明显改善、变差、相同或不确定以后才推进下一级；机器频谱指标不拥有投票权。

## 5. 可复现与权限

学校 Slurm 账户/分区/QOS/CPU/GPU/时长、存储配额由当前登录现场的 `sinfo`、`sacctmgr`、Slurm 分配结果确定。旧学校文档的 `Students` / `qos_stu_medium_2gpu` 只作候选 hint，不当成现时已确认的权益。CPU 严格检查可用单独小 allocation；GPU 新生成/解码单独冻结 scope 与预算，执行前仍需 FFNDS [预执行独立复审](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md) 和 `ACTIVE_RESEARCH.json` 登记。

原始模型/蒸馏模型配对 B3 同样只在学校完成，先检索学校端已有完整视频；旧 worker 中的 G19 原版40步视频只在文献/结果比较里保留，不成为新实验的文件依赖。

**本手册尚未在学校计算节点实测。** 实际 preflight receipt、真实二进制文件、视频观看和新模型结果必须由学校执行产出，不能凭本次 GitHub 写入声称已存在。


## 6. 实用学校端补充：来源发现、注册与原生捕获

**有界发现**：只检查学校项目内明确给定的子目录（深度≤4、最多返回300项，stat-only 而非 hash PASS），可在 `inventory` 中追加参数：

~~~bash
--candidate-dir experiments --candidate-dir outputs
~~~

**登记真正存在的学校视频**：在学校端输出目录下建立私有来源登记，不再引用远端 S2 的 ord898/ord96。以下路径必须按现场已存在的真实文件替换：

~~~bash
python3 "$FFNDS_REPO/tools/fishscale_school_register_pair.py" \
  --config "$FFNDS_REPO/research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json" \
  --out "$SCHOOL_OUT/preflight/SCHOOL_PAIR_REGISTRY.json" \
  --pair-id school_dense_01 --seed 42 \
  --video "$SCHOOL_OUT/school_sources/school_dense_01/original_dense.mp4" \
  --latent "$SCHOOL_OUT/school_sources/school_dense_01/final_latent_bf16.safetensors" \
  --generation-run "$SCHOOL_OUT/school_sources/school_dense_01/CAPTURE_RUN.json" \
  --image "$WAN_ROOT/path/to/real_school_input.png" \
  --prompt-file "$SCHOOL_OUT/school_sources/school_dense_01/prompt.txt"
~~~

这只登记学校文件和 SHA；**还需在 Slurm 计算节点运行上文 `--phase decode` 严格预检**。若学校只有一条旧 MP4、还没有原生 latent 和同一次运行的 RUN，可以先只带 `--video` 登记用于 A0 人眼观看，生成登记状态 `REGISTERED_FOR_REVIEW_ONLY`；A1 不得据此启动。

**学校源输入 manifest**：如需在学校新捕获一个 D-BF16 Dense，保存私有 `$SCHOOL_OUT/preflight/NEW_SOURCE.json`，填写真实的 `image_path`、`image_sha256`、`prompt`、`seed`；占位样例不能直接运行。随后在学校 Slurm 节点执行：

~~~bash
python3 "$FFNDS_REPO/tools/fishscale_school_preflight.py" \
  --phase generate \
  --config "$FFNDS_REPO/research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json" \
  --registry "$SCHOOL_OUT/preflight/SCHOOL_PAIR_REGISTRY.json" \
  --source-manifest "$SCHOOL_OUT/preflight/NEW_SOURCE.json" \
  --out "$SCHOOL_OUT/preflight/GENERATE_FULL_SHA_PREFLIGHT.json"
~~~

它会完整 SHA 校验学校 D high/low、T5、VAE 和输入图。`SCHOOL_CLUSTER_CONFIG.json` 中 T5/VAE 是待验证的学校位置候选，找不到就 BLOCK，不能用旧 worker 目录作为 fallback。

**原生 BF16 捕获组件**：[school capture hook](../../../tools/fishscale_school_native_capture.py) 仅供经批准、真实可运行的学校 LightX2V Dense pipeline 接入；不是完整生成 CLI。在学校 pipeline 初始化后使用：

~~~python
from fishscale_school_native_capture import NativeBF16PostStepObserver
with NativeBF16PostStepObserver(pipe.runner.scheduler, output_dir/"native_latents") as obs:
    pipe.generate(seed=seed, image_path=str(image), prompt=prompt,
                  negative_prompt="", save_result_path=str(output_video))
obs.complete(video=output_video, source_id=source_id, image=image,
             prompt=prompt, seed=seed, high_sha256=high_sha, low_sha256=low_sha,
             vae_sha256=vae_sha, source_commit=source_commit,
             receipt_path=output_dir/"CAPTURE_RUN.json")
~~~

`pipe/output_dir/source_id` 等由学校生产 runner 提供；原始 pipeline 接口不一致时必须单独适配并测试，不得生造生成成功记录。钩子读取原生 BF16 scheduler post latent、保存四步状态，并在完成后将视频 SHA 与 final latent SHA 绑定到同一 CAPTURE_RUN；这也**不表示已经执行了中性 bitwise 验证**。没有 step3 pre/flow 的输入就不能执行 B1 插值实验。

**静态测试**（尚未在本次实际运行）：

~~~bash
cd "$FFNDS_REPO"
python3 -m unittest discover -s tests -p 'test_fishscale_school_preflight.py'
~~~

任何真实模型/ GPU/视频实验仍必须先完成独立评议和 ACTIVE 注册。`
