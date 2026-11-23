# 四步推理默认方法

`wide8192` 用于 Wan2.2-I2V-A14B 四步蒸馏推理。高、低专家各有一个锚点步和一个命中步，命中时固定观测 64 个位置。锚点输出直接写入槽位，两项投影分别计算，每个命中块最多 8,192 行，以 FP32 原址累加读出。

学校已有部署的作业入口为 `sbatch "$FFNDS_PROJECT/fourstep_current/job.sh"`，其中 `FFNDS_PROJECT` 指向学校项目目录。Python 调用 `Experiment().render(source, group)` 与 `make_port(scope)` 同样默认选择 `wide8192`。同一 `Experiment` 会复用端口。显式传入 `method=None` 可以跑原生 Dense，传入 `experiment.build("fast512")` 可以复查旧方法。

## 从学校检查点建立运行目录

在安装了原 LightX2V、Wan 权重、80 份逐层 CAL、校验所需输入图和参考运行器的学校环境中，将 `fourstep-wide8192-185220.zip` 解包。在新建的可写作业目录中执行以下命令。`CHECKPOINT_RUNTIME` 指向解包后的 `runtime/`，`FFNDS_REPO` 指向本仓库。保留冻结运行器供新入口调用。原调度、配置和素材绑定仍由它校验。

```sh
cp -R "$CHECKPOINT_RUNTIME"/. .
sha256sum -c MANIFEST.sha256
mv MANIFEST.sha256 MANIFEST.reference.sha256
mv job.sh job.reference.sh
cp generate.py generate_reference.py
cp "$FFNDS_REPO"/tools/ffnds_fourstep/*.py .
sha256sum *.py *.json job.reference.sh UPLOAD.sha256 MANIFEST.reference.sha256 > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

在分配了 A100 的学校作业中用当前目录作为工作目录，执行 `python generate.py vbi_0392 --group wide8192`。分组名标识输出位置。每条成功的默认成片收据记录 `counters.variant = "wide8192"`、`tile_rows = 8192`、80 次锚点及 80 次命中。

学校 A100 三个开发来源、每源两对，整片平均 41.532→39.440 秒，相对 Dense 下降 5.04%。各来源的顺序及计时见[逐次实测](../../reports/20261122/fourstep-wide8192-185220/timing.json)。

## INT8 锚点缓存变体

在上述运行目录和学校 CUDA 环境下，通过同一入口选择缓存格式。每条命令默认将输出写入对应变体的分组。

```sh
python generate.py vbi_0392 --variant wide8192
python generate.py vbi_0392 --variant wide8192_all_int8_joined
```

Python 入口支持 `Experiment(variant="wide8192_all_int8_joined").run("vbi_0392")` 或 `make_port(scope, "wide8192_all_int8_joined")`。省略变体时使用 BF16 缓存的 `wide8192`。

INT8 变体使用同样的 BF16 W1、W2 和 8,192 行命中分块。它保存输入 `u0`、输出 `y0` 和选中通道预激活 `z0k` 的 INT8 编码，每组 128 通道共享一个 BF16 缩放因子，每层单独保存 64 行精确 BF16 探针。W2 输出一次供缓存编码和门控残差更新共读。运行时需要检查点里的 `fused_ops.py` 和 `exact_port.py` 等冻结模块，以及支持 `torch.cuda._compile_kernel` 的 PyTorch CUDA 环境；学校验证使用 A100 80 GB、PyTorch 2.10.0+cu128。

成功的 INT8 收据包含 `counters.variant = "wide8192_all_int8_joined"`、`joined_y0 = 80`、`joined_fallback = 0`。每片 80 次锚点和 80 次命中，缓存容量 15.40 GiB，单独使用该变体时实测峰值 32.77 GiB。BF16 缓存容量为 30.20 GiB，单独运行峰值为 47.57 GiB。[十片同卡性能与质量记录](../../reports/20261123/wide8192-cache/README.md)
