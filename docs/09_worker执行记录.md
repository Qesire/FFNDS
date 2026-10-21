# worker 续接记录（2026-10-20）

本页记录接手后的实际状态；前八篇文档和归档仍表示原学校服务器的历史快照。
用户最新目标为持续推进到仓库整体验收，不再以早晨时间为截止。
有实质性结果再整理并提交 **本地 main**；按最新目标暂不推送远端。
此前用户曾明确授权直接推送 GitHub main，首两个恢复提交已于 2026-10-20 推送；
后续提交节奏和远端操作以最新目标为准。

## 当前实测与资产

- worker：**2 × NVIDIA A100 80GB PCIe**，驱动 550.54.14，系统 CUDA toolkit 12.4，
  CPU 内存约 503 GiB。不是 H20。首次检查仅有 Xorg、没有计算任务。
- 本地四张 A100 已有其他计算任务，本项目的 GPU 运算只在 worker 运行。
- 两端 `/home/xiang` 为共享 NFS。默认 SSH 跳板曾超时，
  现在通过仓库外的私有SSH配置和`ffnds-worker`别名连接；新入口已实际核验，见[私有执行配置](43_公开元数据与私有执行配置.md)。
- 使用独立 `.venv`，Python 3.11.15，目标 torch 2.10.0+cu128；
  安装已完成，CUDA 可用性与 BF16 GEMM/SDPA 检查已通过。首次提前探测遇到尚未复制完成的
  `libtorch_global_deps.so`，原始记录保留在 `hardware_during_install.json`；重测成功证据为
  `hardware_torch.json` 与 `torch_smoke.json`。
- 模型仅进入 HF 缓存，项目目录不复制权重。`tools/fetch_models.py` 固定完整 revision，
  每个大文件核验完整 SHA256，完成清单位于 `.runtime/models.json`。
- 蒸馏仓库：`lightx2v/Wan2.2-Distill-Models@52699089e1dac804cbedd2400d0d2aea969577de`。
  只下载合同规定的 BF16 high/low 文件；公开 LFS 哈希与交接合同完全一致。
- 两份蒸馏 checkpoint 与 T5、VAE、tokenizer 等全部 **12 个文件已下载且逐文件完整哈希通过**。
  `.runtime/models.json` 为 PASS；HF 缓存内 hardlink view 也已实际建立并通过。
- T5/VAE/tokenizer/config 来源：
  `Wan-AI/Wan2.2-I2V-A14B@206a9ee1b7bfaaf8f7e4d81335650533490646a3`。
  不下载原版 O 的 diffusion 分片。

## 已完成的恢复检查

- 完整 Release ZIP SHA256：
  `b8f78df8018b135bfea627542a3e310aaf81cae0192d875c0ce24ed7b3923d0e`。
- 原包 `verify_package.py`：外层 **4,814** 文件、服务器 **45,372** 文件全部通过。
- 三个上游 ZIP 与公开校对中的固定哈希一致；恢复至 `.runtime/code/`。
  只应用新版 92 个差异文件，恢复后 **1,942** 个受审计源码文件通过，未叠加旧补丁。
- 归档中的当前 v4.9.2 kit、formal/prox 套件及 `work/v5` 已提取至
  `.runtime/archive/server/`，作为只读参考，不直接 source 学校路径配置。
- 机器证据目录：`reports/worker_20261020/`；安装/下载过程日志位于
  `runs/bootstrap_20261020/`，不纳入 Git。

## 运行入口与目录约定

```bash
bash scripts/setup_worker_env.sh
.venv/bin/python tools/restore_sources.py
.venv/bin/python tools/fetch_models.py
.venv/bin/python tools/prepare_model_view.py
.venv/bin/python tools/fetch_research_inputs.py
python3 tools/run_on_worker.py --gpu 0 -- python tools/hardware_probe.py --output runs/hardware.json
python3 tools/run_on_worker.py --gpu 0 -- bash scripts/build_hif4.sh
python3 tools/run_on_worker.py --gpu 0 -- python tools/smoke_hif4.py --output runs/hif4_smoke.json
python3 tools/run_on_worker.py --gpu 0 -- python tools/run_dense_bf16.py --out runs/dense_example
```

远程入口只允许 GPU 0/1，使用节点本地 `flock` 防止本项目重叠，并在启动前检查
已有计算进程。每次 fresh Python 都重建独立 PYTHONPATH，避免继承历史 sitecustomize。
`config/worker.env` 固定本次实测 sm80 架构和每进程 8 CPU threads；编译最多 4 jobs。

`.runtime/` 放恢复源码、参考归档、编译缓存和模型路径清单；`runs/` 放可续跑实验输出；
`reports/` 只保存必要的小型审计证据；`docs/` 解释结论、失败和下一步。
不把模型、下载缓存、视频或大张量提交到 Git。

## 新数值与输入核验

- torch 2.10.0+cu128 在驱动 550.54.14 上实际执行 BF16 GEMM 与 SDPA 通过。
  GEMM 的可精确表示测试输入与 FP64 参考转换结果完全一致；SDPA 对 FP64 参考最大绝对误差约 0.00233。
- 使用系统 nvcc 12.4、sm80 重新编译 `hifxg_quant` 与 `lowbit_quant` 成功。
  构建有 CUDA minor-version mismatch 提示，但没有把提示当成兼容性结论：
  实际原生 HiF4 CUDA 路径和非默认 stream 的四组检查均已通过
  （FP32/BF16 × hifx4/hifx4_1），本轮 native/force_fp32 输出最大差为 0。
  扩展哈希与检查范围见 `hif4_smoke.json`；不代表完整模型或其他量化格式已验证。
- 当前最小依赖 `uv pip check` 通过。旧 decord 0.6.0 wheel 内嵌 CPython 3.6 标签，
  I2V 路径不依赖它，因此移除；历史快照与首次检查失败记录保留。
  按真实导入缺失补入同历史版本的 pyzmq 27.1.0，未安装整套上游 requirements。
- 三个原 V5 来源已从 `BestWishYsh/OpenS2V-Eval@b1e792b1853e5cbffdb7e7cc0d77a2bea7008e9a`
  恢复 3 张图和 6 个种子记录，提示词、首图路径、图像字节数与交接清单一致。
  原始图像哈希缺失，因此不能声称已证明旧/新图像逐字节相同；新下载哈希和此限制均写入 manifest。
- `prepare_model_view.py` 在 HF 缓存内部建立 hardlink 布局，不复制权重字节。
  这是为了满足旧 adapter 在 resolve 后仍要求 high/low expert 目录名的路径合同，
  不修改数值 adapter。
- 使用 high expert 的真实 checkpoint，对 blocks 0/20 做原生 MMWeight 加载、
  FFN0→GELU(tanh)→FFN2、bias 一次及 gate/residual 检查通过。输入为合成 32 行；
  这是加载/layout/算子检查，不是实际轨迹、质量或速度结果。
- 原始 12 条 Smooth 校准清单已从归档恢复，源文件 SHA256 与旧 canonical report 一致；
  12 张公共数据集图像的完整哈希全部匹配旧校准报告，prompt 与固定数据集 revision 一致。
  原校准集包含 task10 开发图，因而 **task10 不作为独立验证来源**；task0/task6 不重叠。
  见 `tools/fetch_calibration_inputs.py` 和 `.runtime/inputs/calibration12/recovery_report.json`。

## 下一步及尚未闭合项

1. task0/seed20261001 的原生 D-BF16 视频基线已完成：81 帧、16 fps、实际 512×768，
   四步 latent 均有限，FP32 形状为 `[16,21,96,64]`，sigma 为
   `[1,0.9375,0.8333333134651184,0.625]`。视频 SHA256 为
   `05919d785ce0078a4b756b5b92c81cf8e23506cc4cb65cd2e563f680025cc87e`。
   源码 hash 与提交 `95c0815` 的原 BF16 runner 一致；证据见
   `reports/worker_20261020/dbf16_task000_seed20261001_plain.json`。
   生成记录约 71.97 秒，包含观测与保存，不作为速度 benchmark。
   同输入的 FFN 边界采集也已通过：blocks 0/10/20/38 × 四步共 16 scopes，
   每个 scope 按实际 `[21,48,32]` 网格均匀无放回抽取 512 行，显式记录 π 与 1/π。
   两次运行四步 latent **逐位一致、最大误差为 0**，视频 SHA256 也一致。
   见 `dbf16_observer_noop.json` 和 `dbf16_capture_summary.json`。
   此结果限定为该输入、种子、原生 BF16 与记录的观测源码，不替代 D-Q 或其他干预的核验。
2. 原 12 输入、种子 42..53 的 Smooth absmax 已全部完成，800 模块覆盖检查通过。
   每条输入保存独立统计、哈希、实际轨迹元数据与恢复点；合并 act SHA256 为
   `5631d50a1c2575d9ef9e80b65b844c9b56e971f779a11b83bfc0059016debfd3`。
3. alpha=0.5、eps=1e-8、BF16 smooth 权重、原生 HiF4 RTN 的完整 W4A4 cache 已重建。
   80 个分片、800 模块与合并文件均通过核验；完整 SHA256 为
   `bfb400f94ce42760070bab5916721fd942b0864907bb4e4e5207c008dc4c3df3`。
   文件在 worker 本机 HF 缓存 `/var/tmp/ffnds-xiang/huggingface/ffnds-derived/` 内，
   共享路径清单为 `.runtime/hif4_rebuilt_cache.json`，小型证据为 `hif4_rebuilt_cache.json`。
   task0/task6 的首种子 D-Q 视频、16-scope 采集及四步逐位 no-op 均已通过；
   每条视频 1,600 次目标线性、1,600 次在线 QDQ、cache miss=0。
   task0 为 512×768，task6 为 624×624，均为 81 帧/16 fps。
   两来源×两种子的四输入矩阵现已完整完成：64 个采集 scope，四个输入各自四步 latent 文件
   SHA256 均与独立 plain 运行相同。AR2 子集为新 8 scopes/4,096 行。
   小型汇总见 `dq_capture_matrix_summary.json`，完整 tensor 路径与 hash 可由汇总定位。
4. 历史约 56 GB D-Q cache、R1 原始大张量及 canonical W2 没有随归档交付，
   用户已明确这些在另一台服务器，授权按现有 HiF4 重新开始。因此不再等待旧资产，
   另立 `WORKER_HIF4_SMOOTH_RTN_REBUILD_20261020` 实验；不能冒充原 AR2 的逐字节续算。
5. AR2 源码阅读发现另一个待修复问题：只为 task0 保存 static 支持，评分阶段却按测试
   scope 查找 static 支持，task6 会缺键；task0 评分也会使用自己的支持而非对侧 donor。
   新可恢复实现必须按冻结规格绑定到对侧 task0 seed，同时保存分片/恢复点。

目前校准、完整新 HiF4 cache、D-BF16 与 D-Q 视频/no-op 已有真实结果；
新 AR2 已完成，C 局部门槛两来源各 4/4 通过；PR1 前 FFN0 核代理未通过。
详见 [重开结果](12_AR2重开结果与下一步.md)。完整 FFN 加速或正式20条质量仍未通过。

## 新入口的验证范围

`run_dense_bf16.py` 与 `run_dense_hif4.py` 共用 `run_distill.py` 的固定模型/轨迹审计。
BF16、HiF4 runner、`hif4_observer.py` 和完整缓存构建已完成上述实际核验。
当前 20 项数值/恢复检查包含：bias-free gain、对侧 donor、FP32 对照、NA 参数恢复，
以及正零/负零不能被误认成逐位一致。此前成功 no-op 的四步文件 SHA256 本来也均完全相同。
新 AR2 runner 和完整矩阵已实际执行；其通过范围限定为局部投影。

## 本机 HF 缓存与 I/O

初次分片写共享 NFS 吞吐较低，主动中断自己的量化进程；退出日志保留
`rebuild_hif4_quantize.log` 的 KeyboardInterrupt，属于 I/O 调整，不是量化数值失败。
只迁移并逐文件验证了 8 个完成分片，未采用中断的临时文件；之后 high/low 分别在两张卡完成。
迁移证据见 `hif4_cache_local_staging.json`。

原始 12 个模型文件也以完整 SHA256 校验后镜像到本机 HF hub cache，snapshot 与 view 为本机
hardlink，不更改逻辑模型身份。见 `node_local_model_view.json`。
runner 仅在 host、逻辑 manifest SHA、12 个 repo/revision/filename/size/SHA 身份均匹配时采用该 I/O view，
并记录实际 view/hash。当前初始化记录从 NFS 的约 903 秒降至本机 view 的约 103 秒；不是 FFN benchmark。
这些 `/var/tmp` 文件仅在 worker 存在；本机清理或节点更换后须重新 staging/构建。
