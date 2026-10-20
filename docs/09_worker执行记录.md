# worker 续接记录（2026-10-20）

本页记录接手后的实际状态；前八篇文档和归档仍表示原学校服务器的历史快照。
用户授权整晚推进、整理仓库，并分阶段直接提交**本地 main**。本轮夜间截止暂定北京时间
2026-10-21 08:00；无需推送远端。

## 当前实测与资产

- worker：**2 × NVIDIA A100 80GB PCIe**，驱动 550.54.14，系统 CUDA toolkit 12.4，
  CPU 内存约 503 GiB。不是 H20。首次检查仅有 Xorg、没有计算任务。
- 本地四张 A100 已有其他计算任务，本项目的 GPU 运算只在 worker 运行。
- 两端 `/home/xiang` 为共享 NFS。默认 SSH 跳板曾超时，
  `ssh -o ProxyCommand=none worker` 实测可用。
- 使用独立 `.venv`，Python 3.11.15，目标 torch 2.10.0+cu128；
  安装阶段的观测不能当作 CUDA 可用性验证。首次提前探测遇到尚未复制完成的
  `libtorch_global_deps.so`，原始记录保留在 `hardware_during_install.json`；安装完成后重测。
- 模型仅进入 HF 缓存，项目目录不复制权重。`tools/fetch_models.py` 固定完整 revision，
  每个大文件核验完整 SHA256，完成清单位于 `.runtime/models.json`。
- 蒸馏仓库：`lightx2v/Wan2.2-Distill-Models@52699089e1dac804cbedd2400d0d2aea969577de`。
  只下载合同规定的 BF16 high/low 文件；公开 LFS 哈希与交接合同完全一致。
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
python3 tools/run_on_worker.py --gpu 0 -- python tools/hardware_probe.py --output runs/hardware.json
python3 tools/run_on_worker.py --gpu 0 -- bash scripts/build_hif4.sh
python3 tools/run_on_worker.py --gpu 0 -- python tools/smoke_hif4.py --output runs/hif4_smoke.json
```

远程入口只允许 GPU 0/1，使用节点本地 `flock` 防止本项目重叠，并在启动前检查
已有计算进程。每次 fresh Python 都重建独立 PYTHONPATH，避免继承历史 sitecustomize。
`config/worker.env` 固定本次实测 sm80 架构和每进程 8 CPU threads；编译最多 4 jobs。

`.runtime/` 放恢复源码、参考归档、编译缓存和模型路径清单；`runs/` 放可续跑实验输出；
`reports/` 只保存必要的小型审计证据；`docs/` 解释结论、失败和下一步。
不把模型、下载缓存、视频或大张量提交到 Git。

## 下一步及尚未闭合项

1. 安装完成后重新做 torch/GPU 探测与 BF16/SDPA 最小运算验证。
2. 在 worker 重新编译两个 HiF4 扩展，要求实际 CUDA 路径和非默认 stream 数值检查通过，
   不允许 Python fallback 冒充编译成功。系统 CUDA 12.4 与 wheel cu128 的组合须实测。
3. 模型下载和哈希完成后，按原四步合同跑一个 D-BF16 dense I2V；记录实际网格、
   scheduler、源码、环境、输入和输出，再扩大采集。
4. 历史约 56 GB D-Q cache、R1 原始大张量及 canonical W2 没有随归档交付，
   本机也尚未发现。因此旧 AR2 不能直接续算；保持为缺资产状态，不能静默换成 BF16 或新量化缓存。
5. AR2 源码阅读发现另一个待修复问题：只为 task0 保存 static 支持，评分阶段却按测试
   scope 查找 static 支持，task6 会缺键；task0 评分也会使用自己的支持而非对侧 donor。
   新可恢复实现必须按冻结规格绑定到对侧 task0 seed，同时保存分片/恢复点。

目前仅恢复/文件核验通过；尚无新 dense 视频、D-Q 复现、完整 FFN 加速或正式20条质量结果。
