# 2026-10-20 上游校对补充包

**本补充包是首版交接归档的必需补充。** 它修正了首版以服务器本地 ZIP 为差异基线造成的 15 个文件遗漏。原 Release `handoff-20261020` 保留原内容与校验值；本包没有合并进旧 ZIP。

本次通过恢复后的学校服务器连接只读核对源码，并从 GitHub 获取固定提交进行逐文件 SHA256 比较。包含 92 个修改/新增文本文件、5 份直接相对公开提交的修改补丁、源码哈希清单、核验结果和恢复工具。不包含完整上游仓库、模型、数据集或编译产物。

| 服务器工作树 | 公开基线 | 不变 | 修改 | 新增 |
|---|---|---:|---:|---:|
| LightX2V | `ModelTC/LightX2V@ed146db64d0b6130c669285fc1aa0b280a6a4823` | 1662 | 1 | 0 |
| Wan2.2-I2V-A14B-W4A4 | `CGCL-codes/Wan2.2-I2V-A14B-W4A4@a49a935677c7f760ef368cd334801416fff5aaa3` | 109 | 2 | 37 |
| HiF4_Sp | `alvinshao0313/HiF4_Sp@9aa2d3be6de265910731aa7c0d053e9c773cf556` | 79 | 2 | 50 |
| HiF4_Sp_wanenv | 同一 HiF4 提交 | 81 | 0 | 0 |

数字限于本次文本源码范围，详见 `AUDIT_SUMMARY.json` 的 `scope`；排除第三方源码目录、权重/数据、编译产物、隐藏目录、egg-info 等，单文件上限 8 MiB。范围内无缺失上游路径、无读取错误、无超限文件。差异身份表示相对该公开基线的本地差异，不证明作者身份。

LightX2V 没有 `.git`、ZIP 注释为空；但 ZIP 范围内全部 1663 个文本文件匹配上述公开提交，因此它是已验证的范围内恢复基线，不能据此证明最初下载的完整 Git checkout 身份。HiF4 的 ZIP 注释明确记录上述提交，81 个范围内文件均匹配；两次 ZIP 的字节哈希不同，不把 ZIP 字节相等作为结论。W4A4 的现场 Git HEAD 和 origin 已匹配公开来源，但 HEAD 不能代表带改动的工作树。

## 恢复源码

在新的工作目录取得上述三个公开仓库并 checkout 固定提交，目录名保持 `LightX2V`、`Wan2.2-I2V-A14B-W4A4`、`HiF4_Sp`。不要将当前 main 当作基线。

本包 `source_deltas/<目录名>/files/` 存放完整差异文件，`patches/` 供审阅或替代性应用。**选择覆盖完整差异文件或应用补丁之一，不要重复叠加修改。** 若只应用补丁，仍需复制全部新增文件。

在补充包根目录执行（`/path/to/code` 是包含三个仓库目录的父目录）：

```bash
python restore_delta.py --repos-root /path/to/code
python restore_delta.py --repos-root /path/to/code --apply
python restore_delta.py --repos-root /path/to/code --verify-only
```

第一条只生成只读计划；第二条先检查全部选定文件的原版/交接哈希，再覆盖差异；第三条验证恢复后的源码。默认处理三个主要工作树，附加 `--repo HiF4_Sp_wanenv` 可单独验证备用树。工具不克隆、不 checkout、不安装环境、不编译、不运行模型，也不检查范围外的额外文件。保留其他本地改动时若发生哈希冲突，应先人工处理；工具不会跳过冲突继续覆盖。

文件哈希按原始字节检查。Windows 的 Git 若启用 `core.autocrlf=true`，检出或应用补丁会改写行尾；请在新的恢复工作副本使用 `core.autocrlf=false` 保留 LF。恢复工具按字节复制差异文件。

通过文件恢复后，继续按首版交接包的环境、资源路径、hook 和运行配置恢复；只有这些差异文件不能恢复模型权重、量化大缓存或全部实验套件。

## A100 → H20 的约束

本次没有接触同事 H20 或运行 GPU 实验。恢复源码只证明文件相符。保持本地四步蒸馏 contract 的 `torch_sdpa`、480×832、block offload、T5 offload 和 HiF4 QDQ 后的 BF16 投影；不要照抄上游示例的 FlashAttention-3、720×1280 和关闭 offload。H20 上按实测 compute capability 和配套工具链重编扩展，保留本地 CUDA current-stream 补丁，并补做非默认 stream 的数值与同步检查。详细见 FFNDS 的 `docs/07_A100到H20迁移必读.md` 和新增 `docs/08_上游源码校对_20261020.md`。

## 验证边界

`VERIFICATION.json` 记录 CPU 文件层的原版→交接源码重建、哈希核验、冲突拒绝和重复应用检查。5 份补丁也经 `git apply --check`、实际应用、结果哈希核对。92 个导出差异文件的哈希均通过；88 个 Python 差异文件经 AST 语法解析。没有据此声称依赖安装成功、模型推理成功或 H20 兼容性通过。

校对明细与 15 个补回文件列表见 `AUDIT_SUMMARY.json`；逐文件上下游哈希见 `SOURCE_MANIFEST.json`。
