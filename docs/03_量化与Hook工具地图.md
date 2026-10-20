# 量化、运行注入与 hook 工具地图

> 本节下面关于ZIP基线与未知版本的记录保留首版语境；2026-10-20已进一步核实公开来源、HiF4 ZIP提交与LightX2V范围内匹配基线，并补回15个W4A4文件。恢复以[上游校对报告](08_上游源码校对_20261020.md)及新版[差异目录](../upstream_audit/README.md)为准，不叠加旧ZIP补丁。

以下路径均相对原服务器 `$ROOT`；本包内层 tar 的 `server/` 下保持相同结构。历史版本很多，按这里的角色定位，不按目录名中的“latest”“SUCCESS”推断权威性。

## 当前优先入口

| 角色 | 路径 | 交接用途 |
|---|---|---|
| 蒸馏 HiF4 量化/部署套件 | `artifacts/wan_hifx4_lightx2v_distill_kit_v4_9_2` | adapters、contract、PTQ pipeline、环境、Slurm、生成工具；当前 D-Q 的基础绑定 |
| PTQ 当前工作树修改 | `code/HiF4_Sp/HiFloat4/ptq` | adapters、algorithms、calibration、formats、runtime、pipeline；仅修改/新增源码与 ZIP 差异随包，完整上游仓库外置 |
| 原版 W4A4 / NVFP4 改动 | `code/Wan2.2-I2V-A14B-W4A4`、`tools/Wan2.2_NVFP4_native_RTN_workflow` | 原版量化、RTN、native backend、历史生成/评测；不要混接 D 的 cache |
| Prox 的权威配置链 | `artifacts/wan_hif4_prox_dit_v1/config/cluster.local.env` | interpreter、base kit、checkpoint、cache、source pool、VBench 实际路径 |
| formal 运行与稀疏仿真 | `artifacts/hif4_scale_sign_proxy_formal_v0/runtime_patch`、`tools`、`scripts` | family runtime、runner、sitecustomize、生成函数；旧 A5 路径从这里进入 |
| 早期激活统计 hook | `analysis_hooks/sitecustomize.py`、`analysis_hooks/wan_activation_hook.py` | 捕获激活统计的原始工具；与新 V5 observer 不是同一代 |
| 早期补丁注入 | `patches/sitecustomize.py` | 部署故障排查/导入补丁历史，不应默认叠加到新采集 |
| 新 D-Q 采集 | `work/v5/dq_p0_runner.py`、`dq_asset_capture_runner.py` | P/C、R、U/V1/H/V2/Y、gate、Delta、QDQ 与有效操作数 |
| 新 BF16 采集/边界审计 | `work/v5/dbf16_boundary_audit_runner.py`、`dbf16_asset_capture_runner.py` | 原生 BF16 callable、写回边界、张量与 dtype；不伪造 QDQ 字段 |
| A/B/C 分析 | `work/v5/analysis_r1` | branch_pair、signed_compute、merge、跨步比较；最终用 clean 汇总 |
| AR1 机制诊断 | `work/v5/discovery_ar1` | `compute_group_probe.py`、`postq2_native_precision_replay.py` 和各 sbatch |
| AR2 支持选择 | `work/v5/discovery_ar2` | `support_probe.py`、`run_support_probe.sbatch`；最后一轮超时未闭合 |

## 量化代码应如何使用

v4.9.2 保留三类东西：配置/资源链、离线校准与量化、推理数值后端。恢复时先确认高低 expert 与 base VAE/T5/tokenizer 的路径，再确认 smooth 与 Hessian 来源，以及实际 cache 变体。`hif4_smooth_rtn` 是当前绑定的变体，目录里还有其他 RTN、GPTQ、magnitude/Hessian 方案，不能任意换 cache 后仍称同一个 baseline。

本地归档的 `native_qdq_wrapper.py` 是服务器 `code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py` 的重命名副本，`native_qdq_backend.py` 对应 `code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py`。10月11日绑定副本位于 `evidence/context_20261011/reference/runtime/`，原路径与哈希见 `metadata/runtime_index.json`。局部回放读取的是有效权重、smooth、bias、gate与实际V2，不能只加载checkpoint tensor模拟HiF4行为。

历史 A5 的 `s1=0.7` 表示 proxy 输入约留30%；`s2=0.436` 表示 hidden 保留7797/13824≈56.4%。其语义 `MASK_NATIVE_FULL_HIDDEN_BEFORE_NATIVE_FFN2` 是先完整计算 W1/GELU 再 mask，因此旧质量结果本身不证明省去了 W1 的计算。

## Hook 的实际风险点

1. `sitecustomize` 在 fresh Python 启动时根据 `PYTHONPATH` 自动加载；在已启动的解释器里修改环境不能重现同样导入顺序。
2. formal 的 sitecustomize 与实验 observer 有组合顺序。不能直接给任意环境添加一个新 hook 路径并假定相同运行行为。
3. `observer/no-op` 的轨迹核验只证明该位置定义的观测不改变结果；不能把所有后续实验都认定为 no-op。
4. 原生 head/cast、gate、residual 写回与 sampling 坐标要保留。仅验证 FFN 内部 tensor 有限，不能替代最终轨迹一致性。
5. 优先读取实际 plan 与 binding；例如目录名包含 threshold 的 R3，实际 `static_hwg_threshold_mode=off`，执行 A5 top-k。

## 上游源码差异与恢复限制

`source_deltas/<repo>/PROVENANCE.json` 记录相对于本地保留 ZIP 的 changed/added/deleted；同目录 `.patch` 是文本修改。完整修改文件在 `server/code/<repo>/`。不变的上游源码不复制。

- Wan W4A4 当前 Git HEAD 已读取为 `a49a935677c7f760ef368cd334801416fff5aaa3`。
- LightX2V、HiF4 两树未发现可读取的 Git HEAD；不能凭 ZIP 文件名猜 commit。
- 当前源码与本地 ZIP 的差异，不能自动解释为“用户原创代码”；它也可能包含版本更新或旧补丁。
- 新机器要先取得匹配上游版本/原 ZIP，再应用差异；缺失上游版本时保留为恢复缺口，不声称一键重建成功。

官方项目入口：[Wan2.2](https://github.com/Wan-Video/Wan2.2)、[LightX2V](https://github.com/ModelTC/LightX2V)。本包记录的是本机实际工作树，不把最新 main 当成历史运行版本。

原项目中安装/提交/取消/清理脚本作为交接源码保留，**本次没有执行**。接手时应先阅读具体脚本；不要把历史脚本中的工作流指令当成新一轮实验必须执行的命令。
