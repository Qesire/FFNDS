# wide8192 运行身份迁移（2026-11-24）

本文件是从 2026-11-23 注意力实验规划升级得到的执行绑定。它更新 FFN 背景，保留注意力机制实验的共同样本、坐标、特征、求解与数据独立性规则。

- 当前核对的 `main`：`13ffb127e71a5d4d1519ecb0df9c7b4d03efe601`。
- `wide8192`：固定 P64、FFN 选中 2560/13824 隐通道、16 共享模态、HHLL、锚点步 0/2、命中步 1/3、命中 tile 最大 8192 token 行；u0、y0、z0k BF16。
- `wide8192_all_int8_joined`：相同 2560 通道与 P64、BF16 W1/W2，三个锚点数据 INT8，每行每128通道 BF16 scale，64 精确探针 BF16，合并 W2 输出编码与 gate/residual。
- 运行入口：`tools/ffnds_fourstep/generate.py` / `tools/ffnds_fourstep/fourstep_backend.py`，学校冻结检查点 `fourstep-wide8192-185220`；校验收据与 manifest。
- 现有真实时间：相对原生 Dense，三源、六对 FFNDS BF16 wide8192 端到端均值 41.532→39.440s（5.04%）；与 BF16 的十片 INT8 比较端到端 37.900→37.935s（INT8略慢），FFN+post 4.83165→4.81399s（INT8较快）。同环境峰值 47.57 vs 32.77 GiB（分别独立测量）。
- 注意力主实验仍为 Dense vs BF16 wide8192 两条 FFN 输入轨迹；候选形成后条件性比较 INT8，自成 cache dtype 匹配对，不改变 192 个机制面板的主体设计。
- 相同注意力额外静态缓存 15.241 GiB 的容量相加：BF16 62.811 GiB；INT8 48.011 GiB；真实联合峰值在候选执行中测量。

## 核对入口

1. 当前依据：`research/CURRENT_HANDOFF.json`、`tools/ffnds_fourstep/README.md`、`reports/20261123/wide8192-cache/README.md`。
2. 本包身份 `wide8192_binding.json`，同时保留 `repository_binding_audit.json` 为历史 `fast512` 来源审计。
3. 同源条件的 Dense 基线明确 `method=None`；BF16 `wide8192` 与 INT8 joined 要有不同的 `variant` receipt，视频亦分组保存。
4. `python compute_asset_budget.py && python validate_plan.py` 检查新预算、采样几何与新身份。若更新基础来源矩阵，沿独立证据链保留 `bases/*.npy` 原字节与 SHA。