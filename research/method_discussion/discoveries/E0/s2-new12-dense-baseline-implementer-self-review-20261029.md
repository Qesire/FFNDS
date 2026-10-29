# New12 Dense baseline：实现者自审，2026-10-29

**身份：实现者自审，不是独立审阅、运行授权、实验完成或质量结论。** 新实现位于 `tools/e0_s2_new12_dense_baseline.py`，软件控制位于 `tests/test_e0_s2_new12_dense_baseline.py`；未冻结、未登记、未运行模型或 GPU。两份本地临时草案位于 `/tmp/ffnds_new12_independent_part0_draft.json` 和 `/tmp/ffnds_new12_independent_part1_draft.json`，仅供审阅；任何代码或绑定资产变化均须重新生成。

每 part 的顺序是先历史 Dense 重复 neutral（part0 ordinal 898，part1 ordinal 96），其十状态校验通过后才进入六张新开发图。新图按 bank 的索引交替分 part；输入均为原 prompt、seed42，完整四步 × 40 block、81 帧。历史 neutral 是重复对照而非新图簇。新图已有输入图像视觉审查和有界历史筛查，真实生成运动和视频质量尚未测量；CORE 和官方质量另行 scope。

数值路径私有导入既有 video20 V2：构造 `SimpleNamespace(mode='dense')` 并使用原 `native_phase`、`OriginalGuard`、`LightTap` 和 pipeline。源码中 `transfer_geometry` 的 Dense 路径只访问 runtime 的 `mode`；未安装 factor runtime。私有模块的计数、资源和 run ID 在新运行时绑定，不改旧文件。新注册计划单臂 Dense，neutral 的历史身份显式记录，调用已有 `collect` 进行生成资产验收。

每 part 限额：7 视频、28 flow、1120 FFN 位置、2240 FFN linear、1134 prefetch、49 stage event pair、70 状态写入及 D2H（425617920 payload bytes）、7 注册写入、10 次历史状态 byte 检查；额外 factor GEMM、训练、量化与官方质量为零。原模型 setup 的 expert / raw-MM / buffer-MM / T5 raw-MM / T5 buffer-MM 加载分别保持 2 / 812 / 40 / 168 / 7；其余原始两视频计数按 7/2 换算。GPU1、CPU4、GPU reserved 40 GiB、主机可用 128 GiB、HF 新增 4 GiB、wall 2400 / outer 2700 秒，双 part 会话最多两卡；另设 6 GiB 磁盘空余门槛。以上是代码合同与推算，**并非真实调用计数或耗时证据**。

实际软件验证：`.venv/bin/python -m unittest tests.test_e0_s2_new12_dense_baseline` 7 项通过，覆盖 part0 顺序与旧控制标识、prompt 变动和重复图 SHA 拒绝、一次性与逐视频限额、单 Dense 注册及历史标记、neutral 失败或不足十状态时的门、禁止 factor 模式；两 part draft 与 inspect 成功，py_compile 成功。标准库只读核验了 12 张新图 SHA 与其 18 图提取和 74 张历史 dHash 筛查的真实 RUN/EXIT；没有导入 Torch/NumPy，也没有打开冻结确认图。此核验早于当前 root 指示；之后未再做真实历史验证或 CPU preflight。

需要 root 独立审阅和之后实际 preflight 的风险：

- `old_control` 通过既有 quality verifier 验证较大历史视频谱系，不只读取单个旧 Dense 控制；其真实读取量与 600 秒 CPU preflight 限额未测。应核对 CPU scope、数据访问范围和实际 wall/IO 后再授权。
- `run` 在输出目录创建前完成注册、CPU proof、master、源文件等验证；这些早期失败不能写本 scope 的 RUN，必须由外层 worker EXIT/stderr 保留。目录创建后的运行失败由内部 RUN 保存；审阅是否满足完整失败接口。
- 仅静态检查复用的 Dense runtime、ledger、prefetch、transfer 和视频验收；没有真实执行过新 runner，单 Dense 单臂 `collect`、异步监测、4 GiB accounting、2400 秒预算及两卡调度的端到端行为均待独立测试/实际证据。
- 实际 source-provenance 校验包含文件 SHA 和 receipt；局部历史 dHash 筛查不证明全球场景独立，也不证明真实运动。十状态 byte 比较先于六个新来源，但尚无新运行的 neutral 结果。
- preflight/runner 的实际冻结代码闭包、worker EXIT 身份、注册 ACTIVE 条目和原生 raw-master proof 应由协调者核准后建立；本文件不提供批准或自动重试。
