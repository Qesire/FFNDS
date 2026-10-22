# R14：三方向融合后端已有实测，转向真实操作数与条件价值

> 后续已获单独授权的 R14N0 完成并关闭，见 [当前交接](HANDOFF_R14N0.md)。本文件保留 R14 合成后端阶段的交接状态。

**当前已完成：**一张 A100 80GB PCIe、固定随机张量完整链路，三方向融合动态版 **28.1745 ms / 1.913×**，同次稠密版 53.8966 ms。主投影 MAC **45.1381%**。同算法未融合动态版 46.9870 ms，说明不能从 MAC 直接推速度。16 项融合控制、14 项原 HiF4 QDQ 控制先通过，49 次定时样本全部保留；峰值分配 8.0797 GiB。零 Wan 模型、拟合、新视频或张量文件。

入口：[说明与公式](../../docs/73_三方向融合算子的实际成本与原生价值下一步.md)、[机制记录](R14_mechanism_record.json)、[冻结后端协议](R14_three_direction_backend_protocol.json)、[原始结果](R14_three_direction_backend_results.json)、[当前决定](../CURRENT_DECISION.json)。按[同前提竞争/Pareto](../../docs/72_与同前提方法竞争的质量速度评价准则_20261022.md)选择后续；旧 R13 直接实现和 D0 原始判定不改写。

## 数值和比较口径

- 固定/动态融合输出与各自未融合三方向参考逐位一致；范数规约相对差 1.93477e-7，3958 个选择 ID 完全相同。
- 动态完整链包括当前 post-Q1 对比范数、top-k/排序/映射。固定融合版 27.9487 ms 仅是高效首 K 放置对照；与动态相差约 0.226 ms，不能单独当作精确 selector 时间，更不是最强固定质量控制。
- 半 hidden＋R256 随机代理 32.2916 ms，不是冻结 UB1 的实际模型结果。共同 norm 是代用品；没有完整 160 FFN、真实 Wan 骨干/响应误差或端到端视频加速。
- 新三方向与旧直接随机输出相对 L2 差 0.00145082。融合自身一致不等于表示改写对 Wan 无误差。
- 四个 Triton 融合边界和执行程序的 SHA 在结果中。首次运行完成，未根据延迟扫布局、seed、q 或点数。不要修改这些执行过的源来保持原结果绑定。

## 准备中的有限原生值问题

[R14N0 草案](R14_native_value_DRAFT.json)尚未登记、尚未获单独执行授权。最多 4 次原冻结 UB1 四步观察运行（来源 0/1/10/11），只在 step2/block20 保存相同 64 个真实四 token 邻域；释放整模型进程后最多 32 次独立原生 FFN、104 次额外 linear。保留原全 M 和点平面 GEMM 形状、真实 Q2/偏置/残差写回，不能将零填充的局部检查叫全场景策略或视频结果。

只用来源 1/11 的真实 one/two-point 损失选精确最佳固定 32/64 放置，先冻结后在 0/10 比较同数量范数动态和标签 oracle。实际执行各策略并验证局部组合恒等式；主指标是实际 sensitive-dtype residual 写回相对同输入教师的 SSE，副指标是 gated-Y。若当前动态胜不过最佳固定或 UB1，就不能推广这个固定规则；若 oracle 也无益则回到算子假设，不扩大符号/来源/位置试到有利。

[分析程序](../../tools/analyze_stripe_native_values.py)和训练/评估隔离测试已准备。[CPU 就绪收据](R14_native_draft_CPU_readiness.json)重验四份旧输入/视频/操作数及仿射 SHA，并核对模型/缓存元数据；未重新哈希全部 93 份大资产，未导入 torch。该检查不是模型登记。真实执行代码和不可变源绑定仍须完成，且要满足根 AGENTS 的单独登记与直接用户授权要求，才能首次运行。

## 执行与复核

```bash
python3 tools/check_research_routes.py
python3 tools/check_r14_backend.py
python3 -m unittest discover -s tests -p 'test_r14*.py' -v
python3 -m unittest discover -s tests -p 'test_stripe_value_analysis.py' -v
```

复测合成程序需要既有环境、原 HiF4 和一张标准独占 GPU；不是默认下一项模型作业。固定源是 `tools/probe_three_direction_backend.py` 与 `tools/spatial_contrast_kernels.py`；程序拒绝覆盖已有结果。公开副本不含模型、缓存、原生操作数或私有 worker 连接信息。模型资格仍为空。

[本轮验证收据](continuation_checks_R14.json)：94 项相关离线测试通过；不含私有 `.runtime`/`runs` 的干净索引导出中，7 个证据 CLI、同 94 项测试和合成 smoke 通过。D0、R13 种子和 R14N0 草案的模型选择均按预期拒绝，公开元数据扫描无私有标识。

[D0 历史交接](HANDOFF_R12_D0.md)和 [R13 历史交接](HANDOFF_R13.md)保留：D0 的 24 次 raw 等预算动作已拒绝，不重跑/扩池/触发 D1；R4E1、R9、R11 各有限失败保持原范围。本轮正面后端证据不自动改变任何闭合模型记录。
