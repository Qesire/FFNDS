# BH–A0 matched head：实际结果独立复核 · 2026-10-28

**SUPPORT，限于本次 CAL24 同状态、同 256 行的 terminal-head 误差结论。** 本工具作者独立于 Terra 汇总，从原始 RUN、实际退出 receipt、操作日志及历史小 JSON 重新核算；未发现数值、计数或输出身份错误。本复核仅使用 Python 标准库，未加载任何 tensor payload、Torch、NumPy、模型或 GPU；不重复原生实验。

## 精确版本

| 对象 | SHA256 |
| --- | --- |
| 冻结协议 `config/e0_bh_a0_matched_head_20261028_v1.json` | `766e83105775bfccc2862a92f7b24b55b28e4a2ebcf93c92c2070ef739994714` |
| 实际 `attempt1/results/RUN.json` | `31baab02b506aa965da6f07e91d72e47bd749f5425ae4faae92bb62f57c27716` |
| 实际 `attempt1/EXIT_RECEIPT.json` | `b4deb31cee2ca0568f1a510831af734a734281886e869663998e6b55e587c619` |
| 实际 `OPERATIONS.jsonl` | `c5791a61f1556933f4de19601a83acdbd967aa618921394353b2f0c5afef85ff` |
| Terra 结果 JSON（参数措辞澄清后） | `9e5aa3e774d617e49b1f2f23f37f3aa02acea468878d75ab43f41f58bf404369` |
| Terra 结果 Markdown（参数措辞澄清后） | `7838db5e6fe5df417c4c81bfec8b8bbbfdb489c9e6bbbbeab6e75680de65d7a6` |

HF 根为 `ffnds-derived/e0_bh_a0_matched_terminal_head_20261028/attempt1`；Terra 报告为 `research/20261028/BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.{json,md}`。以上 SHA 均在本次复核实际重算。未改冻结代码、配置、ACTIVE 或已发布报告。

## 原生身份与操作闭合

- receipt 的实际 child returncode 为 0，`process_exited=true`、`timeout=false`，精确绑定上述协议及 RUN；body 352.016 s，outer 364.986 s。
- 日志全部 28,224 个连续 serial 经逐条复算，每个计数满足 `predebited ≥ invoked ≥ completed`，最终三套计数均等于冻结 limits。96 次 neutral 全部完成于 serial 8694，首次 A0 W1 为 serial 9022，顺序正确。
- 96 个 neutral 的 RUN 记录均 COMPLETE/bitwise_equal；其 expected/actual head SHA 均与原 label manifest 的 `head_rows` SHA 相同。此身份是 **full-M geometry selected-row replay**：其他行固定零，不是完整原状态/完整 head 重建。
- 192 个 A0 实际 M256 输出 SHA 均与原四个终层 RESULT 的 `final_folded_BF16` 逐 source/step 输出 SHA 相同。匹配的是 **A0 自己原来的最终诊断预测**，不是 teacher Y；随后才用原 BF16 gate、residual 和 full-M LN/head 比较共同 target。原 A0 producer 未记录 BF16 reduction flag，本次确切输出 SHA 门已成功满足。
- 96 个 BH saved-head SHA 与历史 FINAL_METRICS 逐 case 一致，两个 prediction pack 的新协议绑定与原 BH RUN 绑定完全一致；没有新 BH forward。历史 FINAL_METRICS 本身未被原 RUN 绑定，Terra 对此披露准确。本复核检查的是 SHA/provenance 元数据链，未独立重读 tensor payload。
- 实际计数为 96 neutral、192 own-Q2 A0 FFN、288 full-M LN/head、288 paired metrics、576 FP64 metric reductions、388 layout restores、1272 H2D tensor copies（1,264,047,360 payload bytes）。Q1、BH forward、teacher-Y numeric read、full-u read、训练、分解、视频均为 0。

## source 等权结论

已独立确认完整 24 source × 4 step 网格、`seed=42+source_index`、原 source/role/expert/rows 绑定；每个 case 三 arm 的 target energy 相同，SSE/energy 和每行 SSE 自洽。先平均 source 内该 expert 的两个 step，再对 source 等权，重算结果与 RUN 和 Terra 全部逐 source 值完全一致：

| Expert / exposed held8 | BH saved | A0 energy94 | A0 random94 |
| --- | ---: | ---: | ---: |
| High | 0.003957802502 | 0.000608788822 | 0.000843396911 |
| Low | 0.003614684166 | 0.000660966153 | 0.000751635295 |

两个 expert 中，BH 均在 0/8 held sources 上优于任一 A0 arm；train16 汇总也全部复算匹配。报告“本匹配终层比较 A0 更准确”成立，不能解释为所有视频、80 函数或未暴露新 source 的结论。

## 解释口径

报告的不同目标/参数量限制正确；本复核期间协调者已把措辞澄清为 **可训练参数**：BH 672,386 对 A0 LoRA 712,704，少 5.657%。上表对应澄清后的报告 SHA，数值与计数重新核对保持相同。A0 还保留原选定的 6016 宽冻结/合并权重，BH 另有固定 head 几何、normalizer 和付费 residual 摘要；该百分比不表示部署总参数或资产量仅差 5.66%。BH 优化 train-only 标准化的 Δq MSE 与 log-slack MSE，A0 优化 gated-Y 误差；相同 split/256 updates 不能使本比较成为单因素消融。

本次成立的是同一 captured native BF16 target 下的函数误差比较：A0 使用真实 M256 FFN 后的 full-M geometry selected rows，BH 使用已有 BF16 head 输出。只覆盖两项 terminal functions；无速度、完整自由轨迹或视频质量证据。Terra 已保留这些边界，无需修改原始 RUN。
