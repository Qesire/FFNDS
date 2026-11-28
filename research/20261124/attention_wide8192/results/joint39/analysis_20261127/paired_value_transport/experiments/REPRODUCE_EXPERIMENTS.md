# 学校实验复算

`EXECUTION.json` 记录学校作业号、方案提交、上传清单指纹和学校结果 SHA-256。`PFRR_PANEL_204479.json` 保留 12 个旧面板的各臂指标、张量指纹与旧指标复算偏差。`FOURARM_204487.json` 保留四臂逐源逐层的 Gram、样本位置及真实轨迹指纹。公开收据的字段范围与哈希记录见 `EXECUTION.json`。

在本目录用 Python 3.11 和 NumPy 复算表格与验收判据。

```bash
python report_attention_paired_value.py PFRR_PANEL_204479.json local_summary.json --fourarm FOURARM_204487.json
```

所有 Gram 的顺序为 A/B/C/D。`value_held_gram` 沿 B/C/D 排列，背景 Gram 的末项为原生 self-attention＋wide8192 FFN 块与同输入 Dense FFN 块之差。`held_base_gram` 的 A 项来自恢复前的 S，其他三项为各自加 P64 校正前的运输底座。由 $G_{ii}+G_{bb}+2G_{ib}$ 得到每臂相对同输入 Dense 出口的平方误差。

学校原始运行依赖封存的模型、源图、初始调度、panel/fit 数据和原运行 release。`attention_sparsepr_panel_control.py` 使用旧 panel 作业 `190739` 和 fit 作业 `192996`，`attention_paired_value_replay.py` 使用同一 fit、原 capture 的 wide8192 轨迹与冻结 C 的稀疏支持代码。运行 release 提供 `generate.py`、`exact_port.py` 和 Wan 参考资产。四臂代码就地复用 [attention 运行](../../../attention_online_structured.py)、[FFN 重放](../../../attention_joint39.py)、[既有 transport](../../../method_transport/attention_defect_transport.py)、[后继模块复算](../../../analysis_20261126/joint_action/run_attention_defect_transport.py)和[中心化状态公式](../../../analysis_20261126/joint_action_review/mass_centered_runtime/attention_mass_centered_low.py)，并随本次实验归档支持多臂 Gram 的 `attention_joint_action.py`。旧面板指标由[面板拟合工具](../../../../attention_panel_fit.py)复算。

新增 value 取样、打乱排列及配对岭回归均在运行前固定。局部一致性检查要求留出 value、恢复后 attention 和同输入 Dense 块出口三项上，C/B 与 C/D 在每个来源的四层几何平均比均小于 1。该检查作为机制诊断保留。固定轨迹输出只作计算图一致性核验。

完整候选轨迹的作业与视频指纹见 [E2E_EXECUTION.json](E2E_EXECUTION.json) 和 [端到端报告](E2E_REPORT.md)。将原始 MP4 保持为 `<source>_<arm>.mp4`，在 Python 3.10+ 使用 OpenCV 与 NumPy，将含 `review_attention_c_schedule.py` 的仓库目录加入 `PYTHONPATH` 后，执行 `python report_attention_paired_value_end_to_end.py E2E_RECEIPT_204740.json <视频目录> local_e2e_scores.json`。
