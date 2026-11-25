# C/A 独立确认的冻结与复算

- [冻结配方](C_CONFIRM_METHOD.json)保留原 C 的注意力执行顺序：原生、OUT-ABS、原生、OUT-D。A 的四步 self-attention 均为原生。两者使用同一 BF16 wide8192 FFN、四步权重与采样配置。
- [来源清单归档](C_CONFIRM_SOURCES.archive.json)保留源、簇、输入图与提示的指纹、历史排除清单和冻结身份。实际生成使用的清单还含服务器文件位置与提示文本，已归档的源表中省去这几个字段。正式索引为 1,118 个来源，具有裁切图的簇为 355 个，历史与已观察排除后剩 299 个簇。三类等额选择各两个。服务器在生成前校验六张输入图的实际字节和六条提示指纹。
- [可用历史记录核对](C_CONFIRM_OBSERVED.json)来自已有生成收据、归档输出与 CAL/DEV 登记，对应 56 簇排除清单。选择仍按原冻结脚本，仅使用候选来源元数据和排除清单。独立性按记录内的来源簇界定。
- [配对运行脚本](attention_c_confirm.py)先在归档 0392 分别 warm A/C，并验证各自四步状态与原收据逐位一致。随后在同卡按清单索引交替先后顺序，一簇一对。按簇重采样区间描述这六个独立来源的变异。
- [配对报告](C_CONFIRM_REPORT.md) · [完整执行收据](C_CONFIRM_RESULT_196504.json) · [保存阶段的逐片时间戳](C_CONFIRM_SAVE_TIMESTAMPS_196504.json) · [度量复算结果](C_CONFIRM_SUMMARY.json)。
- 首步候选另用 [首步执行脚本](attention_phase0_cal.py) 与 [带可选首步特征基的适配器](attention_online_structured_phase0.py)。运行该首步脚本时，将该适配器复制为作业目录内的 attention_online_structured.py；冻结 C 仍使用 [原适配器](attention_online_structured.py)。
