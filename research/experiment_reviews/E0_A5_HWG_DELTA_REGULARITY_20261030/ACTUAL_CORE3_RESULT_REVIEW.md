# 三来源全 M 核心比较：独立实际结果复核

评议者 `/root/sync_abba_actual_review_terra`。本次为只读结果复核，不启动模型、GPU或新的 tensor 数值实验；结论限定于已完成的三来源 teacher-state 付费比较。

实际 CPU 汇总 `E0_A5_HWG_CORE3_CLOSING_COMPARISON_20261030_V2_LOCAL` 的最终 RUN、supervisor EXIT、协议 SHA、job-spec SHA 和 RUN SHA 已逐项核验：实际 child EXIT0、无 timeout、3来源、240站点、13,920行、240个mask/proxy文件，wall 923.889秒。其完成状态和账本记录独立读取与核验了240文件的完整及typed raw SHA；本评议没有再次读取这些raw payload。240文件声明总大小54,704,331,360字节；CPU汇总的显式读账本为54,894,817,578字节，包含元数据，与mask文件大小是不同口径。CPU输出本地JSON和CSV逐字节SHA与真实RUN输出绑定一致。

已再次核对 CAL1、三源 E0、最终汇总、来源96失败、1040首E0失败和两个资源helper的真实 RUN/EXIT 与 protocol/spec SHA 绑定。CAL1和三源完整E0均实际EXIT0；1040首E0因foreign compute EXIT1、仅10对，未混入后续完整80对。96采集monitor timeout实际EXIT1且零生成phase，未纳入平均。两个helper均为 `read exceeded predebit` 实际EXIT1，失败metadata读取留有inflight UNKNOWN；其未向进程返回tensor payload，不等于内核物理I/O为零。summary V1仅保留为未启动的传输失败，V2_LOCAL实际EXIT0。

独立从发布CSV核对13,920个唯一 source/expert/block/arm键、三个正确来源角色和每来源每臂80站点，并以source为独立单位重新计算全部1,507项可定义的来源均值；每一项均与已闭合CPU汇总JSON一致。关键全局K64门后当前输出相对L2：current 0.465694858、delta 0.456755830、static 0.443229877；delta相对current下降1.919503%。两开发来源static/delta分别0.434116408/0.443958734。delta逐token K1误差0.175800412、原M16×K64占用1.0；raster M16的0.442928515/0.370370358与shuffle M16的0.447338469/0.870209240匹配报告。mean-input M64误差0.477000059。四步代数成本按各自动作区分：主项约0.685185，全M proxy合计约1.185185；mean-input 495行proxy约0.693001，后者不含输入归约、选择、packing及真实执行费用。

报告对source1040为校准源、898/979为历史暴露开发源的标识与CSV一致；不把三源bootstrap表述成盲泛化。Q1后全M重放、每状态1344行原生边界逐byte核对、逐token与原布局物理tile的区别、后验对角排序不是signed最优、输出/gate和采样postwrite的范围均保留。所有候选W2均为付费Dense FP32诊断；本次无E1原生skip、E2自由候选视频/八轴质量或公平SYS速度验收，未声称部署通过。

本复核绑定实际报告和源码SHA256：

- `research/20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md`：`9d801c440b5c05b1187b3fcc72b97c6f4aaf5896bf734d05f114db1496987fd0`。
- 同名结果JSON：`8b52d28bf120b7f024e0272d999156b2855abb19c8b051394825eea577d15163`。
- `SOURCE_EQUAL_COMPARISON.json`：`1bd25f4c24fe1f5b18c6421a707798a2a5487af5fd00cee206d2af7b68d52773`；逐site CSV：`cbb4e9f0916c251204681629662ecf46118f0c87894251d9e1f807590aab1bd5`。
- 汇总源码 `tools/a5_hwg_registered_core_summary_20261030.py`：`21c8af390ceee97d77f9078a201d577b5229d223356228b3dff71ef4dcd3cb19`；`tools/a5_hwg_core_summary_math_20261030.py`：`cc9e67fb3ff24e1b593962521f3eb6e4b86a73307c5d2255d9cb03850f9860b4`。
