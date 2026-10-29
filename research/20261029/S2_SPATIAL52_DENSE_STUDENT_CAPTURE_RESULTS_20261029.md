# S2真实空间邻域与自由学生前缀：两来源全160位置采集

2026-10-29 · NATIVE / passive-observation。两个独立曝光诊断来源898（鸟/鱼场景）和96（相机平移）都实际exit0。Dense/value10/dcac10各完成完整四步×40层正常前缀；六臂合计960个原生操作数包、60个历史生成噪声/flow/latent逐字节中性检查通过。现在可在同来源/位置/时空坐标上比较真实teacher与学生h，区分当前W2函数误差与先前状态偏移。

[可重建JSON](S2_SPATIAL52_DENSE_STUDENT_CAPTURE_RESULTS_20261029.json)来自实际RUN、进程EXIT、三臂observation receipt、全部packet metadata与原源/row/typed header，复算命令：

```bash
.venv/bin/python tools/recompute_spatial_student_capture.py --out /tmp/spatial-capture-rebuilt.json
```

## 实际完成的变量与控制

| 项目 | 实际值 |
|---|---:|
| 独立图像来源 | 2，均为已曝光极端Dense运动诊断 |
| 来源×数值臂 | 2×3 |
| 每臂原生位置 | 4去噪步×40层=160 |
| 每位置真实行 | 21 latent τ×52不同xy=1092 |
| 保存数据包／typed张量 | 960／8640 |
| 原始张量payload | 71,961,016,320 bytes，约67.02GiB |
| 历史完整生成的状态byte-neutral | 60／60通过 |
| 新完整模型flow／FFN位置 | 24／960 |
| 新视频decode／视频／拟合／官方评分 | 0／0／0／0 |

52个xy为原Dense 81帧Farneback选择的高流5×5、预冻结中央y12–16/x24–28的5×5、2个远处低流点。全部21τ及所有160位置保留；τ不是四个去噪step。最大流在鸟图背景植物，故这些是空间分层，不能替代主体mask、遮挡或运动对应。

每packet保存原BF16 V1[1092,5120]、postGELU h[1092,13824]、含原b2的Y、FFN前r/后u、gate/shift/scale[5120]、int64原生rows。producer在写出时验证全部typed shape/finite值和raw SHA；本次库存复算仅验证闭合metadata，未重复读取72GB数值。下一独立reader会完整流式SHA读取每包并用原BF16先乘后加逐字节复原u。preGELU a、实际attention/QKV消费者响应、真实latent运动对应和主体/遮挡mask未采。

## 来源与数值血缘

- 来源898：portable V2 part0，worker 1863.796s，真实进程退出0；三臂各160包/10状态通过，GPU监控无违规。协议、RUN、EXIT及各臂收据的完整HF locator/SHA见JSON。
- 来源96：portable V2 part1，worker 1855.991s，真实进程退出0；三臂各160包/10状态通过，GPU监控无违规。协议、RUN、EXIT及各臂收据的完整HF locator/SHA见JSON。

同一seed42、checkpoint、BF16 recipe、80物理函数各候选固定r960库；保持完整W1/GELU和原b2/gate/residual。被动采集对每臂历史完整视频的raw noise/初始noise/四flow/四latent逐字节一致；没有把Dense h当自由student h。采集wall含大数据D2H/序列化，不用于公平SYS比较，视频质量结论仍是上一批二十来源40新视频未通过。

原V1两个等待队列因外来任务占卡由协调者校验自身PID后取消，真实exit−15、0模型/GPU调用、无RUN；未触碰外来任务。另一个已有资产的空闲A100执行器完成69,025,362,058 bytes全SHA/稳定stat/两header CPU proof后，独立portable V2才启动。原V1及迁移/取消证据保留，不覆盖旧scope。

## 新独立设计及下一判别

新上游独立委派指令到达时本次采集已经冻结运行；真实理论子Agent随后独立重建计算图与四组竞争预测、推导L/P/I及数值修正，再评审新reader：[独立推导与修正复核](../method_discussion/discoveries/E0/s2-theory-subagent-s2_theory_design_independent-20261029.md)。它不是本pilot的事前审批，也不是方法SUPPORT投票。

下一新范围用原W2与两固定候选在真实h_D/h_S上做四角M1092读出；保留原生M31668 Y/U锚、g/r差、typed写回、L/P/I与全部signed cross，分高flow/central/far与τ0/DC/AC/逐频点分析。reader V1的两个CPU预检因缺输出parent实际exit1，0GPU/模型、无RUN，证据保留；新的启动修复V2与原数值核心分离、单独登记。两个来源只用于链路及初步机制判别；后续独立新图簇、遮挡/物体对应与正常后缀因果干预按新问题再注册，不把本次采集当作视频方法合格。
