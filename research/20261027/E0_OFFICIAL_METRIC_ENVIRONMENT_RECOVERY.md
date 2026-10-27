# 原版指标的隔离导入与依赖兼容已完成

**结果**：v8在独立指标环境中完成十项原版VBench函数及四项补充模块的导入、119项有效依赖闭包、实际模块来源和完整overlay文件校验。原评分源码、生成`.venv`及旧指标环境保持原样。[实际结果](E0_OFFICIAL_METRIC_ENVIRONMENT_V8_RESULTS.json)绑定原始RUN、完整函数签名和版本；这项结果尚未加载检查点、执行前向或计算分数。

原版CLIP依赖的 `pkg_resources` 来自一个固定SHA的原版setuptools80.9.0 wheel，解到新的HF只读overlay。基础119包保留原始身份，实际生效版本仅setuptools被该overlay覆盖。其余兼容操作限定为源码绑定的八个无参数activation、一个全局MaxPool2d、两条只读字体查询，以及最终SHA核验对原`.pth`文本的一次读取；该文本没有执行。

每个工程范围均单独冻结，失败原记录保留：

| 范围 | 实际停止位置 / 结果 | 官方函数已导入 |
|---|---|---:|
| [v4](E0_OFFICIAL_METRIC_ENVIRONMENT_V4_FAILURE.json) | 原字体查询被子进程守卫拦截 | 2 |
| [v5](E0_OFFICIAL_METRIC_ENVIRONMENT_V5_FAILURE.json) | 原CLIP缺少pkg_resources | 4 |
| [v6](E0_OFFICIAL_METRIC_ENVIRONMENT_V6_FAILURE.json) | pyiqa导入时的无参数MaxPool2d被守卫拦截 | 8 |
| [v7](E0_OFFICIAL_METRIC_ENVIRONMENT_V7_FAILURE.json) | 最终文件校验把原`.pth`文本误归为检查点读取 | 10 |
| [v8](E0_OFFICIAL_METRIC_ENVIRONMENT_V8_RESULTS.json) | 完整导入、依赖、来源与overlay校验通过 | 10 |

最后范围77.022秒，HF新增约4.18MiB，无GPU、学习参数、检查点加载、模型前向或官方分数。实际检查点兼容、原视频评分及完整人工审阅分别需要相应证据；导入成功不改变任何方法的质量认可状态。当前下一项已登记范围针对四段既有UB1/dense视频，只执行原dynamic degree并记录原函数产生的光流幅值。
