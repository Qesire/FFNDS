# C2N2：模型环境修复已完成CPU验收，独立执行授权待定

[C2N1实际启动](C2N1_startup_failure_analysis.json)完成GPU早期占用后，因Agent错误叠加清除项目路径的环境脚本而退出，零原生调用/视频。原单次授权已经结束，原始记录和资格关闭保持不变。

现在[专用模型环境入口](../../scripts/worker_c2_model_env.sh)恢复项目`PYTHONPATH`并保留CUDA UUID，模型启动前检查HiFloat4/LightX2V真实包路径，并禁止把CPU检查的平台跳过标志带入模型。原错误链[负向CPU复现](C2N2_bad_launcher_CPU_reproduction.json)通过；修复链[正向CPU检查](C2N2_environment_CPU_preflight.json)完成适配器/原生推理类导入、组合缓存完整SHA256及全部160参数校验。GPU隐藏、PyTorch CUDA上下文未建立；CPU导入使用LightX2V提供的显式平台检查跳过标志，实际模型不使用它。这些结果不构成原生数值或视频验证。

[固定草案](C2N2_environment_smoke_DRAFT_v2.json)与父实例科学代码/生成AST、source6、seed42、四步832×464/81帧、六臂、预算全部相同：一张空闲A10080GB；6条完整视频、2norm、3selector，零独立FFN/训练/新分解/官方评分；新增结果≤0.5GiB、单次尝试、失败无自动重试。HF模型资产已在私有选定节点就位；不必重复下载或重建量化缓存。采用新HF输出、新预算和新方法执行ID。

入口：[执行器](../../tools/run_c2_environment_smoke.py)、[独立授权检查](../../tools/c2_environment_contract.py)、[结果处理](../../tools/process_c2_environment_results.py)。`python tools/check_c2_environment_preparation.py --check`复核CPU证据和不变的生成循环。当前模型资格为空，旧C2N1未用的前向额度不能自动用于C2N2；需要对本份固定草案新的明确授权，然后完成正式协议及代码/资产绑定再启动。

完成后仍须完整配对视频人工播放、主体/运动/背景/稳定光照审阅，随后才能讨论新来源官方质量与真实重复速度。R16/R17活动方法退役和其他历史关闭保持；[PRD科学方向](HANDOFF_PRD_SUCCESSOR.md)继续保留。
