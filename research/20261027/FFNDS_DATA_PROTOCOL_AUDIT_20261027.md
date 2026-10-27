# 2026-10-27 · FFNDS数据采样、校准与评测源码审查

**结论**：现有项目已经较成熟地检查视频、操作数、seed与原生路径的逐项身份。最优先的修复是把**数据层面的独立性与评价主张**显式绑定到全部新协议，尤其是历史五文件名复制、CAL12/24的固定行数、OP3折与冻结UB1训练预算差异、Student状态坐标重用、二值运动指标的盲区。

## 确认的执行路径

- `tools/evaluate_vbench3.py` 和 `tools/evaluate_disjoint_vbench3.py`：每来源一条真实MP4，由 `hardlink_to` 或 `copy2` 产生5个同内容的文件名；每20来源对应100个名字。`tests/test_vbench_input_index.py` 只核对文件名与旧维度索引覆盖，不检验视频采样独立性。因此历史指标不扩展为 `n=100`。
- `tools/run_fresh_official_metrics.py`：把实际唯一视频放入每来源 `video_list=[path]`，有23 unique fresh sources和实际每arm 23 generated clips的身份检查；它是正式函数在真实自定义来源子集上的执行，不等同每输入五次采样。
- `config/calibration24_ffn_operands.json`：24×256=6144行，和CAL12 12×512相同；160 scope记录原生统一无放回行采样。`tools/audit_calibration_capture.py` 可检查行坐标、FFN形状和81帧@16FPS，验证抽样实现而不验证泛化。
- `tools/audit_calibration_coordinate_exposure.py`：3,840个Dense/Student源×step×block采样坐标一致；相同抽样便于配对，但拟合/评价的统计独立性必须另从来源留出核查。
- `research/20261027/UB1_SOURCE_GENERALIZATION_AUDIT_20261027.md`：formal20/disjoint20成像下降1.345%/3.166%，后者最差来源16.78%；独立来源簇而非sites才是尾部分析统计单位。
- `research/20261027/UB1_DYNAMIC_DEGREE_RESULTS.md`：2个已暴露来源的连续RAFT motion显著下降，四条官方二值结果都0；构成增加continuous-motion观测的直接实测动机。

## 明确的修订

1. 统一协议：[DATA_SAMPLING_CALIBRATION_EVALUATION](../agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)。
2. 数据预检：`tools/ffnds_cohort_preflight.py` 的新schema `ffnds.cohort.v1`，检查来源簇、历史暴露、配对噪声、行抽样、同run多名字；必要时比较本地真实文件字节。
3. Agent：领域Skill与专用实验Skill、根AGENTS、Copilot、视频指标规则统一引用该协议；以发现→机制→判别实验的**肯定式提示**引导，不增加泛化的免责句。
4. 证据边界：本次是已保存证据和源码的`SRC + HISTORICAL-OBS`审查，静态预检工具为新增未见实际 Wan/视频运行的新模块；保留所有旧记录与接受/失败口径。

## 成熟实现的可转移部分

[VBench-I2V official](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)真实重复采样、维度专属来源与 Image Suite；[VBench custom_input](https://github.com/Vchitect/VBench)用于非标准自定义输入的官方函数；[Diffusers reproducibility](https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds)解释随机发生器消费与重置；[LLM Compressor datasets](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/)展示压缩校准数据领域匹配；[Wan2.2 原仓库](https://github.com/Wan-Video/Wan2.2)用于模型/视频推理参数身份。迁移的是**实验设计和统计语义**，实际数值始终属于FFNDS自己的Wan基线。

## 下一次真实研究的最小增量

先用既有CAL24/OP3及formal/disjoint来源账本建立跨批次`source_id,image_id,prompt_id,role,exposure_history`映射，预检后才选新的确认集。校准因素实验、全九维质量与连续运动及速度三者由实际可用计算预算依次独立注册。优先固定来源统计和脚本抽样身份，再比较新算法，以降低无效GPU实验开销。
