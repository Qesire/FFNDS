# 完整下一步实验：ICC / Chipmunk / Q1r2

从 [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) 开始。它统一研究问题、数据角色、实现验收、ICC作者式校准、必要适配诊断、局部选择、五臂视频和有限消融；其它文件是方法与来源附件，不另起队列。

本轮固定主11真实视频来源：作者式CA/CD最多66次模型调用，CAL3的VAE/文本配对适配诊断再加至多30次，主计96；可选22源累计含同一诊断162。主比较55视频槽，两个Dense槽兼作neutral/full-U捕获；完整视频基本68、条件全开74，与校准/诊断共享3次技术重试。联合去噪器前向上界470，局部FFN、编码、分解、搬运与质量评分另账。当前模型/GPU新运行0，真实输入与执行仍待落实。

| 文件 | 用途 |
|---|---|
| [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) | 唯一完整下一步方案与实施顺序 |
| [EXPERIMENT_BUDGET.json](EXPERIMENT_BUDGET.json) | 阶段依赖、NFE/视频/局部格/分解上限与共享重试 |
| [COHORT_PROPOSAL.json](COHORT_PROPOSAL.json) | CAL11、CAL3、可选EXT11、DEV11/3与两个neutral的角色、选择和绑定 |
| [ICC_AUTHOR_CALIBRATION.md](ICC_AUTHOR_CALIBRATION.md) · [ICC_CALIBRATION_RECIPE.json](ICC_CALIBRATION_RECIPE.json) | 作者clean→noise→CA/CD流程及Wan映射的精确合同 |
| [ICC_CALIBRATION_ASSET_AUDIT.md](ICC_CALIBRATION_ASSET_AUDIT.md) · [来源身份](ICC_CALIBRATION_ASSET_SOURCES.json) | 图文、FFN面板、生成视频与真实clean输入缺口 |
| [ICC_STRICT_REFERENCE.md](ICC_STRICT_REFERENCE.md) · [ICC_CODE_CITATIONS.md](ICC_CODE_CITATIONS.md) | 四状态、统计、分解、数值边界与来源 |
| [CHIPMUNK_UPSTREAM_AUDIT.md](CHIPMUNK_UPSTREAM_AUDIT.md) · [CHIP_CODE_CITATIONS.md](CHIP_CODE_CITATIONS.md) | 作者原MLP/selector/状态机、Wan入口与Global-P64区别 |
| [CODE_CITATIONS.json](CODE_CITATIONS.json) · [CITATION_VERIFICATION.json](CITATION_VERIFICATION.json) | 45项主张、91处摘录、42个固定源码文件；复用已核字节身份 |
| [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) · [预算核查](INDEPENDENT_BUDGET_CHECK.json) · [最终审阅身份](FINAL_REVIEW_FILE_IDENTITIES.json) | 本次整合设计的独立回核；不表示模型验收 |
| [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) | 预算、链接、当前元数据与历史记录不变的静态检查 |

## 代码来源与复核

五个固定提交：ICC `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`；Chipmunk `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；引用的FFNDS执行代码 `e7a1e6b7a6babb5cf742072cc07a9961af854416`；LightX2V `9cb6929d83814ff28c4e1386532298fe93a96af1`；Wan2.2 `1ea34ff48f87168174e12956e200b1d908b1c5ff`。取回原字节后可用既有脚本复核：

```bash
python verify_code_citations.py \
  --source icc=/absolute/path/to/icc \
  --source chipmunk=/absolute/path/to/chipmunk \
  --source ffnds=/absolute/path/to/FFNDS \
  --source lightx2v=/absolute/path/to/LightX2V \
  --source wan22=/absolute/path/to/Wan2.2 \
  --output citation-check.json
```

[UPSTREAM_COUNTER_TRACE.json](UPSTREAM_COUNTER_TRACE.json)、[COUNTER_SOURCE_IDENTITY.md](COUNTER_SOURCE_IDENTITY.md)及[audit_chipmunk_counter.py](audit_chipmunk_counter.py)保留原计数器检查身份；本轮没有重跑模型或计数器实验。完整Wan调用链索引为[UPSTREAM_WAN_EXECUTION_SOURCES.json](UPSTREAM_WAN_EXECUTION_SOURCES.json)。

## 已完成证据与未完成工作

A/BW/BWS保持其原实现、数据、运行与结论身份。新的真实clean数据绑定、作者collector与模块/宿主数值验收、三家族选择和主视频仍待执行。ICC/Q1校准来源未配平、DEV已暴露、两种Chip同时存在多处机制差异，解释按统一方案限定。

此前作者式校准修正、旧预算及审阅保留于[固定提交5a5b84a](https://github.com/Qesire/FFNDS/tree/5a5b84a04f71aa1fa6c426febf614a1a4c1d2492/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116)。早期99/897条方案没有本轮活动额度。
