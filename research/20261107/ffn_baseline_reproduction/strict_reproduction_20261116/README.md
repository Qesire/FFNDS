# ICC / Chipmunk：作者校准与精简复现

当前版本承接 `0e29eb7affddc88138337294ffdb5ccc7893f86e`，按用户要求修正 ICC 校准方法。**主规模11个真实视频source，从clean latent前向加噪，CA单调用、CD真实相邻两调用；两专家合计最多66次校准前向，可选22源累计132次。** 五臂主视频仍为55槽位。完整视频合计基本68、条件全开74，与校准共享3次技术重试；全部重试用于视频时最多77条。旧99条D4总账已撤回。本轮模型/GPU运行0，尚未绑定真实clean输入，不能声称已完成数值复现。

| 文件 | 当前用途 |
| --- | --- |
| [ICC_AUTHOR_CALIBRATION.md](ICC_AUTHOR_CALIBRATION.md) | 作者clean/noise/CA-CD配方、Wan必要映射、逐步调用及验收 |
| [ICC_CALIBRATION_RECIPE.json](ICC_CALIBRATION_RECIPE.json) | 11/22规模、实际D4时间、expert、VAE、空文本及统计的机器可读合同 |
| [ICC_CALIBRATION_ASSET_AUDIT.md](ICC_CALIBRATION_ASSET_AUDIT.md) · [资产来源](ICC_CALIBRATION_ASSET_SOURCES.json) | 区分图文条件、FFN面板、生成视频与尚缺的真实clean clip/x0 |
| [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) | 唯一设计：完整三家族局部选择、五臂DEV11开发比较与有限消融 |
| [EXPERIMENT_BUDGET.json](EXPERIMENT_BUDGET.json) | 校准NFE与完整D4分账；离线SVD主184、可选208、含条件闭环最多356 |
| [COHORT_PROPOSAL.json](COHORT_PROPOSAL.json) | 旧图文CAL与Q1身份保留；新ICC真实视频绑定待完成；DEV3/neutral固定 |
| [ICC_CODE_CITATIONS.md](ICC_CODE_CITATIONS.md) | ICC四态、三家族、统计、作者校准调用与历史2R映射 |
| [CHIP_CODE_CITATIONS.md](CHIP_CODE_CITATIONS.md) | 官方Wan入口、原MLP机制、必要接口与Global-P64实际执行 |
| [CODE_CITATIONS.json](CODE_CITATIONS.json) | 45项主张、91处短引述、42个固定源码文件；Git blob/SHA、函数、位置、验收对应 |
| [verify_code_citations.py](verify_code_citations.py) · [CITATION_VERIFICATION.json](CITATION_VERIFICATION.json) | 原字节、引述和固定链接核验，不等于数值/模型验收 |
| [ICC_STRICT_REFERENCE.md](ICC_STRICT_REFERENCE.md) | 四状态、统计/因子数值合同、BF16与缓存要求 |
| [CHIPMUNK_UPSTREAM_AUDIT.md](CHIPMUNK_UPSTREAM_AUDIT.md) · [固定Wan执行源码索引](UPSTREAM_WAN_EXECUTION_SOURCES.json) | 既有作者Wan调用链审计；不重复vendor源码 |
| [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) | 本次真实独立审阅及修订回核；设计/source审查，未执行GPU任务 |
| [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) | 预算、元数据、链接、历史记录不变和来源核验 |

## 核验引述

取回索引中五个固定提交的源码后运行；文件字节必须严格匹配，脚本不自动修换行：

```bash
python verify_code_citations.py \
  --source icc=/absolute/path/to/icc \
  --source chipmunk=/absolute/path/to/chipmunk \
  --source ffnds=/absolute/path/to/FFNDS \
  --source lightx2v=/absolute/path/to/LightX2V \
  --source wan22=/absolute/path/to/Wan2.2 \
  --output citation-check.json
```

ICC固定 `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`；Chip固定 `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；引用的FFNDS执行源码固定 `e7a1e6b7a6babb5cf742072cc07a9961af854416`；LightX2V固定 `9cb6929d83814ff28c4e1386532298fe93a96af1`；Wan2.2固定 `1ea34ff48f87168174e12956e200b1d908b1c5ff`。源码仍从原仓库取得。核验过程中的换行身份修复有索引记录，不把修改后的字节冒充原代码。

## 已有计数器证据

[UPSTREAM_COUNTER_TRACE.json](UPSTREAM_COUNTER_TRACE.json)和[COUNTER_SOURCE_IDENTITY.md](COUNTER_SOURCE_IDENTITY.md)继续复用已核的纯Python接口检查，原源码SHA为 `cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288`；本轮没有重新运行模型或计数器实验。脚本为[audit_chipmunk_counter.py](audit_chipmunk_counter.py)。

## 版本与证据边界

[旧99条D4及rollout校准方案](https://github.com/Qesire/FFNDS/tree/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116)和[更早897次提案](https://github.com/Qesire/FFNDS/tree/9defbbaa9112eee954c24ae92c91831fd2dd69c4/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116)只保留历史身份，不与当前预算相加。A/BW/BWS既有运行和结果不改写。

现有CAL图文名单不能作为已绑定的真实clean视频。旧rollout CA/CD统计和因子不能抵扣新配方；同权重普通SVD可以按完整数值身份复用。Q1保留原CAL11协议；ICC与Q1校准来源尚未配平，五臂属于同DEV的pipeline开发比较。真实clean输入、完整三家族候选、采集器及Wan数值验收完成前，主ICC选择与完整复现均保持未完成。
