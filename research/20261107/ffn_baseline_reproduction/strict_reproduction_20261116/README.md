# ICC / Chipmunk：代码引述与精简复现

当前版本承接 `9defbbaa9112eee954c24ae92c91831fd2dd69c4`，按用户进一步细化、建立代码引述并精简实验的要求更新。**算法和必要接口完整核验；主视频55槽位，所有D4尝试上限99。** 旧897提案已撤回；本轮新增模型/GPU运行0。

| 文件 | 当前用途 |
| --- | --- |
| [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) | 唯一实验设计：主CAL11、六站点选择、五臂DEV11开发比较、条件消融与复用 |
| [EXPERIMENT_BUDGET.json](EXPERIMENT_BUDGET.json) | 55主视频；基本90、条件6、技术重试3；离线SVD/PCG另有上限 |
| [COHORT_PROPOSAL.json](COHORT_PROPOSAL.json) | CAL11、CAL_EXT11、DEV11、DEV3、neutral固定元数据；输入实体/簇/祖先仍待完整绑定 |
| [ICC_CODE_CITATIONS.md](ICC_CODE_CITATIONS.md) | 20项ICC代码判断：四态、三家族、统计、真实校准调用数、历史2R映射 |
| [CHIP_CODE_CITATIONS.md](CHIP_CODE_CITATIONS.md) | 17项Chip判断：官方Wan入口、原MLP机制、必要接口与Global-P64实际执行 |
| [CODE_CITATIONS.json](CODE_CITATIONS.json) | 37项主张、83处短引述、36个固定源码文件；Git blob/SHA、函数、行号、验收对应 |
| [verify_code_citations.py](verify_code_citations.py) · [CITATION_VERIFICATION.json](CITATION_VERIFICATION.json) | 无模型的原字节、摘录位置和永久链接核验；不等于数值/方法验收 |
| [ICC_STRICT_REFERENCE.md](ICC_STRICT_REFERENCE.md) | 四状态、统计/因子数值合同、BF16与缓存部署要求 |
| [CHIPMUNK_UPSTREAM_AUDIT.md](CHIPMUNK_UPSTREAM_AUDIT.md) | 上一轮完整作者Wan调用链审计；执行数量只依当前总账 |
| [UPSTREAM_WAN_EXECUTION_SOURCES.json](UPSTREAM_WAN_EXECUTION_SOURCES.json) | 39份作者Wan执行链源码的固定URL/Git blob/SHA，不重复vendor源码 |
| [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) | 本轮真实独立审阅与修订回核；仅审代码证据和设计，未执行GPU任务 |
| [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) | 预算、元数据、链接、历史资产不变和来源核验记录 |

## 核验引述

取回索引中三个固定提交的源码后运行；工作树中的文件字节必须严格匹配，脚本不自动修换行：

```bash
python verify_code_citations.py \
  --source icc=/absolute/path/to/icc \
  --source chipmunk=/absolute/path/to/chipmunk \
  --source ffnds=/absolute/path/to/FFNDS \
  --output citation-check.json
```

ICC固定 `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`；Chip固定 `b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；引用的FFNDS执行源码固定 `e7a1e6b7a6babb5cf742072cc07a9961af854416`。源码仍从原仓库取得。两个历史工作快照多出的末尾LF已在临时核验副本中恢复，并在索引披露；不会把修改后的哈希冒充原代码。

## 已有计数器证据

[UPSTREAM_COUNTER_TRACE.json](UPSTREAM_COUNTER_TRACE.json) 和 [COUNTER_SOURCE_IDENTITY.md](COUNTER_SOURCE_IDENTITY.md) 继续复用上一轮已核的纯Python接口检查。原源码SHA为 `cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288`；本轮没有重新运行模型或计数器实验。复核脚本为 [audit_chipmunk_counter.py](audit_chipmunk_counter.py)。

## 旧版本

897次提案及其审阅、CAL64/CAL256/EVAL22元数据保留在[固定历史提交](https://github.com/Qesire/FFNDS/tree/9defbbaa9112eee954c24ae92c91831fd2dd69c4/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116)。它们不与当前预算相加。A/BW/BWS既有运行和结果身份不改写。

当前仍需真正执行作者模块与Wan端口验收；输入实体、资源、编译扩展和producer未因文档更新自动得到验证。五臂结果将属于已暴露DEV上的开发比较，校准分布、未选家族、未跑适配及总体方法优胜的结论各按证据范围表达。
