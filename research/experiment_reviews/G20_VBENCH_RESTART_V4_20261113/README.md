# G20 VBench 重启 v4 · 实际独立设计审阅

**状态：三名独立审阅者均已对修订后的实际文件完成 recheck，通过设计复核；不授权执行。**

作者 `/root`。独立审阅者为 `/root/numeric_audit`、`/root/stats_audit`、`/root/scope_review`；审阅者未撰写三份主设计文件。

- [作者自审](AUTHOR_SELF_REVIEW.md)
- [数值/预算初审](REVIEW_NUMERIC_INITIAL.json)：REQUEST_CHANGES_DESIGN_ONLY，2项阻断。
- [统计初审](REVIEW_STATS_INITIAL.json)：CHANGES_REQUESTED，1项与数值审查重合的计时预算问题。
- [范围/资产初审](REVIEW_SCOPE_INITIAL.json)：DESIGN_APPROVED_WITH_NONBLOCKING_NOTES。
- [作者逐项答复及修订后哈希](AUTHOR_RESPONSES.md)
- [数值定向复核](RECHECK_NUMERIC.json)：APPROVE_DESIGN_ONLY，NUM-B1/NUM-B2闭合。
- [统计定向复核](RECHECK_STATS.json)：DESIGN_RECHECK_APPROVED，ST-V4-B01闭合。
- [范围定向复核](RECHECK_SCOPE.json)：DESIGN_RECHECK_APPROVED，两条范围提示已处理。
- [静态身份、计数与链接核验](STATIC_VALIDATION.json)

主设计最终SHA：roadmap `17c632a53d8621db701c835415bb7a03e6db9f8a83acff029b466ab9662451b0`；assets/queue `eec38fff484c9a34a176000c5affce5b9dc404f0f3ceddaa98a6bf7852a9533d`；cohort `8512ec30c9cbb8e1f735bca1046e750edd6942edbc1a11030724ee31812abd30`。

数值reviewer保留的执行解释：若因端点N优于L32而推进在线非线性，仍须纳入同bank的L64/Q32容量控制；同时优于位移非线性不替代这项控制。该比较不增加生成，也不触发更大的来源配额。

原设计的历史文件和G0旧容量判定不修改。本次主张仅为重新划分VBench用途、选择实际元数据、明确有限实验和资产依赖；没有声称图像payload已重核、已采集或已dispatch。
