# 精确 namespace 导入校验修正：有限采集自审

作者 `/root`；真实独立 reviewer 仍为 `/root/sync_abba_actual_review_terra`，本文件不代替其决定。原 PART2_V1 已实际 supervisor EXIT1，完整资产显式 SHA 读取125382492204字节通过，随后原生模型加载到 import origin 门时失败；完成phase数0、视频数0。原 RUN/EXIT、原源码及旧协议保持不变。其余在途原scope不因本修正自动重启。

源码事实：Wan 的 networks.wan、wan.infer 等目录没有 initializer，Python PEP420创建的真实namespace module没有 `__file__`。旧版校验器把所有此类module无条件拒绝。这属于启动检查错误，不是HiF4算法负证据。

本次新增入口复用已冻结原版全部数值采集和独立验证，只替换导入来源检查。精确 prefix→源根映射覆盖 LightX2V、平台及HiFloat4/quant_cy别名；namespace名单仅从协议内逐文件绑定的后代推导，完整名单进入协议。合法namespace要求真实NamespaceLoader、spec和module名称相同、file/origin均None、module和spec只有同一个且完全匹配预期的目录、resolve后不出repo根、没有initializer、至少一个已绑定后代。已导入普通.py/.so仍逐文件核真实路径和SHA，不接纳任意外部第二location。报错带模块名和失败类型。

两个竞争预测：若原失败只是合法namespace误判，修正后来源1040应越过同一启动门，开始原HiF4 plain轨迹；若存在路径污染、额外location、源码变化或其它数值/资源问题，新门应失败并保留具体记录。成功越过import门不等于视频中性通过。

8项无模型/无CUDA fixture实际通过：真实PEP420、空/多location、同树错误目录、symlink逃逸、spec origin/loader/name/file伪装、缺spec/后代、事后新增initializer、别名/extension边界。旧采集12项及代数9项继续适用于未修改原模块。fixture不是原生模型成功证明。

本次先仅来源1040 PART2_NS_V2及其条件CPU verifier：同seed42、同checkpoint/cache/recipe、原4步×40层，plain/observed两视频、十状态+视频byte-neutral、全部160实际postQ1 fullM与1344实坐标行。资源与原科学scope逐字相同：单A100、4 CPU线程、76GiB GPU reserved上限、320GiB RSS、HF新增128GiB、4小时；session所有活跃scope总GPU≤4。原子ACTIVE登记后才启动，新路径一次尝试，失败不自动重试。CPU verifier只在实际producer EXIT0后运行。E0代理比较、E1算子及E2候选视频未包括在此采集范围。

精确协议 `config/e0_a5_hwg_matched_capture_namespace_20261030_part2_frozen_v2.json`；精确private job表 `/home/xiang/.config/ffnds/a5_hwg_capture_namespace_20261030_part2_v2/BOUND_JOBS.json`。parent真实RUN/EXIT SHA在协议绑定；独立复核须检查全部新增文件、原复用模块、namespace名单、精确argv/预算/输入、实际失败事实，再给RUN/条件VERIFY分别写正式记录。当前 REVIEW_REQUESTED，未注册或启动此新scope。
