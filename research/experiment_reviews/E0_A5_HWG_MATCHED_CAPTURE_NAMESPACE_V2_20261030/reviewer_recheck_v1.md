# 独立最终复核：PART2 HiF4 namespace 校验修正

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。**APPROVE 精确 PART2_NS_V2 native 采集和对应条件 VERIFY 范围进入登记**；本review不启动、不登记ACTIVE，也不声称已越过启动门或得到新视频。

原PART2_V1实际producer supervisor EXIT1、无timeout、RUN中phase列表为空，错误为 `native namespace module lacks bound source`；其RUN SHA256 `8055c98b55868186f953b2f43ed1a2c793dc452b15dee7c293590f255600f6d9`，EXIT SHA256 `0a5f792f6ef3e15f6f75fc584046c8992efc2e80a59303ce968ad3901800cef3`，均按新协议实际绑定核验。失败保留，不称neutral或算法负证据；此范围为重新独立登记的版本，不自动重跑旧ID。

真实源码中 `lightx2v.models.networks.wan`、`.wan.infer`、`.wan.weights`、`lightx2v.models.schedulers.wan` 等目录没有 initializer，原实现无条件拒绝 `__file__ is None`。本次入口只覆盖原owner导入来源校验：PEP420 namespace须由冻结source后代导出的精确module→目录映射支持，要求module/spec名、NamespaceLoader、空file/origin、单个相同search location、无initializer且目录在repo根内；普通.py和.so仍要求实际文件路径在冻结清单且SHA相同。HiFloat4与quant_cy合法别名映射同一真实受绑树，不接受额外目录。异常携带模块名/类别；如仍遇到源不符或另一模块错误，应保留失败而非绕过检查。

逐键比较新旧协议，除新scope/输出、实际失败父RUN/EXIT、三项新增源码及依据冻结清单生成的140项namespace映射外，数值路径、source1040、seed42、模型/cache、采集对象及资源完全一致。实际1021项source SHA及完整namespace映射重新核对。两个新私有job从原作业逐项替换入口、协议/作业路径、scope和fresh结果路径重构后一致；私有worker身份未写入公共记录。代码提交 `81d9ffd0377476fd386381449f19555c2c2769ed`；登记时ACTIVE仍须绑定真实调度HEAD。

独立复跑8项namespace与12项原采集软件fixture均PASS，含真实PEP420、空/多location、错误树、symlink逃逸、伪装spec/loader/initializer和字段/读写失败反例；这些不执行真实Wan模型，也不证明启动成功。先前科学及预算审查保持：每来源plain与observer Dense、4×40、十状态/视频byte-neutral，原生Q1后full-M与21个M64/τ；单GPU、4CPU、原有限资源及失败不自动重试。对应CPU VERIFY只可在真实producer EXIT0后启动。四来源均暴露开发源，当前仅采集，不授权E0/E1/E2比较、不证明稀疏收益或质量。
