# 来源96迁移 V3 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。本次针对 PART1_NS_V3 native 及其条件 VERIFY 的新范围进行独立复核，范围为来源96、seed42、plain 和被动 HiF4 Dense 配对采集，不授权 E0/E1/E2 回放、训练或新视频指标实验。

实际旧 PART1_V2 的 RUN、EXIT、原操作员停止请求和后续升级四项原始文件均按协议绑定的大小与 SHA 验证。旧 child 实际退出码 -15、supervisor `process_exited=true`、无 timeout、零生成 phase；RUN 保存了 `Foreign compute appeared on the assigned physical GPU` 实际监控冲突。请求先对自己的精确 child 发 SIGINT，保全失败文件后对尚未退出的同一 child 升级 SIGTERM；记录说明未处理第三方进程。旧 sibling 的 namespace 失败可以解释发出停止请求的动机，但不能把本次实际 foreign-compute 记录或受控停止改写成已经发生的 namespace 失败。旧 scope 与账目保持原身份。

初版 V3 协议只有 1020 源，遗漏被新入口直接导入的 namespace 修正模块及其测试、准备源码，本评议明确提出阻断。修正版重新加入三项：实际 1023 项源码逐个核验大小及 SHA，全部此前已审 NS2 的 1021 项一致，新增两项为本次入口和测试；140 项 namespace 映射与已审 NS2 完全一致。独立重跑两项纯软件 fixture 通过，覆盖已计费读前/每段后检查已有 monitor、冲突时记录部分读取并停止。检查复用已有采样的 monitor，不声称检测时间早于其采样，也不改变数值采集代码。启动、模型执行和中性结果仍以未来实际证据为准。

对照旧 PART1_V2 逐键核对科学设计、来源与模型/cache 绑定、预算不变；更换项限于 scope/新路径、worker profile、CPU affinity、停止父证据及已说明的源码和 namespace 修正。来源图片大小及 SHA 已核验。两份私有 job 的参数、协议和 job SHA、fingerprint、资产 locator 与 worker profile 的实际 SHA 逐项核对；运行选定一张 GPU、CPU 4–7，VERIFY 为空 GPU 且限一个 CPU；新 supervisor 输出目录检查时不存在。运行时现有启动门仍须核实真实机器身份、GPU UUID 与空闲、全资产 SHA 及 host 联合预留；本复核不代报之后的资源状态。host Admission 同时计算既有模型尚未消费的 forecast 和新模型 forecast，保留单 host 两模型及会话四 GPU 上限。

条件 VERIFY 必须等待本 producer 实际 EXIT0；任一异常保留实际结果和扣账，不自动重试。原 scope 的 namespace 设计、ten-state/video byte neutral、160 位置及资源限制仍照既有冻结合同执行，方法质量或稀疏收益不由本 review 推断。

**APPROVE** 精确 PART1_NS_V3 native 采集和其条件 VERIFY 范围进入登记：修正版协议在提交 `1cb47d3267be2ed748124c0fa7046c5085f2dc35` 的 blob SHA 与两份作业绑定一致，源码闭包及两份作业已在该版本再次独立核验。登记和启动时仍需检查真实 HEAD、ACTIVE 及运行资源。
