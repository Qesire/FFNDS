# 三来源 E0 汇总 V2_LOCAL 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。本次范围是已审CORE3汇总的本地CPU启动重绑定，0 GPU、1 CPU，仍只读取已关闭的三来源E0文件，独立核240个mask/proxy的整文件和typed raw SHA，执行source等权统计；没有新模型、视频或候选动作。

逐键比较V1与V2协议，除scope、私有worker/locator、CPU affinity、fresh output、execution placement及未启动前序绑定外，三个父输入、来源角色、全部1039源码、预算及统计算法完全一致。V1的ACTIVE记录为 `NOT_STARTED_TRANSPORT_FAILURE_NO_START_CLAIM_NO_CHILD`，其worker目录检查时不存在，故不将其记为实际运行失败；两次SSH超时的细节依据作者自审，独立核到的是现有状态和目录缺失。V2使用新输出，检查时不存在。

逐项核复V2的protocol committed blob、fingerprint、完整私有job/locator/profile SHA；当前本机与冻结worker身份一致、CPU0在允许亲和集合、CUDA显式为空，预算仍为RSS≤512MiB、显式读≤96GiB、新HF≤256MiB、7200秒外层7320秒，不自动重试。1039源码文件大小和SHA逐一核对。V1独立复核已核三个E0实际EXIT0、240个元数据文件每项58臂、真实逐bit样本检查、调用序号及共同CAL1；本次未重复读取科研payload，实际mask whole/raw SHA由该新scope完成。

已只读复核本地durable启动控制器的最终版本：`--local`的本机身份和0 GPU断言位于构造并写入ACTIVE之前，本次worker身份匹配。检查时控制器SHA `8bbbc29583da326497b2e31d13794fa2cccdfe05862a4f4b8361d7a994a93ec6`；该私有脚本不在protocol源码closure内，调度前仍应确认相同身份和代码。

**APPROVE**：精确V2_LOCAL可进入登记。冻结提交 `079ec392952a36ea3ff1d3c205e666114fa71984` 的protocol blob与私有job绑定一致。source1040是校准图，898/979为历史暴露开发图；不以汇总结果声称盲泛化、实际native skip、SYS倍速或新视频质量。实际启动及输出状态仍须真实收据。
