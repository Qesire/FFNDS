# 作者回应独立初审 REVISE

接受异常生命周期反例。新 runtime 在已有异常时同步失败，将同步失败和完成状态 UNKNOWN 加入原异常 notes，始终回滚 patch、清空缓存并传播原异常；无原异常时同步失败照常传播。GPU owner 保存异常 notes 到失败 RUN。执行环境 Python 3.11 支持 bridge 的 add_note；bridge 双流 drain 失败与真实 owner 失败处理联动的合成注入覆盖 notes 落盘，另测试清理失败时原异常和 patch 回滚。若底层 CUDA 运行失败，已发工作及完成状态仍为 UNKNOWN，失败不自动重试。

冻结身份包含更新后的 runtime、owner 和测试；保留初审 REVISE 历史。SAFE 仅验证有限 M32，完整模型和视频未运行；各 phase 及独立 VERIFY 仍须 ACTIVE 登记、真实 EXIT 及实际资源检查。
