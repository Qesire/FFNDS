# E0 第二轮实现修订复核

评议者 `/root/sync_abba_actual_review_terra`；本文件只记录实现级预审，**不构成实验运行 APPROVE**。所查源码版本：static `bf844e84031d2baa28f165c97a1f3e3740443288ad890218170188c736e36bea`、discovery `aa71b7b9e4554273f75d410c6192a42fe9fcc120cf914cd53b012e0d0db2d0e3`、evidence I/O `f8d58b23627df10cad1c664a91b1e3a52bfbf7402a74ff8cce849def7d5ff32e`、replay `14edfd8edfa503244456c46a045d644788dd9967cb43e701217e3b3ec4109912`、signed math `7f861517a93cf905eae0bcda308831c8ccd8e676f49a5b922985414efc3a93ee`、prepare `2458bdfaf75b035f95cf82ddc05f03c799f7bd30e55f19f59637c89cccea620d`。后续源码变更须重新核查。

对照上一轮反例，`validate_parent` 现经 producer RUN 已绑定的 manifest 重建160个 site→metadata 引用并与协议逐项比对，site 坐标和元数据引用再约束 payload/full-V1；针对同路径 metadata 更换 SHA 的软件反例已加入。首次 replay 之前执行原生根目录 preflight、加载实际 HiF4 backend 并检查实际 import/namespace 来源，之后再次检查；shadow 反例已加入。

全 M 入口新增输入和 gate H2D、坐标、proxy score 算术、mean 扩展与参数 D2H 的带计数区间。support/FP64 signed 统计的调用区间明确包含相关成本；同步 event 与 host wall 不应求和为 kernel 活跃时间。作者的 76 GiB reserved / 96 GiB RSS 数值是尚未实测的预先估计，正式协议必须绑定峰值假设、上限和超限失败行为，实际资源和总 wall 以真实 RUN 测量为准。static 针对 FP32 求得的 Gram/target 二次式，新增逐来源所选组的直接 FP32 W2、FP64 SSE 和预测闭合残差；近抵消 fixture 揭示 FP32 舍入可改变边际。该基线依然是指定 forward greedy 可行方案，非全局 oracle。

用户要求收尾后，限定成功且独立 VERIFY 闭合的来源898/1040/979；来源96实际失败保留为未测，不进入分母。新静态版本仅使用898/1040校准，979是历史暴露开发源而非盲确认；三个来源用于范围内描述，层、step、token 均不冒充独立 source。完整58臂、每源80对与全 M replay 的各自证据身份不变；未保存 full-M 实际 Y，采样匹配不能充当全矩阵逐字节真值。此检查未运行真实模型或科研 tensor。

正式登记前必须另作精确审查：输入的实际 producer 与独立 VERIFY 均已 EXIT0、static 两来源 actual 闭合并冻结、精确 protocol/job/source SHA/资源及机器独立 review 一致，开发读取只能在 static artifact 实际 EXIT0 并冻结后发生。所有诊断是付费 Dense 投影；不由本审查外推 native skip、速度或视频质量。

独立复跑 discovery contract、signed math、evidence I/O 和 native replay 的18项小张量/软件 fixture，全部通过；这些不验证大 M 显存、实际父资产或视频质量。
