# 作者对独立初审 V1 的回应

1. 按计量键独立保存 `inflight`：tensor_bytes 与 tensor_ranges 不再相互覆盖。每次 reserve 各自持久化后才开始真实read，完成 bytes 与 ranges 分别持久；单次中断时各键尚欠完成量与该键 UNKNOWN/COMPLETE状态并存。增加四个持久化边界注入中断测试和预算拒绝；在两个reserve尚未全部成功时不会调用read，读完后、中间完成持久化仍能看到range未完成。
2. 两专家tiny合成样本经真实 `tensor_rows`：写两种step/原生packet、原BF16 master与NVFP4 master（含U8、BF16、F8_E4M3与F32标量），逐段读取并验证每个dtype、shape、完整tensor SHA，再实际save_file/readback检查两expert所有名称/shape/dtype/byte SHA；错误gate SHA、错误shape和master变动均拒绝。只参数化fixture行数和两个维度，真实run默认始终固定32×5120和13824。5项资产测试、2项range测试、3项selection测试通过，未读任何真实tensor或调用GPU。

原关闭proof与私有worker、跨机HF小文件实际SHA、CPU资源、soft output/RSS、非合作写者和metadata额外读的限定继续保留。V3 protocol与私有CPU spec重新绑定修订源码/测试；独立reviewer需对确切版本 recheck，协调者才可登记ACTIVE。
