# SharQ Wan FFN 配对资产 CPU 生产：作者自审

本scope只将已完成的原生捕获与checkpoint字节按真实来源和专家配对并保存；没有模型forward、GPU或新科学tensor统计。两个case均为已暴露开发source911 Dense seed42 block20：step1/high、step2/low。各选真实32行及同一包 `V1` 和 `gate`，metadata中的行号、shape、source、step、expert、grid、raw tensor SHA与完整packet SHA逐一校验；不对32行重排冒充完整M。W1/W2/bias来自各自expert的原BF16 master，另有相同expert/block的NVFP4 checkpoint packed W、scale、bias、alpha和input global scale；两种checkpoint身份不同，后续算法比较必须分别标注。

原master完整SHA历史真实proof与NVFP4两文件完整SHA proof均绑定真实RUN/EXIT；另绑定此前NVFP4 asset CPU probe实际stat/header证明。该CPU范围必须在原master实际所在worker执行：已完成私有小文件核验，两套原BF16和NVFP4 checkpoint当前stat均与各自proof相同；不复制整个master到另一机器。两个worker对同一共享HF小文件SHA一致，但不是未来bundle写入后完整跨节点字节验证，消费者必须独立重新预扣完整bundle SHA。每个读取的master header SHA、fd/path stat前后、dtype/shape/offset和原始tensor字节SHA均核对；打包前再次核原始与staged tensor字节SHA，完整bundle另作一次预扣SHA。无针对非合作写者的强制只读保证，stat/hash间瞬时修改或恶意还原为剩余风险。

预算：34个typed ranges、726,311,456个payload tensor字节；两个包完整SHA 6,182,512字节；最终bundle本身另预扣其实际大小（上限1GiB），64 range/768MiB tensor、8MiB packet的上限均比实际略大。metadata、header、source/runtime RECORD和staged tensor内存SHA的额外读取不归入这些数字。4 CPU、零GPU、3600秒/外层3720秒、8GiB RSS、2GiB HF软输出监控及1GiB单文件RLIMIT；输出扫描/峰值是采样而非硬aggregate quota。写盘出错或SIGKILL时inflight UNKNOWN，尝试保留RUN和emergency失败账，不自动重试。partial写入在scope目录保留，重试须新scope。

可能推翻结果的反例：原BF16和NVFP4看似同专家/block但加载了不同层的weight或错方向；逐key、dtype、shape、range、hash与两类不同master receipt验证防止误标。row坐标即使相同，来自不同step/expert的gate也可能不同，所以分别强制匹配两个原bridge metadata。若NFS有非合作写入或输出预算采样期间瞬时越限，不能声称完全隔离/硬上限；后续consumer重hash。真实独立reviewer应检查老proof字段、stat与不同worker、bundle保存、checkpoint scalar dtype、资源和失败账。

当前仅静态源码/模型header/metadata、同路径小文件跨worker工程验证与合成软件测试；未运行真实tensor提取、模型或GPU。protocol标 FROZEN_PENDING_REGISTRATION 只便于精确复核，必须真实不同Agent审查、作者回应和reviewer recheck后由协调者注册ACTIVE。
