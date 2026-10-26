# E0 · 新配对采集入口的严格视频身份与覆盖检查

针对统一科学审查中归档的[历史门禁偏移](E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)的两个缺口，新增[严格检查层](../../tools/ffnds_paired_video_strict_gate.py)，并接入新双轨入口。**旧v1视频报告、旧门禁及已执行协议均保持原字节**；本结果仅是新入口的软件改进，不把历史核心gate追认为具备这些语义检查。

`STRICT_BUNDLE.json`绑定原`BUNDLE.json`及独立`NATIVE_RUN.json`的SHA。严格层先调用未改的核心gate，再从原生回执的teacher/student两视频清单核对实际路径、SHA、目录中全部MP4及bundle覆盖集合。降低expected_video_count或漏注册一条视频不能通过。每个配对视频都必须保留真实dense reference/reference_identity，删除引用也不能跳过检查。

模型身份从**实际生成回执的assets与真实operator kind**重算，包括checkpoint/cache身份、dense或affine参数族及bank RUN SHA；不复制报告的model_digest。当前新NATIVE_RUN及旧原dense RUN的两种schema分别解析。候选报告误继承candidate digest冒充dense会失败；未知回执schema不能默认为可验证。原始两个dense来源回执已只读复算，均与之前已登记的实际dense digest一致。

8项新增CPU软件测试通过：正确清单、错误dense身份、漏视频并降低计数、目录多视频、缺原生清单、删reference、原生身份/视频SHA错配及旧dense回执独立资产。17项双轨软件测试仍通过，共25项；它们不运行Wan/HiF4/GPU，不证明实际生成中性或画质。

为保持版本历史，[v1草案](../../config/e0_paired_full_depth_20261026.json)保留`6e9e6a…c188`原字节，**从未授权或执行**；另冻[v2执行协议](../../config/e0_paired_full_depth_20261026_v2.json)，SHA `dfefff076066c766766897ab76bb161b1e82fbfb74c5a0031c3f91c68ef95ce0`。v2绑定19份driver/环境/单GPU启动脚本及23份native源码，数学、参数包及六条完整路径预算不变。旧v1源绑定只对应其原实现提交，不用于当前入口启动。

当前三pair仍是DD来源0控制，然后DC来源0/12、seed42/54，单张空闲A100、输出≤0.5GiB，无训练/拟合/分解/官方评分，失败扣账且不重试。生成成本按pair去重；正式质量仍NOT_RUN/false。失败产物继续走逐帧取证或显式缺失/解码失败记录，不冒称严格gate通过。

独立[严格层审阅](../method_discussion/discoveries/E0/independent-paired-driver-final-audit-20261026_v5_strict_gate.md)检查新增工具与接入；前轮[原生组件/控制器审阅](E0_PAIRED_NATIVE_IMPLEMENTATION.md)保留原snapshot范围。**新模型/GPU0，真实全深度采集0；当前v2仍待新固定模型范围的单独明确授权登记。**
