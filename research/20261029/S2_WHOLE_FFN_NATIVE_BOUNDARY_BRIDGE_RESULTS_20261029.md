# S2 · whole-FFN原生预激活桥接资产（2026-10-29）

**source911/456的Dense/value_all四条完整160位置视频实际EXIT0；捕获前后四视频及40状态与各自同SM120历史父结果逐byte相同。** 目标执行端闭环验证PASS。

在(step0,block0)、(step1,block20)、(step2,block20)、(step3,block39)各取32行：8个latent时间槽×4个固定xy。16个数据包共49,460,096字节（含safetensors头），新增的是原完整M=31668调用中W1+bias返回、GELU之前的真实a；不从h反演、不用小M重算。a/x/h/Y/r/u/gate及坐标、实际BF16 bias布局都有身份记录。

这使GELU奇偶分解、共享偶函数组和输入二次表示可在真实teacher/student边界上进一步判别。当前只完成被动采集，未进行新的拟合、二次统计或算子验证；student的Y是实际低秩W2输出，不能代替原W2同输入对照。原生采集不构成质量或速度通过。

CPU首份预检实际EXIT0但RUN输出路径未与控制器收据闭合，原文件保留；另登记一次纯收据补全预检，实际EXIT0并完成绑定后启动原生范围。后续CPU诊断尚未注册，按新优先级降P1，执行前另做独立实验复审。

完整scope/退出证明、16包SHA、行坐标与tensor header见[机器结果](S2_WHOLE_FFN_NATIVE_BOUNDARY_BRIDGE_RESULTS_20261029.json)。
