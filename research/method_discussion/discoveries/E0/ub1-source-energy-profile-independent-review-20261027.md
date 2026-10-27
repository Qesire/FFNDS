# CAL24 source energy profile：独立冻结前审阅

**SRC/CPU 合成结论**：指定原四个位置、24来源、每来源256已保存 teacher rows 的设计与原 contribution 路径一致，6项CPU合成测试通过；草案尚未冻结。实际权重在共享根不可见，必须在指定 worker 对私有权重根做路径和SHA预检。C_mean容差及支持一致性目前只记录布尔，**不作为 COMPLETE 的失败门槛**；若结果不一致，需明确报告该差异，不能把 COMPLETE 解读为重现成功。

审阅工具 SHA256 `87b52a9b7c8af1c572c4dd980c52c5425eb4a27de7ef5a09a10f97d6a34ebe92`、测试 `e55093e8a2470ade192a6b1f30a3b050795910848ecbf91903bfd283ccd6924f`、草案 `d1f4132b5c926289511d9f873c71e3c69319a5276a27518333a9fe26f63fbfb4`；测试 exit 0。共享 HF 的124项CAL资产和4项参考文件路径均存在，4项参考文件SHA一致；96项历史INPUT的 source/seed/step/block/selected rows/payload SHA元数据检查通过。共享根下4项weight shard均不存在，worker的 `--weight-hf-root` 必须指向私有真实根；`--hf-root` 同时承担 CAL 输入及新输出，`--reference-hf-root` 指向共享根。正式运行前仍需在实际worker核验全部文件字节与路径，特别是权重；本次未读 tensor 数值或运行GPU。

原 `contribution_scores` 对每8个64通道组做一次FP32 BMM，再对 gate 乘后的信号转FP64平方归约；每来源27次BMM，96来源位置共2592次、20736逻辑组乘积。原W2读取后维持存储dtype上GPU，在原函数中升FP32，并记录原dtype；不先截断为BF16。V2、gate、b2要求原BF16形状/finite，gate与b2核对固定SHA；b2作为源身份检查，不加入组贡献。每组先在256 rows取均值，再对24来源等权平均，比较四份原C_mean及54/162组支持。来源bank要求64个不同source ID、CAL24与DEV40；只分析4个固定位置，不是全160。

`diagnose` 使用这24个source/image簇计算逐位置 leave-one-cluster-out 和200次bootstrap；应以实际统计及输入暴露状态解释，不将来源内rows当独立样本或泛化为新视频。scope没有新QDQ、W1、模型、拟合或质量评分。GPU预扣、监控和资源限制尚须实际运行收据验证。

## 修订版本复核

工具 `e64ce08b712a51dd2091859a11a3fb185c9528bd7f67eeb320bd5588d46218b7`，测试 `6793aea711f2e2c5c92190685ccce1a0e03a7d7bad924356301e31336c6326f0`，草案配置 `67bbf3e439ea86ea96ef11e586de2c46932394d07d99bacffc753c7cedf9ebd4`；7项CPU测试 exit 0。原C_mean超容差现先保存该位置和24条profile，再抛错停止后续位置；支持差异仍单独报告。运行前检查 torch、numpy 和 safetensors 的精确版本。前节只记录比较不失败的观察适用于旧版。静态无新增阻断，可进入冻结；目标worker的私有权重根及完整资产仍需实际字节预检。
