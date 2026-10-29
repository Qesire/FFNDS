# S2同状态四角重放V3：完整block边界交接及真实控制

ROOT协调者审阅，2026-10-29。旧V2两路按实测约48秒/配对位置、160位置/路，7200秒范围有耗尽风险。旧代码、协议、原生算术与失败身份保留；不扩大旧预算、不重新跑已完成主体。协调者在ACTIVE请求下一完整物理block边界结束旧范围，旧runner的authorize明确拒绝并正常捕获异常、落盘、退出。只有实际EXIT1、timeout=false、原始错误`ValueError: Exact replay not registered`、资源violations为空、4整数倍的非空completed前缀、完整预扣/实测/journal精确相等、原记录SHA及顺序正确，才可冻结新V3。

V3实现者`/root/s2_theory_design_independent`单独完成新runner并自审；其[自审](s2-batched-v3-implementer-self-review-20261029.md)明确是实现者身份，不是独立审批。ROOT逐段对照旧runner和新统计数学，并强化上述父退出条件。实际30项软件控制通过：旧四角/typed/频谱、旧V2启动、新V3预算/父账本/严格比较、batched数学等；新测试还直接数出每gY view+bridge pair的16次真实FFT调用，而u/rowLN为0。合成通过并非真实A100结果。

新协议只引用不可变V2父协议及其输入index，另外绑定全部新旧源码；每路新执行仅为未完成block和三个冻结控制位置：原执行顺序索引0、1及父最后一位置。控制覆盖两来源、两适用step、早/晚层，输出独立CONTROL，不覆盖父记录。每个控制对父记录的全部键、整数、布尔、null、浮点及wrapped phase严格比较（相对1e-7、绝对1e-9、相位1e-8）；低能量仅标记，任何越阈值仍失败。全部三个控制通过后才算剩余位置。两路合计最后需640唯一source×family×step×block记录。

原生部分沿用原始BF16 W2与已存在rank960两GEMM factors、TF32关闭、M1092、原偏置SHA、完整packet SHA及9个原始typed张量、u逐字节中性、先乘后加写回、FP64 gY/rowLN，预测和prefix输入统计定义不变。只把NumPy重复的ROI/频带缩减换为同公式CPU4分块Torch FP64归约；直接形成小差值后算交叉项，避免Gram相消。每位置仍3原W2 GEMM、8factor GEMM、7Y回传/写回、10gY、10rowLN、6view+6bridge、32FFT，控制亦完整计账。

新每路GPU1、CPU4、GPU24GiB、host可用64GiB、3600秒/外层3900秒、HF新增≤512MiB，当前会话合计≤2卡。没有W1、全模型、拟合、视频或官方评分。合并记录保留V2原前缀+V3新后缀，并标记不同统计实现与控制证据；原V2不是完整通过。实际CPU预检/真实三个控制/完整闭账/资源及EXIT仍为启动后验收项，任一步失败即停、不自动重试。
