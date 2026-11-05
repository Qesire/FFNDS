# E4a 精确实验合同 v0（仅待审，不授权运行）

输入仅E1a 02 seed43已独立接受的四个完整field：0/high/0/conditional、10/high/13/unconditional、20/low/26/conditional、30/low/39/unconditional；逐字段绑定原scope、START/EXIT、实际field SHA、原12 shard和对应原A manifest SHA。四field均是anchor状态，不构成同层跨步配对；hit叫anchor-input replay，只分析执行耗时，不评估时间预测/视频质量。

原F32 W1/W2/b1/b2按原native转换为BF16，weight转置布局；实际W1和W2使用torch.empty输出+torch.addmm(out=)，中间tanh-GELU。计时外全M检查重算Dense相对保存F、BF16 u@A相对保存Au（shape/dtype、逐元素精确数值相等和绝对/相对误差；结果字段bit_mismatch是逐元素不相等数量，不能区分正负零，不应解释为字节按位比较），若无法保证native一致性fail closed。A原完整矩阵，不拟合/低秩。

每site独立比较R0(FP32 C)、R16(BF16 C)、R8（每token连续128通道、INT8、FP32 scale、round-to-even/clip）、S（2×2×2真实边界、中心化正交常数系数BF16，真实token加权DC补偿与mean）。host pinned预分配/复用为单site配置，目标与来源是预分配页锁缓冲，不等于全层48GiB页锁可行；另测GPU单site。anchor包括原生Dense W1+GELU+W2、BF16 XA、C形成、编码与完整设备目标写入；hit包括BF16 XA、cache读取/H2D、解码、相加与native BF16 output，host路径包含同步D2H/H2D，不计假想的pageable→pinned staging、预取或overlap。pinned分配/建立及权重/field载入独立计setup。

每现场单次原生Dense D、实际anchor B、hit H、XA仅乘均计时。每site/格式/驻留条件至少5 warmup+20记录，预定顺序分轮交错（包含基线），CUDA event和两端同步wall，报告median/p90、计数和GPU allocated/reserved峰。计时窗内不得hash/写盘/正确性抽样；输出对象至少存活到事件/stream同步。S full-C先构造再pool是参考；额外标注并比较先pool输入/输出再计算A的重排，计时和数值误差分开，BF16批量与舍入不可称等价。报告L=5、10的B+(L-1)H vs L D，D>H时L>1+(B-D)/(D-H)，E=B-D为负、权重载入条件不同或不同采样时间均单列；只是局部组合估算。

每次一个source的一个site及预分配缓存，报告各格式实际payload bytes、peak临时显存、当前site配置和按40层×2CFG的理论容量；GPU单site可驻留不代表40层可驻留。设备转移、初始化、校验和计算身份由实际收据体现；不以此宣称FFN block或视频加速。

资源：1×A100、4CPU、host≤192GiB、1小时wall（soft保留缓冲）、输出有限并记录峰值；固定anode22，不自动更换节点；可与唯一旁作业137957并行，但其必须处于RUNNING或COMPLETING且scontrol确认单节点anode21、1 GPU。允许旁作业已终态并消失；任何其他活跃或等待作业拒绝。启动及每site前后保存squeue和scontrol证据，全局最多2作业2 GPU；共享存储影响不能排除，输入哈希与加载不计入计时窗。仅root dispatcher提交。身份/资源/Slurm严格核验，逐scope作者自审、独立review、recheck、ACTIVE、启动身份复核；失败留存不自动重试。本文件仅计划，不生成批准或ACTIVE。
