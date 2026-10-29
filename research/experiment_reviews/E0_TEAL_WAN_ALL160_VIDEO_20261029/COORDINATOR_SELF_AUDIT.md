# 协调者自审：冻结六源门限的全160自由视频诊断

协调者 `/root`；实现者 `/root/new12_raft_metric_build`；独立复审者 `/root/teal_probe_preexec_reviewer`。四个确切part均已真实独立初审REVISE、作者逐项回应和最终recheck APPROVE，以下只确认注册和执行责任。

**判别问题。** 六源校准门限在两开发来源所有160处局部对照中优于等k随机及静态通道，但完整学生视频还未知。H1预测25%/50%门限下人物动作、潜水运动、晚进入画面的主体和水面变化能随配对Dense保留；H2预测微弱定向特征反复丢失、状态分布漂移导致运动衰减或动作改变。局部SSE不选视频结果、后续自由视频质量不由接触表或单一光流代替。四个来源1040/979/456/923均已暴露开发，不作为盲确认或20源部署验收。

**真实动作。** 原BF16 W1/GELU之后、原W2之前按已闭合CPU六校准来源固定门限置零，原完整W2+b2及当前gate/residual/后缀不变。全4步×40层均执行，两个专家按原high/high/low/low。保留全量Dense W2，明确QUALITY_DIAGNOSTIC_NO_PHYSICAL_SAVING，既不减少W1，也不证明稀疏W2速度。所有完整h的mask/reduction/sync成本均在原生执行内；不能将本scope计时当实现稀疏后的性能。

**匹配与控制。** 每part先独立old/new Dense runtime生成同来源的两条完整81帧视频，全部10状态与视频字节中性通过才继续；研究来源重新生成fresh Dense/threshold25/threshold50，三臂相同原checkpoint、recipe、seed42和raw/initial noise。每part5视频、50状态、320候选mask site；四part总20视频、200状态、1280候选mask site。阈值来源、所有160处实际删除/tie/原零数量、完整生成收据与对照绑定均由闭合校验器检查。缺事件、重复site、错误expert/阈值或噪声差异不得闭合。

**修订与范围。** 接受四项独立意见：每part隐藏CUDA的有限byte-only CPU preflight及CPU/GPU两个实际私有spec；精确CPU/GPU映射与会话并发；whole-attempt 1GiB和原子/外部收据空间；nested W2与完整mode/phase/action closure静态测试。最终四protocol/fingerprint及八spec SHA以作者回应和独立recheck表为准。根协调者不制造投票，分别为各part登记机器review和ACTIVE。

**当前实际调度。** 用户最新直接限制为最多4卡。当前目标端实际可见4张A100、CPU0–15，最近只读观察均空闲；Chipmunk单组probe随后已启动GPU0/CPU0–3。part1/2/3绑定GPU1/2/3及CPU4–7/8–11/12–15，part0在GPU0 holder退出后才能start。CPU preflight与GPU各有真实fresh durable receipt，GPU不能越过preflight成功证明。每次启动前核共享host空闲内存、磁盘、实际GPU进程与锁；空卡不足则保持未启动，已启动失败保留且不自动重试。40GiB GPU与3600/3900秒为限制，不是已测保证；128GiB最低host可用内存会在运行内再次检查。原协议、结果及已完成失败身份保持不变。
