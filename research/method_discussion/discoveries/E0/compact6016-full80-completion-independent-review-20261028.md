# A0 full80补全：独立训练账本审阅

**SRC/CPU合成**：审阅补全工具 `7975d1127df550c9aaabae49a3b535c6e81a3194597abde681ed9b67484a9c5c`、测试 `b99b20078d0392790b02672d87d3ca19378cfbaf95014237ebede369d3aee1`、草案配置 `09bf67275ab5253f93498d7377d4b3119b8886b33fddf417751a9c815b34a83d`；20项合成CPU测试exit0。原工具 `7dc280d8dbc65c14a585d4430f37e162a950b1e71c861148a9e1a9b7fe0b1b28`。草案未冻结；未运行真实模型、训练或GPU。混合资产assembler及loader授权链由另一独立审阅承担。

原三次超时父scope保持失败；20个完整high expert functions对应40个完整cell，以独立审核收据及原父RUN/exit和逐cell身份复用，不载入半完成参数或优化器。其余60个expert/block按确定顺序均分4 part各15函数、两初始化共每part30 cell、全scope120新run ID；旧20函数不重训。新part从24来源×该expert两个step×15函数读取720个原始操作数，每个case原256坐标；16来源训练、8来源开发持出，step嵌套且同expert/block共用一对权重。

随机支持在全80原函数顺序上用seed42连续生成80次，然后取新60函数对应结果；不是把随机序列从剩余函数重新开始。支持选择只用16训练来源×2 step，每函数32次原V2贡献、每次27组BMM。训练热路径与原full80对应230行逐行对照，唯一差异是结束状态标签；每cell8 epoch×16来源×2步=256 Adam更新，原student自己的W1/GELU/Smooth2/native Q2，不把teacher V2当student输出。

每part30 cell预算为原40 cell的3/4：10,620 student forwards/Q2、21,240 BF16 forward GEMM、7,680 Adam更新、480 support source-step、12,960 support BMM；预扣与观察均需在实际结果达到限额。无新teacher FFN、Q1、视频、全模型或官方质量调用。每part一GPU、2CPU、7200秒compute/7500秒outer、6GiB新增HF、32GiB GPU reserved门槛；全四part总24GiB，上述上限为协议而非实测占用。真实资源、父cell完整性、launcher退出和结果由对应收据验证；旧已完成cell不改变父失败状态。
