# 原BF16固定因子完整时间成分复算：协调者审阅

ROOT协调者读完新驱动、原全80谱工具的只读W2 tensor loader、冻结两GEMM factor_forward、实际因子header/原b2/CPU证明与new10源producer、原BF16写回和时空几何核心。本页不是独立Agent支持票，Agent服务额度已用尽，独立审阅未进行。

有辨别力的比较是同一Dense h的 raw M168 W2、shared-r960、step-r960；原始M168矩阵对照单独扣1600 GEMM，分离捕获M31668到小M的舍入差异。每包21τ×8xy不做motion tracking；统计gY、原r下的BF16舍入u和FP64 rowLN(u)，分别报告全21、τ1..20和τ0的DC/AC误差、AC能量与对齐方向。高能量保持而AC方向翻转的合成反例已检验，不能把能量ratio等价为动作保真。

5项合成测试实际exit0，0真实张量、GPU或模型调用：相反相位/消失的AC、零DC/AC显式缺失值、1600包80物理W2与6400因子GEMM预算、先位置后source等权及缺失计数、纯导入没有NumPy/Torch副作用。逐AST核对直接模块调用均可解析。实际metadata-only source gate验证两native范围exit0、10新开发图、全部1600包及same checkpoint SHA，未预看本次真实数值结果。

预检在实际执行worker复用独立退出的8.73GB因子全SHA证明，并对当前stat/header及raw master四文件stat再核验；不重哈希69GB模型。GPU阶段每因子load再核三张量rawSHA，显式记录H2D；W2原矩阵80次读取和80次bias读取均按精确range/full-master-stat绑定。每包仍完整SHA、9个tensor rawSHA、168实际row、CPU重建h rawSHA及原生u逐字节中性检查，不混入Student prefix。

此范围不训练/拟合/分解，不调用W1或full model，不生成视频、不评分，也不测部署速度。1空闲A100、CPU4、GPU reserved≤24GiB、wall1800/outer2100、HF新增≤128MiB；CPU门180/outer240。任何错误保留扣账与退出，不自动重试。输出全160位置与来源尾部，为解释视频运动下降的竞争机制提供局部同输入证据。

精确绑定：

- config/e0_raw_bf16_full21tau_new10_fixed_factor_replay_20261029_draft.json SHA256 `719b58b81dd32e42a1358be825fa88521e89a3ea54dd75e6c226a0c2b7215b03`
- tools/e0_raw_bf16_full21tau_factor_replay.py SHA256 `d2388387cfd6c8433e467cee0f3f0e8a036addd9a5bb36e5173edec9dd6bfa85`
- tests/test_e0_raw_bf16_full21tau_factor_replay.py SHA256 `37ad25807d340bd3833daa7c14b171ade708523440849be17c2fe1ecd82fb706`
