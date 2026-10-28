# 从固定范数补偿转向全80函数的原始 BF16 数据与强基线

**决策：下一次原生范围优先采齐 raw 全160位置的多来源操作数，随后建设“保留完整 raw W1、替换全部80 W2”的强基线；暂停单末层调参。** 新8flow RUN `783e22952f70ab5a8cb0563587f098a674833113b16d31ec65d0acf8ccd72b2c` 已实际exit0：固定step补偿使source12同状态head SSE下降93.67%/82.45%，自身状态step3下降69.59%；剩余Dense传播项0.000161674已大于新同状态局部项0.000069585。它支持保持归一化自由度这条机制，同时把主要研究问题推进到更深层与状态传播。

| 路线 | 对全模型目标的价值 | 当前真正缺口 |
|---|---|---|
| 完整raw W1＋全80 W2低rank＋有限norm-aware recovery | 最直接形成覆盖80函数的质量/执行强基线；避开重新学习GELU特征的混杂 | raw跨深度 h/Y/r/u、源级谱/风险和gain稳定性；其后仍须full80自由视频及匹配驻留测速 |
| B-H65直接学生 | 可研究绕过完整W1的更便宜函数，但固定65充分统计目前只适用于末层读出 | raw学生训练数据缺失；直接扩到前39层没有相同64维固定suffix。只替换两个末层函数最多覆盖1/40的FFN调用，作为单独主线难以验证全80目标 |
| 只采少数深度 | 能先观察不同层的误差；适合极严磁盘预算 | 同样要运行完整Dense前缀/flow，却反复留下未采位置；本轮前向远贵于复制256行，采齐160位置更可复用 |

**最强竞争解释**：末层有固定低维head，使“保方向＋补范数”有效；中层方向仍承担后续attention与FFN，可能在有实际MAC收益的rank预算下根本不够低秩。另一种可能是大部分方向确实可压缩，主要失败来自局部目标/归一化与自身状态失配。先量到同一raw路径的源级谱、gated Y风险及postwrite范数/交叉项，才能区分表示容量与恢复目标，而非默认增加rank或学习率。

## Calibration V2 现有数据检查（已执行纯CPU metadata）

已运行 `ffnds_calibration_design.py coverage`：当前两个raw来源都保留DEVELOPMENT身份，raw CALIBRATION来源为0，各既有来源配额arm均为 `AWAITING_NEW_STRATIFIED_SOURCES`。该工具只证明来源角色覆盖，不证明操作数足够。实际raw V1/Y/r/u行仅覆盖low/block39的step2/3，即2/160 step-sites；连这两个位置也未保存原生GELU后 h。旧HiF4/CAL24不补这158个raw缺口。`diagnose`所需真实raw group/signed统计缺失，按 `NOT_MEASURED` 拒绝，不制作虚拟group数据。当前bank、coverage与错误身份留在 `.runtime/raw_bf16_scope_coverage_20261028/`，没有模型/拟合调用。

## 建议下一固定原生scope：六新raw校准来源＋两旧开发来源，采齐160位置

- 使用已验证consumer20图像簇；排除旧raw0/12、formal20和diverse_confirmation20的图像SHA后，每个subject/background层先按图像SHA排序，用**同一个**Python `Random(42)` 依次置换subject、background，各取3个校准来源。最终helper得到subject组25/27/24、background组36/30/38；这替代此前未冻结的每层重置seed提议，源池/排序规则收口未使用任何新tensor或效果。旧raw0/12留作development transfer；consumer20 index用于定位旧receipt、native source ordinal用于新row_draw，八个图像簇真实去重。所有来源已有HiF4开发曝光，新raw数据角色不改称blind确认。
- 每个source、四步、40层采256个冻结同坐标行：真实pre-W1 V1、GELU后的W2输入h、原W2输出Y、FFN写回前r和写回后u，另存实际gate/modulation及dtype/stride/raw SHA。h是将来完整W1基线合法在线已有信息；Y/u仅为当前付费Dense监督，不进入学生部署。新draw按native source ordinal固定，版本与旧末层draw分开登记。
- 对旧0/12重放完整四步，全部8个flow/post逐字节核对既有raw资产，为采集接缝做neutral，不再decode这两条视频。六个新校准来源各生成一条完整原始BF16 Dense视频，同时采集全部160位置。原noise/encoder/scheduler/offload路径保留；各独立part先通过自己的旧dev四步neutral，再处理该part3个校准来源，无跨part barrier；任一part失败保留PARTIAL范围。无拟合、分解、优化、学生或正式评分；六条新视频各保留纯CPU的81帧CORE报告。
- 合计32完整flow、1280 FFN、2560原FFN linear、6条新视频；8次T5/VAE encode、6次VAE decode、8次原noise draw/prepare。可分两part，各4来源（一个旧neutral＋3新来源）、单张空闲A100/CPU4/≤32GiB/wall1200s/outer1500s，同时最多2卡；所有原始预取仍付费。V1/h/Y/r/u的BF16行payload合计 **22481469440B**，加32flow、原始states/video和manifest，建议总HF≤24GiB、每part≤12GiB。确切调用/磁盘预算由producer冻结前核算。

这不是沿用24×256经验量：六个校准源只是一轮可检验的跨来源pilot，每个物理W2共享的两个step有3072训练行，仍只有6个独立图像簇。两开发来源不能用于反复择优或人口总体结论；源级不稳定时优先补来源，而非把行数当独立样本。

**采集后的固定决策**：先用同一train公共统计、source等权及两个step等权，报告各函数raw响应谱、跨source风险和最弱层；按一个事先冻结的整体W2 MAC预算选择80个rank，禁止根据两个开发视频反复扫rank。例：W2总MAC保留50%的预算，对统一rank的物理上界为1868.108（尚非应选rank），而非硬设512；实际rank分配及norm-aware恢复目标由新增cal数据决定，再冻结唯一full80候选和原版Dense对照。中层恢复保留原AdaLN/gate/residual，在原生gated Y目标外显式检查postwrite u的范数及有符号交叉项，末层的1/κ操作仍只保留其已证适用位置。若谱/留源风险已显示该预算下方向容量明显不足，明确返回这一表示族，不先支付80函数训练。若局部风险可控，下一次部署就直接比较完整raw四步视频、官方分轴质量和同驻留端到端成本，并补学生状态同输入teacher。

未查新论文、未改变G14索引/ACTIVE/冻结源码，未执行新模型或GPU任务。本页是下一producer的范围建议，不是运行注册。
