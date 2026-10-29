# S2四角原生读出：独立建议整合与冻结前协调审阅

2026-10-29 · coordinator `/root`。输入为两个已完成spatial52三臂producer；本审阅不是新视频质量或部署结论。独立研究及实现修正复核见 [真实子Agent产物](s2-theory-subagent-s2_theory_design_independent-20261029.md)。该Agent先独立推导后才读取协调者实现，未参与本代码编写；没有将其标签记作PEER_SUPPORTED。

## 现象、预测与真实动作

十图训练AC改善7.73%，二十开发图value10/dcac10真实视频RAFT却为Dense0.451/0.439。H1：Dense同状态的新区域局部函数失真已较大；H2：学生prefix差及状态依赖交互占主要数值幅度；H3：真实空间运动对应改变可预测性；H4：gate、残差与后缀消费者改变误差影响。此次四角reader直接判别H1/H2的固定状态预测和H4的gate/typed数值部分，H3及attention/视频因果作用另采。两个极端曝光来源是链路与机制pilot，不作人口确认。

每source/site真实 h_D、h_S，算原W2 D(h_D)、D(h_S)，算各固定候选 Q(h_D)、Q(h_S)，原bias逐tensor SHA相同；没有新的W1/模型flow/拟合/分解/视频。T/S/Cs/Ct各用所属state的真实gate/r，再分别计算实数gY、原BF16乘加U、FP64 rowLN(U)。TOTAL=学生local+prefix=L+P+I，全部signed cross保留。此分解的I包含真实reader舍入；gate不同时gY的交互也包含gate差，U的prefix包含残差差。P不是prefix因果效应。

## 独立意见逐条处置

1. **四角与数值锚**：采集原Y/U不冒充小M。所有packet九tensor full SHA及1092坐标复核，每一位置三原生U先逐字节复原后才分析。M1092读出与M31668真实Y差另测；原生gY/U/LN总差等于小M差加两个有符号shape bridge，total/DC/AC/三频带及交叉全部保存。typed四角恒等式不假设BF16线性。
2. **频点相消**：独立审阅指出band复交叉汇总相位可隐藏相反频点；增加逐bin结果、显式band聚合定义与低相干相位null。±π/2双频相消和单频反向的合成反例实际通过。对全部ROI参考AC≤1e-12的频带比值null；τ0单列，频谱是有限短窗口的描述，不推断物体运动方向。
3. **ROI身份及缺变量**：高flow/central/far都是几何分层，非主体mask或latent运动对应。preGELU a和attention/QKV消费者未采，明确不以rowLN替代消费者；必要后续扩大新source、遮挡/主体ROI及正常后缀干预。此次不在线搬运Dense未来信息。
4. **依赖和权限**：新上游独立分派指令到达前pilot已冻结并运行，保留该事实及原协议。现新reader先实际dispatch独立Agent并整合其意见后才冻结。原V1两队列在模型启动前因外来占卡取消，exit−15、0模型调用；新executor全69GB原master SHA/两header CPU proof与portable V2来源全部绑定。

## 有限范围与控制

两独立物理expert分片各一张空闲A100，保持实际两卡session。每part两来源×两step×40block=160个匹配位置，480个真实packet/4320 tensor SHA/480原U字节neutral。每part40次W2+40bias原checkpoint读出，80个固定factor文件/240tensor SHA，480原W2 GEMM+1280 factor GEMM；1120个预测Y D2H，1120次typed postwrite，1600个gY及1600个rowLN，5120次实际rFFT，320来源位置结果。所有真实调用事前扣账并保存失败；没有训练、model flow、新视频或官方评分。

每part GPU reserved≤24GiB、host available≥64GiB、CPU4、wall7200/outer7500s，新增HF≤0.5GiB；输入packet完整SHA流允许40GiB，预计33.51GiB，避免历史32GiB读取器导致错误提前停止。80factor复用portable采集的完整SHA/稳定stat CPU证明，实际加载再查每tensor SHA；全69GB master仅复用同executor已证明stat与header，再对真实W2 range与loaded bytes做SHA。新CPU metadata门不读大张量，不导入Torch/NumPy。失败不自动重试，不覆盖历史RUN。

协调者在项目.venv已实际跑过13项新reader/数学软件控制；独立Agent也用.venv实际复跑13项通过。原capture/proof的7项另已通过。冻结前静态计数逐项核对，频点/bridge新增均已计入。真实新数值结果仍等待新RUN/EXIT。
