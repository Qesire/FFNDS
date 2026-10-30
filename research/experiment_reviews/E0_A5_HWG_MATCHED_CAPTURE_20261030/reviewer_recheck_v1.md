# 独立最终复核：匹配采集与文件验证

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。**APPROVE 四个精确 native 采集范围与其各自单独的 VERIFY 范围进入登记**，条件为调度前实际身份与资源复核、producer真实EXIT0后才启动对应VERIFY。此复核不是 ACTIVE、RUN、质量通过或E0/E1/E2授权。已复核代码提交 `f1d7cdf25213b3ca06fbb894133dee4fd7d5a515`；审查收据另作为文件绑定，不把后续登记用的当前HEAD与这个code commit混淆。

首审指出M64几何、QDQ边界、十状态neutral、全字段预算及同host准入；中间精确审查另指出输出/读取事前扣账、站点与文件唯一、固定独立schema、checkpoint和真实导入身份、bootstrap失败与观察器额外计算、输出哈希以外的video decoder读取。作者多轮回应与代码分别落实这些要求；未经批准的中间版本未成为此次执行对象。decoder内读取量及原生模型/cache loader内部读取量明列UNKNOWN，不合并为已计量的output SHA或身份验证read bytes；视频几何检查每视频固定一次reader、一次metadata和一次count_frames，60秒子期限纳入原总wall并在前后记账。无法核验的物理I/O不作总量主张。

本次重算4份protocol及其各1018项源码绑定，与磁盘SHA一致；使用真实私有profile重建并逐项比对全部8条完整argv与job/protocol/fingerprint、输出位置及资源上限，未在公共记录暴露私有worker身份。独立复跑12项采集与9项M1小软件fixture，共21项PASS；覆盖原始M64、partial文件和读账本UNKNOWN、站点/路径/字段/shape/dtype/native-check反例、文件损坏及decoder期限恢复。这些是软件检查，不证明模型成功。M0报告中的既有cache和镜像SHA、三个worker package文件核验属于独立既有资产检查；本review不重读56GB缓存或原模型，实际producer仍逐文件复核模型、cache、package及已绑定的实际import origin。

每来源严格为plain和observer两条Dense seed42视频；每阶段1600 native linear与1600 QDQ、cache miss0，observer 160站点按原生Q1后实际full-M V1及21个全球对齐M64/τ保存，15项sample字段、精确QDQ/W2/原生checks、同源10状态和完整视频位相等是待实际运行的门。额外观察GELU、index_select、D2H、buffer和memory hash成本在合同单列；采集wall不是FFN sparse speed。每τ只有一个tile，不能代表全部空间或稀有目标；post-Q1 V1不包含完整pre-Q1输入/scale。现存A5参数producer缺失，后续仅可称A5-inspired adapter，重放和方法对照需另审。

每个producer GPU1/CPU4、≤76GiB reserved、≤320GiB RSS、≤128GiB输出、14400s及单次无重试；会话≤4 GPU、同host最多2模型，flock准入将其他模型未消费160GiB预测和128GiB余量纳入，实际余量/资源监测仍可失败。独立VERIFY为GPU0/CPU1，payload读取≤160GiB、metadata≤256MiB、RSS≤512MiB、7200s；producer必须实际EXIT0并绑定RUN/protocol/spec，验证固定schema、位点/路径/inode唯一、文件及raw SHA，并从full V1重提21个tile对sample哈希。显式read/output预扣与inflight UNKNOWN在失败时保留；任一中性、资源或完整性门失败均不能转称成功，不自动重跑。4个暴露来源和单seed只支持开发资产证据；没有科学mask质量、物理skip或独立视频质量结论。
