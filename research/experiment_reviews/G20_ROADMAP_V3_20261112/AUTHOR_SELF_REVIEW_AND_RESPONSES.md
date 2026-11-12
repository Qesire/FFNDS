# G20路线v3：作者自审与独立意见响应

2026-11-12。作者／协调者：`/root`。依据用户本轮要求维护方案、足量多语义图文对、资产树与顺序；没有发起模型、采集、拟合或视频实验。基点为 `1636c83a78fabd4dfd2f08611a6828c7386a366c`。

## 自审对象

当前旧CAL4/DEV5、有限配对行及六站点结果不能决定覆盖、容量、状态非线性、两级读出和真实视频传播中的哪一个主导。v3明确不同预测及最经济的可辨识对照；CAL96/DEV48/CONFIRM96属于前瞻规模，现有manifest真实绑定数为0。source是原始图像簇，seed/row/site/计时重复为嵌套观测。

主设计将来源广度与行密度对照、同bank线性/非线性及ICC/Chipmunk判别并列；原权重×状态归因独立推进；唯一部署候选经过全80站点与完整自由轨迹，最终在同Attention背景与实际最近邻比较。需要付费的native shadow、proxy、恢复、计时重复均进入有限预算。确认N和主合同在看确认结果前冻结。

## 已处理的独立意见

| 提出者 | 意见／反例 | 作者响应与实际修订 |
| --- | --- | --- |
| numeric_audit、scope_review | R1b未保存稳定数学D，直接复用D会依赖不存在的资产 | ACCEPT：L0改为已评分矩阵的实际预测差；插值恒等式与typed BF16部署分开 |
| numeric_audit、scope_review | GELU奇部等于线性，泛写h_odd会破坏非线性控制 | ACCEPT：固定tanh和CAL内尺度，端点/位移臂保留同数量线性与非线性特征 |
| numeric_audit、stats_audit | 2%点门槛、CI排除零和跨2%不确定混用 | ACCEPT：主筛选为源级CI上界<-2%，明确DEV仅用于开发资源筛选 |
| numeric_audit、scope_review | ICC stage-2占比不是可修复份额；交叉项可能抵消 | ACCEPT：补native/surrogate first×second四格及条件混合因子；native-h反事实只解释该具体修复 |
| numeric_audit | N的自由轨迹失败时只采Q1状态不能检验N分布偏移 | ACCEPT：仅一次CAL24的N自身学生采集与native target对照；新增N_STUDENT_CAL24资产 |
| numeric_audit | 幅值只设下界会放过过量或错误方向运动 | ACCEPT：双向幅值、方向、轨迹、覆盖与动作评分并列，冻结可测性规则 |
| scope_review | 按坐标排序后取前缀可能只覆盖早帧/局部区域 | ACCEPT：均匀随机抽签顺序先取前缀，访存排序保留逆映射 |
| scope_review | 局部d不能来自两条不同自由前缀的状态相减 | ACCEPT：T1明确共同Q1父轨迹、同U/residual、候选shadow anchor与cache |
| stats_audit | mean-relative与ratio-of-means可能异号 | ACCEPT：冻结源×expert的先平均绝对MSE再取比及分层聚合；其它口径单列 |
| stats_audit | 技术重复会混淆视频数和独立样本数 | ACCEPT：source×seed×arm默认恰好3次，median后log比；首重复质量，条件数/执行数/n分别记录 |
| scope_review、stats_audit | 36首批/192总捕获条件不包含plain/tap控制执行 | ACCEPT：明确C1=36条件/38执行，总192条件/194执行 |
| stats_audit | Attention改变背景后48条Q/N缺匹配Dense | ACCEPT：缺Dense24时另计24，总72；旧不同背景资产不能替代 |
| stats_audit | DEV、CONFIRM、旧诊断、N自由状态被同一资产ID混用 | ACCEPT：LEGACY_DIRECTION_DIAG、NDEV、DEV_SYSTEM_*与确认资产独立 |
| stats_audit | S2先需要结果审计链而结果又依赖S2，形成循环 | ACCEPT：REVIEW_CHAIN仅事后索引；真实preexec gate由逐scope ACTIVE记录绑定 |
| stats_audit | registry中的CAL/DEV匹配行数有复制残留 | ACCEPT：对照旧原始逐源JSON，CAL517/523、DEV641/637，各自范围与名单分别保存 |
| stats_audit、scope_review | 静止负对照缺失；若静止全排首位会混淆C12曲线 | ACCEPT：CAL12/DEV6/CONF12静止嵌入240，三套CAL前缀静止数2/3/6/12，DEV12前缀2；真实输入与容差在输出前冻结 |
| scope_review、stats_audit | DEV4/8子题方差不覆盖确认8/8，单终点power不等于联合power | ACCEPT：保留规划近似身份、保守SD与敏感性、联合保护power、运动有效n和静止稀有事件界；扩N须结果前冻结 |

## 实际评议分工

`numeric_audit`独立审主稿数学／机制；`stats_audit`独立审主稿、资产树与队列；`scope_review`独立审主稿及由stats_audit初写、root修订的来源规格／manifest。资产树由scope_review初写，因此其独立批准来自stats_audit；cohort由stats_audit初写，因此其独立批准来自scope_review。没有作者自批自己的文件。

初审记录、实际返修和最终输入SHA分别保存在本目录。最终三份复核记录只批准静态设计；后续每个具体scope的源码、真实输入、模型、指标实现、资源与GPU上限仍需真实preexecution review/recheck。设计批准不生成RUN或赋予运行许可。
