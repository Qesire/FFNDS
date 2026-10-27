# disjoint20 提示词运动语义：评分前描述性分层

**对象与时间身份**：以下逐条只读取冻结输入文本，未读取本轮新 candidate 视频或新指标；既有 disjoint20 已属 DEVELOPMENT 暴露来源。20个 source 各一个提示词；这不是正式标签或新确认集，也不改变冻结实验、主终点及阈值。输入 canonical SHA256 `656f85c0415f407f0e07972fd1e756e2559062ae2dd7876741ef0ceb6e4b8392`，所读冻结协议 SHA256 `ad81fffb73256c14e65c4221d255080072bcc9902cf420b541e4da4f61254ca6`；20条 prompt SHA 与协议相符。

分类采用提示词中**明确的主体动作**；`sits`、`parked`描述姿态/状态，不能保证镜头或整个场景静止。**明确静态**要求提示语明确约束整个视频或镜头静止，语义上有资格再核验，不等于已证实静态视频。未提动作不代表画面真实无运动；background 类别也不代表静态。各列是独立语义维度。

| index / source_group | 原始 prompt | 明确主体动作及依据 | 明确相机运动 | 明确整体静态 | 语义归类 |
|---|---|---|---|---|---|
| 00 / i2v_subject:22 | a brown and white cow eating hay | 是：eating | 无 | 无 | 动作明确 |
| 01 / i2v_subject:23 | a seal swimming in the water | 是：swimming | 无 | 无 | 动作明确 |
| 02 / i2v_subject:24 | a panda bear is eating a piece of bamboo | 是：eating | 无 | 无 | 动作明确 |
| 03 / i2v_subject:25 | a small bird sits on a moss covered branch | 无；sits 表姿态 | 无 | 无 | 动作未明确 |
| 04 / i2v_subject:26 | a bird with a fish in its beak flying over a field | 是：flying | 无 | 无 | 动作明确 |
| 05 / i2v_subject:27 | a large flock of birds flying in the sky | 是：flying | 无 | 无 | 动作明确 |
| 06 / i2v_subject:28 | a bald eagle flying over a tree filled forest | 是：flying | 无 | 无 | 动作明确 |
| 07 / i2v_subject:29 | a giraffe walking in a field | 是：walking | 无 | 无 | 动作明确 |
| 08 / i2v_subject:30 | a lioness yawning in a field | 是：yawning | 无 | 无 | 动作明确 |
| 09 / i2v_subject:31 | a little crab scurried on the sandy beach | 是：scurried | 无 | 无 | 动作明确 |
| 10 / i2v_background:30 | a stage door painted with a star on the side of a brick wall | 无；painted描述外观 | 无 | 无 | 动作未明确 |
| 11 / i2v_background:31 | a light house on the edge of the water | 无 | 无 | 无 | 动作未明确 |
| 12 / i2v_background:32 | an asian city street at night with people and bicycles | 无；people and bicycles未指定动作 | 无 | 无 | 动作未明确 |
| 13 / i2v_background:33 | a couple of wooden benches in the middle of a street | 无 | 无 | 无 | 动作未明确 |
| 14 / i2v_background:34 | a pagoda sits on top of a mountain in japan | 无；sits表位置 | 无 | 无 | 动作未明确 |
| 15 / i2v_background:35 | a red bus driving down a snowy street at night | 是：driving | 无 | 无 | 动作明确 |
| 16 / i2v_background:36 | a snow covered street | 无；snow covered描述状态 | 无 | 无 | 动作未明确 |
| 17 / i2v_background:37 | a house with snow on the ground | 无 | 无 | 无 | 动作未明确 |
| 18 / i2v_background:38 | cars parked on the side of the road during a snowstorm | 无；parked描述车辆状态，snowstorm不明确视频动态 | 无 | 无 | 动作未明确 |
| 19 / i2v_background:39 | a group of statues on the side of a building | 无 | 无 | 无 | 动作未明确 |

**分层汇总**：明确主体动作10/20（subject 9、background 1）；动作未明确10/20；明确相机运动0/20；明确整体静态0/20。没有来源仅凭提示词取得静态资格；若以后要报告静态类额外指标，应事先依据明示静态语义和完整实际视频双重核验，而非以背景类别或高 smoothness 替代。RA​​FT全图幅值还可能来自相机或背景，不能单独判定主体动作或提示遵循。

已有图像 SHA 去重只建立字节身份的区别；原图簇、语义近重复及相似场景尚未通过人工或可重现方法确认，不将其宣称为独立确认样本。上述规则和逐源表用于后续解释，不能依观察到的新分数反向重划类别或选优。
