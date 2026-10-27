# UB1 disjoint20 独立视觉复核

**VIDEO-OBS：20/20 来源、180/180 联系表页已逐页查看，每来源帧0–80全覆盖。** JSON列出每页HF相对路径、SHA256、帧范围与查看状态。未读取质量分数，按source序号顺序查看。

三列依次为Dense、历史UB1、fixed-RR consumer；缩略图每列256像素，原始帧832×464。来源×seed为观察单位；这些来源和seed42已暴露，属于非盲开发集。Dense与consumer初始噪声有字节配对记录；历史UB1未保存初始噪声，不能声称三路噪声字节一致。完整联系表体现逐帧视觉证据，不等同于16fps原尺寸连续播放、精确光流或人工验收。

在显式动作提示下，source01、02、04、05、06、08及09可见Dense与另外两路的运动路径或幅度差异；这些具体时段值得检查动作语义与自然性。静态提示下新增人或车（source10、18）是生成分支差异，不能据此判定无该对象的候选不好。保持主体形态和静态背景与保持运动是不同观察；本次未从缩略图断言不可接受的破坏或总体质量通过。

| 来源 | 实际提示词 | 观察（帧号从0计） |
|---|---|---|
| 00 | a brown and white cow eating hay | 小牛：三路形态和背景相近；帧36–80 Dense头部较低，consumer较平，历史UB1介于其间。 |
| 01 | a seal swimming in the water | 水中海豹：帧30–53 Dense转正面并收拢鳍；UB1和consumer至80帧仍偏侧面、鳍张开。 |
| 02 | a panda bear is eating a piece of bamboo | 熊猫：帧9–35 Dense移动竹竿从横向到近竖向送入口，UB1和consumer维持较横向；72–80仍有差异。 |
| 03 | a small bird sits on a moss covered branch | 小鸟：帧60–75 Dense逐渐转向镜头，UB1和consumer保持侧向；苔藓与背景大体相似。 |
| 04 | a bird with a fish in its beak flying over a field | 大鸟：Dense帧0–45翅膀下拍再抬起；consumer至80仍近初始高举，UB1后期有较小且模糊的变化。 |
| 05 | a large flock of birds flying in the sky | 鸟群：Dense从帧0起鸟形和群体排列随帧变化，UB1与consumer在联系表中整体变化较小；小目标细节须原尺寸复核。 |
| 06 | a bald eagle flying over a tree filled forest | 鹰：Dense帧0–35翅从高举转至近水平、后续再抬；历史UB1与consumer至80变化较小，黄树林背景大体保留。 |
| 07 | a giraffe walking in a field | 长颈鹿：三路主体与背景相近；帧20–35历史UB1右前腿伸出，Dense和consumer步态不同，Dense帧54–65尾部明显摆动。 |
| 08 | a lioness yawning in a field | 母狮：Dense帧27–52逐渐闭口，帧53–80保持闭口；UB1和consumer帧80仍张口，背景相近。 |
| 09 | a little crab scurried on the sandy beach | 沙地小蟹：Dense帧30–80腿部及阴影相对位置逐渐变化，UB1/consumer在缩略图中更接近初始姿态；目标较小，动作须原尺寸复核。 |
| 10 | a stage door painted with a star on the side of a brick wall | 舞台门：Dense帧13起人从右侧进入，帧35–80站在门前；UB1与consumer至帧80未见人，砖墙和字样仍保留。该生成分支差异不单独证明提示遵循或因果。 |
| 11 | a light house on the edge of the water | 灯塔：三路灯塔、岩石及海面色调相近；Dense帧30–80左下海浪边缘变化可见，另两路缩略图中变化较小，波纹细节需原尺寸播放。 |
| 12 | an asian city street at night with people and bicycles | 夜间街巷：Dense帧约29起左侧新行人进入，中心行人也向前变化；UB1/consumer行人位置变化较小，灯牌和自行车保留。 |
| 13 | a couple of wooden benches in the middle of a street | 雪夜长椅：三路布局和灯色相近，帧0–80联系表未见明显主体运动差异；落雪及灯光闪烁无法由缩略图可靠判断。 |
| 14 | a pagoda sits on top of a mountain in japan | 远景塔与山：三路轮廓、树冠和山色相近；帧0–80未见显著变化，细微背景动态须原尺寸播放。 |
| 15 | a red bus driving down a snowy street at night | 雪夜街道：Dense帧0–80公交车和旁车逐渐靠近镜头；UB1公交车缓慢靠近，consumer相对位置变化更小。路灯与雪地整体保留。 |
| 16 | a snow covered street | 积雪街道：三路停车车辆、建筑、树木和道路布局相近，帧0–80未见明显主体变化；细微雪或光照变化无法在缩略图判断。 |
| 17 | a house with snow on the ground | 雪中住宅：房屋、门廊、砖柱积雪三路相近；帧0–80未见明显主体运动差异，落雪细节需要播放检查。 |
| 18 | cars parked on the side of the road during a snowstorm | 雪夜街道：Dense约帧42起右下驶入一辆车并在帧54–80远去；另两路未出现该车，灯下背景大体保留。该路径分叉须结合提示判断。 |
| 19 | a group of statues on the side of a building | 石雕建筑：三路柱像、石材纹理及远山相近；帧0–80无明显主体运动差异，微细纹理及光照须原尺寸核查。 |

**真正分歧与下一观察**：同噪声Dense/consumer对照呈现路径分化，但不要求像素或Dense动作逐帧复制。OPEN解释包括运动幅度收缩和扩散轨迹分叉；当前单开发seed联系表不能确定哪一种、也不能定位某个FFN结构。建议人工原尺寸16fps连续观看source01帧30–80、02帧9–80、04帧0–80、08帧27–80、15帧0–80，分别判断游泳、吃竹、飞行、打哈欠及交通动态是否自然和符合各自提示；小蟹09、鸟群05另查小目标。静态源10、18重点观察新对象与光照的合理性而非要求复制。随后用独立source和多seed配对比较判别路径分叉是否稳定。人工完整播放验收未进行。
