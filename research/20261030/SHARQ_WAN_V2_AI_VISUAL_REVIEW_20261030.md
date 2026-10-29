# SharQ Wan V2：两来源三臂全帧 AI 接触表观察

独立观察者 `/root/teal_full_llm_calibration_build_terra`。两项 visual VERIFY 均实际 EXIT0，inspect 均返回 `PASS_ACTUAL_EXIT_SHARQ_QUALITY_CLOSED`；part0 VERIFY RUN SHA `185eb540e47bb19a8f35113aa1610bfdf613f74bca38bb7660744574ae292fa2`，part1 `aa48f95a84f0469c99859da0df65dd7ef19eccfc2a66de3230e3a55432f7614f`。远端绑定 JPEG 原样复制至私有 HF derived 目录，12/12 本地大小和 SHA 与 visual RUN 页面绑定一致。逐页查看全部 12 页；每臂 page0 帧 0–41，page1 帧 42–80，6 个视频共 486 帧。

**1040，双潜水员。** 三臂都保留两名潜水员、蓝色水体及向前游动和摆腿动作，未见明显主体消失。BF16 后半段前景潜水员手臂由下垂转向前伸，Dense32 后半前景肘臂姿势和与后景人物的遮挡关系不同；SharQ 的前景双臂靠近身体后再伸出，后景手臂位置亦不同，约帧 48–80 的姿势差异可见。水体颜色与表面光线大体相近。小图中手指和装备细节不足以定论；遮挡与生成轨迹分叉是动作姿势差异的强替代解释，不能据此断言某臂更准确。

**979，人物与乐器。** 三臂保持同一人物、木桌、植物和右侧粉色弦乐器，主要动作是双手掩面后放下并露出笑容。BF16 约帧 60 后露出脸，末段一手仍靠近脸；Dense32 约帧 60–80 更长时间维持手靠近嘴脸；SharQ 约帧 63–72 双手向下移并在末段靠近膝部，面部微笑较早显露。乐器基本静置，桌面与背景整体保持；接触表不能可靠判别手指解剖、精细面部瑕疵或闪烁。三条同 seed 自由生成轨迹在动作时序上分叉是最强替代解释，不能把更早放手自动视为质量改善或归因于稀疏。

这是 AI 的接触表观察，图格是缩小的 JPEG，不是完整视频播放、人工验收、逐像素相等或运动评分；不能推断不存在细小伪影，也不替代八轴与 RAFT。两个来源均为已暴露开发来源、每来源单 seed，不能外推 heldout。下表 SHA 同时是远端绑定和本地复制身份。

| 来源 | 臂 | 页面 | 帧 | SHA256 |
|---|---|---|---|---|
| 1040 | dense | source1040_dense_page0.jpg | 0–41 | `5c3640d900d6816166bb2cb723ba5535eda6b0c04271e88effbaf5681fdaa5b3` |
| 1040 | dense | source1040_dense_page1.jpg | 42–80 | `c30ff3949be7ad2c1350e8d9a215d5d43b02479b8ecf044db55882bf83233fcb` |
| 1040 | dense32 | source1040_dense32_page0.jpg | 0–41 | `1d1034c45ef984c9bf1e8bd9de325731d481f979925b124b048870f5bfeba411` |
| 1040 | dense32 | source1040_dense32_page1.jpg | 42–80 | `c2598e44c64f1e72b18e5c647ab1e68136641f8adf626c857d40fe4c05d7d42a` |
| 1040 | sharq | source1040_sharq_page0.jpg | 0–41 | `aafa22a60322b06cabd03f946eae000efc61f8724b3d5036bc701092d3f8926c` |
| 1040 | sharq | source1040_sharq_page1.jpg | 42–80 | `97980f0eaa6c44950e717b584107c456314e49d7ac55af5d09873f0594ab9c4e` |
| 979 | dense | source979_dense_page0.jpg | 0–41 | `c2ea12b50bfe63f53b548e72a8dd39dbac018a53291bcd11393d4114a6e88d44` |
| 979 | dense | source979_dense_page1.jpg | 42–80 | `4bfb69edf102062f7eaf9957753f5dba641d3cf6d5f409572d430dcfaeabc6d7` |
| 979 | dense32 | source979_dense32_page0.jpg | 0–41 | `bfdcdc4d1f79b02b7d9124320ae992c815e96a147f1cbe6d29a4fae6611a10cb` |
| 979 | dense32 | source979_dense32_page1.jpg | 42–80 | `c3401588a96affc5ed3c723720df17326b597dabfa31c60575f303ea9ec3cde1` |
| 979 | sharq | source979_sharq_page0.jpg | 0–41 | `d6bb5858c6596be43eb9d119ffdae9dd00b504351abcb82d39f96d516d6400f1` |
| 979 | sharq | source979_sharq_page1.jpg | 42–80 | `b0f73d91ede7d4b3cd2113dba7ba6281469167ad2b578b49b29b29743e761445` |
