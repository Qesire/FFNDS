# E0 · 六条已有视频全帧质量补评与运动测量纠正

新结果：同一 affine54 原型在两个来源上的时序异常不同。熊视频运动估计下降而小镇逐帧变化上升；高频能量并没有统一下降。局部输出 SSE 更低和高频更多，都不能推出自由视频质量合格。此轮只读已有 MP4，零 GPU、零新生成、零官方评分，未做方法选择。

## 实际数据与修正

按照新仓库政策，解码原三臂×两来源全部486帧、480相邻帧对（每视频81帧、16FPS、464×832）。[冻结协议](../../config/e0_existing_video_quality_backfill_20261026.json)、[机器结果](E0_EXISTING_VIDEO_QUALITY_BACKFILL.json)、[六份逐帧原报告与回执](video_quality_backfill_v2/README.md)绑定原视频、原RUN、来源图像/提示、seed、模型及参数包。老六视频和原报告均不改写。

修复了原光流接口的强尺度失真：Farneback改用四舍五入的8bit灰度，旧实现使用[0,1]灰度。192×192平移软件夹具的真实位移为(2,−1)像素，旧输入测得中位(0.00000273,−0.00000136)，修正后为(1.99987,−0.99994)。OpenCV的[原始源码](https://github.com/opencv/opencv/blob/4.13.0/modules/video/src/optflowgf.cpp)含固定行列式正则；这解释了输入尺度的影响，但不能把全流程变化当作固定缩放倍数。夹具仅验证软件，不是Wan实验。

新增双向一致性、运动补偿encoded-luma残差、参考低运动区域和连续窗口；这些仍是编码像素诊断。光流单位为最长边缩小至320后的像素，低运动区域不是语义背景，双向一致性不是光流真值。

| 模式 | 来源 | 光流均值 px | 相邻RGB MAE | Laplacian能量 | 补偿luma MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0 | 0.84601 | 0.02354 | 0.05085 | 0.01351 |
| dense | 12 | 0.12544 | 0.02019 | 0.14087 | 0.01344 |
| whole | 0 | 0.24702 | 0.01464 | 0.03099 | 0.01230 |
| whole | 12 | 0.09614 | 0.01854 | 0.09879 | 0.01333 |
| affine_residual | 0 | 0.07693 | 0.00867 | 0.05356 | 0.00559 |
| affine_residual | 12 | 0.10618 | 0.02954 | 0.16302 | 0.01767 |

## 机制判别

- source0：affine/dense 光流均值比0.0909，相邻RGB变化比0.3684；Laplacian均值比1.0534，逐帧比值>1为53/81。五个16-transition窗口光流比依次为0.180、0.104、0.074、0.069、0.070。
- source12：affine/dense 光流均值比0.8465，相邻RGB变化比1.4630；Laplacian均值比1.1573，逐帧比值>1为81/81。五个16-transition窗口光流比依次为0.920、0.686、0.807、0.813、1.086。

source0五个窗口的运动估计与相邻帧变化均低于dense，符合运动被压低或纹理/对应关系改变的竞争解释；source12五个窗口相邻帧变化均更高，末窗口光流也反高于dense，不能统一解释成冻结。高Laplacian与低运动可以共存，纯平移下卷积模糊也不必改变真实运动；额外纹理、重影和编码噪声均可提高边缘能量。没有光流真值、语义分割或人工完整播放确认，不能把这些指标单独当作运动/照明失真的因果证明。

同一网格下的补偿luma残差只描述编码亮度时变，不能保证跨视频追踪到同一物体。下一步仍需[全40×4 native接口与有符号工具](E0_FULL_DEPTH_TRACE_TOOLING.md)区分部署算子数值、student-prefix函数变化、真实gate/residual写回和scheduler方向。P23/P24保持OPEN。

## 证据缺陷与验收

独立复核发现补评驱动为四个候选报告构造reference_identity时继承了候选model_digest。原paired dense视频SHA、dense RUN SHA、来源/seed/提示及实际像素均正确，六个报告与88组汇总复算最大差0；错误只在附加模型身份字段。保留冻结报告原字节，机器结果逐份给出实际dense digest勘误，不把错误字段作为dense身份真值。未来生成端应直接传入dense身份对象。

核心指标gate通过只说明这次工具规定的文件/字段检查；VBench-I2V、VBench2与用户完整播放审阅均NOT_RUN。零新视频、零模型调用，quality_accepted=false；原型质量失败的结论不变。[独立审查](../method_discussion/discoveries/E0/independent-video-metrics-mechanism-audit-20261026_v2.md)保留公式、反例、勘误和scheduler dtype纠正。
