# 20来源三臂质量与运动补评

Consumer 在这20个已暴露开发来源上相对 fresh Dense 的图像流幅值明显更低：RAFT 连续均值 6.894→1.152（−83.28%），官方 binary dynamic degree 为 6/20→0/20；imaging quality 72.629→71.626（−1.38%）。subject/background consistency 均值上升可能与更静止的生成有关，不能单凭一致性解释为动作质量改善。历史 neutral 的 RAFT 均值为1.969，consumer为1.152（−41.47%），其配对绝对差95% bootstrap区间跨0。

| 指标 | n | Dense | consumer | consumer−Dense 95%来源簇bootstrap区间 |
|---|---:|---:|---:|---:|
| imaging quality | 20 | 72.629 | 71.626 | [−1.872, −0.291] |
| aesthetic quality | 20 | 0.6162 | 0.6160 | [−0.0150, 0.0145] |
| motion smoothness | 20 | 0.9854 | 0.9847 | [−0.00323, 0.00225] |
| subject consistency | 20 | 0.9602 | 0.9809 | [0.01047, 0.03169] |
| background consistency | 20 | 0.9700 | 0.9774 | [0.00080, 0.01459] |
| dynamic degree（二值均值） | 20 | 0.300 | 0.000 | [−0.500, −0.100] |
| RAFT 连续40对均值 | 20 | 6.894 | 1.152 | [−8.501, −3.124] |
| I2V subject | 10 | 0.9822 | 0.9870 | [−0.00038, 0.01071] |
| I2V background | 10 | 0.9872 | 0.9874 | [−0.00232, 0.00327] |

主体和背景 I2V 分别对应各10来源，不应合成20来源同一指标。相对 Dense，imaging有6/20来源下降超2%，相对下降p90为6.41%；RAFT有18/20下降超2%，p90为97.01%。例如叼鱼飞鸟的 RAFT 18.493→1.271，游泳海豹17.898→2.265；雪暴路边汽车的 imaging 57.812→52.811。相对历史neutral，imaging 70.330→71.626、aesthetic 0.6042→0.6160，motion smoothness 0.9863→0.9847；RAFT配对绝对差区间 [−2.009, 0.037]。全部每源数值、两种基线的各轴区间与尾部见 [结果 JSON](UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.json)。区间是固定seed的2000次来源簇bootstrap描述，不是独立盲测或人群通过证据。

实际为60条视频、20来源、每臂每来源单seed：40 fresh Dense/consumer对真实 initial noise 字节一致；20历史neutral未保存初始噪声，不能声明与fresh字节配对。14次原版函数调用覆盖8个独特官方维度，RAFT实际2400 forward、48000 iterations；连续值是图像流幅值，不直接等于主体动作质量或提示遵循。官方每来源五次真实独立采样未运行；正式第九维 camera motion、人工完整视频验收均 NOT_RUN，不能称全面质量通过。

独立核对冻结协议 `d5e72fda97cf405666bdd21fa5828b8a38339b65717057367430a037c94adc49`，两part RUN分别为 `311f382be51fbd9c97335a075c1b7dacab64903bacf3757f726a239794ad1e5b`、`b410957cf5e1fb5b167464de0e20e6f48aa7a119659c057622676d2320aef8ae`；两part及聚类launcher均退出0。逐行核对原始输出与320条配对CSV、各来源簇均值；两part GPU采样无违规，采样峰值均1944 MiB。原始收据和两份聚类报告、CSV SHA 列在结果JSON。
