# 已保存末步 dense 反事实的视频桥：两来源改善异质

**新实测结果是：最后一步改用完整 dense，能在来源12降低多项全帧差异，但来源0几乎没有像素恢复，部分运动/细节代理还变差。** 两个 dense 重渲染控制均逐字节等于历史视频，故可把这组差异接回已保存的原生末步状态。竞争解释是先前近似已改变必要的早期结构/运动，而末步只修复部分剩余方向；本宏干预尚未识别哪一步、哪层或何种表示是最早原因。下一低成本研究应结合有符号状态频谱与独立FFN函数坐标，选择能区分这些解释的固定干预。

## DISCOVERY：从已付费 latent 构造完整可见结果

直接解码`flow_s3.safetensors:z_dense_at_student_after`得到明确的`S,S,S,D`路径：前三个原生候选步骤、最后一个完整dense步骤。对照是原`S,S,S,S`和原全dense视频。末步一次改变40个FFN及其内部后续输入，故只对该整步宏干预成立；它没有独立隔离某层或数值/容量/状态机制。控制另解码`z_dense_after`，得到相同VAE/媒体链的逐字节中性检查。

## EVIDENCE：全帧、运动与细节分别看

| 指标（同来源原dense参照） | 来源0：原候选 → 末步dense | 来源12：原候选 → 末步dense |
| --- | --- | --- |
| 平均逐帧RGB RMSE | 0.154160 → 0.153464 | 0.130529 → 0.112803 |
| 编码luma RMSE | 0.151253 → 0.150705 | 0.124198 → 0.106173 |
| 有符号编码luma偏移 | 0.006670 → 0.007226 | 0.015196 → 0.016059 |
| 相邻帧差RGB RMSE | 0.056418 → 0.055109 | 0.065115 → 0.054826 |
| Farneback位移L2差（下采样像素） | 0.854121 → 0.860995 | 0.178075 → 0.158782 |
| 估计运动幅度比（理想参照1） | 0.106258 → 0.108312 | 0.867577 → 0.706978 |
| Laplacian能量比（参照1） | 1.065137 → 0.911274 | 1.156966 → 0.974558 |
| 运动补偿luma变化差 | 0.015574 → 0.015048 | 0.023234 → 0.019946 |

来源0：latent SSE下降4.22%，平均RGB RMSE仅下降0.45%；Farneback位移误差上升0.80%，估计运动幅度比仍仅约0.108，细节比从1.065变0.911并未向参照恢复。来源12：latent SSE下降16.30%，RGB RMSE下降13.58%，相邻帧差RMSE下降15.80%，Farneback位移差下降10.83%，细节比更接近1；但运动幅度比从0.868降到0.707，有符号luma偏移在两个来源都上升。不存在所有代理一致改善的结论。

所有新clip分别完整评估81帧/80帧对，共324帧/320帧对；两个中性控制的已定义配对差为0、比例为1。指标使用编码RGB/luma及uint8尺度Farneback，光流/补偿/细节均为诊断代理，不是真实运动、语义背景或感知质量评分。原四报告完整字段、逐帧条目和SHA保留在[指标目录](video_metrics/saved_flow_decode/)。

[汇总JSON](E0_SAVED_FLOW_DECODE_RESULTS.json) SHA256：`540e03e11b53e8cdbf97e7479bc6eef3ff87c05669a74007c21d67e7262438be`；原生`DECODE_RUN`：`24943a6febd28e88c3abcabacafff5701b6c718ba9be51565c9dab7d8b769fcf`。完整core gate及独立VAE inventory均通过。官方VBench-I2V/VBench2、语义背景及人工全视频审阅按实际情况NOT_RUN，`quality_accepted=false`。旧原始视频/报告/协议保持不变。

## EXECUTION：四次VAE推理完成，0新增denoiser/FFN

单张空闲A100、一次有限attempt：4次VAE网络decode/4个新MP4，0新增denoiser NFE/FFN、训练、拟合、官方评分、失败或重试。VAE权重SHA `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`；原生WanVAE、BF16、81×464×832、16fps/ffmpeg，与现存媒体链对齐。setup 63.921秒；四次render wall总和18.804秒；物理显存峰值6693MiB；HF输出8,330,343字节，低于0.1GiB。

这里setup只计VAE构造与同步，先前预检/模块导入的总启动wall未单独测量。VAE/render成本与原四步宏路径的历史FFN成本分开；这里FFN_calls=0不表示该宏路径无需FFN。指标绑定immutable DECODE_RUN后再生成，逐clip render去重键与共享setup键分离。全部实际MP4由独立清单核对，hybrid模型身份、原始BF16 tensor key/native hash、decoder及参照从实际父收据重建。

[设计/源码审查v8](../method_discussion/discoveries/E0/independent-saved-final-latent-vae-bridge-audit-20261026_v8.md)、[最终独立源码审查v5](../method_discussion/discoveries/E0/independent-saved-flow-decode-source-review-20261026_v5.md)明确区分10项CPU软件测试与本轮真正原生中性验证；[独立实证复核v6](../method_discussion/discoveries/E0/independent-saved-flow-decode-empirical-review-20261026_v6.md)另外完整解码4个新clip与2个旧候选，2418项逐帧数值复算差为0。[固定协议](../../config/e0_saved_flow_decode_20261026_v1.json)按临时持续授权登记。

对FFN研究的含义是，后期完整dense只提供局部挽回，单次末步替代尚未达到质量目标。当前优先保持必要的早期/有符号功能方向，并继续检验独立的低维非线性坐标；任何真实省算仍需实际GEMM、观测代价与全视频联合比较。P24→G21/D09/P25新增两来源宏干预VIDEO-OBS，根因和下一函数表示仍OPEN。
