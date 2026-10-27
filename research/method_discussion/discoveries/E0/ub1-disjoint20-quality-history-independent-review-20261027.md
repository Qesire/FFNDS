# disjoint20 三臂官方质量工具：独立冻结前审阅

**结论（SRC/CPU 合成测试）**：当前草案已有逐 part 原子 attempt 和预扣、producer 完成与 launch 退出核验；一项身份检查应前移：60 条视频的 resolved path 唯一性只在评分后的 `cluster_rows` 校验，建议 preflight 在评分前检查并添加重复路径负例。跨来源/臂相同 SHA 是否属于禁止 alias 应在冻结清单中明确。当前 config 尚未填资产、未冻结，因此真实 recovery 0 + original 1–3 与 60 视频绑定仍需最终复核。

审阅工具 SHA256 `95e54feb0fc561c61fc7e809952dc420a82a8368fbf062cb0eb911fbd844d075`，测试 `e10a9ab5ae634429f7be2aaf15cd117f01c108abe0913a03e921f75a7c0e195b`；项目虚拟环境独立运行13个 CPU 合成测试通过，exit 0。未运行模型、GPU、评分或读取真实 tensor。

每 part 对10个来源的三臂调用六项共同维度和对应的一个 I2V 维度，共14次官方函数；两个 part 总计2400次 RAFT forward、48000次迭代，实际 forward 包装计数并限制每 part 1200次。AMT、DINO等使用维度级前置预算账本，外层不声称观察到内部每次调用。两 part 的输入视频逐个解码验证81帧及分辨率；正式 camera motion 未运行，历史 neutral 无已保存 initial noise，不能冒充字节配对。

直接 `compute_i2v_subject/background` 使用上游 `load_i2v_dimension_info` 的 `image_name` 加相对 `vbench2_beta_i2v/data/crop/16-9`；工具切换到 work 目录并按来源 SHA 绑定建同路径链接，因此不依赖 VBench 上层新映射步骤。实际结果仍取原函数返回的 per-video 数值。聚合调用 `paired_report(records, replicates=2000, seed=42)` 与当前接口匹配，按 image cluster 等权、条件和 seed 嵌套；20 image 身份与近重复边界须按冻结数据核实，不将样本视作独立人群。

producer 检查要求四个分区、phase 覆盖、40个唯一 registered IDs、原 bank/method 绑定及成功 launch；part0 的协议 SHA 必须不同于原协议，其余三个须相同。最终清单必须核对 part0 是指定成功 recovery，且原失败 part0 仍隔离。每 part 1 GPU、2 CPU、1800秒内部/2100秒外部，双方最多2 GPU；scope 限额及最终 monitor 异常的非零传播有源码路径。结果是已暴露开发来源的有限官方轴比较，不是完整视频质量或人工验收。

## 最终资产复核

修订工具 `78cf4464e2874b1fc32bbe5f4d2add19ef40460efc4125d33633991764600cd2`、测试 `e3a21f69f9b92494779ed10360e428c8234b33b82ef21391596ca70278210a30`、冻结配置 `d5e72fda97cf405666bdd21fa5828b8a38339b65717057367430a037c94adc49`：14项合成测试 exit 0。60路径/inode唯一预检已加入；对实际冻结HF输入调用只读 `preflight` 返回 True，涵盖恢复part 0及原part 1–3的生产者收据、40 fresh和20 historical身份。未执行评分或解码张量；相同内容的独立文件按实际记录，不凭SHA不同证明语义独立。
