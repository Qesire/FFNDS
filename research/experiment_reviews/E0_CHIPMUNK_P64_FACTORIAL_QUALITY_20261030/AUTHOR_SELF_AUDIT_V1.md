# 四臂 factorial 质量：作者自审，待独立审阅

作者 `/root/sm120_full_quality_build_terra`。6个协议 `FROZEN_PENDING_REGISTRATION`，没有ACTIVE授权或评分运行。新factorial parent GPU及独立验证实际EXIT0；draft实际调用两个父owner的inspect并绑定RUN/EXIT/verification，逐source校验NONE与旧Dense视频字节、全部初始噪声和source image/prompt，BOTH引用旧已闭合V2原video而不再生成。每臂仅每个暴露开发source、seed42；NONE/BOTH新旧评分是同视频重复测量，不能增加独立N。

每source八轴8原函数、32视频维度结果；MUSIQ324、CLIP L14/head各12、AMT160、原dynamic RAFT160/3200、DINO subject324、CLIP B32 4、I2V DINO/DreamSim各328。独立连续RAFT按0,2,...,80相邻抽帧，4×40=160 forward、3200 iter，另实算不视为官方dynamic重复可免账。视觉只解码HIGH/LOW两条各81帧、四页；已闭合原Dense/BOTH contact sheet通过旧视觉verifier和完整video SHA匹配复用。图像不是连续播放或人审。

四臂逐来源输出H−N、L−N与B−H−L+N，之后两来源等权；它衡量动态policy级交互，B/L在不同student前缀的step3 mask可以不同，不是固定mask线性分解。旧质量结果报告作为源绑定并与NONE/BOTH新分数逐轴比较，保留是否精确一致和差值；连续RAFT单独比较旧均值，不把大运动解释成质量更好，不合成八轴总分。

预检、GPU和独立VERIFY分阶段各需真实ACTIVE、精确job及退出回执，GPU assigned物理0/1质量、2/3独立RAFT，CPU视觉；实际设备UUID及profile为私有，动态启动前重查。沿用原VBench权重/隔离环境及SHA journal：每phase whole-file SHA预算16GiB；worker sidecar另64MiB软监测，嵌套child前/子进程小型父身份读取、模型loader内部读取UNKNOWN。GPU summary及完成复核各读取旧分数报告一次、VERIFY再次读取；视觉老闭合身份检查的嵌套小读UNKNOWN。软限额不是硬文件系统配额。source closure包含各实际父、执行、测试、官方engine和关联身份；没有新模型下载。

合成10项CPU测试、所有三个入口静态编译与visual隔离help通过。初版private job helper在 `-I` 下丢失tools路径，实际无模型进程退出并已修复路径；未执行旧协议移至私有archive，新V2重新生成。科学反例：原度量类包含调用路径/采样假设，四视频预算翻倍不等于真实forward都成功；多个模型内部加载可能不可精确观察。请独立审核每项调用数和失败预扣、4路径与I2V配对、重复分数时验证再读预算、旧闭合与source closure、visual只两个新视频、CUDA隔离和独立VERIFY。任何错误须修订后重新生成协议/jobs，真实结果前不报告质量结论。
