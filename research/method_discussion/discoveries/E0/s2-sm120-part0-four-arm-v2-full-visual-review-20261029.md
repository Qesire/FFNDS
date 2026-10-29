# SM120 part0 四臂全81帧接触表：独立视觉观察

2026-10-29。实际逐页查看 911、1039、616 各 Dense / 全压缩 / 前10层 Dense / 后10层 Dense 的 **24 张 JPEG**，并按 manifest 逐页重新计算 SHA，全部相符；每臂 page0 是帧0–41，page1 是帧42–80，末尾三格为空。V2 RUN 的 SHA 与 actual EXIT0 收据一致，root 的目标机 closed verification 已记录 `PASS_CLOSED_SM120_ALL81_VISUAL_CPU`。这是 AI 对全部81帧缩图接触表的观察，不是完整速度的视频播放、人类审核或细微运动定量。只有3个开发来源，不作质量通过或泛化结论。

同源同 SM120 fresh Dense 为对照；不能将旧 A100 同 seed 的事件当成本次 Dense 的预测，因为已核实两次原始噪声不同。

## 逐源可见现象

**911，青蛙与木桩。** Dense 中青蛙朝向、头部轮廓及背景构图在0–80大体稳定，末段可能有很小的姿态/视线方向变化；在缩图上不能可靠判定眼睑、呼吸或嘴部细节。三候选同样维持青蛙和木桩，视觉上大体近静；难以由这些页面可靠排序哪个候选更接近 Dense 的细小动作。浅景深背景和木桩基本保持位置，没有清楚的大范围镜头移动。微小姿态变化、纹理生成变化和缩图/编码差异都可能影响观感。

**1039，车前两人。** Dense 在约0–40帧中右侧人从低头逐渐抬头并转向左侧人物，左侧人的头部/身体朝向也发生变化；后段左侧人物姿态继续改变。警灯红蓝亮度随帧显著变化，车辆与大树在画面中的总体位置较稳定。全压缩和后10层 Dense 的两人相对较固定，警灯仍发生变化；前10层 Dense 约55–65帧可见左侧人物伸出的手臂向下变化，不能称其完全无动作。缩图不能可靠判断手部接触、视线、遮挡下的肢体或语义互动；人物形变、局部运动与生成外观变化是竞争解释，警灯闪烁也不等同人物运动。

**616，山脊、橙色坡地和云。** Dense 约帧8–20，坡地和前景出现并扩大明显暗区，之后亮暗边界继续变化；山峰和远山轮廓总体稳定，顶部云形也有变化。三候选保留山脊与橙色坡地，但在这些缩图中没有看到 Dense 那样明显横跨前景的大暗区变化，云和局部亮度仍可能略变。最直接可见的是亮暗变化；云遮阴、生成光照变化、云自身运动及相机/几何微变均是待区分解释，不能仅凭接触表断定物理云影或独立测出运动。

root 提供的已闭合 RAFT v2 part0 来源归一比为全压缩 0.1314、前10层 0.1761、后10层 0.1193；其整图光流量与上述局部视觉现象不是同一对象，光照/云/警灯、相机和主体都可贡献变化，不能用数字补充截图中看不清的动作。三个来源结果也不足以给出一般性的恢复排序。

## 证据身份与逐页 SHA

V2 RUN SHA256 `e9a21f7400541b91d0ce9e78cf91b6bdee45987c67322d79e61bddde3a9d0014`；MANIFEST SHA256 `6b8547eae7722465593de0c0a8252132cbaf3c2562959307d883d8a0f1041afa`；EXIT SHA256 `78c78fd4dc469f81898bdf303577269558b95d5e7405831783907db5bee4f8aa`。路径根为 `/home/xiang/.cache/huggingface/ffnds-derived/e0_s2_sm120_fixed_depth_all81_visual_cpu_v2_20261029/part0/attempt1/worker/results/pages`；文件名 `source{ordinal}_{mode}_page{0,1}.jpg`。

| source | mode | page0 SHA256 | page1 SHA256 |
|---:|---|---|---|
| 911 | dense | `304626876c3dbd119d4b9dd46f1ff23668c4685aed3a25dc9f9fb4996d5922cb` | `9fe729337ec057a889d1f198adc4634e45353790004ed50f8b25b3453c0de21c` |
| 911 | value_all | `3dcc91a035faaaf936764b68d0ede7a8a6224ae198c3897fd13e1701da357d06` | `80ff53a4c910283b965a000c518a2ce4be41a826561a070743eeffa18b87f018` |
| 911 | value_early10_dense | `d4d3e0f877b8055b9ba47cd07c523b99cb3423d514de6d5938e6b5b2cb4c1c77` | `ffad9e31cdbdf3d0b9eec2c34e1824221a0f7864e23359b2c2f8dda510cbffea` |
| 911 | value_late10_dense | `c333caf35971c948bf909aab81de94695c4d76e6f96013e622b8da0abef5c6a6` | `1856298b830d3b26292ae6987a98dae528237ae7981e41b1113b4ea73f50bede` |
| 1039 | dense | `d91dd48a3015e55c502ad8250fe7e5dc9856219cce44a9d58a14459bf2ebf5c8` | `0f78f9f24486d0ae544962f6885611963681dfa090c99be33ac7363dff2e7705` |
| 1039 | value_late10_dense | `428aa4cd54a4ae07b867acaef93cd495cc193a8ec53361f81a996ab648e1bfdf` | `e15e3b24ba42e396d03e392eb0d20d4cba731d9e35c870a2956ed000c12aa487` |
| 1039 | value_all | `be370e94aff66954a94badc4df9311b2011c8e1553db4bb2e0173024a2278a7e` | `14d155ea6371b818999c29a529556aaed1e257179292142afad4362f4cd98713` |
| 1039 | value_early10_dense | `e509f3793bcbe786440f31115a8ab989e1b6675bc9bb4f5563c6b1db597e67b1` | `781a07ac62305fac01da0dfa5b6101b9fe1a2939282865fe18df19a9435bd338` |
| 616 | dense | `7a0d42934db841657d75afd00eb084699a3fdabdefa39a299382d29763c7ce81` | `2e6989ea4c76e493e3ba68358ac37f66ab0f1d8e3650e897c3279e8f4f619295` |
| 616 | value_early10_dense | `f477e51750d11b02f54344ca55856acb030a985339036fd22b0a2fac78bcc34d` | `48b8c5dc0b57fe16e70ca2e2e90424394f40ad7bdb1c32525a16d0611ee4d71f` |
| 616 | value_late10_dense | `2007e248433c7ad4a8cb2835de1d9037fd61026358c710fc6dd668c983d31513` | `76423a59f6a4d23ef64c7a504a6eff74e74dce4db50fbd22ac4c5030eabd2c73` |
| 616 | value_all | `2a6ab3c5ccdf03809ba0727197e693a6eb6cdbedaeede916a4d059ab77223a20` | `c188b6644d74cbc451fa603ae2ae1afa3d314d6e4736924873949df56129cde9` |
