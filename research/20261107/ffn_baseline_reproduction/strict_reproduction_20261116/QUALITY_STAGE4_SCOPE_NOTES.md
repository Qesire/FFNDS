# 五臂质量评测：候选接口与资源绑定

状态：代码候选、工程测试通过；尚无本轮VBench/RAFT实测结果，也不是ACTIVE或独立审核通过记录。现有两源Dense可先作为真实输入身份资产；近似臂必须等主家族、80参数与必要neutral冻结并按方案完成。

`quality_vbench_batch.run` 每个受审scope处理一种维度及1–55条真实视频。每条必须绑定源ID、方法、seed42、图像/文本/视频SHA、source cluster、相同输入身份（包括原始噪声/配方/权重）及真实generation成功收据；调用方的 `verify_generation(record,scope)` 必须独立验证这些文件/执行ID并返回匹配的proof SHA。不得用没有验证的布尔值、拼接视频或同文件多次别名冒充生成。不同方法按固定编号创建一对一软链接供VBench识别；最终只读取每视频记录，混合方法aggregate不用于结论，也不声称官方五次采样或榜单成绩。

独立guard必须冻结代码/依赖/输入SHA、真实Slurm分配、REVIEW/ACTIVE、批次及最大预算，调用worker时传入实际 `authorize`、`verify_generation` 和资源检查；输出要有 `RUN`/`EXIT`（exit_code=0、scope SHA、result内per_video SHA和video_count）、原始metric输出、effective_loader、逐视频JSON，失败或未完成留缺失。`quality_pair_report.load_verified_runs` 检查逐视频文件与EXIT哈希、状态、scope、数量和维度；配对报表必须使用事先冻结且按SHA确认的 `motion_dense_floor` 和11源cluster映射。标准汇总为源内相对Dense差，先cluster内源等权再cluster等权，保留逐源范围、cluster范围与leave-one-cluster-out；不可将帧/方法或重复来源当独立样本，不声明未预设margin的非劣。

实际参考资源待当前作业逐项重验，不能把旧scope的路径存在当作本scope权限：

- `runner_metrics.py` SHA256 `090f5e483c1cb7765dbda296baf01c858e7eb733ec433889e1f480e5fbd5d287`；`metric_contract.py` `1f77d121d86d8e9b661be7cdac80cdc7467558df51216fe01fe40b37a2f50944`；`metric_loaders.py` `9699de62e9d06cb6c09544b0681352dfa46b222809e99ad7dbdd4879dd1dcbba`。这三份原始文件按字节重用；实际package代码、全部轴所需权重与仓库依赖逐路径核SHA并拒绝网络下载。
- 旧G19环境：`/public/home/scc/pb23061276/miniconda3/envs/vbench_i2v_official/bin/python3.10`，torch `2.7.1+cu118`、numpy `1.26.4`、vbench `0.1.5`、dreamsim `0.2.1`、open_clip_torch `2.24.0`、peft `0.5.0`、pyiqa `0.1.10`；需重新绑定Python及实际package/source闭包。旧metric_seed `20261103`；再次固定在scope中。
- ffmpeg `/public/home/scc/pb23061276/miniconda3/envs/hif4/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`，SHA256 `e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99`；每个输入实际解码81帧、832×464、16fps，检测到缺帧/错误尺寸即失败。
- 历史小批次预算为每作业1张A100、4 CPU、64GiB host、GPU上限60GiB、wall 3600秒、soft 3480秒、空闲盘≥10GiB；这是旧2源×4方法某个维度的范围，不能推为55视频的足额运行时间。新scope按实际批次、轴模型/权重、当前分配与首个有效批次记录调整，不额外制造warmup视频；完整视频不作为独立source数。

十轴与范围：

| 轴 | 固定资源/语义 | 适用和解释边界 |
|---|---|---|
| subject_consistency | DINO ViT-B/16 | 逐视频主体时序特征相似度；主体运动和身份特征影响结果 |
| background_consistency | CLIP ViT-B/32 | 背景时序特征相似度；相机移动可影响 |
| motion_smoothness | AMT-S | 插帧/运动平滑代理；平滑不等于动作保持 |
| dynamic_degree | RAFT-things | 原版布尔值另存；严格40个两帧间隔pair、20次迭代、连续top-5% flow均值与5个8-pair窗口 |
| aesthetic_quality | CLIP ViT-L/14及审美头 | 模型审美代理，非客观质量标准 |
| imaging_quality | MUSIQ-SPAQ | 图像质量代理，按固定实现采样 |
| i2v_subject | DINO | 相对源图主体一致性，不能单独衡量连续动作 |
| i2v_background | DreamSim/CLIP/DINO与ensemble LoRA | 相对源图背景一致性；源图/分辨率绑定 |
| camera_motion | CoTracker | 只对明确匹配原实现支持的camera instruction来源计算，其余NOT_APPLICABLE；无指令不得猜测 |
| temporal_flickering | 原版逐视频时域代理 | 相机、物体运动会混入；按明确静态/动态/未说明三类分别呈现 |

连续运动比只在Dense绝对值大于事先冻结阈值时提供，否则NA并保留两臂绝对值。任何质量结果之后不得改阈值、来源、采样帧或聚合规则。`quality_dev3_contact.create` 固定DEV3(vbi_1013/0886/0800)、所有五臂及帧0/20/40/60/80（16fps），对缺失视频留空，保留来源、视频SHA、proof SHA和ffmpeg身份；仅为视觉观察辅助，不自动生成审美/运动判词。用户可在同等时间位置做连续播放；接触表不替代视频。

## 配对批次、消融和收据

`quality_guard.py --help` 是受控入口，不负责提交任务。scope 与 ACTIVE、独立 REVIEW 的SHA必须交叉绑定，先审阅再运行；scope 固定 actual Slurm A100 分配、预算、Python、源及模型资源、视频与图文条件、可执行 verifier 的SHA和全部metric代码/权重。另有单独的 `metric_protocol_path`/`metric_protocol_sha256`，其 dimension、seed、运行时和package版本、代码/资产、旧参考文件、ffmpeg及工作目录字段必须与scope逐项相同。各视频生成收据由scope所绑定的 verifier 校验；`quality_dense_proof.py` 仅实现已审计Dense资产，非Dense会直接拒绝，须接入真实近似臂生成收据的独立 verifier 并审核，不可虚构PASS。

相同维度、相同源的配对结果还要求 metric protocol SHA 和 metric batch identity 完全相同。main 批次需在同批中包含需比较的 Dense 与近似臂。E4/E6 在各自DEV3批次中同时计算已有 parent 和 child 视频，允许复用parent视频但须在该批次重新计算其指标；E6另外明确且仅固定 `tail` 或 `precision`。这些附加指标批次不产生新视频。camera_motion 对当前DEV11全部为NOT_APPLICABLE，不提交或解释无相机指令的数值。

V2条件图与旧VBench loader采用不同绝对目录，实际逐文件读取核查后11个源图字节SHA相同；当前scope仍须按输入路径重验，不能只依据文件名。旧九条Dense有视频且可用于质量比较，但原始最终latent未保存；严格的video+latent复用只通过新Stage1的两条。旧metric小批次资源及时间不得直接外推至完整55视频或所有维度；本候选尚无新指标运行或性能估计。
