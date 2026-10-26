# DISCOVERY · 按真实图像与历史用途定义来源独立性

**可直接利用的结构是：phase1 四个新增 native 来源确实是原 Affine54 bank 的校准来源；而 CAL24 全部24个条目都有更早的原生采集与 student 暴露记录。** 因而 phase1 能测已暴露内容上的稳健性，phase2 的严格新来源候选应从 CAL24 外建立。另一种可能是现有其它冻结输入池还有未执行条目；本轮把这类条目保留为有身份的候选、未完成独立性认证，不从“本次还没运行”推导“历史从未暴露”。最便宜的后续是按 image/prompt/group 联接完整执行台账，再冻结新来源清单。

作者：independent_e0_audit；日期：2026-10-26。范围：只读 Git blob、源码、JSON/日志元数据及图片文件 SHA；未读取保存张量作数值分析、未调用 GPU/模型/SSH，未改 ACTIVE、atlas、旧协议或旧审阅。前一轮固定坐标 heldout 设计仍为 OPEN。

## EVIDENCE · 精确审阅对象

本人以 `git show origin/main:...` 阅读了多来源报告、计划和静态 checker，并核对这些内容在 `e0a8ecd` 与此次实际读到的远端树中 SHA 相同：

| 对象 | SHA256 |
| --- | --- |
| `research/20261026/E0_MULTISOURCE_DIRECTIONAL_EVALUATION_20261026.md` | `23069d0d5470cef322b44b76ed2230b8c9a7089113861331eb0c653e43686e20` |
| `config/e0_multisource_directional_evaluation_20261026.json` | `945ea604b37e81f433c936dcd679fd799303b813e40afed1f58e2750a525bac6` |
| `.runtime/inputs/calibration24/canonical.json` | `3c24f59ca5960c3c3ca9bc69fcb91ccccaf6c01644ede7361877e8733aa4efcb` |
| CAL24 `recovery_report.json` | `3dd5766105b200efe72b28e05036948261e07583e34dca6f6d3f42b50a40ab38` |
| `research/20261025/E0_NATIVE_OBSERVATIONS.json` | `6bab763310c38f399c57da7106f4d01c1b2800843a4b3fc4f8f6171923df93c5` |

这里的 `index` 是 canonical 列表位置。其内部 `source_index` 则是原官方元数据行号，`source_group` 又有单独的过滤后编号；三者不同。例如 CAL24[12] 为原行64、`i2v_background:8`，CAL24[16] 为原行176、`i2v_background:22`。后续输入记录应同时绑定 canonical SHA/位置、image SHA、prompt SHA、group 和真实 seed；不同池中的 `source16` 不能互换。

## CAL24 身份与逐来源回执已实际闭合的部分

本轮重新用普通 CPU 元数据/哈希完成：

1. 24份 canonical prompt 的 UTF-8 SHA 均等于各自 `input_provenance.prompt_sha256`；24张实际现存图片的文件 SHA 均等于 canonical image SHA。此步不查看图片内容或推断运动语义。
2. dense 两个 collection shard 合计 index0–23，无重复/遗漏；每个 `input_report` 文件的 SHA 与 collection 完全相符，且其中 `input` 与相同位置 canonical 完全相等。24份均为 `PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP`，各有160个 scopes。
3. student/onpolicy 两个 shard 同样覆盖 index0–23；24份子回执 SHA/完整 canonical 身份全部匹配，各160 scopes，状态均为 `PASS_NATIVE_ONPOLICY_FULL160_AND_NOOP`。
4. 因此48份子回执的原始图像/文字身份已核对；未重读其任何 operand payload 或复算模型输出。旧坐标复用审计另记录3840/3840位置复用，这里只读取了其回执和当前源码。

| 历史证据 | 实际 SHA256 |
| --- | --- |
| dense collection shard0 | `5155c007ca7238c40ccbf182394c10b3ae0be672a2a24c22879c504a7303fbb2` |
| dense collection shard1 | `27ff8a26e2f4a67d97539ce95068e909b4fa7108f44eaa00257461228b9f89c2` |
| onpolicy collection shard0 | `37644a278f4543b16ddcfc2b049b3f5d1ac1211821ca88ca3bb00ec64e8829d8` |
| onpolicy collection shard1 | `c34c08761056d29a157acc3871ac9175a960c05faafe80ace5a7e0dae3e9048d` |
| `reports/worker_20261020/calibration_row_coordinate_exposure_audit.json` | `c560089b4d914984e25346e73f3c6e9eb08bac0a1d5a6cd1f39ebb095d8d2f05` |

collection 的仓库相对位置是 `runs/worker_restart_20261020/{calibration24_ffn_operands,onpolicy_calibration24}/shard{0,1}/RUN.json`；子回执路径和 SHA 由它们直接给出。该递归绑定使下附24项身份表可复核，不需要公开任何私有模型路径或硬件标识。

## 拟合、选择与评价分别留下了什么历史

CAL24 原计划在 `config/calibration24_ffn_operands.json` 中明确固定24个 animal/architecture 来源，排除当时 formal20、旧calibration12和research源。**那时的 source-disjoint 说明它们与更早输入不同；不代表今天仍是未用测试集。** 同计划下一步是使用全部24×256行训练 C-static108 的支持/中心增益/偏置和固定 rank256/lambda1 残差。

本人亲读 `fit_cstatic_centered.py`：`load_training_reports(expected_inputs=24,expected_rows=256)` 要求 `set(range(24))`，逐一核对 capture 输入与 canonical；`fit_positions` 在两专家/全部160位置循环所有24来源，并以训练 contribution score 选择固定108组支持。亲读 `fit_residual_full.py --calibration24-fixed`：固定 rank/lambda沿用旧12来源的选择，新的所有24来源参与残差参数拟合。这区分了**CAL24贡献支持/参数**与**rank/lambda来自旧12的选择**；不应把二者合并成“CAL24重新CV所有参数”。

现有完成回执：

| 完成记录 | 状态/范围 | SHA256 |
| --- | --- | --- |
| `calibration24_centered_fit/high_RUN.json` | `COMPLETED_CALIBRATION_PARAMETER_FIT`，80位置 | `db60dba76b2d0a697129cc27e6ec58c3f2e53b4dc84cbae79f34fbb756422783` |
| 同目录 `low_RUN.json` | 同状态，80位置 | `f1b591addb6607897dc0f34804d828d33cf2337ce12558b85881d78d04f9c81a` |
| `calibration24_residual_fit/RUN.json` | `COMPLETED_RR1_FULL160_CALIBRATION_FIT`，160位置 | `f845586b57b205c4cec12198c416a834ae036631a849f6f9f72e8e3c61fc1771` |
| `calibration24_pipeline/RUN.json` | `COMPLETED_CALIBRATION24_FROZEN_FORMAL_COMPARISON` | `650e3464386a54f036862613d9d150ec1a63d9409390518c28576e9332e1938d` |

上述路径同以 `runs/worker_restart_20261020/` 为根。更晚 onpolicy 的24份实际完成回执本身已说明它们被用于原生 student 状态观察，而非只冻结了未执行计划。

**未闭合的旧拟合源码细节（OPEN）**：三个 fit RUN 指向的原 checkpoint-root `identity.json` 在当前节点不可直接读取，故本轮没有逐一复核旧拟合参数文件与 training-collection identity；不把源码阅读冒充一次完整 artifact replay。旧 pipeline 记录的执行源码 SHA 为 `04545e2e602a07489b9152ede13e47931f2bf4e88d90afcdfd9a225f586f9955`，当前 `continue_calibration24.py` 已为 `ebbe9d101db0e0c7c4b39d8ab3e1d3aaf5b1fd06a70c7aa82f6000fe7bf1e8c4`，两者不同。本轮保留这处版本差异，不声称当前脚本逐字复现旧启动器。CAL24 全24已暴露的结论同时有上述实际 dense/onpolicy 回执，故不依赖补齐这个旧拟合来源细节才成立。

## Phase1 的四来源角色：已用校准，适合稳健性复验

多来源计划给出的4/8/16/20 image SHA、seed46/50/58/62和 source_group 与 canonical 全部一致；0/12为seed42/54。固定六个位置确为3 subject、3 background、0 camera。`source_group` 不能从视频画面是否移动反推成 camera 类别。

对当前 Affine54 bank，证据比历史总暴露更直接：

- `E0_RECONSTRUCTION_BANK.json` SHA `9731a3dac0213a463337d6a3c9b51a7645b4a8ba2ecf75e5c664f127fb9d8918` 明列校准 `[4,8,16,20]`，`held_sources_read_for_fit=[]`。
- 本轮实际读取 HF `e0_function_basis_20261026/prototype_banks_attempt1/RUN.json`，SHA匹配 `fd38f0c148a9284da26adf35bc43c06e720ecb89b00c35500e5b4d375c32dc81`，状态 `PASS_FIXED_RECONSTRUCTION_BANKS`；全部160个 pack 的 `training_sources` 均为 `(4,8,16,20)`。
- 本人逐段阅读 bank builder：这四来源两步组成256行，参与 affine 系数、group support 分数和 recovered readout；所以“只是尚未跑 paired native，因而是独立新测试源”是明确反例。

phase1标注 `EXPOSED_AFFINE54_CALIBRATION` 正确，新增640 site/8视频是该固定bank训练来源上的内容复验。0/12虽然是该bank的拟合外来源，也已经参与历次模型/指标/机制研究；`held from this fit` 与 `never exposed across project` 是不同事实。

## Phase2 真正可以认证的候选与 OPEN

**本轮可认证的未暴露 CAL24 候选集合为 `[]`，不是其余18个 index。** 24项均已有实际 dense/student暴露，且旧CAL24训练链覆盖全体。换一个seed也不能把已见图像变成新的独立source；seed变化可作为同source内配对重复，并保留source聚类。

其它现存输入池中，本轮只做以下有限定位：

| 输入池 | 已读身份 | 本轮判断 |
| --- | --- | --- |
| `calibration12`、`formal20`、`disjoint20`、`diverse_confirmation20` | canonical确实存在；旧研究与R17历史排除清单已引用 | 不能凭文件名中的 disjoint/confirmation 授予全历史未暴露标签；未重新审计其全部实际用途 |
| `r17_fresh23` | canonical SHA `758f74e79dc9ebbf75a42aa64db024ac79a8a0c180d01cd80cf728de5f455c10`；原输入审计 SHA `80bb6f7264b0127fabf56a874a90ef2ceec6108b2b1e7556b1a597916b70e16c`，真实8 subject/8 background/7 camera | 是实际存在的外部CAL24候选池，但“fresh”是旧冻结时状态，不是当前认证 |

R17 interruption 审计 SHA `53c8f1e75a62d66f98596326e9f82032e8eaed92b190c93071a3eae7135f44b0` 记录：R17池[0..4]完成全部五臂，[5]部分完成；[6..22]的生成job在**该次批次**未尝试，官方metric calls0。该池真实camera标签可用于后续核验来源类别。**本轮没有逐一联接R17之后所有C1N/C2N/其它拟合、选择和评价回执，因而[6..22]仅为下一次历史联接的候选，不授予“从未暴露”资格，不把R17索引改名成CAL24索引。**

所以当前 phase2 没有由本审阅授予的4个具体“已认证新源”。最便宜的后续可先审上述真实R17未尝试条目的全部后续用途；若严格全历史未暴露无法闭合，再从既有合法数据源选新图像，冻结真实元数据/图片/prompt SHA，并对所有历史 image/name/prompt/group 集合做碰撞检查。输入从公开合法数据源取得不自动意味着未参与项目研究。记录可选池和 OPEN 比凭空补 `[1,2,3,5]` 更能保住后续证据含义。

## 对新 checker 与科学解释的具体边界

亲读 `check_e0_multisource_design.py`：它验证六个整数、SHA字符串格式、四个校准role、预算和状态，但没有加载 canonical/collection/bank 递归核对身份，也没有验证 phase2历史暴露。因此 checker PASS 是静态计划一致性，不能作为 source独立性证明。建议未来冻结phase2记录时增加真实来源清单与亲读执行回执的 exposure joins；这是实现建议，不修改当前冻结工具。

图锚点 G0/G1（真实条件输入与source身份）连接 F04/F06/F08/F09（校准函数与值读出）及 D09（配对视频证据）。来源暴露是 empirical evidence 的适用范围，不是更改原算子的计算事实。上述 audit 不改变已测 Affine54 FFN成本和原视频结果；它改变的是新site/video能否支撑训练外内容泛化的解释。

本轮实际原始来源均为仓库协议/源码、canonical及图片字节、48份子回执、完成fit/onpolicy/bank日志与R17中断审计；没有新增外部论文或网页检索，也没有为此身份核查虚构新的科学数值实验。

## CAL24 全24身份索引（全部已暴露）

下表由本轮实际读取的 canonical 生成；所有 image/prompt SHA 与实际图片/文字字节核验通过。完整身份由上述 canonical SHA 和 collection递归绑定固定。

| CAL24位置 | 原metadata行 | source_group | 实际seed | image SHA256 | prompt SHA256 |
| ---: | ---: | --- | ---: | --- | --- |
| 0 | 872 | i2v_subject:0 | 42 | `bbcf851ef52d17f8e1eccce9d449109e61a35d5548d1556d3b8f57e9830b23df` | `e11ffd1fbbce9c3408c9f4259ef309a2a659c2fcbf4f42ef8e001b2f6261c640` |
| 1 | 873 | i2v_subject:1 | 43 | `a36db0ce7deee9e3db53e8af67c19189bf8366221239de1d0829c349e68bfea1` | `bc8c93f2d434b6de4bf00df1695136b2b2ac40e053c18a6b2a74c948bbd1f95f` |
| 2 | 874 | i2v_subject:2 | 44 | `d5a383c998e54616f5bc224bee0d9ccd8dd1418164ee7a7914e82a62a1d97285` | `db64c33396c73edc9a0e16ea006d14cb552ffc612f336a1507332e18b2bfbfcd` |
| 3 | 875 | i2v_subject:3 | 45 | `9dac556bfb55036cb18ec6768ee4783ae840058df6e251516894a7b6f3ef4e46` | `3c2f3af683dc588e296644e114c9d2ee58545d86265878b260f63d609dc5cf97` |
| 4 | 876 | i2v_subject:4 | 46 | `64bca1e56a73f87957cefa96ffe7801db9e80506be015da23fcaec1655042f65` | `150db25b6d43b1cf625841ad7755bca5f3d4462e6700b93698c35b68a400e862` |
| 5 | 877 | i2v_subject:5 | 47 | `2a4441fa15d42a03d7fd5debd24fc2f9f96c4add9c4dbbb32d4be811b180b26c` | `555c04882df5b03c614dd4b85873f601b5cd0022d547140444b6ec378eb4826f` |
| 6 | 878 | i2v_subject:6 | 48 | `cdbd5493f1e032e28ddd8ac41a3716c721ae09bc3e7c50bec77c318e8709829b` | `c7ac03106121c0e1f01cd7418302057481af6eea21849ee7a750513927d0f505` |
| 7 | 879 | i2v_subject:7 | 49 | `5f7e7471f37623b391f7fdb58ac2e2db07d6792853ab10b343899fb0157d6fdb` | `0f67a70803e97b2a32f23c73aa0b72446a88bff596d82f477d62b434a82cb5c4` |
| 8 | 880 | i2v_subject:8 | 50 | `0f15f7bf76dc1612a30f02f7ae7b109c9489aa4c0197c987cdf534f39a59ae7b` | `c5f003b91685020718563b5b42fcb50027271dfd17b39e93d8685aa0a7d2a072` |
| 9 | 881 | i2v_subject:9 | 51 | `7d13f39f845df916dd98927e352c741c55946b2edd1ec9d0461a2dda3b0f4123` | `a3e2a2eb7951cf973cb060860d949dbdcff825418c027bb8a620a772177d865d` |
| 10 | 892 | i2v_subject:20 | 52 | `510d323937c0f8d38d9a2da593ba08cafbfeeee7d9ba5711f115cced30a11a2e` | `25b21f5bb7b2270094c0c2c78142af118d785d5ffa10139b88fa7eda65f3a7a8` |
| 11 | 893 | i2v_subject:21 | 53 | `f85f66ead6353f9194b48c6be1fca2c06df77ce553249b6823e0869cd8623890` | `4ff13721fc9882f13b683238bbd9e3cc7aa1eefef2e5095c776e11ac4f3b23f0` |
| 12 | 64 | i2v_background:8 | 54 | `6c1e3a9d4c9b89fe5ecc2f8df80791e634094f522b7fba5ad8aacb6458abdfde` | `a37ea2e01f336f4b7f97ab7f6ff9bd8f5da1250304feb87fb7f0c2f39f7421f5` |
| 13 | 72 | i2v_background:9 | 55 | `9a487eeda0799fbf808270e70d9cf4be895f7914bb991107965b4045eace90f6` | `193eeb4d35a15fd568728057432ab17d4134ac15146333dcb84d9959a991399e` |
| 14 | 160 | i2v_background:20 | 56 | `b79ffa3ce87b9d4504b5a015f18387c944f4de27e06fd2ece3518ceceefb2592` | `c4bd7922fe8a96d607d613d0b79c6b5c051ee53b2052ab234ab7b142637f0a65` |
| 15 | 168 | i2v_background:21 | 57 | `6c77d2ab65d497f08bef88e6f758357afe670de4aff5b46367667d0245940c35` | `245595479174bec53c2087c636fe78a9b1cb1fe5fb56bd8b584530a3370543bf` |
| 16 | 176 | i2v_background:22 | 58 | `1efcfeb84db1336150f319b2e1701f310e941cdd7eab900af25a3e8bd588d6c1` | `48defe2135d36c85d8d1e73c3bfb2eab991967d8346b1a394ecd776d9e2a58f1` |
| 17 | 184 | i2v_background:23 | 59 | `53e635a131f526da436f66ba753880b90e342495460bb533c337814490e8b146` | `d7c15ebd1c1eb7e4fdf351f7047b0f11f7e6f6a77c17499286ab3b2e5cbea877` |
| 18 | 192 | i2v_background:24 | 60 | `bc49c9bd817b373d22eb00e3caaea5502c98fad7ea744c585efd498381de0fdb` | `6eea500dd70685d9841486db608fc51f8553f8ce7b90007868b49ab09f37467b` |
| 19 | 200 | i2v_background:25 | 61 | `4d6d4bed3a7f204e6472c2fc07a42575ddb22a094424f26c86edf0381eed95fd` | `afc3f5f1e8e448f2c8e3b3914ee1d7db0a1fbd1b546b174c04a935a381dd0d84` |
| 20 | 208 | i2v_background:26 | 62 | `451a3c3fb18a12fa9792b9a7cc80d1b4a118d1ece38a91ee604bc8de12b8f0bb` | `46d0c103a7e34c3df65cc0a1ced3915f4a7c268c320234fa06c0e52ea9bfb827` |
| 21 | 216 | i2v_background:27 | 63 | `f0e81464a1532647bea22a00845fba3f2463c54c19acba4efa98ba37df7f86e5` | `9df3f61e9f570928f5f27b7f91c8d17e939e2c85722741bbcf257350233d57ae` |
| 22 | 224 | i2v_background:28 | 64 | `201eeba5bccf1f929c47f4a6b33b0d52876e76d7b6966f20b527db33762b0442` | `e326daf7a4dfa552510b9d234a97e6ba0f6b81c80477792b4dd47a46f1305367` |
| 23 | 232 | i2v_background:29 | 65 | `0f12b95b45634f0a61256ed74c2470759fb59f9516488d43d975fb7ee7ac91e7` | `895069138f10bb96c4de27681cfe550777c8db0a8e896620800f82e5e577f11d` |
