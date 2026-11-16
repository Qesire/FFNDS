# ICC 作者式校准：现有数据资产核查

核查基准：FFNDS `main` **0e29eb7affddc88138337294ffdb5ccc7893f86e**，提交时间 2026-11-15 18:06:36 UTC（北京时间 2026-11-16 02:06:36）。本轮通过远端 branch/tree 读取确认 head，并逐字节核对下述 10 个资料文件与该树中的 Git blob；全部匹配。明细见 [ICC_CALIBRATION_ASSET_SOURCES.json](ICC_CALIBRATION_ASSET_SOURCES.json)。这是资产来源核查，没有新增模型或采集运行；此记录随ICC校准修正一起发布。

## 决定性结论

**当前 CAL11/CAL_EXT11 是 VBench 的静态输入图片和文本条件名单。公开资料没有为这 22 个具名来源绑定可验证的真实干净视频、其 VAE 编码 `x0` 或 Dense 最终 raw-latent tensor。** 已验证的 `VB_DENSE22` 是 **CAL11＋DEV11** 的去噪途中 FFN `U/Y/gate`，不是 22 条 CAL 干净视频。当前 CAL_EXT11 对应旧 CAL_B11，仍标为未生产。

因此，现有 bank 不能直接充当 ICC 作者式“干净样本 → 随机 t 加噪 → 单次/相邻一步统计”的源数据。该统计流程本身不需要完整生成轨迹；当前最小缺口是 **11 条真实 clean clip 或与真实 clip 绑定的可重载 VAE 编码 x0**。CAL22 只是在确需扩展时再增加 11 条，不必恢复 CAL64/CAL256，也不需要先采 11 条或 22 条完整 D4 轨迹。[当前名单](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/COHORT_PROPOSAL.json) · [资产状态](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json) · [bank 绑定](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/B_BINDINGS.json)

## 1. CAL11 与 CAL_EXT11 实际指向什么

当前 `COHORT_PROPOSAL.json` 中的 CAL11 与原配置 `wave=CAL_A_INITIAL` 的 11 个 ID 完全相同；CAL_EXT11 与 `wave=CAL_B_RESERVED_EXTENSION` 完全相同。每条包含 `image_name`、`prompt_en` 和 prompt SHA256；当前 proposal 的 `image_bytes_sha256`、`source_cluster_id` 均为 null。原设计配置另给 JPEG 的 archive member，形如 `data/crop/16-9/<image_name>`，没有原始视频 URI、帧范围、视频 SHA 或 clean latent URI。[名单](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/COHORT_PROPOSAL.json) · [原始选择配置](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/config/g20_vbench_restart_cohort_v1_20261113.json)

| 官方类别 | CAL11：ID／输入图片 | CAL_EXT11：ID／输入图片 |
|---|---|---|
| single-human | `vbi_1005` — a man is running on a dirt road.jpg | `vbi_1024` — a woman smiles while holding a yellow flower.jpg |
| multiple-human | `vbi_1039` — an indian police officer talking to an old woman.jpg | `vbi_1064` — an older woman and a young girl are knitting together.jpg |
| animal | `vbi_0895` — a seal swimming in the water.jpg | `vbi_0901` — a giraffe walking in a field.jpg |
| transportation | `vbi_0937` — the red Alfa sports car is speeding down the road.jpg | `vbi_0922` — a blue car driving down a dirt road near train tracks.jpg |
| food | `vbi_1094` — The meat is picked up from the grill with tongs.jpg | `vbi_1107` — a person is holding up a burger with his hands.jpg |
| plant | `vbi_1082` — reeds blowing in the wind against a cloudy sky.jpg | `vbi_0967` — a pink rose is blooming in a garden.jpg |
| other | `vbi_0966` — a person holding a sparkler in front of a sunset.jpg | `vbi_0958` — a statue of a lion is sitting on a pedestal.jpg |
| architecture | `vbi_0304` — cars parked on the side of the road during a snowstorm.jpg | `vbi_0152` — a beach with a lot of buildings on the side of a cliff.jpg |
| scenery | `vbi_0648` — a flock of birds flying over a tree at sunset.jpg | `vbi_0760` — a boat sits on the shore of a lake with mt fuji in the background.jpg |
| indoor | `vbi_0456` — a kitchen with a sink and a lot of glasses on the counter.jpg | `vbi_0448` — a living room with a christmas tree and a rocking chair.jpg |
| abstract | `vbi_0032` — a visually captivating abstract video, rich in color, set against a dramatic black background.jpg | `vbi_0040` — a purple and yellow abstract painting with a black background.jpg |

这些是 22 个具名图片条件的预算与角色，不自动成为 22 个已验证 clean-video source。图片内容中含“video”一词也不改变其 `.jpg` 资产身份。

原 VBench 资源身份：

- 元数据：`Vchitect/VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/vbench2_i2v_full_info.json`；SHA256 `8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`。
- 图片 archive：`lxl-158/vbench-i2v@577ba45c1653ede26e29996737f42841f2ec9eea/vbench2_beta_i2v/data.tar.gz`；2,607,670,436 bytes；SHA256 `aa6af89775242f35dbcd1f4c53421a6547d0700b1a99206e98b87a28ea393865`。当前记录明确 archive 为历史 hash 绑定，本轮未重新读取远端大文件。
- 已公开 BWS 条件中有 CAL 的 `vbi_1005/vbi_0895/vbi_0648` 三张图片的真实集群路径和 SHA；剩余 CAL8 的逐图字节凭据未在本轮检到的公开 V0 资料中完整展开。CAL_EXT11 目前只保留所选图片条件和 archive member。以上限制不需要重跑 V0/V2；如需消耗图片，先读取既有绑定。[元数据和 archive 状态](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json) · [已公开的六条 BWS 条件](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/bws_conditions.json)

已公开的三个 CAL 图片路径均位于：

`/public/home/scc/pb23061276/projects/wan-quant/data/VBench/vbench2_beta_i2v/crop/16-9/`

| CAL ID | 文件名 | 图片 SHA256 |
|---|---|---|
| `vbi_1005` | a man is running on a dirt road.jpg | `16b2f0d4f1b408720d46c6e1b617435649005e2ba4b98f37854a437dd24f621e` |
| `vbi_0895` | a seal swimming in the water.jpg | `c4c0d8a8c97fd91f1c65010c5c1743c90aff53d911fd9fd8372ccbd811dd685f` |
| `vbi_0648` | a flock of birds flying over a tree at sunset.jpg | `70a5935d80202df41584ed0b78ac0099a134b327eeaa5b03f40c3969d80021bd` |

## 2. 可消耗资产与缺项

| 对象 | 当前证据／producer／locator | 能否直接支持作者式 clean→random-t 校准 |
|---|---|---|
| CAL11 图片＋文本 | 上表。V0 已登记完成；部分逐图身份由 BWS 条件展开；原公开 V0 凭据尚不完整 | 可提供 I2V 条件，**不能提供动态 clean clip** |
| CAL_EXT11 图片＋文本 | 当前 proposal 和原 CAL_B 角色名单；无该 11 条的新增捕获完成记录 | 仍是保留条件；不能当已生产 CAL22 |
| `VB_DENSE22` | V2 job 161964，登记 EXIT 0；`BANK_MANIFEST.json` 及 hash 由 B_BINDINGS 核验；CAL11＋DEV11、4 步×40 层，每次抽 512 行 U/Y 及 gate | U 是被扩散模型变换后的局部 FFN 输入，Y 是 FFN 输出；不能恢复 clean x0，也不等价于对真实 clean clip 随机 t 加噪 |
| `VB_DENSE_CALB11` | 当前资产表 `NOT_PRODUCED_NOT_REQUESTED`，locator=null；producer V3 未产出 | 不能沿用为已生产 CAL_EXT11 |
| BWS O40 输入 | job 166225；6 来源×12 份 FFN 输入；72 `.pt`＋sidecar；`CLUSTER_ASSETS_MANIFEST` 逐文件绑定 | 是原生 O40 去噪途中的 FFN 输入；仍不是 clean x0 |
| BWS Dense 最终视频 | 6 个实际生成 MP4，逐文件 path/bytes/SHA；其中 **CAL3、DEV3** | 可作为显式 **generated-video** 备选数据；需要解码＋VAE 再编码，且只有3个CAL。不能等同 real-clean 主参考；DEV3不可挪作CAL |
| CAL11 Dense D4 最终视频 | 原设计允许复用匹配的 Dense capture video；本轮公开 V2 记录只明确 bank，未发现 CAL11 最终 MP4 的完整 manifest/路径/hash | 先读取已有 V2 output/receipt 清单确认；目前不登记为“11条已可消耗视频”，也不据此要求重生成 |
| CAL11/CAL_EXT11 的真实 clean clip | 当前名单/asset registry/BWS manifest 均无 source-video URI、原始帧范围、视频 hash 绑定 | **本轮没有可验证资产** |
| 真实 clip 的 VAE 编码 x0 | 未找到可重载 tensor 路径、shape/dtype/scale、VAE与预处理身份和 payload hash | **本轮没有可验证资产** |
| Dense 生成终态 raw latent | 公开 bank/BWS producer/manifest 没有最终 latent tensor 存盘项；旧 G19 资产记录明确 latent hash 不是可恢复 payload | 当前不能声明已存在可直接读的 raw x0；若集群另有未登记文件，先绑定，身份应为 **model-generated terminal latent** |
| Wan VAE／权重 | BWS scope 绑定实际模型目录和 Wan2.1 VAE 文件 hash | 编码算子可定位；这不等于已有 clean x0 数据 |

依据：[当前资产表](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)、[B_BINDINGS](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/B_BINDINGS.json)、[BWS cluster manifest](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/output/CLUSTER_ASSETS_MANIFEST.json)、[实际 capture runner](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/bws_runner.py)、[FFN 捕获字段](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/b_capture_o40.py)、[旧资产 registry 的 G19_VIDEO_METRICS 项](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/research/20261112/G20_ASSET_REGISTRY_20261112.json)。

### Bank 的精确身份

集群根目录：

`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v2_capture_20261113_rev2/stage/output/bank`

manifest：

`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v2_capture_20261113_rev2/stage/output/BANK_MANIFEST.json`

SHA256：`f6f8e100da6f75b2db612edd8c8c4d21729ed9bf99bfb5092941f2381a4d0bee`。

payload 格式：`{source}/{source}__seed42__s{step:02d}__b{layer:02d}.pt`。B_BINDINGS 的 `bank.cal11` 为：`vbi_0032, vbi_0304, vbi_0456, vbi_0648, vbi_0895, vbi_0937, vbi_0966, vbi_1005, vbi_1039, vbi_1082, vbi_1094`；另有 11 个 DEV，`n_sources=22`。这份22来源数量不表示CAL22。

### 已有最终生成视频的精确身份

BWS stage：`/public/home/scc/pb23061276/projects/wan-quant/work/g20_two_track_core_20261114/BWS/v1/stage`。

下列路径相对于该 stage 的 **`output/`**。这个相对基准由 `bws_runner.py` 中 `out=stage/'output'` 与 `byproduct=out/'byproduct_video'` 确认。

| 角色 | ID | 相对 output 路径 | bytes | SHA256 |
|---|---|---|---:|---|
| CAL | `vbi_0648` | `byproduct_video/vbi_0648__o40.mp4` | 564403 | `63cf612ddf8a3f177c934a391fd25e9c28d9db077a273ca80b121d53ef23b8e3` |
| DEV | `vbi_0800` | `byproduct_video/vbi_0800__o40.mp4` | 3591295 | `52cc820ac122a15dcb68422ac394f7d564092ecb1a387bda8b1fb0f9ec55e003` |
| DEV | `vbi_0886` | `byproduct_video/vbi_0886__o40.mp4` | 692657 | `96df954b4980fab72caf04c579e81d3c18de763f0acd2e048c7c23d31c14cc18` |
| CAL | `vbi_0895` | `byproduct_video/vbi_0895__o40.mp4` | 2011255 | `93400333ccfcf7c8810fc420adbaa740d0fbd7210d54d8488e3e37c84aa2664a` |
| CAL | `vbi_1005` | `byproduct_video/vbi_1005__o40.mp4` | 2352656 | `9f0ffbda5d46ea90c28ffca331d195983c1d3f0b76fea9b99ef5e8408e55d7f4` |
| DEV | `vbi_1013` | `byproduct_video/vbi_1013__o40.mp4` | 1488580 | `2877fc2ce51eb2c3a4594d4525da3335e1d651458b44a2b192b27326372dba5c` |

这六个 MP4 是 `pipe.generate(..., save_result_path=...)` 的输出，不是输入的原始拍摄视频。捕获回执中的 `initial_noise` 只记录张量 hash；`O40_CAPTURE` 没有最终 latent payload URI，cluster manifest 的类别为 params、O40输入、sidecar、byproduct_video，没有 raw final-latent 项。[runner](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/source/bws_runner.py) · [manifest](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/output/CLUSTER_ASSETS_MANIFEST.json)

VAE 可定位为：

`/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-bf16/Wan2.1_VAE.pth`

507,609,880 bytes；SHA256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。[冻结 scope 的 base_assets_sha256](https://github.com/Qesire/FFNDS/blob/0e29eb7affddc88138337294ffdb5ccc7893f86e/reports/20261114/g20_two_track_core/BWS/v1/evidence/scope.json)

## 3. 零新增完整轨迹的可行范围

ICC 固定作者代码 `ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4` 从 ImageFolder 读干净图像，VAE 编码后采随机 t 并 `q_sample`；CA 的 `mag` 分支做一次模型求值，CD 的 `delta_mag` 分支从随机 t2 出发，先求一步 x_t1，再在 t1 求值统计差分。这并非先跑完整采样轨迹。256 在 `gen_decomp.py` 中是输入数据采样条数，不是256条完整去噪轨迹。[作者 gen_decomp.py](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py) · [作者统计循环](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py)

对应到 Wan 视频时，若已有 **真实干净完整 clip** 或其 **正确 Wan VAE 编码 x0**，可直接执行 forward-noise 与随机 t／相邻一步统计，无需完整 D4/O40 rollout。需要支付 VAE 编码（如未缓存）、条件编码和相应模型前向的离线成本。适配的 noise/sigma、时间索引、expert 切换与相邻一步定义由机制实现文件单独确定；它们不能由目前 native D4 两端点替代。

单幅 I2V 图片只能形成条件支路。将首帧 repeat 成一段视频，会人为产生静态时间分布；第一帧＋零帧的 VAE 条件张量也仍是条件支路。二者均不能作为真实动态 clip 的 clean x0。当前 FFN U/Y、最终生成视频与真实 clean clip 保留不同数据身份。

## 4. 最小资产缺项与直接行动

1. **先绑定11条真实 clean clip，而不是先生成11条D4。** 可以读取已有集群真实视频资源；每条绑定可读路径、payload hash、clip帧范围/fps、图像条件与文本、来源/重复簇及CAL角色。若现有VBench JPG没有对应原片，保留CAL11/CAL22的规模，另立明确的clean-video校准来源名单；不能凭JPG文件名推测或构造原片。
2. **有clip而没有x0时，直接按Wan VAE编码并缓存。** 记录 clip hash、VAE hash、裁剪/采帧规则、缩放、shape、dtype 和 x0 payload hash。已有正确编码则直接复用；这一步不要求完整视频生成。
3. **复用现有11/22预算，重新定义校准样本数的含义。** 主校准11条clean clip；CAL22仅扩11条。每条随机t与噪声抽样是该clip上的重复观测，不增加独立来源数量。CA/CD采样和统计规则按作者机制执行，不使用先前的D4端点择一哈希规则作为主参考。
4. **现有Dense生成资产作为明确的备选来源单列。** 若集群已有CAL11终态raw latent，可省去重新生成和MP4再编码，但需先拿到可重载payload与身份；所得数据是模型生成分布。若只有CAL11生成MP4，可以VAE重编码，同样属于generated-video分布。当前只公开核实到BWS的3个CAL生成MP4，不能用3个DEV补成6个CAL，更不能声称已具备CAL11。是否采用这种备选来源需有明确研究决定，本审计没有自动选用。

资产检查结论是 **真实clean数据尚未绑定**，不是要求补跑D4/O40。现成bank和已有生成视频全部保留原身份，可继续支持旧局部结果、条件复用与显式生成分布的对照。
