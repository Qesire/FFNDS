# S2 来源扩容元数据审计（独立草案）

2026-10-29 · DRAFT_METADATA_ONLY · `/root/s2_theory_design_independent`。本表是待核查的元数据定位示例，并非已选cohort、已证明独立来源、已观测动作或运行授权。科学选择待当前L/P/I结果。没有打开冻结确认图像、解压归档、生成视频、读取大张量或调用模型/GPU。

现有[库存报告](../../../20261029/RAW_BF16_FRESH_SOURCE_INVENTORY_20261029.md)扫描7个source bank（含diverse_confirmation20）记录123个唯一SHA，原84个crop缓存均匹配这些已知SHA；244个原始候选仅做名称/行号排除。再以[已用于新10拟合的来源](../../../../config/e0_raw_bf16_fresh10_full21tau_sources_20261029_v1.json)行号及pilot 898/96排除，剩余234条旧库存预候选。其已扫描库覆盖不是全历史；图像SHA未知，不能声称与训练、开发或冻结确认完全不重叠。10张原source现已参与拟合，旧bank中fitting=false表示当时状态。

存量归档locator：`/home/xiang/.cache/huggingface/hub/models--lxl-158--vbench-i2v/snapshots/577ba45c1653ede26e29996737f42841f2ec9eea/vbench2_beta_i2v/data.tar.gz`，stat存在且目标2,607,670,436 bytes；此前[冻结归档协议](../../../../config/e0_raw_bf16_fresh10_source_archive_cpu_20261029_v2.json)记录归档SHA `aa6af89775242f35dbcd1f4c53421a6547d0700b1a99206e98b87a28ea393865`，本次未重读或重算。元数据locator：`/home/xiang/.cache/huggingface/hub/models--lxl-158--vbench-i2v/blobs/33c3e3f66b5a88534d2665001934ca5f5cfa76ca`，stat存在364,832 bytes；此前库存记录SHA `8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`。原始revision `577ba45c1653ede26e29996737f42841f2ec9eea`。表中member是依据历史producer命名规则构造，尚未打开归档核验存在、成员bytes或SHA。

每个候选当前：image SHA = **NOT_MEASURED**；与7个库按名称/旧行号无已知匹配，fresh10/pilot按行号排除；跨全部训练、开发和冻结确认的精确SHA/近重复 = **NOT_VERIFIED**；实际文件 = **NOT_EXTRACTED**；subject/camera/local/background/occlusion真实行为 = **NOT_OBSERVED**。表按旧元数据中六种图像类型各取首3项展示可定位性，并非动作分层或最终选择。

| 元数据row | 元数据image_type | 预期归档成员 locator（待核） | image SHA / 行为分层 |
|---:|---|---|---|
| 910 | animal | `data/crop/16-9/a group of fish swimming over a coral reef.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 911 | animal | `data/crop/16-9/a toad is sitting on top of some moss.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 912 | animal | `data/crop/16-9/a great white shark swimming in the ocean.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 977 | single-human | `data/crop/16-9/a person riding a skateboard on a concrete floor.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 978 | single-human | `data/crop/16-9/a woman with long black hair is posing for a picture.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 979 | single-human | `data/crop/16-9/a woman sitting on the ground in front of a guitar.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 1039 | multiple-human | `data/crop/16-9/an indian police officer talking to an old woman.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 1040 | multiple-human | `data/crop/16-9/two people scuba diving in the ocean.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 1041 | multiple-human | `data/crop/16-9/A man and woman dressed as sugar skulls in a field of flowers, sharing a loving gaze with each other.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 922 | transportation | `data/crop/16-9/a blue car driving down a dirt road near train tracks.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 923 | transportation | `data/crop/16-9/a sailboat is drifting on the ocean.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 924 | transportation | `data/crop/16-9/a couple of boats floating on a body of water.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 616 | scenery | `data/crop/16-9/a mountain range with a sky background.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 624 | scenery | `data/crop/16-9/a large bonfire is burning in the night sky.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 632 | scenery | `data/crop/16-9/a close-up view of the flames of a fireplace.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 456 | indoor | `data/crop/16-9/a kitchen with a sink and a lot of glasses on the counter.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 464 | indoor | `data/crop/16-9/a wooden table in front of a brick wall with bottles on the wall.jpg` | NOT_MEASURED / NOT_OBSERVED |
| 472 | indoor | `data/crop/16-9/a room filled with paintings and statues.jpg` | NOT_MEASURED / NOT_OBSERVED |

## 需要获得的证据与可执行下一producer（未注册）

1. 依据当前L/P/I决定来源级假设、source数量和预先固定的、与结果无关的选择规则；将所有已知训练/开发/冻结确认canonical的图像SHA/名称/原图簇键，以及新10、pilot和其他历史source manifests合并成版本化排除集合。不得读取确认图像；对缺hash的旧manifest停在不确定，先补可信元数据。

2. 单独注册有限CPU archive producer：读取上面的精确归档版本并核算流式I/O与磁盘预算，只提取预声明的候选成员至新独立目录，保存成员名、源图字节SHA、source row、revision、prompt ID、原尺寸、提取RUN/EXIT；遇重复或缺成员按事前规则排除，不重写旧来源。精确匹配所有历史及保留确认的SHA；跨不同裁剪/编码同一图簇仍要原始identity和近重复审查。无比对通过记录即不能计作新源。

3. 做新候选输入人工图簇与可见性审查，再在独立冻结scope生成同checkpoint/seed Dense参考视频；仅在真实输出上标记对象运动、相机全局运动、局部动作、背景动态、遮挡/突发及静态对照，可多标签；比较前后向flow一致性、对象/背景区域和人工核查。prompt指令与静图类别只作选择意图，不能充作行为标签。每类缺口要公开并扩来源，而非用多个token或同图多seed填足12个image clusters。

4. 原生采集按实际批准覆盖4step×40层×21τ的teacher及两个自由student，记录row、g/r/u、足够邻域、source图像SHA/seed及成本；若关键对象运动超出5×5邻域或受遮挡，预定义跨邻域/扩大采样与失配标记。参考Dense光流是离线oracle，在线信息与成本另计。后续报告以image cluster等权，source-held与冻结确认严格分离。

最直接缺口：当前可证明符合“与全部历史及冻结确认SHA不重叠”的候选数 = **未测量**，不能从本表声称已有12个；全历史排除集、18个成员真实byte SHA/near-duplicate、真实Dense视频行为和完整资源收据均待未来producer。
