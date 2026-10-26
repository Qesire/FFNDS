# DISCOVERY · 真新phase2来源与unused-outcome来源可以分别保留

**从原始官方元数据中，已按结果观察前冻结的顺序找到四对不在任何历史冻结输入 cohort 中的真实图像/prompt，并从现存 HF 原归档核实了全部输入字节。** 原官方行908/376/425/437分别为 animal subject、architecture background、camera pans left、camera zooms in。它们使 phase2 可以满足更严格的新cohort定义；R17 剩余条目仍可作为 unused-outcome-holdout 独立资源，而不混称成同一种证据。

作者：independent_e0_audit。日期：2026-10-26。本轮仅元数据/源码/回执、普通文件SHA、JPEG格式/尺寸与一手许可页面；0模型/GPU/保存张量数值分析/视频/新素材下载/独立图片文件写入。新写本MD及同名主题JSON，不改v10、ACTIVE、atlas、旧协议或共享canonical；输入原件继续只存在HF原缓存归档中。

## EVIDENCE · R17[6..22] 后续用途的实际闭合

R17 canonical 身份仍为 `758f74e79dc9ebbf75a42aa64db024ac79a8a0c180d01cd80cf728de5f455c10`。原R17批次实际完成57次生成，fresh quality只涉及[0..5]；其后 C1N0 实际完成8条生成请求，去除两个旧CAL24控制后为 R17 `[0,2,8,10,16,20]`。72个 finite scope 的来源集合也完全相同。它们不是仅写在未执行计划中的“潜在暴露”。

| 原始记录 | 本轮读取的事实 | SHA256 |
| --- | --- | --- |
| `R17E0_partial_generations.json` | 57 completed，quality来源0–5 | `5f867aebbe4acb226330f3ff4dabf9ae407a08daf851baca74b881701c1de014` |
| `R17E0_partial_budget.json` | 57 full generations，metric calls0 | `b16557fc90900951ab11397277ef44257a63934365d5a7cdbd15785f67129597` |
| `C1N0_native_results.json` | 六个fresh来源，72 finite scopes，实际8请求 | `8b52040c84e229cd87f4983da66060d9d48034dc309740ec5074bae4005f3469` |
| `C2N0_native_results.json` | requests空，预算0 forward | `9dc381e0f57ef15140d4c2e42a3724b111f3d82d0142c6521f0c876ca6639129` |
| `C2N1_native_results.json` | requests空，预算0 forward | `c4be9ba86d222c108f8b106fd29804ce27f49bf963fc92fc37e4201c8f8d46c4` |
| `C2N2_native_results.json` | requests空，预算0 forward | `0cd3d6b5b0ae4e5919080e1359b96d4bc4d61bb3e772519dc74e0a06414a4bf8` |

这些路径均位于 `research/20261023/`。本人亲读 C1 driver 的实际jobs构造，及 C2N0/N1/N2 三个driver：后者均指定 R17[6]，并保留旧CAL24[0]身份控制。虽然三个实际 C2 回执都没有完成生成/FFN/selector调用，[6]已参与试验选择和工程/内容预审，因此保守地排除出新的phase2候选。

剩余 R17集合为

\[
U=\{7,9,11,12,13,14,15,17,18,19,21,22\}.
\]

本轮对这些条目的真实image SHA及**带JSON引号的精确source_group**作全文联接：仓库 `research/config/tools/reports/runs/.runtime/inputs` 的 JSON/JSONL/MD/Python/log 中，只命中原canonical和原 `R17_fresh23_input_audit.json`；HF的两个 derived metadata根中只命中其 `INPUT_AUDIT.json` 镜像。精确匹配很重要：`i2v_background:2` 的未加边界子串也会命中历史 `:20/:22`，不能当真实碰撞。

另外检索了 canonical SHA 的全部后续引用，实际执行入口只有 R17 generation/evaluator、C1和三个C2；官方R17 evaluator要求115条quality视频，实际完成27条且预算metric0，所以其全23来源的官方评分从未执行。当前普通fit源码没有新增R17训练入口：旧C-static/RR1等绑定12/24来源训练数据；当前Affine54 bank明确只校准CAL24[4,8,16,20]。这里的阴性证据限于上述可追溯项目记录，不声称检查了任意外部未记录作业或原模型预训练集合。

U中12张实际HF图片字节均匹配 canonical，且其 image/prompt/group 与旧R17历史排除清单的98个图像/文字身份、87个group无碰撞。**但全部23个R17条目都曾写入原23来源/九维metric的cohort设计。** 按协调者本轮明确采用的最严格 literal 规则，U可标 `UNUSED_OUTCOME_HOLDOUT`，不可替代从未进入历史cohort/metric设计的新phase2。v10的OPEN因此细化为：历史结果用途已查清，严格新cohort资格仍不授予R17。

## 严格新四来源：固定选择顺序与真实身份

本轮亲读现存原始 `vbench2_i2v_full_info.json`，1118行，实际文件SHA为

`8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`。

在读取任何新模型结果前，先固定四个有序stratum：

1. `i2v_subject` 且 `image_type=animal`；
2. `i2v_background` 且 `image_type=architecture`；
3. `camera_motion` 且原prompt含 `camera pans left`；
4. `camera_motion` 且原prompt含 `camera zooms in`。

每层取原metadata行号最小、image_name/prompt/source_group未在全部历史输入集合中出现的条目，选过的底层图片名立即从后续候选排除。历史集合来自当前全部38份 `.runtime/inputs/**/*.json`，含CAL12、CAL24、formal20、disjoint20、diverse-confirmation20、research任务输入及**全部R17**；记录了每个文件的确切SHA。合并后排除121个image SHA、121个image_name、121个prompt、110个group。图片字节相撞即保留失败，不在看到结果后自动换source。

| 原metadata行 | 原filtered/group | 原prompt | 图片尺寸 | 图片SHA256 |
| ---: | --- | --- | --- | --- |
| 908 | `i2v_subject:36` | a small monkey holding a piece of food in it's mouth | 5328×2997 | `fa3b72dae6689006f129de64d0ece1de551b64b9bc03c009ae50bd3a23826592` |
| 376 | `i2v_background:47` | a reflection of a city with buildings in the water | 3136×1764 | `cad88b8fe1a0d0882c3c428f0d6b03ae22be5c7f4f50177f377cc8bd58749ac0` |
| 425 | `camera_motion:371` | an art gallery with paintings on the walls, camera pans left | 1920×1080 | `ecc932d9e1fb13b08912b67484052da51a1bf780d16fbf9bad69baf1c64af958` |
| 437 | `camera_motion:382` | a room with a lot of pictures on the walls, camera zooms in | 6000×3375 | `6b82a9a2cc6f212dce6d979931d647646921fe523175ecf53715b1c48a122df0` |

四个prompt SHA、文件bytes、archive_member、原图裁剪/URL元数据、38项历史文件SHA和排除集合全部在[本次JSON](independent-phase2-official-four-input-audit-20261026_v11.json)中。该JSON最终SHA：

`0575e63303620736ab7b8bca38237d1cdb80491e4eab32249ec9a10865e4bb6b`。

`source_index`在这里始终是原官方metadata行，不伪造CAL24[24..27]；若协调者冻结新本地0–3顺序，必须保留以上原始行号与group、图像/prompt SHA。两个camera记录的官方 `dimension` 确为仅 `camera_motion`，指令逐字来自原prompt；这证明任务标签，不证明尚未生成的视频实际遵守了相机动作。

## 原始素材、许可和缓存取得方案

本轮原字节来自已经下载的 `lxl-158/vbench-i2v@577ba45c1653ede26e29996737f42841f2ec9eea` 原归档 `vbench2_beta_i2v/data.tar.gz`，2,607,670,436 bytes。本人重新完成整档SHA核对：

`aa6af89775242f35dbcd1f4c53421a6547d0700b1a99206e98b87a28ea393865`。

只读取 `data/crop/16-9/<原image_name>` 对应的四个regular member到内存，核对byte长度、SHA、JPEG格式及16:9尺寸，并确认与121旧image SHA及相互之间均无重复。未从图片网站重新下载或替换裁剪，也没有将原图写入仓库。四张共4,648,655 bytes。

原归档 `data/i2v-bench-info.json` 实际454,612 bytes，SHA

`7dbe47684eadb2b4ace3fe452a4d8e8ceffa9c788d9c1ff048af3bb2585f96bf`。

本人按原file_name唯一联接，获得以下原图页面并实际打开：

- [Jimmy Chan 的monkey照片](https://www.pexels.com/photo/photography-of-a-baby-monkey-eating-vegetable-1000529/)：原页标免费使用，连接Pexels许可。
- [Pixabay 的city reflection照片](https://www.pexels.com/photo/yellow-and-brown-house-painting-164455/)：原页标 CC0。
- [Matheus Viana 的gallery照片](https://www.pexels.com/photo/colorful-arts-hanging-on-wall-2372978/)：原页标免费使用，连接Pexels许可。
- [Thomas balabaud 的room照片](https://www.pexels.com/photo/empty-dining-tables-and-chairs-1579739/)：原页标免费使用，连接Pexels许可。

本人亲读[Pexels原许可页](https://www.pexels.com/license/)：描述免费使用/修改及可选署名，并列明原样转售、股票照片/壁纸平台再分发、暗示代言等限制。本审计记录这些实际条款和原图来源，不把[VBench代码仓库Apache2.0](https://github.com/Vchitect/VBench/blob/master/LICENSE)替代成所有照片或画中作品的许可证。原HF镜像README通过浏览器取回失败；直接原档URL→原Pexels页面链已闭合，不编造镜像license。官方[I2V Image Suite说明](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md)也明确原metadata包含URL/caption/crop信息、提供16:9裁剪，并把camera评估定义为prompt相机指令遵循。

未来独立冻结输入时，最省成本的取得方式是复用上述HF原归档，使用逐member读取而非 `extractall`，将这四个已经绑定SHA的原字节写入新的 **HF derived** cohort目录；例如以 `official908/input.jpg` 等原行号命名，再核对同一SHA/大小。原素材保持HF内；仓库只保存manifest/哈希/报告。缓存缺失时，按同repo/revision/filename下载到HF、验证完整archive SHA后执行完全相同的成员读取；不回到网站重新请求可能变化的JPEG，不自动用别的图片补缺。该步骤不含模型或视频授权，当前尚未写共享canonical或staging资产。

## 竞争预测、下一观察与证据边界

phase1六来源全部已暴露，新四对则能检验同一冻结bank在新conditioning image/prompt上的泛化：若历史两来源的状态放大/注入符号仍成立，四条新来源应保留对应预登记方向；若控制随内容/seed反转，现有机制应以来源条件化重新表达。无论哪种结果，都保留source-level完整误差和视频指标向量，不能把960个嵌套site当独立视频样本。

至少两个新source可在随后协议中固定公共seed42及第二个预定seed54以交叉比较内容与随机轨迹；本文件没有生成任何初始噪声或赋予重复试验权限。保留原prompt，不为“更明显运动”改写文本。新四个覆盖1 subject/1 background/2明确camera任务，仍是探索下限，不代表所有类别或正式质量通过。

图锚为 G0/G1 条件输入/身份 → F04/F06/F08/F09 固定FFN函数 → D09视频结果。当前建立的是source身份、历史暴露与输入字节证据；通向原生新来源试验和画质结论的边均 conditional/OPEN。旧v10关于CAL24全暴露和R17后续用途未全审计的历史状态保持原字节，本v11补齐其后续判断。
