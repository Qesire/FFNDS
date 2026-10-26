# E0 paired video strict gate — independent static review, 2026-10-26

结论：在本次绑定的新增工具与 driver 接入范围内，原审计指出的 dense 语义身份与实际生成清单覆盖两个缺口已形成明确的软件检查；本审查发现的删去 reference 绕过也已关闭。此结论限于静态源码，不是实际视频证据或模型执行许可。

| 对象 | SHA-256 |
| --- | --- |
| `tools/ffnds_paired_video_strict_gate.py` | `59d6596cff43cc6197d4b1ab76eb90e9ca92855901b6d3394b7eabc257c9a6d3` |
| `tools/run_e0_paired_full_depth.py` | `82df60878f7a010e1682977aa404049d2d217573ac30b352545c030821ed9b2e` |
| `tests/test_ffnds_paired_video_strict_gate.py` | `9a0272c1733b3dbb1e093e0726eefa6accfc41b4ff1476cf5bd4175dacc7a451` |
| `config/e0_paired_full_depth_20261026_v2.json` | `dfefff076066c766766897ab76bb161b1e82fbfb74c5a0031c3f91c68ef95ce0` |
| 原 `config/e0_paired_full_depth_20261026.json` | `6e9e6a39898d12c90f8378cec7ee77a7f26a909e5690d8e1547896bdaf67c188` |

我实际读取新增 strict 工具、driver 的 evaluate_pair/不可变 NATIVE_RUN 写出路径、旧核心 gate/validate 及 strict fixture 源码；没有运行这些程序。问题来源为 `E0_TRACKING_RESULT_AND_DRIFT_AUDIT_20261026.md` 第三节，不把旧门禁 PASS 或既有四处错误 digest 当作 dense 身份真值。

- driver 在原 BUNDLE/core gate 通过后写 `STRICT_BUNDLE.json`，绑定 BUNDLE SHA，并另列已先行写出的 `NATIVE_RUN.json` 路径和 SHA；strict 失败抛异常，driver 不到达最终成功状态。strict 工具已加入 driver 必备源码集合。
- inventory 不取 bundle 自报的 expected_count：strict 从各个 SHA 已核对的 native receipt 强制读取 teacher/student 两分支，逐视频核对真实文件 SHA，构造 `(receipt SHA, resolved video path)` 权威集合。core 条目必须集合相等且长度相等，重复或漏报失败；receipt 同目录实际 `*.mp4` 也须等于该 receipt 清单。当前 driver 每 pair 独立目录、两条平铺 MP4，符合这项文件布局假设。
- dense 身份不复制 report：解析实际引用 receipt，先唯一匹配真实路径及视频 SHA。原生 paired schema 根据分支对应 `pair.modes` 得到 dense/candidate；历史 schema 仅接受 `mode=dense` 并在 phases 唯一匹配视频。其 model_digest 使用该实际 receipt 的 checkpoint/cache manifest 摘要，并将 dense 的 prototype_family 固定为 dense、bank 摘要置 None；candidate 使用其实际 bank 摘要。
- strict 将报告 model_digest、可取得的 source_id 与 seed 对回上述身份，并要求参考 operator 确为 dense。旧核心 gate 继续负责报告/视频/运行与参考 receipt 的 SHA，以及完整逐帧指标；新增检查与它组合使用。历史 dense schema 此处只提取 seed，source/index/prompt 的历史对应仍依赖 driver 原有冻结输入与 reference 交叉核对，不能把 strict 独立工具说成完整历史来源审计。
- 首次源码中，`if report.get('reference')` 允许清空 reference 跳过参考身份检查。已即时反馈；最终源码对每个 paired native clip 强制非空 reference/reference_identity，因此清空任一项都会留下失败，不能绕过 dense 义务。原生两分支现在都必须有配对参考。

我读到八个 strict fixture，覆盖正常配对、复制 candidate digest 冒充 dense、降低 expected_count 漏片、额外 MP4、无独立 inventory、删 reference、实际身份/文件漂移，以及历史 dense 用自身资产重算。fixture mock 旧 core gate，仅证明新增义务的设计，不能当作组合 gate 或真实解码的端到端执行。root 报告八项 strict 与十七项 paired CPU 测试 PASS；本审查未独立运行测试。

旧 gate 和 evaluator 的实际 SHA 仍分别为 `3f1b70fa25b7c961f8141ba396ee3b9507e028d60fd168e642e0548de99a2d07`、`0c6ab066674adb2a04097978ced52abcd7cbf1125918b74d59cf976a56ffec66`，与 v1 协议绑定一致；旧报告未在本审查中改写。新增 driver 已不同于 v4 的 `093324…b977`，不能继承 v1 原绑定或 v4 整体快照结论。收尾已核对另冻 v2 的 19 个 driver/env/launcher 与 23 个 native 文件摘要均匹配，六条路径/单 GPU budget 未变；最后新增的 environment 字段仅记录 torch/CUDA/GPU/Python/TF32/BF16 reduction 设置，未修改算法设置。本次内容审阅仍只针对 strict 接入及这段新增记录，不声称重新审计整个 v2 launcher/environment。

信任边界是已冻结、由真实生产过程写出的 generation receipt，而非防止攻击者同时伪造 receipt 与所有摘要；inventory 也只覆盖调用者提供的 native receipts 和规定目录。当前每 pair driver 显式固定唯一 NATIVE_RUN，避免 bundle 自选少报该 pair 的视频。正式质量仍须官方/人工证据，此 gate 不授予质量接受；无 native 作业、GPU 或新视频，原始双轨实测仍为零。

保留 v4 审查 `independent-paired-driver-final-audit-20261026_v4.md`，SHA `3972c2ab289a5c2a7a4255b1080e770eaf5499907f81dfc8e94b0bcea5099c1a`。本文仅新增独立审阅，不修改旧文件、实现、ACTIVE 或注册表。
