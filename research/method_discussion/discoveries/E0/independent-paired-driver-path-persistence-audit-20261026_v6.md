# E0 paired driver path and persistence — independent static audit, 2026-10-26

结论：对修复历史路径及异常证据留存的最终源码未发现新增阻断。本文保存已经完成的只读审查，不重新运行测试、模型或视频，不修改旧审阅与 ACTIVE。

| 已审对象 | SHA-256 |
| --- | --- |
| `tools/run_e0_paired_full_depth.py` | `c4d362c5b7dca25edc5eb63e6dd769bffde236517ac147d9445be36799d7d0ca` |
| `tests/test_e0_paired_driver.py` | `70fb630ea0ef2f14417b7106d102694c63c721f292946a86fada438af0eebdad` |

审查背景：root 报告 v2 的 DD 已完成四步与两个视频，但 driver 将 `HF_relative` 错当作相对 HF root，在历史 receipt 检查处失败；原先 TRACE 写出位于该检查之后，内存中的 160 个 site 没有落盘。两个视频与旧 dense 视频的 SHA 相等是 root 的只读核对结果，不能据此宣称保存了 DD 全深度 trace 或完成了旧 latent 一致性验证。失败 RUN 保留，后两 pair 停止且不重试。

- `historical_branch` 统一解析 `hf_root/ffnds-derived/HF_relative`，核对实际 receipt SHA，要求唯一 source_index 与正确 seed，再核对真实视频 SHA。解析后的 receipt 不得逃逸 derived root，视频不得逃逸 receipt 目录。
- CPU preflight 遍历所有新 pair 所需的 dense/candidate 历史分支，先读取并核验实际 receipt、seed 和视频；execute 要求 preflight 的历史核验标志，先于预约模型 attempt。这里没有 torch/native 导入或模型调用。
- `pipe.generate` 返回后先保存 TRACE、OBSERVATION、两轨 trajectory 和已有调用计数，再进行 teacher 解码；两视频完成后又先写运行计数与成本，然后才检查计数预期及历史 latent/video 一致性。因此这些后置检查失败不再阻止此前 trace 落盘。
- Python 可捕获异常进入 except 时，`persist_observation` 再尝试保留已有 site/step、trajectory、observer/runtime 计数及 residency。部分记录使用明确 partial schema/status，不冒充完整原生证据；`recorded_site_geometries` 与 observer 的 `fully_completed_sites` 分开，`incomplete_or_unverified_tail_sites` 标出最后可能未完成 strata/vector 写出的记录。
- 新的 prior-control preflight 绑定失败 v2 RUN、两个视频 SHA、原协议/source/seed 与数学组件；此复用限于已有 DD 视频对照，不能恢复其未保存的全深度记录。新四视频范围仍须另行冻结并获得用户授权，本文不授予执行权限。

已独立核对 controller、observer、weight residency 与 trace consumer 四个源码 SHA 均与 v2 一致；本次改动是路径、preflight 与证据保存时序，数学调用未变。没有把修复后的保存能力追溯赋给已经结束的 v2 失败 attempt。

读过八项 driver 软件测试源码，其中新增检查覆盖 derived-root 路径、真实视频/seed、路径逃逸与 partial tail 语义。八项 PASS 为 root 报告，本审查未执行测试；fixture 不等于实际模型验证。

留存仍是 Python 可捕获异常下的 best effort；SIGKILL、断电或 OOM 硬终止可能丢失尚未写出的内存记录。完整 trace 的保存也不等于 reference 一致性通过或视频质量接受。旧 v2 丢失的 trace 不可补称已保存，本文件不启动、不重跑任何模型。

此审阅绑定上表快照，未等待或审定未来 v3 协议；后续源码变更不自动继承结论。原 v4/v5 审阅与所有旧 receipt 保持原字节。
