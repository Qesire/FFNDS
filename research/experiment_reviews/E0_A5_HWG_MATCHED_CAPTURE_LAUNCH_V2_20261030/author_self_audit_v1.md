# PART1 启动登记修复：新有限范围自审

作者 `/root`。原PART1_V1的实际supervisor收据为JSONDecodeError、child_pid=null、process_exited=false、returncode=null；没有启动子进程或模型，没有新视频。root逐任务更新ACTIVE时使用了非原子write_text，与前一supervisor的重复validate读取竞争，形成短暂不完整JSON。这是协调者写入错误；旧失败收据/日志完整保留，不当成EXIT0或自动重跑。

root已使用临时文件写完整JSON后os.replace的原子登记更新。新的PART1_V2只补未启动的来源96 plain+observed两视频，scope、source、seed42、模型/cache、代码1018项、数值recipe、21项fixture、逐file读写账、十状态/视频中性、预算全部与已独立批准的PART1_V1相同。只新增scope ID、fresh HF输出目录、实际无child失败的绑定，并给RUN和条件VERIFY各建一个新private job。不会重跑其它三个正在执行的scope，不修改其代码/协议或数据。

独立review需核对新旧差异、真实失败parent和两个精确argv/SHA；不能把原批准直接充当新scope批准。root在登记/启动本次时先一次原子登记全部状态，再启动单个job；不在supervisor启动窗口进行非原子写。执行并发仍≤4 GPU、单job1张B worker的同一空闲A100；CPU验证仍只在真实producer EXIT0后。

竞争风险沿用原审：真实全部160采集可能因数值、内存或文件中性失败；此修复不保证成功。新scope失败仍不自动重试。此登记修复本身不执行新科学数值，后续采集和验证的证据边界保持原合同。

精确新协议为 `config/e0_a5_hwg_matched_capture_launch_20261030_part1_frozen_v2.json`；私有job表 `/home/xiang/.config/ffnds/a5_hwg_capture_launch_20261030_v2/BOUND_JOBS.json`。当前REVIEW_REQUESTED，未注册/未启动V2。
