# Author self-audit: input-video visual V2 parts 2 and 3

Author: /root/teal_full_llm_calibration_build_terra. Old visual jobs for both parts actually exited 1 at taskset before RUN and before decoding any frame. Root omitted the CPU availability precheck; old attempt receipts remain unchanged. This is a new bounded scope, not an automatic retry. The actual worker affinity was checked as CPUs 0–15; V2 assigns part 2 to 12–13 and part 3 to 14–15.

Each scope is CPU-only: 2 threads, 300 seconds (600 outer), 4 GiB RSS, 32 MiB output; exactly 3 parent videos, 243 decoded frames, 3 EOF checks and 6 JPEG pages; no model, GPU or RAFT. Both actual parent native producer and independent typed verifier EXIT0 are bound by the protocol. A separate independent verifier requires its own registered scope and actual EXIT0.

| part | protocol SHA256 | fingerprint | run job SHA256 | verify job SHA256 |
|---|---|---|---|---|
| 2 | 66500c9ac5aa27e1a6abcd7a46620e84a25e3950e4cba66791b06ffdfe31dab8 | 4dfde8234d6b18229a5d81ef14178ed3ec7cd1bd05d91cff39f4fc750bd2f670 | cf7cb90ea2eb6a9fa6d3f8a7d9ce52b3947a177043ea2eacfad84bbf6ec607b4 | 3763f201cef0b29afe37e1b08f75a4a1e3a15218d49ba5e821e15d8774fc704b |
| 3 | e8105fc8336087be7c1070b6e204a8258dd7d7e669bf1273be2a5d0cb6d90f72 | 0bf8ea5a2c70585498a0d2160bb12cc93b4d3ce751c5092dd60a159b61ea1711 | 6f79d1faf35ef0806b2c5a01f96a1e85a12fdf0fb13e09cf792f52c1c5ee0ca1 | 5b97921910c74cbbf30d89ec1569c94a9c926465233ca08216b45151730ef3f1 |

The exact source closure includes new visual engine, execution, jobs, freezer, independent verifier and tests. The verifier loads the V2 visual engine. Per-phase SHA cap 16 GiB, journal 16 MiB and worker-extra 64 MiB sampled soft; nested parent metadata and other uninstrumented I/O remain UNKNOWN.

`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest tests.test_e0_chipmunk_input_video_quality_execution_visual_v2 tests.test_e0_chipmunk_input_video_quality_visual_v2` — 6 CPU synthetic tests passed. The real run and independent verifier reject missing ACTIVE, bad producer EXIT and wrong CPU affinity; no actual frame decoding was performed. Exact review and root registration remain required.
