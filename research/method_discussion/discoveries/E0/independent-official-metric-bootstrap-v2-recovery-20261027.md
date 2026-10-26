# Volatile-cache accounting now separates disappearing paths from real I/O failures

**DISCOVERY.** A fresh isolated import attempt can reuse the complete verified package set while accounting for uv's changing cache tree. The v1 failure identifies an enumeration race: a temporary directory vanished during `pathlib.rglob`. A competing explanation would be a permission, disk or other I/O failure; v2 explicitly propagates those errors. Two real concurrent-deletion regressions and an actual unreadable-directory test distinguish the cases before another bounded attempt.

**EVIDENCE.** The executed v1 stopped during wheel installation, before the metric import child. Its frozen `RUN.json` records `FAILED_SCOPE_DEBIT_PRESERVED`, `FileNotFoundError`,67.92224083840847 seconds and510,036,785 new file bytes. All91 package archives completed download and SHA verification. Recorded checkpoint/model/forward/official-score/GPU counts are0. This failure provides no evidence about pretrained metric compatibility or video quality. The v1 source/protocol/lock/tests, RUN, ATTEMPT and bootstrap log were independently checked and preserved.

The new v2 artifact set is independent:

| Object | Path / SHA256 |
| --- | --- |
| Tool | `tools/e0_official_metric_environment_v2.py` / `0981c79162497ed29ee941aeefd2011180d6e930010b1c090fc8d8bc3c9c1a79` |
| Protocol draft | `config/e0_official_metric_environment_20261026_v2.json` / `73fa61364e8d92964ae887306eb12cd1d427adbea332fef59b09f50eaa5194cb` |
| Tests | `tests/test_e0_official_metric_environment_v2.py` / `ed29dba12778e4ef5b34da5670561cace3aad11399c6e30b390142b346356b32` |
| Unchanged package lock | `config/e0_official_metric_environment_lock_20261026_v1.json` / `a0a715cb9070e9f029c6c91d00e8bd4a59ee1ebdf231a5800abb88ff199c4dbc` |
| Original failed v1 RUN | HF-relative `derived/ffnds/E0_official_metric_environment_v1/RUN.json` / `84e12a50f7baa9af5bd19219c8333a822a201b48e617265622e5cd36626a608e` |

The unchanged v1 lock supplies package/source/runtime identities only; v2 has its own ID, protocol registration, attempt debit and HF output directory. The tool copies the finite set of needed runtime functions into the new module; it does not modify old function globals or execute the old bootstrap entry point. There is no resolver or downloader entry point in v2.

The accounting walk uses `os.scandir` and `DirEntry.stat(follow_symlinks=False)`, counts regular files, and queues real directories. It tolerates only `errno.ENOENT` from a disappearing enumerated entry or queued child directory. `EACCES`, `EIO`, `ENOTDIR` and every other error propagate. Symlinks to the shared runtime are excluded from new-file bytes. The scan is a non-atomic snapshot of a volatile tree, so a disappeared temporary file need not contribute to that snapshot. Disk reserve and wall limits remain independently checked. A final accounting error produces an explicit failed receipt with its errno and unknown total, then is rethrown; it is not converted to0bytes.

All91 original archives were independently rehashed during this preparation and matched the immutable lock. v2 references those regular files directly under the v1 artifact directory, with no duplicate archive download or copy. It verifies the old receipt and source hashes, checks each reused archive before installation, and checks them again after installation. Offline uv consumes only the locked wheel files; the seven reviewed source-only package subtrees retain their existing exact recipes. The same28 declared runtime distributions and all original metric/constructor/load/forward/CUDA/network import guards are preserved. Root may inspect the v1 partial environment as historical evidence; v2 uses a fresh directory.

**Verification:** all17 software tests passed. Two coordinated background threads remove real files/directories at the two accounting race points; the subsequent real kernel `stat`/`scandir` calls produce ENOENT. The tests do not inject a replacement FileNotFoundError. Permission coverage includes an actual chmod000 directory on this non-root process and explicit EACCES/EIO/ENOTDIR propagation checks. Further tests retain helper-SHA/origin/path precedence, checkpoint/network/subprocess rejection, source-subtree integrity, package closure, symlink accounting exclusion and immutable archive reuse. These are software/physical-file observations, not metric-model or video evidence.

**EXECUTION boundary.** Scope ID is `E0_OFFICIAL_I2V_IMPORT_COMPATIBILITY_V2`; HF-relative output is `derived/ffnds/E0_official_metric_environment_v2`. Root must register its exact final protocol SHA before its one new attempt. Limits remain20minutes,2CPUs,4GiB new files,≥10GiB free reserve,0GPU assignments and0checkpoint reads/model constructions/parameter creations/forwards/scores. The same application-level guards apply; no kernel sandbox is claimed. No automatic retry or package-family fallback is provided. This agent prepared and tested the draft without creating a v2 attempt or environment, changing ACTIVE, installing packages, or importing actual Torch/metric modules.

After registration, the root-controlled command is:

```bash
.venv/bin/python tools/e0_official_metric_environment_v2.py run --protocol config/e0_official_metric_environment_20261026_v2.json
```

The scientific consumer remains **P23 → P24 → P25 → D09**. Official metrics would be **posthoc** source-bound appearance/motion/detail observations; their relationship to a specific FFN/latent mechanism is still **conditional**. Passing this import scope would support the next bounded model-compatibility check, without adding FFN savings, official clip scores, human acceptance or stage promotion. `quality_accepted=false`; realtime remains paused.
