# Compact6016 isolated quality v2 · independent review

Reviewer: `/root/compact6016_six_video_build`. Verdict: **SUPPORT** for this exact software/draft, based on SRC, synthetic CPU controls and an actual stdlib-only isolated interpreter check. No scoring model, real official score, video decoder, weight tensor or GPU was executed by the reviewer. Full eight-module imports remain a separately registered worker CPU preflight, followed by its actual exit receipt.

| Artifact | SHA256 |
| --- | --- |
| `tools/e0_compact6016_consumer_quality.py` | `0e047e79cc77b91fa0bb45b88c065867f181315c92558ab56957c5f9920c187f` |
| `tests/test_e0_compact6016_consumer_quality.py` | `064b9d2f374cda9fe3a86796b2bd4766a54d23a2e93ab3f64fddb71722c5ca21` |
| `config/e0_compact6016_consumer_quality_20261028_v2_draft.json` | `f63296d1902fde0e9647437d6558bc5f0ddfea037a19458f478e52fb889059e4` |

The failure-specific change is concrete: choose the original metric interpreter with `-I -S`, rebuild its import path from the bound source view, setuptools overlay, metric site, project `tools` and interpreter stdlib, and resolve NumPy/pyiqa/Torch before importing package code. The project site directory is absent; the two explicitly shared Torch/torchvision packages still resolve through their registered symlinks to their actual project package files. Actual module origin, bytes and distribution version are checked after import. This prevents the v1 NumPy2-first import path from silently replacing the original NumPy1.26.4 dependency.

I independently launched the actual bound metric venv interpreter with `-I -S` for `inspect`, and separately called only `configure_import_environment` through stdlib `runpy`. Both exited0. The latter selected the actual HF NumPy1.26.4 and pyiqa0.1.10 package files and the exact project Torch2.10.0+cu128 file. It reported isolated/no-site/ignore-environment all1 and no project-site fallback; neither NumPy nor Torch was imported. All seventeen software bindings matched actual bytes/SHA. This is a real interpreter/path check, separate from the synthetic mocks, and is not an eight-module import result.

The model execution path requires both a separately exited byte-only CPU receipt and a separately exited eight-official-module import receipt before NumPy/Torch import. Each binds the exact protocol and source/environment digests plus its actual RUN SHA and exit0. The import receipt additionally checks all eight function origins/bytes, exact module origins/versions, zero denied attempts, hidden/uninitialized CUDA, bounded wall/output bytes and the exact eight activation plus one pool stateless wrappers. A missing receipt, different protocol/RUN, NumPy2 version, changed function file, partial import coverage or nonzero NN/checkpoint/CUDA work is rejected.

I read the actual imports-only guard chain. Before importing official modules it blocks checkpoint opens/deserialization, Torch hub/network/subprocess access, parameter/buffer creation and registration, ordinary NN construction and module calls, and CUDA lazy/native initialization. The two existing source-and-callsite-bound exceptions retain only the eight Transformers stateless activation defaults and one pyiqa stateless MaxPool wrapper; their finalizers check identities and empty parameter/buffer/child state. These are application/source-path guards, not a claim to sandbox arbitrary adversarial native I/O. The actual original module import closure still must pass on the worker.

All six existing clips are rescored in the same isolated process: six common dimensions × six videos plus two source-specific I2V dimensions × three modes =42 results. The unchanged original dynamic-degree sampling yields forty two-frame RAFT pairs per clip, so six clips pay exactly240 observed RAFT forwards and4800 iterations. The original threshold/count, top-five-percent flow statistic and Boolean score remain separate from continuous means/five fixed windows. All candidate/Dense deltas and ratios use the newly scored Dense values; old same-byte Dense scores are historical sanity only. Historical module origins remain NOT_RECORDED. Two exposed image clusters remain the independent units, and the output does not assert video quality acceptance or five independent samples.

The new scope/output preserves the failed v1 protocol `bf571e1b…`, RUN `dfd8f972…` and its actual exit1. It explicitly requires all old official-call/new-score/RAFT counters to remain zero and retains the old failed identity instead of overwriting or reclassifying it. No automatic retry is introduced.

Independent verification on 2026-10-28:

```text
.venv/bin/python -m unittest discover -s tests -p test_e0_compact6016_consumer_quality.py -v
37 tests; actual exit0; 0.328s
.venv/bin/python -m py_compile tools/e0_compact6016_consumer_quality.py tests/test_e0_compact6016_consumer_quality.py
exit0
<bound original metric venv>/bin/python -I -S tools/e0_compact6016_consumer_quality.py inspect --protocol config/e0_compact6016_consumer_quality_20261028_v2_draft.json
exit0; draft pending freeze; model_calls_executed=0
<same interpreter> -I -S <stdlib-only runpy/configure check>
exit0; NumPy/Torch both absent from sys.modules
```

Six focused existing v8 guard controls also independently exited0 in0.041s: exact eight stateless defaults, outside-callsite rejection, changed source/buffer rejection, checkpoint/network/subprocess rejection, and the two pool identity/order/state controls. They import no Torch/metric package. No blocking defect was found in this exact v2 version; registration and real exited preflights remain required execution gates.
