# Parameter-free import defaults have a precise boundary before pretrained models

**DISCOVERY.** Eight original Transformers activation objects are bookkeeping defaults with no learned state. Their construction can be separated from pretrained model construction by exact source/type/call-site/count bindings while continuing to reject parameters, buffers, forwards and checkpoints. The competing outcome is that another import requests a different object or action; v4 records that first denial, including its stack, rather than expanding the exception automatically. The next useful observation is the independently registered v4 import attempt, followed by a separately bounded original-loader/clip study if all imports pass.

## EVIDENCE: the completed v3 result and exact original source

The root registered and executed v3 protocol SHA **`80ee6f0a2e163ed7653b6c471f3d79b6d5fbc5c219784f5633a0dfce5c485fcb`**. Before registration the root corrected `isolation.fresh_venv=false`, matching the actual read-only reuse of the complete v2 environment. The earlier candidate SHA `da5e3b987e67df0c27582cd4e4f469685ce4eb52534f221023e12d202dbbcab4` was not the executed protocol. V3 tool/tests bytes were unchanged.

V3 successfully imported `i2v_subject`, then stopped during `i2v_background` import. The actual path was DreamSim → `load_mae_as_vit` → Transformers ViTMAE module → `transformers.activations`. That module's backwards-compatible globals at lines244–251 instantiate these exact defaults:

| Global | Actual type in `transformers.activations` |
| --- | --- |
| `gelu_python` | `GELUActivation` |
| `gelu_new` | `NewGELUActivation` |
| `gelu` | `GELUActivation` |
| `gelu_fast` | `FastGELUActivation` |
| `quick_gelu` | `QuickGELUActivation` |
| `silu` | `SiLUActivation` |
| `mish` | `MishActivation` |
| `linear_act` | `LinearActivation` |

The actual installed `transformers/activations.py` SHA256 is **`37450e9d543aef0495ed67a6fd2017db63938e9995aa2cff16c79b4533ba10bd`**. Its constructors initialize only ordinary module bookkeeping plus Python callable attributes. V3 correctly rejected the first generic `nn.Module.__init__`; it did not construct a pretrained model, create parameters, load checkpoints, execute a forward or initialize CUDA. The earlier narrow AST scan of four other packages' *function default expressions* did not cover these Transformers module globals and supplies no contradictory evidence.

V3 also recorded a denied `socket.__new__` before the fatal constructor denial. Its audit did not preserve a stack for that event. The inspected `urllib3/util/connection.py` has source SHA **`2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74`**: module-global `HAS_IPV6=_has_ipv6('::1')` calls `socket.socket` and local bind only inside `if socket.has_ipv6`, catching any exception. This explains a plausible caught probe; attribution of the v3 event remains **source-supported inference**. The exact false branch is independently testable without making a socket.

V3 original RUN SHA is `aab003e7e7b0123646243138c1cdc28c5e8d66efb6e8918a4c36147a93b7b009`; IMPORT_RESULT SHA is `e6b00a8ce22faa7a1b70ea81ac9fb2c6395e779db2deca28d5956e370e884f9e`. Both are preserved with the executed source and protocol.

## EXECUTION: v4 is a new, explicit engineering scope

The root explicitly requested this scoped allowance. The new artifacts are:

- `tools/e0_official_metric_environment_v4.py`, SHA **`bcaf3aa989b3523cad2c505e26722514969548c453db6e4879c2715f1769040f`**.
- `config/e0_official_metric_environment_20261026_v4.json`, candidate SHA **`77525bcd9585499a1ceb8b8d3d1fbabd1e73831c932bb1be3acad484205a3b36`**.
- `tests/test_e0_official_metric_environment_v4.py`, SHA **`10fa6e4c6bbb4c4b638a205c95d28e6196ce886cca8780d4217804107ada2802`**.

Scope ID is `E0_OFFICIAL_I2V_IMPORT_COMPATIBILITY_V4`; output uses the independent HF-relative directory `derived/ffnds/E0_official_metric_environment_v4`. Root registration must bind the final protocol SHA before execution. Existing v1–v3 files, receipts and installed packages remain unchanged.

The whitelist checks the live class identity, module, exact file path and SHA, original module-global line, order and total count. It calls the original `nn.Module.__init__` only for those eight objects, then checks empty parameter/buffer/child dictionaries. It checks the final globals again after imports. Other NN module construction remains denied; `Parameter`, `UninitializedParameter`, parameter registration, buffer registration, `Buffer`, `UninitializedBuffer`, every NN forward and every checkpoint/CUDA path remain denied. Receipts separate eight permitted stateless wrappers from0pretrained-model constructions.

Before importing Torch or any metric dependency, v4 sets the process-local stdlib flag `socket.has_ipv6=False`. The pinned `_has_ipv6` false branch consequently creates no socket. The original urllib3 and official metric source bytes are unchanged. All socket creation, connect, DNS and subprocess audit denials remain active. Denied actions now retain source stacks, so a later failure can be localized without assuming it came from the same probe.

Read-only reuse of the v2 installed119-distribution environment and922 bound metadata files avoids another package installation. V4 retains scope-local Python/TorchInductor/debug/font/HF cache paths, two-CPU affinity/thread limits,20minutes,4GiB new files and≥10GiB free reserve. It permits0package operations,0network fetches,0GPU assignments,0parameters,0buffers,0pretrained-model constructions,0forwards,0checkpoint reads and0scores, with exactly8stateless wrapper initializations. The guards are application-level checks, not kernel confinement. One attempt is registered; no retry or fallback is implicit.

All **23 software tests pass**. New tests cover the exact eight ordered defaults, ninth/other/outside-site construction rejection, source changes, buffer-state detection, the actual urllib3 function's false/true capability branches, and continued socket/connect/DNS rejection. They use Python-only fake module instances and source extraction; they do not import actual Torch/metric models. Final read-only checks confirmed preserved v1–v3 identities and no agent-created v4 attempt.

The root-controlled entry point after registration is:

```bash
.venv/bin/python tools/e0_official_metric_environment_v4.py run --protocol config/e0_official_metric_environment_20261026_v4.json
```

## Next quality object and ready checkpoint evidence

The separate [asset SHA audit](independent-official-metric-checkpoint-sha-audit-20261027.json) actually rehashed all19 manifest records: weights, source archives and adapter configuration, totaling4,954,433,866bytes. Every SHA and size matched; record SHA is `67920d2d4c8134d4f8f90de465fab5d8d6a81e8adce6a342219d6a8c10ae08de`. No checkpoint was deserialized. That is byte-identity evidence, with loader/shape/key/numerical compatibility still untested.

After imports, the useful model study should obtain original source-bound appearance, motion and frame-quality observations on immutable existing clips. The fresh official sources425/437 have real `camera pans left` / `camera zooms in` generation instructions, so CoTracker has an actual labeled consumer when those clips exist. Root serializes the exact loader/forward/video scope separately. Original-function scores, paired engineering metrics, human full-video acceptance and FFN physical costs remain distinct evidence. The relevant **P23 → P24 → P25 → D09** connection is posthoc measurement; specific latent-to-quality causal claims remain conditional. This import work changes no FFN arithmetic or accepted video-quality frontier.
