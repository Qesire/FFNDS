# Independent review: SM120 kernel overlay export

**SUPPORT — CPU byte-export source and synthetic control.** This review binds these SHA-256 identities:

- `tools/export_sm120_kernel_overlay.py`: `8bd1ba90576c3441d7966eb20cc4798e4cf6e09882fc7052945ae0d0643d93f6`.
- `tests/test_export_sm120_kernel_overlay.py`: `dc1f5d989cbadedb7b3f5e68205bb6c2f3c562d2b770d7509440a1d7b72f32ab`.
- `config/e0_sm120_kernel_overlay_export_20261029_draft.json`: `d3dcfad6ff0459fc55bb667d18191d90b1dd1b2a12afbb2c86297bd858258c6e`.

The code binds the completed CPU build protocol, RUN and actual exit receipt, the five original Python source identities and the built ELF identity. It checks regular files, byte lengths and SHA-256, verifies the ELF signature, and constructs a deterministic archive. The registered run checks CPU-only affinity and ACTIVE scope identity, requires fresh confined output, and retains a failure receipt. The internal 120-second wall check follows archive construction; the registered outer 150-second timeout bounds the subsequent write as well.

The synthetic test command `python3 -m unittest discover -s tests -p test_export_sm120_kernel_overlay.py` completed with **1 test, exit 0**. It checks deterministic archive bytes and rejects a symlink and a non-ELF payload. No `boundsource`, actual overlay export, model, GPU or native kernel execution was performed for this review. The review supports the specified export procedure; native execution remains a separate evidence scope.
