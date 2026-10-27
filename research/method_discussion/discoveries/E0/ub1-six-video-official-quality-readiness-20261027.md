# UB1 six-video official quality: source and asset readiness

The fixed subject and background inputs each have dense, neutral UB1 and candidate outputs at seed 42. Once those six output bytes and their generation receipts are complete, a compact official evaluation can assess imaging quality, aesthetic quality, motion smoothness, dynamic degree, subject consistency and background consistency on all six, plus i2v subject on the three subject clips and i2v background on the three background clips. Grouping clips in each original dimension function gives eight official function calls, with per-video outputs retained. The two sources are development-exposed; results describe these clips rather than population quality.

The locally staged VBench 0.1.5 source view and all 12 necessary model files exist and their full SHA-256 values match the original asset manifest. This includes MUSIQ-SPAQ, CLIP ViT-L/14 and its aesthetic linear head, AMT-S, RAFT-things, DINO ViT-B/16, CLIP ViT-B/32, and four DreamSim ensemble files. The private receipt `.runtime/one_time_repair_20261027/UB1_QUALITY_ASSETS_READINESS_PRIVATE.json` records their exact relative paths, sizes and hashes and the checked dimension-source hashes. No download is needed for those staged files. V8's isolated imports-only receipt is `PASS_IMPORTS_ONLY` with SHA `c2c40eacc6a3337cd42eb7a312f43785e171b245d5d116bf34c8cfd37bde774a`; it establishes import compatibility, not model scoring. Preserve the generation environment and use the existing isolated metric interpreter and V8 overlay.

For exactly 81 decoded frames at 16 fps, original VBench sampling gives these outer calls, conditional on successful full decoding and one function invocation per dimension:

| Dimension | Clips | Original outer model calls |
| --- | ---: | ---: |
| Imaging quality | 6 | 486 MUSIQ frame calls |
| Aesthetic quality | 6 | 18 CLIP image-encoding batches and 18 aesthetic-head batches, batch size 32 |
| Motion smoothness | 6 | 240 AMT midpoint predictions, 41 even-indexed frames per clip |
| Dynamic degree | 6 | 240 RAFT forwards, each 20 iterations; 41 sampled frames per clip |
| Subject consistency | 6 | 486 DINO frame calls |
| Background consistency | 6 | 6 CLIP image-encoding batches |
| I2V subject | 3 | 246 DINO calls including each source image |
| I2V background | 3 | 246 DreamSim `embed` calls including each source image |

The DreamSim and other model internals have additional submodule calls that are not enumerated here. RAFT's official binary dynamic degree and its original per-pair continuous flow summaries should both be retained: low motion alone is not a quality failure. The original temporal-flickering source explicitly requires a static video; an image of houses does not establish that condition. Apply it only to clips admitted by a fixed, defensible static-video screen and report it separately from semantic background consistency.

Reuse the original dimension functions and checkpoint-resolution conventions in [offline_i2v_metric_loaders.py](/home/xiang/projects/FFNDS/tools/offline_i2v_metric_loaders.py) and the transparent RAFT score and forward observers in [e0_ub1_dynamic_degree.py](/home/xiang/projects/FFNDS/tools/e0_ub1_dynamic_degree.py), adapted to a separately frozen six-clip ledger. [run_fresh_official_metrics.py](/home/xiang/projects/FFNDS/tools/run_fresh_official_metrics.py) illustrates original functions and FULL_INFO construction but is fixed to a different 145-generation cohort and cannot run this scope unchanged. Its broad loader context imports other dimensions and enforces its historical package versions, so reuse only the compatible verified loader pieces under the isolated environment. Preserve original function arguments, including device identity: motion-smoothness source branches on the literal string `cuda`, whereas the older runner passes a `torch.device`, affecting its original resize branch. Record the actual branch rather than silently altering it.

This is static preparation only. The six new video SHA values and final output receipts must be bound after generation; no metric model, GPU, score, download or environment modification was performed here.
