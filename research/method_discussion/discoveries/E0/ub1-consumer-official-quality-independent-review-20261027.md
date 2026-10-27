# Independent source review: eight official UB1 video dimensions

Reviewed executable version: `tools/e0_ub1_consumer_official_quality.py`, SHA-256 `70d7b6ca36674389a4e889b647fb80b42e7a0f83e7f593d6100b0f69af75e422`. The review read the staged original VBench 0.1.5 functions, the isolated V8 environment's installed PEFT/DreamSim source, asset JSON, checkpoint ZIP pickle metadata and the adapter safetensors header. Eight software tests passed. This review constructed no metric model, initialized no GPU and computed no learned score.

The new driver preserves the eight original scoring functions and their `(aggregate, per_video_details)` results. All eight return per-video dictionaries with `video_path` and `video_results`; the driver's recursive NumPy scalar conversion and exact path-cohort validation match those actual return forms. Imaging-quality per-video results retain the original raw MOS scale, while its aggregate is divided by 100 by the author function. Comparisons should keep that original scale identity.

Two concrete loader failures were identified and corrected before execution:

1. Original subject-consistency and i2v-subject call `torch.hub.load(**submodules_list)` with the key `repo_or_dir`. The initial replacement required a positional parameter named `repo`. The reviewed `make_local_hub_loader` accepts both original keyword and DreamSim's positional invocation, restricts the original DINO source/model, and removes evaluator-only path/read-frame/resolution metadata before calling the local original hub loader.
2. The immutable DreamSim adapter JSON contains eight extension fields that the locked isolated PEFT `LoraConfig` does not accept: `alpha_pattern`, `rank_pattern`, `layer_replication`, `loftq_config`, `megatron_config`, `megatron_core`, `use_dora` and `use_rslora`. Original PEFT config deserialization passes them directly to the dataclass constructor. The reviewed adapter compatibility context acts only on the exact directory and JSON SHA, verifies their inactive default values, and supplies the same five LoRA settings that original DreamSim itself already reads: rank, alpha, dropout, bias and target modules. Original `PeftModel.from_pretrained` accepts this explicit configuration and skips the incompatible JSON reconstruction. Its existing default adapter also makes `load_adapter` skip a second config read. Original adapter-weight loading and state-dictionary application remain in place, and the classmethod descriptor is restored afterward.

The injected configuration preserves the active model mathematics. Rank 16, alpha 1, dropout 0.3, bias `none` and target `qkv` match the original JSON. Other supported settings match the original defaults: fan-in/fan-out false, normal LoRA initialization and no extra saved modules/layer filters. `from_pretrained` changes inference mode to true, matching the original JSON. The safetensors header contains exactly 72 qkv LoRA A/B tensors of shapes `(16, 768)` and `(2304, 16)`, covering 36 attention modules. The extra JSON auto-mapping is class/save metadata for this path, where the original model is already supplied.

The staged checkpoint routes were followed separately:

| Dimension | Original loader route | Registered outer work |
| --- | --- | ---: |
| imaging_quality | MUSIQ with the explicit staged SPAQ checkpoint | 486 frame forwards |
| aesthetic_quality | local CLIP L/14 plus the staged `emb_reader` linear head | 18 CLIP batches and 18 head batches |
| motion_smoothness | staged AMT-S YAML, original dynamic `networks.AMT-S.Model`, staged state dictionary | 240 midpoint predictions |
| dynamic_degree | original RAFT construction and staged RAFT-things state dictionary | 240 forwards, 4800 iterations |
| subject_consistency | original DINO backbone through local hub and its exact URL-to-local checkpoint map | 486 frame forwards |
| background_consistency | local CLIP B/32 checkpoint | 6 encoding batches |
| i2v_subject | same original DINO route; original 16:9 source-image lookup | 246 image/frame forwards |
| i2v_background | original DreamSim ensemble, staged backbone/projection and exact LoRA files | 246 outer `embed` calls |

DreamSim's DINO backbone uses the separately staged original backbone checkpoint; its larger DreamSim DINO file is additionally read for the original student projection parameters. Those assets have distinct identities. ZIP pickle metadata for all three DreamSim `.pth`/`.pth.tar` files has CPU storage locations and only the expected tensor/OrderedDict reconstruction globals, compatible with their explicit `weights_only=True` reads. The existing DreamSim cache contains the three required backbone files and ensemble adapter directory, so the original cache-completeness branch selects the staged files. Local open-CLIP conversion uses its supplied state dictionary and does not call a remote pretrained-model factory.

The actual pinned `load_i2v_dimension_info` needs prompt, dimension, `image_name` and video list; it does not require an additional `image_type` key. The driver's scoped `vbench2_beta_i2v/data/crop/16-9` aliases match that exact lookup. Read-only image metadata confirms both frozen JPEG inputs are RGB, with sizes 5472×3078 and 5184×2916. Original motion-smoothness branches on the literal string `cuda`; the chosen `torch.device('cuda:0')` follows its documented alternate anchor branch while placing the actual tensors/model on the assigned CUDA device, as recorded by this protocol.

Whole-dimension work is prepaid before the original function dispatch and stays debited if a call fails. These predebits are distinct from observed RAFT dispatch/iteration counts and its original continuous flow scores. Other opaque internal module calls remain marked unknown. Exact ACTIVE protocol authorization and read-only source/asset verification precede CUDA reservation/model work; one assigned A100, two CPU cores, one attempt, the wall/output caps and denied network/write paths remain explicit. The six frozen video identities are development-exposed examples. Low flow alone is not a quality failure, temporal flickering remains unrun without a defensible static-video screen, and the report retains pending quality acceptance.

After the two loader repairs, this independent source review found no further deterministic blocker in the eight staged computation paths. This is source/metadata and software-test evidence; the subsequently executed single-GPU result must establish actual checkpoint/model/score completion under its own frozen protocol SHA.
