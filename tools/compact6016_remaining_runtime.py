"""Deploy an audited 94-old/66-new ordinary6016 bank without relabeling failures.

The native interception/residency implementation is reused read-only. This
loader validates the new mixed lineage, then loads only eight deployment
fields per student; it neither resumes training nor reads LoRA factors.
"""
import e0_compact6016_remaining66 as remaining
import e0_compact6016_full80_completion as completion

from compact6016_native_runtime import (INITIALIZATIONS, EXPERTS, DEPLOY_KEYS,
    TRAINING_KEYS, steps_for, function_key, validate_deployment_pack,
    Compact6016Residency, Compact6016Runtime)
from ub1_consumer_runtime import tensor_sha, verify_binding


def load_bank(manifest, initialization, hf_root, base, native_biases, *, repo_root, loader=None):
    """Read only the eight deployment tensors for each of eighty functions.

    The caller binds the immutable manifest itself before invoking this loader.
    Original native biases come from the verified checkpoint CPU loader. The
    original cache remains intact for the matched Dense replay.
    """
    import json
    if initialization not in INITIALIZATIONS or manifest.get("schema") != "ffnds.compact6016.deployment_bank.mixed_lineage.v1":
        raise ValueError("Frozen deployment bank/init required")
    if manifest.get("status") != "METADATA_COMPLETE_TENSOR_LOAD_PENDING" or len(manifest.get("cells", [])) != 160:
        raise ValueError("Exactly 160 final cells required before deployment")
    if manifest.get("producer_id") != remaining.ID:
        raise ValueError("Remaining66 assembler producer required")
    verified = remaining.assemble_manifest(manifest["training_protocol"], manifest["parts"], hf_root, repo_root=repo_root)
    if verified != manifest:
        raise ValueError("Frozen deployment manifest differs from actual complete producer receipts")
    # A cell retains the exact scope and protocol representation of its producer.
    old_plan = json.loads(verify_binding(repo_root, manifest["completion_protocol"]).read_text())
    if old_plan.get("id") != completion.ID:
        raise ValueError("Original completion protocol scope differs")
    producer_protocols = {completion.ORIGINAL_ID: old_plan["original_protocol"],
                          completion.ID: manifest["completion_protocol"],
                          remaining.ID: manifest["training_protocol"]}
    for cell in manifest["cells"]:
        key = function_key(cell["expert"], cell["block"])
        init = cell["initialization"]
        producer = cell.get("producer_id")
        if init not in INITIALIZATIONS or producer not in producer_protocols:
            raise ValueError("Unknown producer or initialization")
        expected_id = (completion._original_run_id(*key, init) if producer == completion.ORIGINAL_ID
                       else completion.function_id(*key) + "_" + init if producer == completion.ID
                       else remaining.function_id(*key) + "_" + init)
        if (cell.get("run_id") != expected_id or cell.get("producer_protocol") != producer_protocols[producer]
                or cell.get("steps") != list(steps_for(key[0]))):
            raise ValueError("Cell producer, run ID, protocol or steps differs")
    import hashlib
    bank_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if loader is None:
        from safetensors import safe_open
        def loader(path):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != DEPLOY_KEYS | TRAINING_KEYS:
                    raise ValueError("Original folded export keys differ")
                return {k: handle.get_tensor(k).contiguous() for k in DEPLOY_KEYS}
    view = json.loads(verify_binding(hf_root, manifest["asset_view"]).read_text())
    specs = {(r["expert"], r["block"]): r for r in view["shards"]}
    expected = {(e, b) for e in EXPERTS for b in range(40)}
    if set(specs) != expected or len(view["shards"]) != 80:
        raise ValueError("Original full80 native asset identity required")
    positions = {}
    cells = [c for c in manifest["cells"] if c["initialization"] == initialization]
    if len(cells) != 80 or {(c["expert"], c["block"]) for c in cells} != expected:
        raise ValueError("Exactly eighty complete same-init folded FFNs required")
    for cell in cells:
        key = function_key(cell["expert"], cell["block"])
        path = verify_binding(hf_root, cell["export"])
        pack = loader(path)
        if set(pack) != DEPLOY_KEYS:
            raise ValueError("Deploy loader read unexpected tensors")
        for name, value in pack.items():
            recorded = cell["export"]["tensors"][name]
            if (list(value.shape) != recorded["shape"] or str(value.dtype) != recorded["dtype"]
                    or tensor_sha(value) != recorded["sha256"]):
                raise ValueError("Actual folded tensor bytes differ from producer export")
        up = base.records[key[0]][f"blocks.{key[1]}.ffn.0"]
        down = base.records[key[0]][f"blocks.{key[1]}.ffn.2"]
        for name, value in (("W1", up["weight_qdq"]), ("W2", down["weight_qdq"]),
                            ("smooth1", up["smooth_scale"]), ("smooth2", down["smooth_scale"])):
            spec = specs[key]["tensors"][name]
            if list(value.shape) != spec["shape"] or tensor_sha(value) != spec["sha256"]:
                raise ValueError("Actual original native cache differs from bank parent")
        biases = native_biases[key]
        metadata = validate_deployment_pack(pack, cell, up, down, biases["b1"], biases["b2"])
        positions[key] = {**pack, **metadata, "expert": key[0], "block": key[1], "initialization": initialization,
                          "run_id": cell["run_id"], "export_binding": cell["export"],
                          "producer_result_binding": cell["result"], "support_binding": cell["support_file"],
                          "producer_scope": cell["producer_id"], "producer_protocol": cell["producer_protocol"],
                          "bank_semantic_sha256": bank_sha}
    return positions
