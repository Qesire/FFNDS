"""Verify the immutable school-cluster roots used by the six-site experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


SITES = tuple((expert, layer) for expert in ("high", "low") for layer in (4, 19, 34))
STEPS = {"high": (0, 1), "low": (2, 3)}
PANEL_SHAPE = (512, 5120)
GATE_SHAPE = (1, 1, 5120)
WEIGHTS = {
    "0.weight": (13824, 5120),
    "0.bias": (13824,),
    "2.weight": (5120, 13824),
    "2.bias": (5120,),
}
INPUT_HASHES = {
    "frozen_comparison": "c804efa9e9e15cc2440389e734520344b549958ed93e63568f42b49a0303b9c0",
    "strict_q1_manifest": "dc2d2b2289ff119d432c5a347facc446601e5ef2526eb10c09f86f06df88e247",
    "old_bindings": "58cec6415e1eb6142b989ed7dad2bdb5dbbc2bdf38270a7e48e156d9081ccc48",
    "old_params": "d373130313612b4bf14bfbbe8e71273d9651943797e413cd795181c33a770ca3",
    "bank_manifest": "f6f8e100da6f75b2db612edd8c8c4d21729ed9bf99bfb5092941f2381a4d0bee",
    "capture_scope": "69078c0f758267c657d325103376f5f36e5643616ed3dc542f731bab3ff430d9",
    "cal_pairs": "0d2cf0d2d9fc21bc1913e96758ff5c3790c4dc536f615ca427d19d70decd9af4",
    "strict_historical_dev_pairs": "b743eb66f3087cf478f6849d737ba638d290d1c74988e1831106d176499d08ca",
    "old_scope": "ec7558a110831daaece640b8c45ed24a22ead513906e598bb2120e954cd364dd",
    "strict_q1_job": "7bd2c00f7628a39a34fb93f8103583189f81f2652aa9217b9a95cebeb167f9a6",
    "strict_backend": "60831fa1f1f8f4c670000356adba52a679ea6464efec15da1ea2b354d00b997b",
    "capture_backend": "60831fa1f1f8f4c670000356adba52a679ea6464efec15da1ea2b354d00b997b",
}


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def spec(path, expected=None):
    path = Path(path)
    found = sha_file(path)
    if expected is not None and found != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {found} != {expected}")
    return {"path": str(path), "sha256": found, "bytes": path.stat().st_size}


def read(specification):
    return json.loads(Path(specification["path"]).read_bytes())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def bits(tensor):
    return tensor.contiguous().view(torch.int16)


def tensor_bytes(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def load_tensor_file(path):
    return torch.load(path, weights_only=True, mmap=True, map_location="cpu")


def tensor_contract(value, shape, dtype):
    return tuple(value.shape) == tuple(shape) and value.dtype == dtype and value.is_contiguous()


def validate_panel(path, source, site, step, frozen_source, weight, capture, expected):
    label = f"{source}/{site}/{step}"
    receipt_path = path.with_suffix(".pt.json")
    receipt_ref = spec(receipt_path)
    receipt = read(receipt_ref)
    ref = spec(path, receipt["file_sha256"])
    require(receipt["file"] == path.name and receipt["bytes"] == ref["bytes"] and
            receipt["consumer_contract"] == "PANEL512_TYPED_V1", f"{label}: producer receipt")
    panel = load_tensor_file(path)
    meta = panel["metadata"]
    expert, layer = site
    require((meta["source"], meta["expert"], meta["layer"], meta["step"], meta["seed"]) ==
            (source, expert, layer, step, 42), f"{label}: producer role")
    require(meta["scope_sha256"] == capture["sha256"], f"{label}: producer scope")
    for key in ("image_sha256", "prompt_sha256", "source_cluster_id", "cycle_role"):
        require(meta[key] == frozen_source[key], f"{label}: {key}")
    require(all(receipt[k] == v for k, v in meta.items()), f"{label}: serialized metadata differs from receipt")
    require(meta["checkpoint_sha256"] == weight["sha256"], f"{label}: checkpoint")
    require(meta["grid"] == [21, 29, 52] and meta["flattened_tokens"] == 31668, f"{label}: geometry")
    require(meta["Y_semantics"].startswith("native dense W2 apply output includes b2; gate not yet applied"),
            f"{label}: expected Dense pre-gate target")
    for key, shape in (("U", PANEL_SHAPE), ("Y_pre_gate", PANEL_SHAPE), ("gate", GATE_SHAPE),
                       ("shift", GATE_SHAPE), ("scale", GATE_SHAPE)):
        require(tensor_contract(panel[key], shape, torch.bfloat16), f"{label}: {key} shape/dtype/stride")
    ids, coords = panel["token_indices"], panel["token_thw"]
    require(tensor_contract(ids, (512,), torch.int64) and tensor_contract(coords, (512, 3), torch.int64),
            f"{label}: token index shapes")
    require(len(ids.unique()) == 512 and int(ids.min()) >= 0 and int(ids.max()) < 31668, f"{label}: valid indices")
    rebuilt = torch.stack((ids // (29 * 52), (ids // 52) % 29, ids % 52), dim=1)
    require(torch.equal(coords, rebuilt), f"{label}: raster reconstruction")
    token_sha = tensor_bytes(ids)
    require(token_sha == meta["token_id_sha256_le_int64"], f"{label}: token order hash")
    if expected is not None:
        require(str(path) == expected["panel"]["path"] and ref["sha256"] == expected["panel"]["sha256"] and
                token_sha == expected["token_id_sha256_le_int64"], f"{label}: strict panel reference")
    return dict(ref, step=step, token_sha256=token_sha, gate_sha256=tensor_bytes(panel["gate"]),
                timestep=meta["timestep"], receipt=receipt_ref)


def run(work, output):
    a = work / "g20_two_track_core_20261114/A_Q1_RESIDUAL/v1/stage"
    s = work / "ffnds_strict_reproduction_20261116_336b7e1"
    v = work / "g20_v2_capture_20261113_rev2/stage"
    q = s / "wan_main_q1_01_v1"
    binding = s / "stage3_binding_v1"
    files = {
        "frozen_comparison": s / "wan_freeze_v1/FROZEN_COMPARISON.json",
        "strict_q1_manifest": s / "q1_v4_fixed_remaining_v2/output/Q1_MANIFEST.json",
        "old_bindings": a / "B_BINDINGS.json",
        "old_params": a / "output/A_Q1_RESIDUAL_PARAMS.json",
        "bank_manifest": v / "output/BANK_MANIFEST.json",
        "capture_scope": v / "scope.json",
        "cal_pairs": binding / "Q1_CAL_PAIRS.json",
        "strict_historical_dev_pairs": binding / "DEV_PAIRS.json",
        "old_scope": a / "scope.json",
        "strict_q1_job": q / "scope.json",
        "strict_backend": q / "output/EFFECTIVE_CONFIG.json",
        "capture_backend": v / "output/EFFECTIVE_CONFIG.json",
    }
    refs = {k: spec(p, INPUT_HASHES[k]) for k, p in files.items()}
    obj = {k: read(ref) for k, ref in refs.items()}
    strict = obj["strict_q1_job"]
    old = obj["old_scope"]
    capture = obj["capture_scope"]
    frozen_ref = obj["frozen_comparison"]["parameters"]["Q1r2"]
    require(all(frozen_ref[k] == refs["strict_q1_manifest"][k] for k in ("path", "sha256"))
            and frozen_ref == strict["parameters"], "Q1 manifest identity")
    for name, digest in strict["reference_sha256"].items():
        spec(v / name, digest)
    for name in ("a_q1_residual_runner.py", "a_q1_residual_repr.py"):
        spec(a / name, old["code_sha256"][name])
    for name in ("wan_ffn_comparator_ports.py", "wan_comparison_job.py", "wan_ffn_port.py"):
        spec(q / name, strict["bundle_sha256"][name])
    require(strict["v2_reference_dir"] == str(v) and strict["seed"] == capture["seed"] == 42,
            "strict runner references capture scope and seed")
    for key in ("model_path", "model_cls", "low_noise_original_ckpt", "high_noise_original_ckpt",
                "enable_cfg", "infer_steps", "target_video_length", "target_width", "target_height",
                "use_bfloat16", "norm_modulate_backend", "resize_mode", "denoising_step_list",
                "feature_caching", "self_attn_1_type", "cross_attn_1_type", "cross_attn_2_type"):
        require(obj["strict_backend"][key] == obj["capture_backend"][key], f"runtime recipe: {key}")
    require(len(strict["schedule"]) == 4 and capture["expert_schedule"] == "HHLL" and
            capture["anchor_steps"] == [0, 2] and capture["hit_steps"] == [1, 3], "HHLL schedule")
    spec(strict["python_executable"], strict["python_executable_sha256"])

    original_weights = old["checkpoints"]
    require(original_weights == capture["checkpoints"], "capture and A checkpoints differ")
    checkpoints = {expert: spec(ref["path"], ref["sha256"]) for expert, ref in original_weights.items()}
    weight_tensors = {}
    for expert, layer in SITES:
        site = f"{expert}{layer:02d}"
        with safe_open(checkpoints[expert]["path"], framework="pt", device="cpu") as container:
            weight_tensors[site] = {}
            for suffix, shape in WEIGHTS.items():
                key = f"blocks.{layer}.ffn.{suffix}"
                value = container.get_tensor(key)
                require(tensor_contract(value, shape, torch.bfloat16), f"original FFN checkpoint: {key}")
                weight_tensors[site][key] = tensor_bytes(value)
    print("checkpoint verified", flush=True)

    banks = obj["old_bindings"]["bank"]
    require(banks["manifest_sha256"] == refs["bank_manifest"]["sha256"] == old["bank_manifest_sha256"],
            "bank manifest identity")
    bank = obj["bank_manifest"]
    cal, dev = banks["cal11"], banks["dev11"]
    require(len(cal) == len(dev) == 11 and len(set(cal + dev)) == 22 and set(cal + dev) == set(bank),
            "frozen CAL11 and DEV11 source lists")
    require(all(bank[x]["cycle_role"] == group and bank[x]["sites"] == 160 for group, lst in
                (("CAL", cal), ("DEV", dev)) for x in lst), "source split identity")
    for key in ("source_cluster_id", "image_sha256"):
        require(len({r[key] for r in bank.values()}) == 22, f"duplicate source identity: {key}")
    for name, cohort in (("cal_pairs", cal), ("strict_historical_dev_pairs", dev)):
        archived = obj[name]
        require(set(archived["sources"]) == set(cohort) and archived["v2_bank_manifest_sha256"] ==
                refs["bank_manifest"]["sha256"] and archived["bindings_sha256"] == refs["old_bindings"]["sha256"],
                f"{name}: inheritance and cohort")
    cal_refs = {(r["source"], r["expert"], r["layer"]): r for r in obj["cal_pairs"]["input_pairs"]}
    require(len(cal_refs) == len(obj["cal_pairs"]["input_pairs"]) == 880, "strict calibration pair count")
    dev_refs = {(r["source"], r["expert"], r["layer"]): r for r in obj["strict_historical_dev_pairs"]["input_pairs"]}
    require(len(dev_refs) == len(obj["strict_historical_dev_pairs"]["input_pairs"]) == 66,
            "strict historical DEV pair count")
    q1 = obj["strict_q1_manifest"]
    sites = {(r["expert"], r["layer"]): r for r in q1["sites"]}
    require(q1["status"] in ("ALL_80_CONVERGED", "ALL_80_FINITE_FIXED_ITERATES") and
            len(sites) == len(q1["sites"]) == q1["num_sites"] == 80 and
            set(sites) == {(e, l) for e in STEPS for l in range(40)}, "strict full-80 Q1 coverage")
    old_params = obj["old_params"]["params"]
    require(set(old_params) == {f"{e}{l:02d}" for e, l in SITES}, "old kit coverage")
    full80, comparisons = [], {}
    for (expert, layer), row in sites.items():
        key = f"{expert}{layer:02d}"
        archive = row["artifact"]
        current = spec(archive["path"], archive["sha256"])
        identity = archive["identity"]
        require((identity["expert"], identity["layer"], identity["checkpoint_sha256"]) ==
                (expert, layer, checkpoints[expert]["sha256"]) and
                identity["calibration_manifest_sha256"] == refs["cal_pairs"]["sha256"] and
                identity["calibration_sources"] == cal, f"{key}: strict Q1 identity")
        full80.append(dict(current, expert=expert, layer=layer, status=row["status"]))
        if (expert, layer) not in SITES:
            continue
        prior = spec(old_params[key]["path"], old_params[key]["sha256"])
        strict_payload = load_tensor_file(current["path"])
        old_payload = load_tensor_file(prior["path"])
        payload_identity = strict_payload["identity"]
        # The torch artifact preserves the Python tuple, while the JSON manifest stores an array.
        require(dict(payload_identity, calibration_sources=list(payload_identity["calibration_sources"])) == identity,
                f"{key}: strict tensor identity")
        a32 = strict_payload["A_q1_f32"]
        historical = old_payload["A_Q1_BF16"]
        require(tensor_contract(a32, (5120, 5120), torch.float32) and
                tensor_contract(historical, (5120, 5120), torch.bfloat16), f"{key}: matrix contract")
        runtime = a32.to(torch.bfloat16)
        difference = bits(runtime) != bits(historical)
        comparisons[key] = {
            "expert": expert, "layer": layer, "old_params": prior, "strict_q1": current,
            "bf16_bitwise_equal": not bool(difference.any()), "bf16_unequal_elements": int(difference.sum()),
            "bf16_value_unequal_elements": int((runtime != historical).sum()),
            "bf16_max_abs_difference": float((runtime.float() - historical.float()).abs().max()),
            "strict_bf16_tensor_sha256": tensor_bytes(runtime), "old_bf16_tensor_sha256": tensor_bytes(historical),
            "ffn_tensor_sha256": weight_tensors[key],
        }
        print(f"{key} Q1 BF16 unequal elements: {comparisons[key]['bf16_unequal_elements']}", flush=True)

    root = Path(banks["bank_root"])
    require(root == v / "output/bank", "bank locator")
    pair_rows = []
    source_panels = obj["old_bindings"]["panel"]["per_source"]
    for split, names in (("CAL", cal), ("DEV", dev)):
        for source in names:
            source_orders = set()
            for site in SITES:
                expert, layer = site
                steps = STEPS[expert]
                row = {"source": source, "split": split, "expert": expert, "layer": layer}
                for role, step in zip(("anchor", "hit"), steps):
                    panel = root / source / f"{source}__seed42__s{step:02d}__b{layer:02d}.pt"
                    archived = (cal_refs if split == "CAL" else dev_refs)[(source, *site)][role]
                    require((archived["source"], archived["seed"], archived["step"], archived["layer"]) ==
                            (source, 42, step, layer), f"{source}/{site}/{step}: archived role")
                    for key in ("image_sha256", "prompt_sha256", "source_cluster_id"):
                        require(archived["input_identity"][key] == bank[source][key],
                                f"{source}/{site}/{step}: archived {key}")
                    require(archived["input_identity"]["checkpoint_sha256"] == checkpoints[expert]["sha256"],
                            f"{source}/{site}/{step}: archived checkpoint")
                    expected = archived if archived["kind"] == "v2_panel" else None
                    endpoint = validate_panel(panel, source, site, step, bank[source], checkpoints[expert],
                                              refs["capture_scope"], expected)
                    require(float(endpoint["timestep"]) == float(strict["schedule"][step]["timestep"]),
                            f"{source}/{site}/{step}: timestep")
                    require(endpoint["token_sha256"] == source_panels[source]["token_id_sha256_le_int64"] ==
                            archived["token_id_sha256_le_int64"], f"{source}/{site}/{step}: archived coordinates")
                    source_orders.add(endpoint["token_sha256"])
                    row[role] = dict(endpoint, archived_reference_kind=archived["kind"])
                pair_rows.append(row)
            require(len(source_orders) == 1, f"{source}: inconsistent token order across sites/steps")
    require(len(pair_rows) == 132, "validated six-site pair count")
    reused = all(entry["bf16_bitwise_equal"] for entry in comparisons.values())
    result = {
        "schema": "FFNDS_STATE_OPERATOR_BINDINGS_V1", "status": "PASS",
        "base_commit": "3011ca907fac612a9d64f7f0177d7794ebc3621b",
        "bank_root": str(root), "bank_manifest": refs["bank_manifest"],
        "cal_sources": cal, "dev_sources": dev, "checkpoints": checkpoints,
        "schedule": strict["schedule"], "input_manifests": refs,
        "l64_policy": "REUSE_ALL_SIX" if reused else "REFIT_ALL_SIX",
        "strict_q1_full80": full80, "sites": comparisons, "pairs": pair_rows,
        "new_d4": 0,
    }
    pending = output.with_suffix(".partial")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    pending.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.work, args.output)


if __name__ == "__main__":
    main()

