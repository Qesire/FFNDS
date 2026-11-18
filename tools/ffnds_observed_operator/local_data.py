"""Frozen six-site Wan endpoint identities and native BF16 observations."""

from hashlib import file_digest, sha256
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F


SITES = tuple((expert, layer) for expert in ("high", "low") for layer in (4, 19, 34))
STEPS = {"high": (0, 1), "low": (2, 3)}
SUFFIXES = ("0.weight", "0.bias", "2.weight", "2.bias")
SHAPES = ((13824, 5120), (13824,), (5120, 13824), (5120,))
ORIGINAL_BINDINGS = "8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6"


def sha_file(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(ref):
    path = Path(ref["path"])
    require(path.is_file(), f"ASSET_NOT_AVAILABLE: {path}")
    if "bytes" in ref:
        require(path.stat().st_size == ref["bytes"], f"Asset size differs: {path}")
    require(sha_file(path) == ref["sha256"], f"Asset hash differs: {path}")
    return path


def bound_json(ref):
    return json.loads(checked(ref).read_bytes())


def tensor_sha(value):
    return sha256(value.cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def equal_bits(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def tensor_contract(value, shape, dtype):
    return value.shape == shape and value.dtype == dtype and value.is_contiguous()


def folds_for(source, token_ids, device):
    """Assign by global ID, independent of serialized row order and site."""
    ids = token_ids.tolist()
    require(len(ids) == len(set(ids)) == 512, f"{source}: expected 512 unique IDs")
    prefix = f"observed-secant-20261118|{source}|"
    ordered = sorted(ids, key=lambda g: (sha256(f"{prefix}{g}".encode("utf-8")).digest(), g))
    index = {g: i for i, g in enumerate(ids)}
    halves = (ordered[:256], ordered[256:])
    result = []
    for fold_id, (pool, query) in enumerate((halves, halves[::-1]), 1):
        probes = np.random.default_rng(42).permutation(sorted(pool))[:64].tolist()
        record = {"fold": fold_id, "pool_ids": pool, "query_ids": query, "probe_ids": probes}
        for name, values in (("pool", pool), ("query", query), ("probe", probes)):
            record[name] = torch.tensor([index[g] for g in values], dtype=torch.long, device=device)
        result.append(record)
    return result


def archive_inventory(scope, spec):
    require(scope["bindings"]["sha256"] == ORIGINAL_BINDINGS, "Wrong generation of original BINDINGS")
    archive = bound_json(scope["bindings"])
    require(archive["schema"] == "FFNDS_STATE_OPERATOR_BINDINGS_V1" and archive["status"] == "PASS"
            and archive["l64_policy"] == "REUSE_ALL_SIX", "Archived P0 has no approved L64 reuse")
    require(archive["cal_sources"] == spec["cohorts"]["CAL"]
            and archive["dev_sources"] == spec["cohorts"]["DEV"], "Archive source split changed")
    require([item["timestep"] for item in archive["schedule"]] == [1000.0, 937.5, 833.3333129882812, 625.0],
            "Archive timestep schedule changed")
    bank = bound_json(archive["bank_manifest"])
    roles = {source: split for split, sources in (("CAL", archive["cal_sources"]), ("DEV", archive["dev_sources"]))
             for source in sources}
    require(len(roles) == len(bank) == 22 and set(bank) == set(roles), "Archive source bank coverage changed")
    require(all(row["cycle_role"] == roles[source] for source, row in bank.items()), "Archive CAL/DEV role changed")
    require(all(len({row[key] for row in bank.values()}) == 22 for key in ("source_cluster_id", "image_sha256")),
            "Repeated image-cluster identity in cohort")
    expected_sites = {(expert, layer) for expert in STEPS for layer in range(40)}
    q1_sites = {(row["expert"], row["layer"]) for row in archive["strict_q1_full80"]}
    require(len(archive["strict_q1_full80"]) == 80 and q1_sites == expected_sites, "Strict Q1 coverage changed")
    rows = {(row["expert"], row["layer"], row["source"]): row for row in archive["pairs"]}
    expected = {(expert, layer, source) for expert, layer in SITES for source in roles}
    require(len(archive["pairs"]) == 132 and set(rows) == expected, "Frozen endpoint Cartesian coverage changed")
    require(set(archive["sites"]) == {f"{expert}{layer:02d}" for expert, layer in SITES}, "Site binding changed")

    missing = []
    assets = list(archive["checkpoints"].values()) + list(archive["strict_q1_full80"])
    assets += [site["old_params"] for site in archive["sites"].values()]
    endpoint_paths = set()
    for row in rows.values():
        require(row["split"] == roles[row["source"]], "Endpoint pair split differs from cohort")
        for role in ("anchor", "hit"):
            ref = row[role]
            require(ref["path"] not in endpoint_paths, f"Aliased endpoint file: {ref['path']}")
            endpoint_paths.add(ref["path"])
            assets.extend((ref, ref["receipt"]))
    for ref in assets:
        path = Path(ref["path"])
        if not path.is_file() or path.stat().st_size != ref["bytes"]:
            missing.append(str(path))
    require(not missing, "ASSET_NOT_AVAILABLE: " + ", ".join(missing))
    return archive, bank, rows


def load_endpoint(ref, row, step, archive, bank):
    path = checked(ref)
    receipt = bound_json(ref["receipt"])
    require(receipt["file_sha256"] == ref["sha256"] and receipt["file"] == path.name
            and receipt["bytes"] == ref["bytes"] and receipt["consumer_contract"] == "PANEL512_TYPED_V1",
            f"Endpoint receipt differs: {path}")
    panel = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
    meta = panel["metadata"]
    require(all(receipt.get(k) == v for k, v in meta.items()), f"Panel receipt metadata differs: {path}")
    require((meta["source"], meta["expert"], meta["layer"], meta["step"], meta["seed"]) ==
            (row["source"], row["expert"], row["layer"], step, 42), f"Panel identity differs: {path}")
    require(ref["step"] == step and ref["timestep"] == meta["timestep"] == archive["schedule"][step]["timestep"],
            f"Panel timestep differs: {path}")
    require(meta["checkpoint_sha256"] == archive["checkpoints"][row["expert"]]["sha256"]
            and meta["scope_sha256"] == archive["input_manifests"]["capture_scope"]["sha256"],
            f"Panel checkpoint or capture scope differs: {path}")
    fields = ("image_sha256", "prompt_sha256", "source_cluster_id", "cycle_role")
    require(all(meta[k] == bank[row["source"]][k] for k in fields), f"Panel source identity differs: {path}")
    require(meta["grid"] == [21, 29, 52] and meta["flattened_tokens"] == 31668 and
            meta["Y_semantics"].startswith("native dense W2 apply output includes b2; gate not yet applied"),
            f"Panel coordinate or target convention differs: {path}")
    for key, shape in (("U", (512, 5120)), ("Y_pre_gate", (512, 5120)), ("gate", (1, 1, 5120))):
        require(tensor_contract(panel[key], shape, torch.bfloat16), f"Panel {key} contract differs: {path}")
    ids, coords = panel["token_indices"], panel["token_thw"]
    require(tensor_contract(ids, (512,), torch.int64) and tensor_contract(coords, (512, 3), torch.int64)
            and ids.unique().numel() == 512 and 0 <= ids.min() and ids.max() < 31668,
            f"Panel token IDs differ: {path}")
    require(torch.equal(coords, torch.stack((ids // (29 * 52), ids // 52 % 29, ids % 52), dim=1))
            and tensor_sha(ids) == ref["token_sha256"] == meta["token_id_sha256_le_int64"]
            and tensor_sha(panel["gate"]) == ref["gate_sha256"], f"Panel coordinate or gate hash differs: {path}")
    return panel


def load_pair(row, archive, bank, known_ids, device):
    anchor, hit = [load_endpoint(row[role], row, step, archive, bank)
                   for role, step in zip(("anchor", "hit"), STEPS[row["expert"]], strict=True)]
    ids = anchor["token_indices"].clone()
    require(torch.equal(ids, hit["token_indices"]), f"Mixed anchor/hit token rows at {row['source']}")
    source = row["source"]
    require(torch.equal(known_ids.setdefault(source, ids), ids), f"Token order changed across sites for {source}")
    operands = {"u0": anchor["U"], "u1": hit["U"], "y0": anchor["Y_pre_gate"], "y1": hit["Y_pre_gate"],
                "gate_bf16": hit["gate"], "anchor_gate_bf16": anchor["gate"]}
    pair = {key: value.to(device) for key, value in operands.items()}
    require(all(bool(torch.isfinite(value).all()) for value in pair.values()), f"Nonfinite endpoint {source}")
    pair.update(source=source, split=row["split"], gate=pair["gate_bf16"].reshape(-1).float(),
                token_ids=ids.to(device), folds=folds_for(source, ids, device))
    return pair


def native_runtime(scope, archive):
    config = bound_json(archive["input_manifests"]["strict_backend"])
    # The config requests 832x480. The registered crop produces the 29x52 latent raster.
    for key, expected in {"use_bfloat16": True, "enable_cfg": False, "infer_steps": 4, "target_video_length": 81,
                          "target_width": 832, "target_height": 480, "lora_configs": None}.items():
        require(config[key] == expected, f"Strict runtime recipe differs at {key}")
    require(config.get("dit_quant_scheme", "Default") == "Default" and not config.get("dit_quantized", False)
            and not config.get("clean_cuda_cache", False), "Strict FFN block is no longer Default BF16")
    capture = bound_json(archive["input_manifests"]["capture_scope"])
    require(capture["runtime_versions"]["torch"] == torch.__version__, "Capture Torch build changed")
    for path in reversed(capture["python_import_paths"]):
        sys.path.insert(0, path)
    for recorded in scope["native_sources"].values():
        checked(recorded)
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

    post = WanTransformerInfer.post_process
    for function, key in ((post, "transformer_infer"), (MMWeight, "mm_weight"), (GET_DTYPE, "envs")):
        require(Path(inspect.getfile(inspect.unwrap(function))).samefile(scope["native_sources"][key]["path"]),
                f"Imported a different LightX2V {key}")
    dtypes = GET_DTYPE(), GET_SENSITIVE_DTYPE()
    require(dtypes == (torch.bfloat16, torch.bfloat16), "Native infer or sensitive dtype is not BF16")
    host = SimpleNamespace(infer_dtype=dtypes[0], sensitive_layer_dtype=dtypes[1], clean_cuda_cache=False)
    return MMWeight, post, host


def native_linear(weight, bias, name, factory):
    module = factory(f"{name}.weight", f"{name}.bias", lora_prefix="blocks")
    # The Default block loader transposes checkpoint weights and preserves their strides on CUDA.
    module.weight = weight.T
    module.bias = bias
    require(not module.has_lora_branch and not module.has_diff, "Native observer unexpectedly has LoRA")
    return module.apply


def load_site(expert, layer, archive, checkpoint, device, runtime):
    key = f"{expert}{layer:02d}"
    binding = archive["sites"][key]
    require((binding["expert"], binding["layer"]) == (expert, layer) and binding["bf16_bitwise_equal"],
            f"No frozen L64 reuse at {key}")
    old_path = checked(binding["old_params"])
    old = torch.load(old_path, mmap=True, map_location="cpu", weights_only=True)
    q1_ref = next(row for row in archive["strict_q1_full80"] if (row["expert"], row["layer"]) == (expert, layer))
    require(all(q1_ref[k] == binding["strict_q1"][k] for k in ("path", "sha256", "bytes")),
            f"Strict Q1 archive identity differs at {key}")
    strict = torch.load(checked(q1_ref), mmap=True, map_location="cpu", weights_only=True)["A_q1_f32"]
    q1 = strict.bfloat16()
    require(equal_bits(q1, old["A_Q1_BF16"]) and tensor_sha(q1) == binding["strict_bf16_tensor_sha256"],
            f"Strict Q1 does not match bound L64 production input at {key}")
    a, q32 = old["A_L_BF16"], old["Q32"]
    require(tensor_contract(a, (5120, 5120), torch.bfloat16) and q32.shape == (5120, 32)
            and bool(torch.isfinite(a).all()) and bool(torch.isfinite(q32).all()), f"L64/Q32 invalid at {key}")
    with safe_open(checkpoint, framework="pt", device="cpu") as container:
        prefix = f"blocks.{layer}.ffn"
        values = [container.get_tensor(f"{prefix}.{name}") for name in SUFFIXES]
        for name, value, shape in zip(SUFFIXES, values, SHAPES, strict=True):
            require(tensor_contract(value, shape, torch.bfloat16) and
                    tensor_sha(value) == binding["ffn_tensor_sha256"][f"{prefix}.{name}"], f"Weight differs: {key}/{name}")
        w1, b1, w2, b2 = (value.to(device) for value in values)
    factory, post, host = runtime
    first = native_linear(w1, b1, f"{prefix}.0", factory)
    second = native_linear(w2, b2, f"{prefix}.2", factory)

    def observe(value):
        require(value.shape == (64, 5120) and value.dtype == torch.bfloat16 and value.device == device,
                f"{key}: native observer expects exactly 64 BF16 rows")
        out = second(F.gelu(first(value), approximate="tanh"))
        require(out.shape == value.shape and out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all()),
                f"{key}: native observer output contract differs")
        return out

    def gate_witness(value, pair):
        require(value.shape == pair["u1"].shape and value.dtype == torch.bfloat16, f"{key}: pre-gate output contract")
        residual = pair["u1"]  # Witness addend only. True pre-gate target and current gate are archived.
        gate = pair["gate_bf16"]
        expected = torch.add(residual, torch.mul(value, gate.squeeze()))
        return equal_bits(post(host, residual.clone(), value, gate), expected)

    identity = {"old_params": binding["old_params"], "strict_q1": q1_ref, "l64_tensor_sha256": tensor_sha(a),
                "q32_tensor_sha256": tensor_sha(q32), "ffn_tensor_sha256": binding["ffn_tensor_sha256"]}
    return {"key": key, "A": a.to(device), "q32": q32.to(device=device, dtype=torch.float64), "w1": w1, "b1": b1,
            "w2": w2, "b2": b2, "observe": observe, "gate_witness": gate_witness, "identity": identity}


def fill_probes(site, pair):
    pair["probe_y1"] = [site["observe"](pair["u1"].index_select(0, fold["probe"])) for fold in pair["folds"]]
