"""Prepare four review-only P2 stages from completed P1 and strict evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ARMS = {"L0": "l0", "L+": "lplus", "Q32": "q32", "N32": "n32"}
DEV3 = ("vbi_0392", "vbi_1013", "vbi_0800")
SITES = {(e, n) for e in ("high", "low") for n in (4, 19, 34)}
GEOMETRY = (81, 832, 464, 16)
NOISE_SHAPE = [16, 21, 58, 104]
STRICT_Q1 = "dc2d2b2289ff119d432c5a347facc446601e5ef2526eb10c09f86f06df88e247"
STRICT_TRANSFORMER = "5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225"
P1_PRODUCER = "afa7b57ac5ddb3d16d89ecc71790a0e9084b7bdda6479c618bdf41841ae674dc"
P2_PRODUCER = "7156a6b686b83c58a26357a7aaed3211a7f2a5286e8f76a2cfebc35118ef8e30"
RENDERER = "154a75968d2f21bf473d28b9bf4e5ef5d9930bdb41704b8eeee4066d4fc2876b"
STRICT_SOURCES = {
    "icc_native_job.py": "f233fe6994b1d27def5a3f2f976797f73e0573d290699468a67cc860099c1a4a",
    "wan_comparison_job.py": "4474cf184ed34a266e9401f432226553fe9ed5ac1fafc13f0fc7f3219e3f4f42",
    "wan_ffn_port.py": "38e196015e59eaf4900067927ec38498b2ea3bbeacc1f3218d2a37c34e4b78ed",
    "wan_ffn_comparator_ports.py": "36c37453635e8684147918cb4a4c3257298fe94e5ff28d3de838debd1d650c79",
}
STAGE_FILES = (
    "wan_state_operator_job.py",
    "ffnds_state_operator/analysis.py",
    "ffnds_state_operator/features.py",
    "ffnds_state_operator/fit.py",
    "ffnds_state_operator/bf16_replay.py",
    "ffnds_state_operator/native.py",
)


def require(check, message):
    if not check:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asset(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    require(expected is None or digest == expected, f"changed asset: {path}")
    return {"path": str(path), "sha256": digest}


def verify(spec):
    return asset(spec["path"], spec["sha256"])


def write(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def geometry(row):
    return tuple(row[key] for key in ("frames", "width", "height", "fps"))


def completed_review(stage, scope, scope_spec):
    review = asset(stage / "REVIEW.json")
    active = asset(stage / "ACTIVE_RESEARCH.json")
    decisions = read(review["path"])
    registry = read(active["path"])
    registry = registry.get(scope["scope_id"].lower(), registry)
    require(decisions["scope_sha256"] == scope_spec["sha256"]
            and decisions["recheck_disposition"] == "APPROVE"
            and registry["scope_sha256"] == scope_spec["sha256"]
            and registry["review_sha256"] == review["sha256"],
            f"review and ACTIVE receipts do not bind {stage}")
    return {"review": review, "active": active}


def verified_p1(p1):
    p1_scope_spec = asset(p1 / "scope.json")
    scope = read(p1_scope_spec["path"])
    p1_review = completed_review(p1, scope, p1_scope_spec)
    require(scope["method"] == "STATE_OPERATOR_P1"
            and scope["l64_policy"] == "REUSE_ALL_SIX"
            and (scope["max_nfe"], scope["max_d4"], scope["retries"]) == (0, 0, 0),
            "P2 is scoped to the reviewed strict reuse P1")
    p0 = verify(scope["bindings"])
    p0_binding = read(p0["path"])
    p0_exit = verify(scope["p0_exit"])
    p0_run = asset(Path(p0_exit["path"]).with_name("RUN.json"))
    p0_scope = asset(Path(p0_exit["path"]).with_name("scope.json"))
    p0_review = completed_review(Path(p0_scope["path"]).parent, read(p0_scope["path"]), p0_scope)
    exited, started = read(p0_exit["path"]), read(p0_run["path"])
    require(exited["exit_code"] == 0 and exited["slurm_job_id"] == started["slurm_job_id"]
            and exited["output_sha256"]["BINDINGS.json"] == p0["sha256"]
            and started["scope_sha256"] == p0_scope["sha256"]
            and p0_binding["status"] == "PASS" and p0_binding["l64_policy"] == "REUSE_ALL_SIX"
            and set(p0_binding["sites"]) == {f"{expert}{layer:02d}" for expert, layer in SITES}
            and all(item["bf16_bitwise_equal"] is True for item in p0_binding["sites"].values()),
            "P0 scope/receipt or six-site exact-Q1 identity differs")
    fit = asset(p1 / "output/fit/FIT_MANIFEST.json")
    bf16 = asset(p1 / "output/BF16_ACCEPTANCE.json")
    run_spec, exit_spec = asset(p1 / "receipt/RUN.json"), asset(p1 / "receipt/EXIT.json")
    result, run, end = read(bf16["path"]), read(run_spec["path"]), read(exit_spec["path"])
    job = str(end["slurm_job_id"])
    require(run["scope_sha256"] == result["scope_sha256"] == p1_scope_spec["sha256"]
            and str(run["allocation"]["job_id"]) == str(result["slurm_job_id"]) == job
            and result["producer_sha256"] == scope["bundle_sha256"]
            and result["fit"] == fit and result["strict_q1"] == scope["parameters"]
            and result["native_transformer_sha256"] == STRICT_TRANSFORMER
            and result["schema"] == "FFNDS_STATE_OPERATOR_BF16_V1"
            and result["status"] == "PASS" and (result["nfe"], result["d4"], end["exit_code"]) == (0, 0, 0),
            "P1 result did not complete under the reviewed allocation")
    witnesses = result["sites"]
    require(len(witnesses) == 24 and {(row["expert"], row["layer"], row["arm"]) for row in witnesses}
            == {(expert, layer, arm) for expert, layer in SITES for arm in ARMS},
            "P1 BF16 witness coverage differs")
    for row in witnesses:
        for field in ("exact_deployed_order", "native_gate_write_bitwise_equal", "q1_shared_arithmetic_bitwise_equal"):
            require(row[field] is True, f"failed BF16 acceptance field: {field}")
        require(len(row["pairs"]) == 11 and {p["source"] for p in row["pairs"]} == set(p0_binding["dev_sources"]),
                "P1 BF16 witness lacks frozen DEV11")
    manifest = read(fit["path"])
    rows = list(manifest["sites"].values())
    require(manifest["schema"] == "FFNDS_STATE_OPERATOR_FIT_V1" and manifest["bindings"] == p0
            and manifest["l64_policy"] == "REUSE_ALL_SIX"
            and tuple(manifest["arms"]) == tuple(ARMS) and len(rows) == 6
            and {(row["expert"], row["layer"]) for row in rows} == SITES,
            "fit manifest and reviewed P0/P1 contract differ")
    for row in rows:
        frozen = verify(row["params"])
        require(Path(frozen["path"]).parent == Path(fit["path"]).parent, "fit parameter escaped completed P1")
    for name, digest in manifest["derived"].items():
        asset(Path(fit["path"]).parent / name, digest)
    return scope, p0, fit, {"scope_sha256": p1_scope_spec["sha256"], "slurm_job_id": job,
                            "scope": p1_scope_spec, "result": bf16, "run": run_spec, "exit": exit_spec, **p1_review}, {
        "scope": p0_scope, "run": p0_run, "exit": p0_exit, **p0_review,
    }


def strict_evidence(root, p1_scope):
    q1 = root / "wan_main_q1_01_v1"
    historical = read(q1 / "scope.json")
    strict = asset(root / "q1_v4_fixed_remaining_v2/output/Q1_MANIFEST.json", STRICT_Q1)
    noise = verify(historical["dense_initial_noise"])
    config = verify(p1_scope["strict_effective_config"])
    require(config["path"] == str(q1 / "output/EFFECTIVE_CONFIG.json"), "strict Q1 config path differs")
    frozen = verify(historical["frozen_comparison"])
    require(read(frozen["path"])["parameters"]["Q1r2"] == historical["parameters"] == p1_scope["parameters"] == strict,
            "P0/P1 and strict publication Q1 manifest differ")
    identity_keys = ("v2_reference_dir", "reference_sha256", "python_executable", "python_executable_sha256",
                     "scheduler_sha256", "schedule", "native_transformer_sha256")
    identical = {key: historical[key] for key in identity_keys}
    require(identical == {key: p1_scope[key] for key in identity_keys}
            and identical["native_transformer_sha256"] == STRICT_TRANSFORMER, "historical runtime identity changed")
    reference = Path(identical["v2_reference_dir"])
    for filename, digest in identical["reference_sha256"].items():
        asset(reference / filename, digest)
    sources = {row["record_id"]: row for row in read(reference / "v2_conditions.json")}
    native_scope = read(reference / "scope.json")
    effective = read(config["path"])
    require(effective["use_bfloat16"] is True and all(effective.get(key) == expected
            for key, expected in native_scope["required_runtime_config"].items()),
            "strict effective Wan config differs from the frozen BF16 recipe")
    dense_stage = root / "wan_dense_export_nine_v1"
    dense_scope_spec = asset(dense_stage / "scope.json")
    dense_scope = read(dense_scope_spec["path"])
    admission = verify(dense_scope["dense_admission"])
    rows = {row["source"]: row for row in read(admission["path"])["slots"]}
    actual_noise = read(noise["path"])
    source_map = {}
    dense_receipts = completed_video_run(dense_stage, "Dense", 36)
    for source in DEV3:
        condition, slot, initial = sources[source], rows[source], actual_noise[source]
        image = asset(condition["image_path"], condition["image_sha256"])
        expected_prompt = hashlib.sha256(condition["prompt"].encode("utf-8")).hexdigest()
        checkpoint_ids = {expert: spec["sha256"] for expert, spec in native_scope["checkpoints"].items()}
        require(condition["seed"] == slot["seed"] == 42 and condition["cycle_role"] == "DEV"
                and condition["orient"] == "landscape"
                and condition["prompt_sha256"] == expected_prompt == slot["input_identity"]["prompt_sha256"]
                and image["sha256"] == slot["input_identity"]["image_sha256"]
                and checkpoint_ids == slot["input_identity"]["checkpoint_sha256"]
                and initial == slot["initial_noise"] and initial["shape"] == NOISE_SHAPE
                and initial["dtype"] == "torch.bfloat16"
                and slot["scheduler"] == historical["schedule"] and geometry(slot["decoded"]) == GEOMETRY,
                f"Dense source geometry/noise/condition differs at {source}")
        receipt = verify(slot["producer_receipt"])
        original = read(receipt["path"])
        require(original.get("record_id", original.get("source")) == source and original["initial_noise"] == initial
                and original.get("video", original)["decoded_rgb24_sha256"] == slot["decoded"]["decoded_rgb24_sha256"],
                f"original Dense receipt differs at {source}")
        dense = {"receipt": receipt, "video": verify(slot["video"])}
        if slot["main_slot_complete"]:
            dense["latent"] = verify(slot["final_latent"])
        else:
            exported = dense_stage / "output" / f"{source}.json"
            emitted = asset(exported, dense_receipts[source])
            row = read(emitted["path"])
            require(row["source"] == source and row["arm"] == "Dense" and row["seed"] == 42
                    and row["initial_noise"] == initial and row["schedule"] == historical["schedule"]
                    and row["prior_dense_rgb_equal"] is True and row["prior_dense_latents_equal"] is True
                    and row["video"]["decoded_rgb24_sha256"] == slot["decoded"]["decoded_rgb24_sha256"],
                    f"Dense latent export failed historical parity at {source}")
            dense = {"receipt": emitted, "video": asset(exported.with_suffix(".mp4"), row["video"]["mp4_sha256"]),
                     "latent": verify(row["latent_file"]), "prior_receipt": receipt}
        source_map[source] = {"image": image, "prompt_sha256": expected_prompt, "seed": 42,
                              "source_cluster_id": condition["source_cluster_id"], "initial_noise": initial, "dense": dense}
    require(len({source_map[s]["source_cluster_id"] for s in DEV3}) == len(DEV3), "duplicate image cluster")
    for arm, directory in (("Q1r2", "wan_main_q1_01_v1"), ("Global-P64", "wan_main_gp64_01_v1")):
        directory = root / directory
        digests = completed_video_run(directory, arm, 44)
        for source in DEV3:
            receipt = asset(directory / "output" / f"{source}.json", digests[source])
            row = read(receipt["path"])
            require(row["source"] == source and row["arm"] == arm and row["seed"] == 42
                    and row["initial_noise"] == actual_noise[source] and row["schedule"] == historical["schedule"]
                    and geometry(row["video"]) == GEOMETRY and len(row["sites"]) == 160,
                    f"strict {arm} comparator differs at {source}")
            source_map[source][arm] = {
                "receipt": receipt, "video": asset(Path(receipt["path"]).with_suffix(".mp4"), row["video"]["mp4_sha256"]),
                "latent": verify(row["latent_file"]),
            }
    return identical, strict, noise, config, frozen, admission, source_map


def completed_video_run(stage, arm, nfe):
    summary = read(stage / "output/SUMMARY.json")
    started = read(stage / "receipt/RUN.json")
    exited = read(stage / "receipt/EXIT.json")
    require(started["scope_sha256"] == sha(stage / "scope.json")
            and str(started["allocation"]["job_id"]) == str(exited["slurm_job_id"])
            and exited["exit_code"] == 0
            and summary["status"] == "COMPLETE" and summary["arm"] == arm and summary["nfe"] == nfe
            and len(summary["records"]) * 4 == nfe,
            f"archived generation receipts do not close: {stage}")
    return {row["source"]: row["receipt_sha256"] for row in summary["records"]}


def build_p2_producer(base, destination):
    """Extend the frozen P1 companion only in the generated P2 bundle."""
    require(base["sha256"] == P1_PRODUCER, "unexpected P1 producer source")
    source = Path(verify(base)["path"]).read_bytes().decode("utf-8").replace("\r\n", "\n")
    before = "        pipe, output, prior = build_pipeline(stage, scope, reference)\n"
    check = """        live_config_path = output / "EFFECTIVE_CONFIG.json"
        expected_config = scope["strict_effective_config"]
        bound_json(expected_config)
        live_config = {"path": str(live_config_path), "sha256": sha(live_config_path)}
        parity = {
            "schema": "FFNDS_STATE_OPERATOR_RECIPE_PARITY_V1",
            "scope_sha256": args.scope_sha,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "live": live_config,
            "strict": expected_config,
            "exact_bytes_equal": live_config["sha256"] == expected_config["sha256"],
            "nfe_at_check": 0,
        }
        parity_path = output / "RECIPE_PARITY.json"
        write_json(parity_path, parity)
        if not parity["exact_bytes_equal"]:
            raise RuntimeError("live effective configuration differs byte-for-byte from strict Q1")
"""
    receipt_line = '            "checkpoints": prior["checkpoints"], "producer_sha256": scope["bundle_sha256"],\n'
    parity_line = '            "recipe_parity": {"path": str(parity_path), "sha256": sha(parity_path)},\n'
    require(source.count(before) == source.count(receipt_line) == 1, "P1 producer insertion sites changed")
    source = source.replace(before, before + check).replace(receipt_line, receipt_line + parity_line)
    compile(source, str(destination), "exec")
    with destination.open("xb") as stream:
        stream.write(source.encode("utf-8"))

def prepare(p1, strict_root, out):
    require(not out.exists(), f"staging destination already exists: {out}")
    p1_scope, p0, fit, bf16, p0_evidence = verified_p1(p1)
    runtime, q1, noise, config, frozen, dense_admission, sources = strict_evidence(strict_root, p1_scope)
    tool_dir = Path(__file__).resolve().parent
    generator = asset(Path(__file__).resolve())
    renderer_generator = asset(tool_dir / "stage_renderer.py",
                               "b5d0dd837ef0137aad9dba97d2e473222e5002401bd9988c69335cf36f961ed9")
    launcher = asset(tool_dir / "wan_state_operator_job.sh",
                     "198b95e74d4a6bcb060d76514f066e33e56a05fd64f2682aa210ac73add3379a")
    historical_root = strict_root / "wan_main_q1_01_v1"
    producer_files = {name: asset(historical_root / name, digest)
                      for name, digest in STRICT_SOURCES.items()}
    for file in STAGE_FILES:
        producer_files[file] = asset(p1 / file, p1_scope["bundle_sha256"][file])
    p2_scope_base = {
        **runtime, "git_base": p1_scope["git_base"], "method": "WAN_FREE_GENERATION",
        "force_dense": False, "seed": 42, "sources": list(DEV3), "max_d4": 3, "max_nfe": 12, "retries": 0,
        "parameters": q1, "bindings": p0, "fit": fit, "bf16_acceptance": bf16, "p0_evidence": p0_evidence,
        "frozen_comparison": frozen, "dense_initial_noise": noise,
        "strict_effective_config": config, "dense_admission": dense_admission,
        "candidate_sites": [[expert, layer] for expert, layer in sorted(SITES)],
        "l64_policy": "REUSE_ALL_SIX", "untouched_parameter_sites": 74,
        "expected_ffn_calls": [40, 40, 40, 40], "expected_candidate_ffn_calls": 12,
        "expected_strict_ffn_calls": 148, "geometry": list(GEOMETRY), "source_identity": sources,
        "resources": {
            "gpus": 1, "cpu_cores": 4, "host_memory_gib": 160, "gpu_memory_gib": 78,
            "wall_seconds": 5400, "soft_seconds": 5100, "min_free_disk_gib": 38,
            "in_flight_disk_floor_gib": 12, "partition": "P107-A100",
            "qos": "qos_p107-a100", "account": "competition", "session_max_gpus": 2,
        },
    }
    with TemporaryDirectory(prefix=".p2-preparation-", dir=out.parent) as tmp:
        tmp = Path(tmp)
        draft = tmp / "stages"
        draft.mkdir()
        renderer = tmp / "wan_comparison_job.py"
        subprocess.run([sys.executable, renderer_generator["path"], producer_files["wan_comparison_job.py"]["path"],
                        str(renderer)], check=True)
        p2_producer = tmp / "wan_state_operator_job.py"
        build_p2_producer(producer_files["wan_state_operator_job.py"], p2_producer)
        staged_files = dict(producer_files, **{
            "wan_state_operator_job.py": asset(p2_producer, P2_PRODUCER),
            "wan_comparison_job.py": asset(renderer, RENDERER),
            "wan_state_operator_job.sh": launcher,
        })
        inventory = {"status": "AWAITING_INDEPENDENT_REVIEW", "max_experiment_concurrency": 2,
                     "total_d4": 12, "total_nfe": 48, "stage_generator": generator,
                     "base_renderer": producer_files["wan_comparison_job.py"],
                     "base_p1_producer": producer_files["wan_state_operator_job.py"],
                     "p2_producer_sha256": staged_files["wan_state_operator_job.py"]["sha256"],
                     "renderer_generator": renderer_generator, "stages": {}}
        for arm, slug in ARMS.items():
            stage = draft / slug
            stage.mkdir()
            digests = {}
            for relative, spec in staged_files.items():
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(spec["path"], path)
                digests[relative] = asset(path, spec["sha256"])["sha256"]
            write(stage / "scope.json", {
                **p2_scope_base, "scope_id": f"FFNDS_STATE_OPERATOR_P2_{slug.upper()}_20261118_V1",
                "remote_stage": str(out / slug), "arm": arm, "bundle_sha256": digests,
            })
            scope_digest = sha(stage / "scope.json")
            with (stage / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as manifest:
                for relative, digest in sorted(dict(digests, **{"scope.json": scope_digest}).items()):
                    manifest.write(f"{digest}  {relative}\n")
            inventory["stages"][arm] = {"path": str(out / slug), "scope_sha256": scope_digest}
        write(draft / "PREPARED.json", inventory)
        require(not out.exists(), f"staging destination was concurrently created: {out}")
        draft.rename(out)
    return inventory


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--p1", required=True, type=Path, help="completed combined P1 stage")
    cli.add_argument("--strict", required=True, type=Path, help="completed strict reproduction root")
    cli.add_argument("--out", required=True, type=Path, help="new P2 stage parent directory")
    args = cli.parse_args()
    output = args.out.parent.resolve(strict=True) / args.out.name
    inventory = prepare(args.p1.resolve(strict=True), args.strict.resolve(strict=True), output)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
