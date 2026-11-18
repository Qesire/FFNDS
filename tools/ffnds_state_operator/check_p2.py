"""Verify the successful finite four-arm production jobs and original media."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path


ARMS = {"L0": "l0", "L+": "lplus", "Q32": "q32", "N32": "n32"}
SOURCES = ("vbi_0392", "vbi_1013", "vbi_0800")
CANDIDATE_LAYERS = {4, 19, 34}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, subject):
    if not condition:
        raise ValueError(f"P2 evidence mismatch: {subject}")


def check_source(stage, scope, item):
    source = item["source"]
    out = stage / "output"
    path = out / f"{source}.json"
    require(sha(path) == item["receipt_sha256"], source)
    row = read(path)
    require((row["source"], row["arm"], row["seed"]) == (source, scope["arm"], 42)
            and row["schedule"] == scope["schedule"]
            and row["initial_noise"] == scope["source_identity"][source]["initial_noise"]
            and row["dispatch"] == [{"step": i, "expert": i // 2} for i in range(4)], (source, "dispatch and initial noise"))
    require(row["ffn_calls_by_step"] == [40] * 4 and len(row["sites"]) == 160, (source, "call count"))
    for index, site in enumerate(row["sites"]):
        step, layer = divmod(index, 40)
        require((site["step"], site["layer"]) == (step, layer) and
                site["expert"] == ("high" if step < 2 else "low")
                and site["kind"] == ("anchor" if step in (0, 2) else "hit")
                and site.get("operator") == (scope["arm"] if layer in CANDIDATE_LAYERS else None),
                (source, step, layer))
    gates = row["gate_witness"]
    require(len(gates) == 12 and {(g["step"], g["layer"]) for g in gates} ==
            {(step, layer) for step in range(4) for layer in CANDIDATE_LAYERS} and
            all(g["full_write_bitwise_equal"] for g in gates), (source, "actual current native gate"))
    require([entry["step"] for entry in row["steps"]] == [0, 1, 2, 3] and
            all(len(entry[k]["sha256"]) == 64 for entry in row["steps"] for k in ("flow", "latent")) and
            row["steps"][-1]["latent"] == row["final_latent"], (source, "trajectory hashes"))
    video, latent = row["video"], row["latent_file"]
    require(tuple(video[k] for k in ("frames", "width", "height", "fps")) == (81, 832, 464, 16)
            and sha(out / f"{source}.mp4") == video["mp4_sha256"] and
            Path(latent["path"]) == out / f"{source}_latent.pt"
            and sha(latent["path"]) == latent["sha256"], (source, "media and latent"))
    return {"source": source, "recipe_exact": True, "initial_noise_sha256": row["initial_noise"]["sha256"],
            "native_gate_checks": 12, "strict_q1_calls": 148, "total_ffn": 160,
            "video_sha256": video["mp4_sha256"], "latent_sha256": latent["sha256"],
            "receipt_sha256": item["receipt_sha256"], "instrumented_wall_s": row["wall_seconds"],
            "peak_gpu_allocated": row["peak_gpu_allocated"]}


def verify(root):
    inventory = read(root / "PREPARED.json")
    results = {}
    for arm, slug in ARMS.items():
        stage = root / slug
        scope = read(stage / "scope.json")
        receipt = stage / "receipt"
        run, end = read(receipt / "RUN.json"), read(receipt / "EXIT.json")
        summary = read(stage / "output/SUMMARY.json")
        scope_hash = sha(stage / "scope.json")
        require(scope_hash == inventory["stages"][arm]["scope_sha256"] == run["scope_sha256"],
                (slug, "scope identity"))
        require(end["exit_code"] == 0 and str(run["allocation"]["job_id"]) == str(end["slurm_job_id"]),
                (slug, "job identity"))
        require((summary["status"], summary["arm"], summary["nfe"]) == ("COMPLETE", arm, 12) and
                tuple(item["source"] for item in summary["records"]) == SOURCES,
                (slug, "three completed trajectories"))
        require(sha(stage / "output/EFFECTIVE_CONFIG.json") == scope["strict_effective_config"]["sha256"]
                and read(stage / "output/RECIPE_PARITY.json")["exact_bytes_equal"] is True,
                (slug, "native effective config parity"))
        sources = [check_source(stage, scope, item) for item in summary["records"]]
        results[arm] = {"job_id": end["slurm_job_id"], "scope_sha256": scope_hash,
                        "run_sha256": sha(receipt / "RUN.json"), "exit_sha256": sha(receipt / "EXIT.json"),
                        "records": sources, "exit_code": 0}
    return {"status": "PASS", "new_d4": 12, "nfe": 48, "ffn_calls": 1920,
            "candidate_native_gate_witnesses": 144, "arms": results}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    result = verify(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["status"], result["new_d4"], result["nfe"], result["candidate_native_gate_witnesses"])

