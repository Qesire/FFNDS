"""Run the frozen 50-source, three-arm evaluation in the existing school runtime.

The generation backend is reused, not reimplemented. Quality runs save only the
final latent; the separate timing runs do not add tensor capture or replay panels.
This file has been checked locally without launching model inference.
"""
import argparse
import gzip
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re

import numpy as np

from centered_runtime import FrozenCenteredRuntime, file_hash
from summarize_medium_eval import SEED, summarize_timing


ARMS = ("dense", "wide8192", "centered")
IDENTITY_SHA256 = "979e68efc18c1f999e0370c7698e435880190598aa79add6d23567d7eea4ff20"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def read_identity(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    if sha256(data).hexdigest() != IDENTITY_SHA256:
        raise ValueError("Expected the original, decompressed job-201500 receipt")
    receipt = json.loads(data)
    return {r["source"]: r["arms"]["centered"]["record"]
            for r in receipt["pairs"] if r["repeat"] == 0}


def timing_order(index, repeat):
    # Each source sees every arm in each ordinal position over its three rounds.
    base = ARMS if index % 2 == 0 else (ARMS[0], ARMS[2], ARMS[1])
    shift = (index // 2 + repeat) % 3
    return base[shift:] + base[:shift]


def check_pair(arms):
    records = [arms[arm]["record"] for arm in ARMS]
    if len({record["initial"] for record in records}) != 1:
        raise RuntimeError("Paired initial noise differs")
    for name in ("flows", "latents"):
        if len({record[name][0] for record in records}) != 1:
            raise RuntimeError(f"Native first-anchor {name} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Resolve every local argument before the archived runtime changes cwd.
    args = argparse.Namespace(**{k: v.resolve() for k, v in vars(args).items()})
    if args.out.exists():
        raise FileExistsError("Choose a new run output; partial results remain intact")
    cohort = json.loads(args.cohort.read_text())
    people = cohort["sources"]
    ids = [p["source_id"] for p in people]
    if len(people) != 50 or len(set(ids)) != 50:
        raise ValueError("Expected 50 distinct sources")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", sid) for sid in ids):
        raise ValueError("Source ids must be safe file-name components")
    if len({p["source_cluster_id"] for p in people}) != 50:
        raise ValueError("Each source must be an independent image cluster")
    if len({p["image_sha256"] for p in people}) != 50:
        raise ValueError("Duplicate image bytes in the cohort")
    timed = cohort["timing_source_ids"]
    if len(timed) != 12 or len(set(timed)) != 12 or not set(timed) <= set(ids):
        raise ValueError("Expected a frozen 12-source timing subset")
    if cohort.get("seed", 42) != 42:
        raise ValueError("This version is frozen to seed 42")
    identity = read_identity(args.identity_receipt)
    runtime = FrozenCenteredRuntime(args.runtime_dir, args.fit_dir)
    if cohort.get("method_binding_sha256") != runtime.binding_sha256 or cohort.get("candidate_id") != runtime.binding["candidate_id"]:
        raise ValueError("Cohort was frozen for another method binding")
    runtime.register_manifest(args.cohort)
    exp = runtime.experiment
    prefix = f"medium_{exp.job}_{file_hash(args.cohort)[:12]}"
    state = dict(phase="identity", schema="FFNDS_CENTERED_MEDIUM_RUN_V1", status="running",
                 arms=list(ARMS), seed=42, cohort=str(args.cohort),
                 cohort_sha256=file_hash(args.cohort), driver_sha256=file_hash(__file__),
                 method_binding_sha256=runtime.binding_sha256,
                 runtime_generate_sha256=runtime.runtime_generate_sha256,
                 job=exp.job, gpu=exp.state["gpu"], output_root=str(exp.output),
                 cold_init_seconds=exp.state["init_seconds"], setup=exp.state["setup"],
                 timing_scope="warm full request with symmetric backend trace witnesses; no tensor archival",
                 memory_scope="same process with wide8192 port resident for all arms",
                 identity_checks=[], warmup=[], quality=[], timing=[])
    save(args.out, state)
    # Reuse the first centered identity run as its warmup. These four requests
    # are preparation, outside the 150 quality + 108 timing count.
    for sid in ("vbi_0392", "vbi_1013"):
        row = runtime.render(sid, prefix + "_identity_centered", "centered")
        matches = runtime.generate.equal_trace(row["record"], identity[sid])
        state["identity_checks"].append(dict(source_id=sid, matches_201500=matches, output=row))
        save(args.out, state)
        if not matches:
            raise RuntimeError("Thin entry point differs from the archived centered method")
    for arm in ("dense", "wide8192"):
        state["warmup"].append(runtime.render("vbi_0392", prefix + "_warm_" + arm, arm))
        save(args.out, state)
    lookup = {p["source_id"]: p for p in people}
    # Method warmup is complete; quality-only latent capture starts after timing.
    state["phase"] = "formal_timing"
    save(args.out, state)
    for repeat in range(3):
        for index, sid in enumerate(timed):
            order = timing_order(index, repeat)
            outputs = {arm: runtime.render(sid, f"{prefix}_timing_{repeat}_{arm}", arm)
                       for arm in order}
            check_pair(outputs)
            if any(o.get("quality_capture_in_timing_window") is not False for o in outputs.values()):
                raise RuntimeError("Formal timing cannot include quality capture")
            state["timing"].append(dict(source_id=sid, source_cluster_id=lookup[sid]["source_cluster_id"],
                                        repeat=repeat, order=list(order), arms=outputs))
            save(args.out, state)
    timing_summary = summarize_timing(state, cohort, np.random.default_rng(SEED))
    save(args.out.with_name("TIMING_SUMMARY.json"), dict(job=exp.job, cohort_sha256=state["cohort_sha256"], timing=timing_summary))
    state["phase"] = "quality"
    save(args.out, state)
    orders = tuple(itertools.permutations(ARMS))
    for index, person in enumerate(people):
        sid = person["source_id"]
        order = orders[index % len(orders)]
        outputs = {arm: runtime.render(sid, f"{prefix}_quality_{arm}", arm,
                                       capture_final_latent=True) for arm in order}
        check_pair(outputs)
        state["quality"].append(dict(source_id=sid, source_cluster_id=person["source_cluster_id"],
                                     order=list(order), arms=outputs))
        save(args.out, state)
    state["phase"] = "complete"
    state["status"] = "complete"
    state["generation_requests"] = dict(quality=150, formal_timing=108, preparation=4, total=262)
    save(args.out, state)


if __name__ == "__main__":
    main()
