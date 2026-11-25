"""Bind ten-source outputs, score paired decoded RGB, and run native VBench dimensions."""
import argparse
from contextlib import closing
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ARMS = ("dense", "wide8192", "wide8192_u0_int8", "wide8192_y0_int8",
        "wide8192_u0_z0k_int8", "wide8192_all_int8", "wide8192_all_int4")
REFS = ARMS[:3]
DIMS = ("subject_consistency", "background_consistency", "aesthetic_quality",
        "motion_smoothness", "imaging_quality", "i2v_subject", "i2v_background")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    pending = path.with_name(path.name + ".%d.pending" % os.getpid())
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf8")
    pending.replace(path)


def verify(path, digest):
    require(sha(path) == digest, "Digest mismatch: " + str(path))


def checked_video(record, video, receipt, receipt_hash, video_hash, rgb_hash, geometry):
    verify(receipt, receipt_hash)
    verify(video, video_hash)
    rec = read(receipt)
    rec_video = rec["video"]
    require(rec["source"] == record["source"] and rec["experts"] == [0, 0, 1, 1]
            and len(rec["flows"]) == len(rec["latents"]) == 4, "Unmatched trajectory")
    require((rec["initial"], rec["flows"][0], rec["latents"][0]) ==
            (record["initial"], record["first_flow"], record["first_latent"]), "First anchor differs")
    require((video_hash, rgb_hash) == (rec_video["mp4_sha256"], rec_video["decoded_rgb24_sha256"])
            and list(geometry) == [rec_video[k] for k in ("width", "height", "frames", "fps")],
            "Unmatched rendered video")
    return dict(path=str(video), sha256=video_hash, rgb_sha256=rgb_hash,
                receipt=str(receipt), receipt_sha256=receipt_hash)


def prepare(args):
    cohort_path = args.cohort.resolve()
    cohort_hash = sha(cohort_path)
    cohort = read(cohort_path)
    sources = cohort["sources"]
    require(len(set(sources)) == len(sources) == 10 and sources == [r["source"] for r in cohort["records"]],
            "Expected ten distinct ordered sources")
    verify(cohort["metric_scope"], cohort["metric_scope_sha256"])
    geometry = cohort["assessed_geometry"]
    require(geometry == [832, 464, 81, 16] and cohort["seed"] == 42, "Unmatched protocol")
    arm_dirs = dict(pair.split("=", 1) for pair in args.arm)
    require(set(arm_dirs) == set(ARMS[1:]) and len(args.arm) == len(arm_dirs), "Expected six unique arm roots")
    info, digests = {}, {}
    for arm in ARMS[1:]:
        folder = Path(arm_dirs[arm]).resolve()
        state_path, seal = folder/"ARM_RESULT.json", folder/"ARM_RESULTS.sha256"
        sealed = {rel: digest for digest, rel in (s.split(maxsplit=1) for s in seal.read_text().splitlines())}
        verify(state_path, sealed[state_path.name])
        state = read(state_path)
        require(state["cohort_sha256"] == cohort_hash and state["status"] == "complete"
                and state["sources"] == sources and state["seed"] == 42 and state["mode"] == arm,
                "Unmatched completed arm: " + arm)
        rows = [row for row in state["rows"] if row["formal"]]
        require([r["source"] for r in rows] == sources, "Unmatched formal source order")
        for row in rows:
            rel = "output/formal/%s.json" % row["source"]
            require(row["receipt_sha256"] == sealed[rel] and row["mode"] == arm and row["group"] == "formal",
                    "Unsealed formal receipt")
        verify(folder.parent.parent/"MANIFEST.sha256", state["manifest_sha256"])
        digests[arm] = dict(root=str(folder), manifest_sha256=state["manifest_sha256"],
                            arm_result_sha256=sealed[state_path.name], arm_results_seal_sha256=sha(seal),
                            generation_job=state["job"])
        info[arm] = (folder, rows)
    require(len({entry["manifest_sha256"] for entry in digests.values()}) == 1, "Mixed code manifests")
    records = []
    for i, r in enumerate(cohort["records"]):
        verify(r["image_path"], r["image_sha256"])
        require(hashlib.sha256(r["prompt"].encode("utf8")).hexdigest() == r["prompt_sha256"],
                "Source prompt differs")
        entry = {k: r[k] for k in ("source", "image_name", "image_sha256", "prompt", "category", "dimensions")}
        videos = entry["videos"] = {}
        videos["dense"] = checked_video(r, Path(r["dense_video"]), Path(r["dense_receipt"]),
                                       r["dense_receipt_sha256"], r["dense_mp4_sha256"], r["dense_rgb_sha256"], geometry)
        for arm, (folder, rows) in info.items():
            row = rows[i]
            receipt = folder/"output"/"formal"/(r["source"]+".json")
            video = receipt.with_suffix(".mp4")
            require(folder.parent.parent/row["receipt"] == receipt and folder.parent.parent/row["video"] == video,
                    "Unexpected output path")
            videos[arm] = checked_video(r, video, receipt, row["receipt_sha256"], row["video_sha256"],
                                        row["decoded_rgb_sha256"], geometry)
        records.append(entry)
    save(args.binding, dict(schema="FFNDS_ANCHOR10_METRIC_BINDING_V1", cohort=str(cohort_path),
                            cohort_sha256=cohort_hash, metric_scope=cohort["metric_scope"],
                            metric_scope_sha256=cohort["metric_scope_sha256"], seed=cohort["seed"],
                            evaluator_sha256=sha(__file__), arms=list(ARMS), geometry=geometry,
                            arm_artifacts=digests, sources=sources, records=records))


def load_binding(path):
    binding = read(path)
    verify(__file__, binding["evaluator_sha256"])
    verify(binding["cohort"], binding["cohort_sha256"])
    verify(binding["metric_scope"], binding["metric_scope_sha256"])
    return binding


def decode(video, geometry):
    import imageio_ffmpeg
    import numpy as np
    verify(video["path"], video["sha256"])
    w, h, count, fps = geometry
    clip = np.empty((count, h, w, 3), dtype=np.uint8)
    digest = hashlib.sha256()
    stream = imageio_ffmpeg.read_frames(video["path"], pix_fmt="rgb24", input_params=["-threads", "1"],
                                        output_params=["-threads", "1"])
    with closing(stream):
        meta = next(stream)
        require(tuple(meta["size"]) == (w, h) and meta["fps"] == fps, "Unexpected decoded geometry")
        seen = 0
        for frame in stream:
            require(seen < count and len(frame) == w*h*3, "Unexpected decoded frame")
            clip[seen] = np.frombuffer(frame, np.uint8).reshape(h, w, 3)
            digest.update(frame)
            seen += 1
        require(seen == count and digest.hexdigest() == video["rgb_sha256"], "RGB hash or frame count differs")
    return clip


def pixel_distance(clip, ref):
    import numpy as np
    absolute, squared, temporal = 0, 0, 0
    previous = None
    for frame, target in zip(clip, ref):
        residual = np.subtract(frame, target, dtype=np.int16)
        absolute += int(np.abs(residual).sum(dtype=np.int64))
        squared += int(np.square(residual, dtype=np.int32).sum(dtype=np.int64))
        if previous is not None:
            temporal += int(np.abs(residual - previous).sum(dtype=np.int64))
        previous = residual
    mse = squared / clip.size
    return dict(rgb_mae=absolute/clip.size, temporal_delta_mae=temporal/(clip.size-clip[0].size),
                mse=mse, psnr_db=10*math.log10(255**2/mse) if mse else None)


def pixels(args):
    binding = load_binding(args.binding)
    arms = {a: {r: {"rows": []} for r in REFS} for a in ARMS}
    for record in binding["records"]:
        clips = {a: decode(record["videos"][a], binding["geometry"]) for a in ARMS}
        memo = {}
        for arm in ARMS:
            for ref in REFS:
                key = tuple(sorted((arm, ref)))
                if key not in memo:
                    memo[key] = (dict(rgb_mae=0., temporal_delta_mae=0., mse=0., psnr_db=None) if arm == ref
                                 else pixel_distance(clips[arm], clips[ref]))
                arms[arm][ref]["rows"].append(dict(source=record["source"], **memo[key]))
        del clips
        print("SCORED RGB", record["source"], flush=True)
    for per_arm in arms.values():
        for item in per_arm.values():
            rows = item["rows"]
            item["n"] = len(rows)
            item["mean"] = {k: (sum(r[k] for r in rows) / len(rows) if all(r[k] is not None for r in rows) else None)
                            for k in ("rgb_mae", "temporal_delta_mae", "mse", "psnr_db")}
    save(args.out/"pixels.json", dict(binding_sha256=sha(args.binding), evaluator_sha256=sha(__file__),
                                      references=REFS, units="decoded RGB24, levels 0..255; PSNR in dB; null PSNR denotes positive infinity",
                                      aggregation="equal-source means over 81 frames or 80 adjacent transitions",
                                      arms=arms, job=os.getenv("SLURM_JOB_ID")))


def aggregate(dim, rows):
    if dim == "background_consistency":
        return sum(r["video_sim"] for r in rows) / sum(r["cnt_per_video"] for r in rows)
    return sum(float(r["video_results"]) for r in rows) / len(rows) / (100 if dim == "imaging_quality" else 1)


def vbench(args):
    import numpy as np
    import torch
    import vbench2_beta_i2v as i2v
    binding_file, out = args.binding.resolve(), args.out.resolve()
    binding_hash = sha(binding_file)
    binding = load_binding(binding_file)
    scope = read(binding["metric_scope"])
    dim = args.dim
    require(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(), "Expected one A100")
    ref = Path(scope["quality_reference_dir"])
    for name in ("metric_loaders.py", "metric_contract.py"):
        verify(ref/name, scope["quality_reference_sha256"][name])
    for name, version in scope["metric_package_versions"].items():
        require(importlib.metadata.version(name) == version, "Package version differs: " + name)
    sys.path.insert(0, str(ref))
    from metric_loaders import safe_loaders
    from metric_contract import loader_spec
    project = Path(scope["metric_project_root"])
    os.chdir(project)
    mapping, weights, _ = loader_spec(dim, project, scope["metric_cache_root"], scope["metric_site_packages"])
    for p in weights:
        verify(p, scope["metric_asset_sha256"][str(p)])
    module = importlib.import_module(("vbench2_beta_i2v" if dim.startswith("i2v_") else "vbench") + "." + dim)
    verify(module.__file__, scope["metric_code_sha256"][module.__file__])
    compute = getattr(module, "compute_" + dim)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed = scope["metric_seed"]
    for row in binding["records"]:
        verify(project/"vbench2_beta_i2v/data/crop/16-9"/row["image_name"], row["image_sha256"])
    arms = {}
    order = binding["arms"]
    require(bool(order) and len(set(order)) == len(order), "Expected distinct bound arms")
    for arm in order:
        table = {row["videos"][arm]["path"]: row for row in binding["records"]}
        for path, row in table.items():
            verify(path, row["videos"][arm]["sha256"])
        part_path = out/"parts"/(dim+"_"+arm+".json")
        if part_path.exists():
            part = read(part_path)
            require(part["binding_sha256"] == binding_hash and part["dim"] == dim and part["arm"] == arm,
                    "Partial result disagrees with binding")
            arms[arm] = part["result"]
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        info = out/"info"/(dim+"_"+arm+".json")
        save(info, [dict(prompt_en=row["prompt"], image_name=row["image_name"], dimension=[dim], video_list=[path])
                    for path, row in table.items()])
        started = time.monotonic()
        with torch.no_grad(), safe_loaders(torch, i2v, dim, mapping, scope):
            score, details = compute(str(info), "cuda", mapping)
        require(len(details) == len(table) == len(binding["sources"]) and {r["video_path"] for r in details} == set(table),
                "Native source count or paths differ: " + arm)
        require(math.isclose(aggregate(dim, details), float(score), rel_tol=1e-12, abs_tol=1e-12),
                "Native reduction differs")
        by_path = {r["video_path"]: r for r in details}
        rows, eligible = [], []
        for path, record in table.items():
            raw = by_path[path]
            normalized = aggregate(dim, [raw])
            require(math.isfinite(normalized), "Nonfinite native source score")
            tagged = dim in record["dimensions"]
            if tagged:
                eligible.append(raw)
            rows.append(dict(source=record["source"], score=normalized, eligible=tagged, native=raw))
        result = dict(n_all=len(rows), n_eligible=len(eligible), all_score=float(score),
                      eligible_score=aggregate(dim, eligible), rows=rows, elapsed_s=time.monotonic()-started,
                      job=os.getenv("SLURM_JOB_ID"))
        save(part_path, dict(binding_sha256=binding_hash, dim=dim, arm=arm, result=result))
        arms[arm] = result
        print("SCORED", dim, arm, len(eligible), result["eligible_score"], flush=True)
    save(out/(dim+".json"), dict(dimension=dim, binding_sha256=binding_hash, evaluator_sha256=sha(__file__),
                                 scope_sha256=binding["metric_scope_sha256"], seed=seed, arms=arms,
                                 package_versions=scope["metric_package_versions"], site_module_sha256=sha(module.__file__),
                                 weights_sha256={str(p): scope["metric_asset_sha256"][str(p)] for p in weights}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--binding", type=Path, required=True)
    prep.add_argument("--cohort", type=Path, required=True)
    prep.add_argument("--arm", action="append", required=True, metavar="MODE=DIRECTORY")
    for name in ("pixels", "vbench"):
        parser = sub.add_parser(name)
        parser.add_argument("--binding", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        if name == "vbench":
            parser.add_argument("--dim", required=True, choices=DIMS)
    arguments = cli.parse_args()
    {"prepare": prepare, "pixels": pixels, "vbench": vbench}[arguments.action](arguments)
