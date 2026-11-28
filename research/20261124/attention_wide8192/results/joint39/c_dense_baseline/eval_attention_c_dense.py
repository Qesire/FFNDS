"""Verify the timed videos, bind the first repetition, and run paired RGB diagnostics."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

from eval_anchor10 import DIMS, checked_video, decode, load_binding, pixel_distance, read, save, sha, verify


def prepare(generation, metric_scope, metric_scope_sha, output):
    sealed = {rel: digest for digest, rel in (line.split(maxsplit=1) for line in (generation / "RESULTS.sha256").read_text().splitlines())}
    producer = generation / "output/DENSE_C_RESULT.json"
    verify(producer, sealed["output/DENSE_C_RESULT.json"])
    verify(metric_scope, metric_scope_sha)
    result, protocol = read(producer), read(generation / "DENSE_C_PROTOCOL.json")
    verify(generation / "DENSE_C_PROTOCOL.json", result["protocol_sha256"])
    sources_file = generation / "CONFIRM_SOURCES.json"
    verify(sources_file, result["manifest_sha256"])
    frozen = read(sources_file)
    geometry = [832, 464, 81, 16]
    assert result["status"] == "complete" and len(result["pairs"]) == 12 and protocol["seed"] == 42
    assert set(protocol["score_dimensions"]) == set(DIMS)
    score_root = Path(read(metric_scope)["metric_project_root"]) / "vbench2_beta_i2v/data/crop/16-9"
    arms = ("dense", "C")
    rows = []
    repeat_zero = [pair for pair in result["pairs"] if pair["repeat"] == 0]
    for reference, pair in zip(frozen["sources"], repeat_zero, strict=True):
        sid = reference["source_id"]
        assert sid == pair["source_id"] and reference["source_cluster_id"] == pair["cluster_id"]
        image = Path(reference["image_path"])
        verify(image, reference["image_sha256"])
        verify(score_root / image.name, reference["image_sha256"])
        assert sha256(reference["prompt"].encode()).hexdigest() == reference["prompt_sha256"]
        row = dict(source=sid, image_name=image.name, image_sha256=reference["image_sha256"],
                   prompt=reference["prompt"], category=reference["category"], dimensions=DIMS, videos={})
        for arm in arms:
            rec = pair["arms"][arm]["record"]
            assert rec["group"] == f"paired_0_{arm}" and rec["job"] == result["job"]
            artifact = f"output/{rec['group']}/{sid}"
            inputs = dict(source=sid, initial=rec["initial"], first_flow=rec["flows"][0], first_latent=rec["latents"][0])
            assert sealed[artifact + ".mp4"] == rec["video"]["mp4_sha256"]
            row["videos"][arm] = checked_video(inputs, generation / (artifact + ".mp4"), generation / (artifact + ".json"),
                sealed[artifact + ".json"], rec["video"]["mp4_sha256"], rec["video"]["decoded_rgb24_sha256"], geometry)
        rows.append(row)
    assert [r["source"] for r in rows] == protocol["source_ids"]
    save(output, dict(schema="FFNDS_C_DENSE_V1", geometry=geometry, seed=42, arms=arms, records=rows,
        sources=protocol["source_ids"], cohort=str(sources_file), cohort_sha256=result["manifest_sha256"],
        metric_scope=str(metric_scope), metric_scope_sha256=metric_scope_sha,
        evaluator_sha256=sha(Path(__file__).with_name("eval_anchor10.py")), generation_job=result["job"],
        generation_receipt_sha256=sha(producer)))


def pixels(binding_path, output):
    binding = load_binding(binding_path)
    rows = []
    for item in binding["records"]:
        c = decode(item["videos"]["C"], binding["geometry"])
        d = decode(item["videos"]["dense"], binding["geometry"])
        rows.append(dict(source=item["source"], **pixel_distance(c, d)))
    save(output, dict(pairs=rows, job=os.environ["SLURM_JOB_ID"], binding_sha256=sha(binding_path),
                     generation_receipt_sha256=binding["generation_receipt_sha256"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "pixels"))
    cli.add_argument("--binding", type=Path, required=True)
    cli.add_argument("--generation", type=Path)
    cli.add_argument("--metric-scope", type=Path)
    cli.add_argument("--metric-scope-sha")
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args.generation.resolve(), args.metric_scope.resolve(), args.metric_scope_sha, args.binding)
    else:
        pixels(args.binding, args.out)
