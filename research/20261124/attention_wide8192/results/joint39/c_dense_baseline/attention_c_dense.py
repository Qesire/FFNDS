"""Counterbalanced, two-repeat, six-source Dense versus frozen C comparison."""
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import sys

import torch

from attention_online_structured import Experiment, OnlineAttention, digest, equal_trace, load, save


ARMS = ("dense", "C")
GEOMETRY = (832, 464, 81, 16)


@torch.inference_mode()
def main(fit):
    config = load("DENSE_C_PROTOCOL.json")
    method = load("CONFIRM_METHOD.json")
    manifest = load("CONFIRM_SOURCES.json")
    assert digest("CONFIRM_METHOD.json") == config["method_sha256"] == manifest["method_recipe_sha256"]
    assert digest("CONFIRM_SOURCES.json") == config["source_manifest_sha256"]
    assert digest("attention_online_structured.py") == method["attention_source_sha256"]
    assert digest(fit / "FIT.sha256") == method["frozen_fit_manifest_sha256"]
    archive = Path(config["historical_dev"])
    previous = Path(config["historical_confirmation"])
    assert digest(archive) == method["quality_receipt_sha256"]
    assert digest(previous) == config["prior_confirmation_sha256"]
    historical = load(archive)
    registered = load(previous)
    assert registered["status"] == "complete" and registered["manifest_sha256"] == config["source_manifest_sha256"]
    people = manifest["sources"]
    assert [p["source_id"] for p in people] == config["source_ids"]
    assert len(people) == len({p["source_cluster_id"] for p in people}) == 6

    experiment = Experiment()
    for row in people:
        sid = row["source_id"]
        assert digest(row["image_path"]) == row["image_sha256"]
        assert sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
        assert sid not in experiment.rows
        experiment.rows[sid] = dict(row, record_id=sid, seed=config["seed"])
    attention = OnlineAttention(experiment, fit)
    assert attention.calibration == method["calibration"]
    wide = experiment.build("wide8192")
    result = experiment.output / "DENSE_C_RESULT.json"
    state = dict(status="running", job=experiment.job, gpu=experiment.state["gpu"],
                 plan_commit=config["plan_commit"], protocol_sha256=digest("DENSE_C_PROTOCOL.json"),
                 manifest_sha256=config["source_manifest_sha256"], method_sha256=config["method_sha256"],
                 cold_init_seconds=experiment.state["init_seconds"], setup=experiment.state["setup"],
                 warmup={}, pairs=[])

    def render(sid, arm, group):
        context = attention.bind("OUT-D") if arm == "C" else nullcontext(({}, [], []))
        with context as (counts, _, _):
            row = experiment.render(sid, group, wide if arm == "C" else None)
        video = row["video"]
        assert tuple(video[k] for k in ("width", "height", "frames", "fps")) == GEOMETRY
        if arm == "C":
            ffn = row["counters"]
            assert (ffn["variant"], ffn["tile_rows"], ffn["anchors"], ffn["hits"]) == ("wide8192", 8192, 80, 80)
            assert tuple(counts[k] for k in ("attention_sites", "anchors", "hits", "native_probe_queries")) == (160, 80, 80, 5120)
        else:
            assert "counters" not in row and not counts
        return dict(record=row, attention=dict(counts))

    dev = "vbi_0392"
    references = dict(dense=historical["control"][dev]["dense"], C=next(
        r["record"] for r in historical["renders"] if r["source"] == dev and r["arm"] == "C"))
    for arm in ARMS:
        warm = render(dev, arm, f"warm_{arm}")
        assert equal_trace(warm["record"], references[arm]), (arm, "archived DEV replay")
        state["warmup"][arm] = warm
        save(result, state)

    exact_prior = {r["source_id"]: r["arms"]["C"]["record"] for r in registered["pairs"]}
    originals = {}
    for rep in range(config["repeats"]):
        for index, source in enumerate(people):
            sid, cluster = source["source_id"], source["source_cluster_id"]
            order = ARMS if (index + rep) % 2 == 0 else ARMS[::-1]
            arms = {arm: render(sid, arm, f"paired_{rep}_{arm}") for arm in order}
            dense, c = (arms[arm]["record"] for arm in ARMS)
            assert dense["initial"] == c["initial"]
            repetition = {}
            for arm in ARMS:
                value = arms[arm]["record"]
                key = (sid, arm)
                repetition[arm] = None if rep == 0 else equal_trace(value, originals[key])
                if rep == 0:
                    originals[key] = value
            state["pairs"].append(dict(source_id=sid, cluster_id=cluster, stratum=source["stratum"],
                repeat=rep, order=order, arms=arms, repeat_trace_exact=repetition,
                historical_c_trace_exact=equal_trace(c, exact_prior[sid]),
                first_step_exact=all(dense[k][0] == c[k][0] for k in ("flows", "latents")),
                wall_ratio=c["wall"] / dense["wall"], gpu_ratio=sum(c["step_ms"]) / sum(dense["step_ms"])))
            save(result, state)
    state["status"] = "complete"
    save(result, state)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
