"""Pair the released C schedule with a single change to high attention step 0."""
from contextlib import nullcontext
from pathlib import Path
from statistics import median
import sys

from safetensors.torch import load_file
import torch

from attention_online_structured import (
    Experiment, OnlineAttention, SOURCES, checked_controls, digest, equal_trace,
    load, save, save_trajectory,
)


PLAN = "090ae3aea20f1d33d19cdc91dd9ac910d8e8bdee"


def checked_trace(directory, label, expected):
    path = directory / (label + ".safetensors")
    assert path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"]
    return load_file(str(path), device="cpu")


def differences(source, archive, candidate, reference, current):
    old = archive / "output/candidate_C" / (source + "_trajectory")
    new = candidate / (source + "_trajectory")
    first = checked_trace(old, "initial", reference["trace_tensors"]["initial"])
    second = checked_trace(new, "initial", current["trace_tensors"]["initial"])
    assert torch.equal(first["noise"], second["noise"])
    rows = []
    for step in range(4):
        name = f"step{step}"
        a = checked_trace(old, name, reference["trace_tensors"][name])
        b = checked_trace(new, name, current["trace_tensors"][name])
        costs = {}
        for key in ("flow", "latent"):
            assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
            expected, observed = a[key].double(), b[key].double()
            assert torch.isfinite(observed).all()
            costs[key] = float((expected - observed).norm() / expected.norm().clamp_min(1e-12))
        rows.append(costs)
    return rows


@torch.inference_mode()
def main(fit, panels, config_path):
    config = load(config_path)
    assert config["plan_commit"] == PLAN
    assert digest(panels / "RESULTS.sha256") == config["panel_results_sha256"]
    controls = checked_controls(load(Path("base.json")), load(Path("cohort.json")), config)
    archive = Path(config["quality_stage"])
    old_receipt = archive / "output/ONLINE_RESULT.json"
    assert digest(old_receipt) == config["quality_sha256"]
    history = load(old_receipt)
    assert history["status"] == "complete" and history["job"] == "194188"
    references = {r["source"]: r for r in history["renders"] if r["arm"] == "C"}
    assert set(references) == set(SOURCES)
    for src, row in references.items():
        assert row["record"]["initial"] == controls[src]["wide8192"]["initial"]
        clip = archive / "output/candidate_C" / (src + ".mp4")
        assert digest(clip) == row["record"]["video"]["mp4_sha256"]

    exp = Experiment()
    wide = exp.build("wide8192")
    online = OnlineAttention(exp, fit)
    assert online.calibration == history["calibration"]
    state = dict(status="running", plan_commit=PLAN, job=exp.job, gpu=exp.state["gpu"],
                 implementation_sha256=digest(Path(__file__)),
                 attention_implementation_sha256=digest(Path("attention_online_structured.py")),
                 reference_quality_sha256=digest(old_receipt), reference_quality_job=history["job"],
                 arms={"C": ["native", "OUT-ABS", "native", "OUT-D"],
                       "C0": ["OUT-ABS", "OUT-ABS", "native", "OUT-D"]},
                 calibration=online.calibration, setup=exp.state["setup"], init_seconds=exp.state["init_seconds"],
                 sources=list(SOURCES), seed=42, quality={}, warmup=[], renders=[])
    result_path = exp.output / "SCHEDULE_RESULT.json"

    def render(source, arm, group, trace=False):
        is_new = arm == "C0"
        root = exp.output / group
        capture = save_trajectory(exp, root / (source + "_trajectory")) if trace else nullcontext({})
        with online.bind("OUT-D", high_step0=is_new, archive_routes=trace) as (counts, _, routes), capture as states:
            record = exp.render(source, group, wide)
        original = references[source]["record"]
        assert original["initial"] == record["initial"]
        assert record["counters"]["variant"] == "wide8192" and record["counters"]["tile_rows"] == 8192
        assert record["counters"]["anchors"] == record["counters"]["hits"] == 80
        if not is_new:
            assert equal_trace(record, original), (source, group, "C archive mismatch")
        row = dict(source=source, arm=arm, record=record, attention=dict(counts))
        if trace:
            assert len(routes) == 80
            ref_routes = references[source]["attention_anchor_identity"]
            assert routes[0] == ref_routes[0], (source, "layer 0 step 0 support mismatch")
            route_matches = {str(step): sum(a["support_sha256"] == b["support_sha256"]
                                          for a,b in zip(routes,ref_routes,strict=True) if a["step"]==step)
                             for step in (0,2)}
            row.update(trace_tensors=states, route_matches=route_matches, attention_anchor_identity=routes)
        elif is_new:
            assert equal_trace(record, state["quality"][source]["record"]), (source, group, "C0 quality mismatch")
        return row

    for source in SOURCES:
        row = render(source, "C0", "candidate_C0", trace=True)
        row["relative_l2_vs_C"] = differences(source, archive, exp.output / "candidate_C0", references[source], row)
        state["quality"][source] = row
        save(result_path, state)
    quality = dict(status="complete", plan_commit=PLAN, job=exp.job, sources=list(SOURCES),
                   quality=state["quality"], references={s:r["record"]["video"] for s,r in references.items()})
    save(exp.output / "SCHEDULE_QUALITY.json", quality)
    print("QUALITY_READY", flush=True)

    # Fresh C replay verifies the reference path and warms the same process.
    state["warmup"].append(render(SOURCES[0], "C", "warm_C"))
    save(result_path, state)
    paired = []
    for repeat in range(3):
        for i,source in enumerate(SOURCES):
            order = ("C", "C0") if (repeat+i)%2==0 else ("C0", "C")
            rows = {}
            for arm in order:
                row = render(source, arm, f"timing_{repeat}_{arm}")
                rows[arm] = row["record"]
                state["renders"].append(dict(repeat=repeat, **row))
                save(result_path, state)
            c,c0 = rows["C"],rows["C0"]
            paired.append(dict(source=source, repeat=repeat, order=list(order), wall_ratio=c0["wall"]/c["wall"],
                               gpu_ratio=sum(c0["step_ms"])/sum(c["step_ms"]),
                               step_delta_ms=[p-r for p,r in zip(c0["step_ms"],c["step_ms"],strict=True)]))
    state["pairs"] = paired
    state["paired_wall_ratio"] = median(row["wall_ratio"] for row in paired)
    state["paired_gpu_ratio"] = median(row["gpu_ratio"] for row in paired)
    state["status"] = "complete"
    save(result_path, state)


if __name__ == "__main__":
    main(*(Path(arg).resolve() for arg in sys.argv[1:4]))
