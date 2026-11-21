"""Closed-loop A/B rollout of exactly the four high-step readouts fitted on CAL3."""
from pathlib import Path
from time import perf_counter

import torch

from generate import Experiment, digest, load, save


LAYERS = (0, 13, 26, 39)
FIELDS = ("A0", "w2k")


class Switch:
    def __init__(self, method, fitted, report):
        self.method = method
        self.files = {}
        self.active = None
        self.loaded = set()
        for arm in ("A", "B"):
            for item in report["layers"]:
                name = item["site"]
                path = fitted / arm / (name + ".pt")
                assert digest(path) == item["fold"][arm]["sha256"]
                self.files[arm, int(name[4:])] = path
        assert set(self.files) == {(arm, layer) for arm in ("A","B") for layer in LAYERS}

        # First render is cold. The port allocates the packed weights lazily at
        # each native anchor, before the first step-1 hit uses its fold.
        prepare = method.prepare_static
        def cold_prepare(phase, pack, device):
            live = prepare(phase, pack, device)
            step, layer = divmod(method.counters["sites"], 40)
            if step == 0 and layer in LAYERS:
                self.install(layer, live)
            return live
        method.prepare_static = cold_prepare

    def install(self, layer, live):
        packed = torch.load(self.files[self.active, layer], weights_only=True, map_location="cpu", mmap=True)
        assert set(packed) == set(FIELDS)
        for key in FIELDS:
            assert packed[key].dtype == live[key].dtype == torch.bfloat16
            assert packed[key].shape == live[key].shape
            live[key].copy_(packed[key])
        self.loaded.add(layer)

    def select(self, arm):
        assert not self.method.busy
        torch.cuda.synchronize()
        start = perf_counter()
        if self.active != arm:
            self.active = arm
            self.loaded.clear()
            for layer in LAYERS:
                key = "high", layer
                if key in self.method.weights:
                    self.install(layer, self.method.weights[key])
        torch.cuda.synchronize()
        return perf_counter() - start


@torch.inference_mode()
def main():
    data = load("render_bindings.json")
    protocol = load("protocol.json")
    assert data["protocol_sha256"] == digest("protocol.json")
    fit_dir, cap_dir = (Path(data[key]) / "output" for key in ("fitted", "capture"))
    assert digest(fit_dir / "RESULT.json") == data["fit_result_sha256"]
    assert digest(cap_dir / "CAPTURE_INDEX.json") == data["capture_index_sha256"]
    fit, capture = load(fit_dir / "RESULT.json"), load(cap_dir / "CAPTURE_INDEX.json")
    assert fit["status"] == "PASS" and capture["status"] == "complete"
    assert fit["protocol_sha256"] == capture["protocol_sha256"] == data["protocol_sha256"]
    val = [r for r in protocol["trajectories"] if r["split"] == "validation"]
    assert len(val) == 4 and all(r["seed"] == 44 and not r["arms"] for r in val)
    native = {r["trajectory_id"]: r for r in capture["trajectories"]}
    exp = Experiment()
    exp.rows = {r["trajectory_id"]: r for r in val}
    for row in val:
        golden = native[row["trajectory_id"]]
        assert golden["prompt_sha256"] == row["prompt_sha256"]
        assert golden["image_sha256"] == row["image_sha256"] == digest(row["image_path"])
    # Keep the original resident FFN program for the other 76 sites and both
    # native anchor calls. There is one allocation for each hit weight.
    method = exp.build("fast512")
    switch = Switch(method, fit_dir, fit)
    records = []
    exp.state.update(seed=44, order=list(exp.rows), attempts=records, training=fit["job"],
                     graph="Identical fast512 port at 80 hit sites; only high layers 0,13,26,39 use the arm-specific folded A0 and w2k.",
                     timing="Synchronized full pipe.generate window includes encoder, decoder, export and traces; warm weight switches separately timed. First video also charges first-use packing.")
    baseline = exp.base["cal_manifest"]["sha256"]
    assert fit["frozen_manifest_sha256"] == baseline
    exp.checkpoint()
    for i, trajectory in enumerate(exp.rows):
        pair = {}
        for arm in (("A", "B") if i in (0, 3) else ("B", "A")):
            switch_time = switch.select(arm)
            r = exp.render(trajectory, arm, method)
            cold = len(records) == 0
            assert r["counters"]["new_packed_sites"] == (80 if cold else 0)
            assert switch.loaded == set(LAYERS)
            golden = native[trajectory]
            for field in ("flows", "latents"):
                assert r[field][0] == golden[field][0], (trajectory, arm, field)
            assert r["initial"] == golden["initial"]
            video = r["video"]
            assert (video["frames"], video["width"], video["height"], video["fps"]) == (81, 832, 464, 16)
            r.update(seed=44, capture_job=capture["job"], training_job=fit["job"], cold=cold,
                     initial_and_native_prefix_matched=True, switch_seconds=switch_time,
                     prompt_sha256=golden["prompt_sha256"], image_sha256=golden["image_sha256"],
                     fold_sites={s["site"]:s["fold"][arm]["sha256"] for s in fit["layers"]})
            save(exp.output / arm / (trajectory + ".json"), r)
            records.append(dict(trajectory=trajectory, arm=arm, cold=cold, sha256=video["mp4_sha256"], wall=r["wall"], switch_seconds=switch_time))
            pair[arm] = r
            exp.checkpoint()
        assert pair["A"]["initial"] == pair["B"]["initial"] and pair["A"]["flows"][0] == pair["B"]["flows"][0]
    assert exp.state["nfe_this_job"] == 32 and len(records) == 8
    exp.state.update(status="complete", seconds=perf_counter()-exp.start)
    save(exp.output / "RESULT.json", exp.state)


if __name__ == "__main__":
    main()
