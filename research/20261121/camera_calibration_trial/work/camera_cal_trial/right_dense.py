"""Render two prospective opposing-camera Dense controls with shared seed 44."""
from hashlib import sha256
from time import perf_counter

import torch

from generate import Experiment, digest, load, save


@torch.inference_mode()
def main():
    cases = load("right_protocol.json")
    assert len(cases) == 2
    exp = Experiment()
    exp.rows = {r["trajectory_id"]: r for r in cases}
    exp.state.update(seed=44, order=list(exp.rows), pairing="Two Dense right controls; archived ordinary and pan-left controls share seed, geometry and image.", attempts=[])
    exp.checkpoint()
    for r in cases:
        stem = r["trajectory_id"]
        assert r["seed"] == 44 and digest(r["image_path"]) == r["image_sha256"]
        assert r["prompt"] == r["base_prompt"] + ", camera pans right"
        assert sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
        out = exp.render(stem, "dense")
        assert out["initial"] == r["initial"]
        assert len(set(r["prior_first_flows"]+[out["flows"][0]])) == 3
        assert (out["video"]["frames"],out["video"]["fps"],out["video"]["width"],out["video"]["height"]) == (81,16,832,464)
        out.update(trajectory_id=stem,source_id=r["source_id"],seed=44,condition="right",prompt=r["prompt"],
                   prompt_sha256=r["prompt_sha256"],image_sha256=r["image_sha256"],initial_matches_base_left=True,first_flow_distinct_from_base_left=True)
        save(exp.output/"dense"/(stem+".json"),out)
        exp.state["attempts"].append(dict(trajectory_id=stem,sha256=out["video"]["mp4_sha256"],wall=out["wall"]))
        exp.checkpoint()
    assert exp.state["nfe_this_job"] == 8
    exp.state.update(status="complete",seconds=perf_counter()-exp.start)
    save(exp.output/"RESULT.json",exp.state)


if __name__=="__main__":
    main()
