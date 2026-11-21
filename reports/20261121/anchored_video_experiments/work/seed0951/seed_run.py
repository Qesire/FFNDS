"""Pair the frozen 0951 left-pan Dense and original fast512 runs at four new seeds."""
from pathlib import Path
from time import perf_counter
import torch
from generate import Experiment, digest, load, save


@torch.inference_mode()
def run():
    protocol = load("seed_protocol.json")
    exp = Experiment()
    assert digest(exp.output / "EFFECTIVE_CONFIG.json") == protocol["reference_config"]
    exp.rows = {r["record_id"]:r for r in protocol["rows"]}
    exp.state.update(seed=None, seeds=[43,44,45,46], order=list(exp.rows), seed42_reference=protocol["reference"],
                     pairing="Dense first for seeds 43 and 46. Original fast512 first for seeds 44 and 45.")
    assert [r["seed"] for r in exp.rows.values()] == [43,44,45,46]
    method = exp.build("fast512")
    initial = {protocol["reference"]["initial"]}
    for key, row in exp.rows.items():
        assert digest(row["image_path"]) == row["image_sha256"]
        arms = ("dense","candidate") if row["seed"] in (43,46) else ("candidate","dense")
        pair = {}
        for arm in arms:
            result = exp.render(key, arm, method if arm == "candidate" else None)
            if arm == "candidate":
                assert result["counters"]["new_packed_sites"] == (80 if row["seed"] == 43 else 0)
            vid = result["video"]
            assert (vid["width"],vid["height"],vid["frames"],vid["fps"]) == (832,464,81,16)
            result.update(seed=row["seed"],prompt_sha256=row["prompt_sha256"], image_sha256=row["image_sha256"],
                          algorithm="dense" if arm == "dense" else "old_fast512")
            save(exp.output/arm/(key+".json"), result)
            pair[arm]=result
        a,d = pair["candidate"],pair["dense"]
        for field in ("latents","flows"):
            assert a[field][0] == d[field][0] and a[field][1] != d[field][1]
        assert a["initial"] == d["initial"] and d["initial"] not in initial
        initial.add(d["initial"])
        exp.state["attempts"].append(dict(source=key,seed=row["seed"],initial=d["initial"],order=arms,
                                        sha256={g:pair[g]["video"]["mp4_sha256"] for g in arms}))
        exp.checkpoint()
    assert len(initial) == 5 and exp.state["nfe_this_job"] == 32
    exp.state.update(status="complete", seconds=perf_counter()-exp.start)
    save(exp.output/"RESULT.json",exp.state)


if __name__=="__main__":
    run()
