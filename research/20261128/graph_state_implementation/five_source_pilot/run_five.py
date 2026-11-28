"""Render five fixed sources in one warmed, uninstrumented optimization port."""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from ffnds_optim.config import OptimizationConfig
from ffnds_optim.runtime import OptimizedRuntime, digest


PLAN = "a366f68950c8d5afcd053d9c040061d230648eac"
SOURCES = ("vbi_0632", "vbi_0016", "vbi_1095", "vbi_0440", "vbi_0616")


def load(path):
    return json.loads(Path(path).read_bytes())


def run(args):
    args = vars(args)
    role = args.pop("role")
    paths = {k:Path(v).resolve() for k,v in args.items()}
    config = OptimizationConfig.load(paths["config"])
    spec = load(paths["manifest"])
    assert spec["plan_commit"] == PLAN and tuple(r["source_id"] for r in spec["sources"]) == SOURCES
    assert digest(paths["reference"]) == spec["generation_206224_sha256"]
    prior = {r["source_id"]:r["arms"]["centered"]["record"] for r in load(paths["reference"])["quality"]}
    start = time.perf_counter()
    model = OptimizedRuntime(paths["runtime_dir"], paths["fit_dir"], config, binding_path=paths["binding"])
    model.register_manifest(paths["manifest"])
    experiment = model.experiment
    torch.cuda.synchronize()
    capture = dict(plan_commit=PLAN,status="running",job=experiment.job,role=role,gpu=experiment.state["gpu"],
                   profile=config.to_dict(),profile_sha256=config.fingerprint(),semantics=config.semantics(),
                   cold_init_and_port_seconds=time.perf_counter()-start,sources=SOURCES,records={},
                   gpu_hardware=torch.cuda.get_device_properties(0).name,torch=torch.__version__)
    model.generate.save(paths["out"],capture)
    warm = model.render("vbi_0392","warm")
    capture["warmup"] = warm
    model.generate.save(paths["out"],capture)
    order = SOURCES if role=="candidate" else SOURCES[::-1]
    for source in order:
        item = model.render(source,"quality")
        record = item["record"]
        assert (record["video"]["width"], record["video"]["height"], record["video"]["frames"], record["video"]["fps"]) == (832,464,81,16)
        assert prior[source]["initial"] == record["initial"] and len(record["step_ms"])==4
        assert all(item["attention"].get(k)==v for k,v in (("attention_sites",160),("anchors",80),("hits",80),("native_probe_queries",5120)))
        assert all(record["counters"].get(k)==v for k,v in (("anchors",80),("hits",80),("tile_rows",8192)))
        # The archived timer has stopped, and saving/hashing the video has finished.
        # Reading the retained scheduler value here adds no work to the measured window.
        latent = experiment.pipe.runner.scheduler.latents
        final = experiment.v2.finite_tensor_sha(latent,torch)["sha256"]
        item["final_latent"] = None
        if final == record["latents"][-1]:
            target = experiment.output/"quality"/(source+".final.safetensors")
            save_file({"latent":latent.detach().cpu().contiguous()},str(target))
            item["final_latent"] = dict(file=target.name,sha256=digest(target),latent_trace_sha256=final)
        item["exact_206224_centered_trace"] = model.generate.equal_trace(record,prior[source])
        if config.name == "frozen" and not item["exact_206224_centered_trace"]:
            raise RuntimeError(f"Frozen five-source reference differs from 206224 on {source}")
        capture["records"][source] = item
        model.generate.save(paths["out"],capture)
        print("PILOT_SOURCE_COMPLETE",config.name,role,source,flush=True)
    capture["status"] = "complete"
    model.generate.save(paths["out"],capture)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config","runtime-dir","fit-dir","binding","manifest","reference","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--role",required=True,choices=("candidate","same_gpu_reference"))
    run(p.parse_args())
