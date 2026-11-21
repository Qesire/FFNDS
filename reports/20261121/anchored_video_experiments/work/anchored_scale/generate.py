"""Generate paired native videos, retaining only a backend that reproduces the reviewed operator."""
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
import gc
import hashlib
import json
import os
import re
import shutil
import sys

import torch
from icc_native_job import build_pipeline


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, data):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def equal_trace(a, b):
    keys = ("initial", "flows", "latents")
    return (all(a[k] == b[k] for k in keys) and
            a["video"]["decoded_rgb24_sha256"] == b["video"]["decoded_rgb24_sha256"])


class Experiment:
    def __init__(self):
        self.stage = Path.cwd()
        self.output = self.stage/"output"
        self.output.mkdir(exist_ok=True)
        self.job = os.environ["SLURM_JOB_ID"]
        self.base = load("base.json")
        ref = Path(self.base["v2_reference_dir"])
        assert os.environ["SLURM_JOB_PARTITION"] == "P107-A100"
        assert all(digest(ref/k) == v for k, v in self.base["reference_sha256"].items())
        sys.path.insert(0, str(ref))
        import runner_v2
        self.v2 = runner_v2
        self.cohort = load("cohort.json")
        self.reviewed = load("reviewed.json")
        self.rows = {r["record_id"]:r for r in self.cohort}
        assert len(self.rows) == len({r["source_cluster_id"] for r in self.cohort}) == 24
        assert set(self.reviewed) == {"vbi_0392", "vbi_0800", "vbi_1013"}
        assert set(self.reviewed) <= set(self.rows)
        assert len({r["image_sha256"] for r in self.cohort}) == 24
        for s, r in self.rows.items():
            assert re.fullmatch("vbi_[0-9]{4}", s) and r["seed"] == 42
            assert digest(r["image_path"]) == r["image_sha256"]
            assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
        self.order = sorted(self.rows, key=lambda s:hashlib.sha256(("FFNDS-anchor-scale24-pair-v1|"+s).encode()).digest())
        self.start = perf_counter()
        runtime = self.stage/("runtime_"+self.job)
        runtime.mkdir()
        self.pipe, config_dir, _ = build_pipeline(runtime, self.base, ref)
        effective = config_dir/"EFFECTIVE_CONFIG.json"
        assert digest(effective) == self.base["strict_effective_config"]["sha256"]
        shutil.copyfile(effective, self.output/"EFFECTIVE_CONFIG.json")
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
        self.owner, self.struct = WanTransformerInfer, MultiDistillModelStruct
        self.state = dict(status="running", job=self.job, order=self.order, protocol_sha=digest("MANIFEST.sha256"),
                          init_seconds=perf_counter()-self.start, nfe_this_job=0, attempts=[], setup=[],
                          geometry=[832,464,81,16], seed=42, gpu=torch.cuda.get_device_name(),
                          pairing="ordered by hash; alternate dense/candidate first. Synchronized generate window includes encode and symmetric on-GPU trace witnesses; all hashing after stop.")
        self.checkpoint()

    def checkpoint(self):
        save(self.output/("progress_"+self.job+".json"), self.state)

    def build(self, mode):
        from exact_port import make_port
        gc.collect()
        torch.cuda.empty_cache()
        start = perf_counter()
        method = make_port(self.base, mode)
        torch.cuda.synchronize()
        self.state["setup"].append(dict(mode=mode, seconds=perf_counter()-start))
        return method

    def render(self, source, group, method=None, reuse=False):
        dest = self.output/group
        dest.mkdir(exist_ok=True)
        rec, video = dest/(source+".json"), dest/(source+".mp4")
        if reuse and rec.exists() and video.exists():
            previous = load(rec)
            if previous["video"]["mp4_sha256"] == digest(video):
                return previous
        row = self.rows[source]
        sched = self.pipe.runner.scheduler
        get_index = self.struct.get_current_model_index
        step_post = sched.step_post
        initial, flows, latents, experts, events = [], [], [], [], []

        def dispatch(model):
            step = int(sched.step_index)
            assert step == len(experts) and step < 4
            if step == 0:
                initial.append(sched.latents.detach().clone())
            value = get_index(model)
            experts.append(int(model.cur_model_index))
            assert experts[-1] == step//2
            pair = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            pair[0].record()
            events.append(pair)
            return value

        def post():
            step = int(sched.step_index)
            assert step == len(latents)
            assert self.base["schedule"][step] == dict(timestep=float(sched.timesteps[step]),sigma=float(sched.sigmas[step]))
            flows.append(sched.noise_pred.detach().clone())
            value = step_post()
            latents.append(sched.latents.detach().clone())
            events[step][1].record()
            return value

        print("START", group, source, flush=True)
        try:
            self.struct.get_current_model_index, sched.step_post = dispatch, post
            self.v2.reset_scheduler_generator(sched, self.pipe.runner.model)
            with ExitStack() as stack:
                if method is not None:
                    stack.enter_context(method.trajectory())
                    stack.enter_context(method.bind(self.owner,self.base["native_transformer_sha256"]))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                self.pipe.generate(seed=42,image_path=row["image_path"],prompt=row["prompt"],
                                   negative_prompt="",save_result_path=str(video))
                torch.cuda.synchronize()
                wall = perf_counter()-start
                peak = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
        finally:
            self.struct.get_current_model_index, sched.step_post = get_index, step_post
        assert experts == [0,0,1,1] and len(latents) == len(flows) == 4
        hashes = lambda tensors:[self.v2.finite_tensor_sha(t,torch)["sha256"] for t in tensors]
        result = dict(job=self.job,source=source,group=group,initial=hashes(initial)[0],
                      flows=hashes(flows),latents=hashes(latents),video=self.v2.video_digest(video),
                      experts=experts,wall=wall,step_ms=[a.elapsed_time(b) for a,b in events],
                      peak_allocated=peak,peak_reserved=reserved)
        if method is not None:
            c = dict(method.counters)
            assert c["anchors"] == c["hits"] == 80 and c["sites"] == 160
            assert c["gate_native"]+c["gate_fused"] == 160
            result["counters"] = c
        save(rec, result)
        self.state["nfe_this_job"] += 4
        self.checkpoint()
        print("DONE", group, source, round(wall,3), flush=True)
        return result

    def run(self):
        self.render("vbi_0392","warm_dense")
        references = dict(self.reviewed)
        for mode in ("fast512", "resident512", "original512"):
            method = self.build(mode)
            trial = dict(mode=mode,calibration=[],formals=[],failures=[])
            self.state["attempts"].append(trial)
            for s, reference in self.reviewed.items():
                actual = self.render(s,"gate_"+mode,method)
                trial["calibration"].append(dict(source=s,match=equal_trace(actual,reference)))
                if not equal_trace(actual,reference):
                    trial["failures"].append(s)
                    break
            self.checkpoint()
            if trial["failures"]:
                print("GATE MISMATCH",mode,trial["failures"],flush=True)
                del method
                continue
            print("GATE PASSED",mode,flush=True)
            for i, s in enumerate(self.order):
                arms = ("dense","candidate") if i%2 == 0 else ("candidate","dense")
                pair = {a:self.render(s,a+"_"+mode,method if a=="candidate" else None) for a in arms}
                d, a = pair["dense"], pair["candidate"]
                assert d["initial"] == a["initial"] and d["latents"][0] == a["latents"][0] and d["flows"][0] == a["flows"][0]
                if s in references and not equal_trace(a, references[s]):
                    trial["failures"].append(s)
                    break
                trial["formals"].append(s)
                self.checkpoint()
            del method
            gc.collect()
            torch.cuda.empty_cache()
            if trial["failures"]:
                continue
            if len(references) != len(self.rows):
                original = self.build("original512")
                historical_check = self.render("vbi_0392","original_check",original,reuse=True)
                assert equal_trace(historical_check,self.reviewed["vbi_0392"]), "Original operator or native environment differs from reviewed 0392"
                for s in self.order:
                    if s not in references:
                        references[s] = self.render(s,"original_reference",original,reuse=True)
                del original
                gc.collect()
                torch.cuda.empty_cache()
            matches = {s:equal_trace(load(self.output/("candidate_"+mode)/(s+".json")),references[s]) for s in self.rows}
            trial["failures"] = [s for s in matches if not matches[s]]
            self.checkpoint()
            if all(matches.values()):
                self.state.update(status="complete",backend=mode,exact=matches,seconds=perf_counter()-self.start)
                self.checkpoint()
                save(self.output/"RESULT.json",self.state)
                return
        raise RuntimeError("No backend reproduced the reference trajectory")


if __name__=="__main__":
    with torch.inference_mode():
        Experiment().run()
