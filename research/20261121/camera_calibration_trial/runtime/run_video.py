"""Generate the three original matched Wan D4 videos with the joint secant FFN."""
from functools import lru_cache
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import resource
import shutil
from time import monotonic

from icc_native_job import authorized_stage, build_pipeline, read_json, sha, write_json
from wan_comparison_job import bound_json, render


@contextmanager
def bind_probe_gate(owner,port,expected_sha):
    """Verify gate handoff in the native post on 64 rows without materializing a full reference."""
    import torch
    from wan_ffn_port import source_digest
    original_ffn, original_post=owner.infer_ffn,owner.post_process
    if source_digest(inspect.getfile(original_post))!=expected_sha:
        raise RuntimeError("Native transformer post implementation changed")
    pending={}
    checks=[]
    ids=torch.as_tensor(port.probe,device="cuda",dtype=torch.long)
    def ffn(model,phase,x,attn,shift,scale,gate=None):
        if port.current_gate is not None or pending:
            raise RuntimeError("Unexpected nested FFN")
        port.current_gate=gate
        try:
            y=original_ffn(model,phase,x,attn,shift,scale,gate)
        finally:
            port.current_gate=None
        if "operator" in port.seen[-1]:
            pending.update(model=model,y=y,gate=gate)
        return y
    def post(model,x,y,gate,pre_infer_out=None):
        if not pending:
            return original_post(model,x,y,gate,pre_infer_out)
        if pending["model"] is not model or pending["y"] is not y or pending["gate"] is not gate:
            raise RuntimeError("Native post did not receive current pre-gate FFN output")
        dtype=model.sensitive_layer_dtype if model.sensitive_layer_dtype!=model.infer_dtype else x.dtype
        if model.sensitive_layer_dtype!=model.infer_dtype:
            expected=x[ids].to(dtype)+y[ids].to(dtype)*gate.squeeze()
        else:
            expected=x[ids].add_(y[ids]*gate.squeeze())
        result=original_post(model,x,y,gate,pre_infer_out)
        if result.dtype!=expected.dtype:
            raise RuntimeError("Native post changed the expected output dtype")
        dtype=result.dtype
        int_dtype={torch.bfloat16:torch.int16,torch.float32:torch.int32}[dtype]
        checks.append((result[ids].view(int_dtype)==expected.view(int_dtype)).all())
        row=port.seen[-1]
        port.gate_witness.append(dict(step=row["step"],layer=row["layer"],probe_rows=64))
        pending.clear()
        return result
    owner.infer_ffn,owner.post_process=ffn,post
    try:
        yield
        if pending or not checks or not bool(torch.stack(checks).all()):
            raise RuntimeError("Native gate write failed sampled bitwise equality")
        for row in port.gate_witness:
            row["sampled_write_bitwise_equal"]=True
    finally:
        owner.infer_ffn,owner.post_process=original_ffn,original_post


def main():
    stage=Path.cwd().resolve()
    scope_sha=os.environ["FFNDS_SCOPE_SHA256"]
    scope,reference=authorized_stage(stage,scope_sha)
    import torch
    import runner_v2 as v2
    from native_joint import JointPort

    assert (scope["sources"],scope["max_d4"],scope["max_nfe"],scope["seed"])==(
        ["vbi_0392","vbi_1013","vbi_0800"],3,12,42)
    assert scope["geometry"]==[81,832,464,16] and scope["force_dense"] is False
    slots=set(scope["active_sites"])
    archive=bound_json(scope["cal_manifest"])
    assert archive["status"]=="PASS" and slots==set(archive["sites"])
    noise=bound_json(scope["dense_initial_noise"])
    historic=bound_json(scope["r32_bindings"])
    probe=scope["observer_ids"]
    assert probe==historic["fixed_probe_ids"] and len(set(probe))==len(probe)==64
    start=monotonic()
    peak={"load_seconds":0.0,"loads":0,"resident_file_bytes":0}

    @lru_cache(maxsize=len(slots))
    def site_loader(expert,layer):
        site=f"{expert}{layer:02d}"
        ref=archive["sites"][site]
        path=Path(ref["path"])
        t=monotonic()
        if path.stat().st_size!=ref["bytes"] or sha(path)!=ref["sha256"]:
            raise RuntimeError("Frozen parameter identity mismatch: "+site)
        params=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        peak["load_seconds"]+=monotonic()-t
        peak["loads"]+=1
        peak["resident_file_bytes"]+=ref["bytes"]
        return params

    active={(s[:-2],int(s[-2:])) for s in slots}
    assert all(e in ("low","high") and 0<=l<40 for e,l in active)
    limits=scope["resources"]
    def check_resources():
        if monotonic()-start>limits["soft_seconds"]:
            raise TimeoutError("Reviewed wall time reached")
        if shutil.disk_usage(stage).free<limits["in_flight_disk_floor_gib"]*2**30:
            raise OSError("Insufficient scratch space")
        if (torch.cuda.max_memory_reserved()>limits["gpu_memory_gib"]*2**30 or
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limits["host_memory_gib"]*2**30):
            raise MemoryError("Reviewed resource ceiling reached")

    with torch.inference_mode():
        all_conditions=read_json(reference/"v2_conditions.json")
        v2.verify_conditions(all_conditions)
        conditions={r["record_id"]:r for r in all_conditions}
        for source in scope["sources"]:
            row,identity=conditions[source],scope["source_identity"][source]
            assert sha(row["image_path"])==identity["image"]["sha256"]
            assert hashlib.sha256(row["prompt"].encode()).hexdigest()==identity["prompt_sha256"]
            assert noise[source]==identity["initial_noise"]
        pipe,output,_=build_pipeline(stage,scope,reference)
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("Native pipeline re-enabled TF32")
        expected=scope["strict_effective_config"]
        bound_json(expected)
        actual=sha(output/"EFFECTIVE_CONFIG.json")
        assert actual==expected["sha256"]
        for expert,layer in sorted(active):
            site_loader(expert,layer)
        check_resources()
        write_json(output/"RECIPE_PARITY.json",dict(exact_bytes_equal=True,actual=actual,expected=expected,nfe_at_check=0))
        write_json(output/"BINDINGS.json",dict(scope_sha256=scope_sha,cal_manifest=scope["cal_manifest"],
                   active_sites=sorted(slots),fixed64_r32_probe_ids=probe,sources=scope["sources"],
                   sources_identity=scope["source_identity"],schedule=scope["schedule"],seed=42,
                   selected_model=scope["arm"],residency=scope["residency"],
                   torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        class RecordedPort(JointPort):
            def finish(self):
                result=super().finish()
                # The pinned comparison writer serializes these two fields
                # verbatim. Keep all the port's other counters alongside them.
                counter=result["cache_io"][0]
                counter.update({key:val for key,val in result.items()
                                if key not in ("sites","cache_io","gate_witness")})
                counter.update(scope["residency"])
                return result

        def make_port(source):
            return RecordedPort(site_loader,probe,active_sites=active,**scope["residency"])
        def bind(owner,port):
            return bind_probe_gate(owner,port,scope["native_transformer_sha256"])
        render(pipe,scope,output,conditions,make_port,check_resources,extra_binding=bind,trace_flows=True)
        assert site_loader.cache_info().currsize==len(slots)
        peak["wall_with_setup_seconds"]=monotonic()-start
        write_json(output/"PARAMETER_RESIDENCY.json",peak)


if __name__=="__main__":
    main()
