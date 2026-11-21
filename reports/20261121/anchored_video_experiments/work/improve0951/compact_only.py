"""Isolate physical memory measurements from the full-port experiment."""
from pathlib import Path
import gc
import torch
from camera_run import CameraRun
from compact_run import run as compact
from generate import load,save,digest,equal_trace

with torch.inference_mode():
    exp=CameraRun()
    row=load("base_row.json")
    assert digest(row["image_path"])==row["image_sha256"]
    exp.rows[row["record_id"]]=row
    left="vbi_0951_pan_left"
    reference=load("references.json")
    warm=exp.render(left,"dense_warm")
    assert equal_trace(warm,reference[left]["dense"])
    for s in (left,row["record_id"]):
        folder=exp.output/"low_only"
        folder.mkdir(exist_ok=True)
        save(folder/(s+".json"),dict(**reference[s]["low_only"],group="archived_low_only"))
    gc.collect()
    torch.cuda.empty_cache()
    report=compact(exp)
    dense=load(exp.output/"compact_dense"/(left+".json"))
    assert equal_trace(dense,warm)
    assert exp.state["nfe_this_job"]==20
    exp.state.update(status="complete",gpu=torch.cuda.get_device_name(),
                     warmed_dense_equal_archive=True,compact_output=report)
    save(exp.output/"compact_job_result.json",exp.state)
