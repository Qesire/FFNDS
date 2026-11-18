import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description="Add optional tracing hooks to the exact strict runner")
parser.add_argument("reference", type=Path)
parser.add_argument("stage_copy", type=Path)
args = parser.parse_args()
raw = args.reference.read_bytes()
if hashlib.sha256(raw).hexdigest() != "4474cf184ed34a266e9401f432226553fe9ed5ac1fafc13f0fc7f3219e3f4f42":
    raise RuntimeError("expected the exact completed strict Q1 runner")
draft = raw.decode("utf-8").replace("\r\n", "\n")
begin = draft.index("def render(")
tail = draft.index("\ndef main()", begin)
reference = draft[begin:tail]
def replace(old, new):
    global reference
    if reference.count(old) != 1:
        raise ValueError(old)
    reference = reference.replace(old, new)

replace("def render(pipe, scope, output, conditions, make_port, check_resources):",
        "def render(pipe, scope, output, conditions, make_port, check_resources, *, extra_binding=None, trace_flows=False):\n    from contextlib import nullcontext")
replace("            if forced:\n                flow =","            if forced or trace_flows:\n                flow =")
replace("            if forced:\n                flow_trace.append","            if forced or trace_flows:\n                flow_trace.append")
replace("            with bind_native_ffn(WanTransformerInfer, measured, exact_source_sha=scope['native_transformer_sha256']):",
        "            with bind_native_ffn(WanTransformerInfer, measured, exact_source_sha=scope['native_transformer_sha256']), (\n                nullcontext() if extra_binding is None else extra_binding(WanTransformerInfer, port)\n            ):")
replace("        if isinstance(result, dict):\n            receipt['cache_io'] = result['cache_io']",
        "        if isinstance(result, dict):\n            receipt['cache_io'] = result['cache_io']\n            if 'gate_witness' in result:\n                receipt['gate_witness'] = result['gate_witness']\n        if trace_flows:\n            receipt['steps'] = [{'step': step, **row} for step, row in enumerate(flow_trace)]\n            receipt['ffn_calls_by_step'] = [sum(row['step'] == step for row in sites) for step in range(4)]")
replace("        video = v2.video_digest(path)",
        "        video = v2.video_digest(path)\n        if trace_flows and (len(flow_trace) != 4 or tuple(video[k] for k in ('frames', 'width', 'height', 'fps')) != (81, 832, 464, 16)):\n            raise RuntimeError('incomplete flow, latent, or physical video evidence')")
with args.stage_copy.open("xb") as output:
    output.write((draft[:begin] + reference + draft[tail:]).encode("utf-8"))
