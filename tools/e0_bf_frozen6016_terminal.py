"""Four fixed-feature B-F students, two real terminal functions, fixed rank512.

Inspection/preflight import no numerical library. A separately frozen exact
scope and exited CPU byte receipt precede GPU execution. This is inherited-A0
output-space compression: it is not matched total offline training cost to a
fresh B-H/A0 student. No complete model, encoder or video is run here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import bf_frozen6016_output as core
import e0_bh_a0_matched_head as matched

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_BF_FROZEN6016_R512_TERMINAL_20261028"
MATCHED_PROTOCOL_SHA = "766e83105775bfccc2862a92f7b24b55b28e4a2ebcf93c92c2070ef739994714"
MATCHED_RUN_SHA = "31baab02b506aa965da6f07e91d72e47bd749f5425ae4faae92bb62f57c27716"
MATCHED_EXIT_SHA = "b4deb31cee2ca0568f1a510831af734a734281886e869663998e6b55e587c619"
EXPERTS = matched.EXPERTS
RESOURCES = {"GPU":1, "session_max_GPUs":2, "repository_GPU_policy":"DYNAMIC_MAX_ACTUALLY_CALLABLE",
             "CPU_threads":4, "compute_seconds":1800, "outer_seconds":2100,
             "GPU_reserved_bytes":15<<30, "output_bytes":1<<30,
             "free_disk_bytes":2<<30, "automatic_retry":False}
REQUIRED_SOURCES = matched.REQUIRED_SOURCES | {
    "tools/bf_frozen6016_output.py", "tools/e0_bf_frozen6016_terminal.py",
    "tests/test_bf_frozen6016_terminal.py", "tools/compact6016_adapter.py"}


def exact_limits():
    # All dispatched aten.mm/addmm/bmm/mv products are separately counted by
    # actual dtype, including autograd, decomposition checks and full-M heads.
    limit = {"head_pack_loads":2, "head_tensor_checks":4, "A0_energy_pack_loads":2,
        "native_quantizer_setup":1,
        "A0_tensor_checks":16, "label_pack_loads":96, "label_tensor_checks":672,
        "V1_Y_gate_pack_loads":96, "original_operand_checks":384,
        "A0_own_feature_cache_calls":96, "A0_GELU":96, "A0_Q2":96,
        "A0_original_Q2_SHA_checks":96, "A0_original_Y_SHA_checks":96,
        "conditioning_cross_source_checks":4,
        "geometry_BF16_gain_add":4, "geometry_head_build":4,
        "train_mean_fits":2, "train_covariance_cases":64,
        "step_conditioning_identity_checks":2, "two_step_head_QR":2,
        "head_projector_build":2, "complement_covariance_build":2,
        "output_covariance_eigh":4, "final_head_space_QR":2,
        "output_basis_orthogonality_checks":4, "head_span_preservation_checks":2,
        "coefficient_initialization":4, "adapter_random_draws":4,
        "adapter_random_H2D_copies":4, "adapter_random_H2D_bytes":4*32*6016*4,
        "coefficient_merges":1032, "factored_forward_calls":1408,
        "gated_Y_loss_calls":1024, "backward_calls":1024, "finite_gradient_checks":2048,
        "Adam_updates":1024, "training_batches":1024,
        "oracle_projection_cases":192, "oracle_precast_q_preservation_cases":192,
        "BF16_gate_multiply":576, "BF16_residual_add":576,
        "neutral_cases":96, "neutral_head_byte_checks":96,
        "padded_zero_states":672, "selected_row_writes":672,
        "fullM_LN":672, "BF16_head_gain_add":672, "BF16_head_modulation":672,
        "fullM_head_GEMM":672, "selected_u_unchanged_checks":672, "selected_head_hashes":672,
        "GPU_original_layout_restores":12,
        "FP64_q_variance_cases":672, "head_error_decomposition_cases":576,
        "gated_Y_diagnostic_cases":576,
        "BF16_matrix_products":5728, "FP32_matrix_products":3080,
        "FP64_matrix_products":1330,
        "student_export_writes":4, "student_export_reload_checks":4,
        "analysis_geometry_writes":2, "analysis_geometry_reload_checks":2,
        "head_prediction_writes":2, "head_prediction_reload_checks":2,
        "head_prediction_row_copies":576,
        "Q1_calls":0, "teacher_V2_numeric_reads":0, "online_teacher_Y_reads":0,
        "new_weight_quantizations":0, "intermediate_new_quantizations":0,
        "full_model_calls":0, "encoder_calls":0, "video_generations":0,
        "official_quality_calls":0}
    # Uploads happen exactly once and persist through the fixed four students.
    # 4 head tensors + 10 energy tensors + 96*(r,u,head,gate,rows,V1,Y)
    # + one head_scale/shift pair per step. W1,b1,smooth2 are referenced by
    # the frozen A0 exports; only required five tensors are uploaded per expert.
    n,d,h,w = 256,5120,64,6016
    limit["asset_H2D_tensor_copies"] = 4+10+96*7+8
    limit["asset_H2D_payload_bytes"] = (2*(2*d*h+2*h) + 2*(4*w*d+6*w+2*d)
        +96*(8*n*d+2*n*h+2*d+8*n)+8*2*d)
    return limit


def write(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def read_bound(root, binding):
    return json.loads(matched.labels.bound(root,binding).read_text())


def validate_plan(plan, frozen=False):
    core.require(plan.get("id")==ID and plan.get("parameters")==core.P
        and plan.get("resources")==RESOURCES and plan.get("limits")==exact_limits()
        and plan.get("basis_modes")==list(core.BASIS_MODES)
        and plan.get("source_split")=={"training":list(core.TRAIN),"diagnostic_heldout":list(core.HELD)}
        and plan.get("allowed_case_ids")==list(matched.CASE_IDS), "Fixed BF pilot/split/budget differs")
    core.require(plan.get("bank_manifest",{}).get("sha256")==matched.BANK_SHA
        and plan.get("parent_protocol",{}).get("sha256")==matched.labels.PARENT_SHA
        and plan.get("BH_protocol",{}).get("sha256")==matched.BH_PROTOCOL_SHA
        and plan.get("matched_head_protocol",{}).get("sha256")==MATCHED_PROTOCOL_SHA
        and plan.get("matched_head_RUN",{}).get("sha256")==MATCHED_RUN_SHA
        and plan.get("matched_head_exit",{}).get("sha256")==MATCHED_EXIT_SHA,
        "Exact completed A0/CAL24/BH parents required")
    if frozen:
        core.require(plan.get("assets_finalized") is True and plan.get("binding_status")=="BOUND_READY_FOR_REGISTRATION",
                     "BF draft assets NOT_BOUND")
        paths=[b["path"] for b in plan["repo_sources"]]
        core.require(len(paths)==len(set(paths)) and REQUIRED_SOURCES<=set(paths)
            and sum(p.endswith(".so") and any(k in p for k in ("hifxg_quant","lowbit_quant")) for p in paths)==2,
            "Required frozen source closure/two Q2 extensions missing")


def verify_assets(plan,hf):
    manifest,cells,bh_run=matched.verify_assets(plan,hf)
    previous=read_bound(hf,plan["matched_head_RUN"])
    exited=read_bound(hf,plan["matched_head_exit"])
    core.require(previous.get("id")==matched.ID
        and previous.get("status")=="COMPLETE_MATCHED_CAL24_HEAD_ERRORS_NATIVE_NEUTRAL96_A0_192_BH_SAVED96"
        and previous.get("protocol_sha256")==plan["matched_head_protocol"]["sha256"]
        and all(previous.get(k)==matched.exact_limits() for k in ("predebited","invoked","completed")),
        "Completed matched-head comparison differs")
    core.require(exited.get("id")==matched.ID and exited.get("process_exited") is True
        and type(exited.get("child_returncode")) is int and exited["child_returncode"]==0
        and exited.get("timeout") is False and exited.get("automatic_retry") is False
        and exited.get("run_sha256")==plan["matched_head_RUN"]["sha256"]
        and exited.get("protocol_sha256")==previous["protocol_sha256"], "Prior matched-head actual exit differs")
    core.require(not previous.get("assigned_GPU_resources",{}).get("violations"), "Prior matched-head monitor violation")
    matched.labels.bound(ROOT,plan["matched_head_protocol"])
    energy=[c for c in cells if c["initialization"]=="energy94"]
    core.require(len(energy)==2 and {c["expert"] for c in energy}==set(EXPERTS), "Two completed energy39 cells required")
    for cell in energy:
        result=read_bound(hf,cell["result"])
        rows=result["diagnostics"]["final_folded_BF16"]
        core.require(len(rows)==48 and {(r["index"],r["step"]) for r in rows}==
            {(i,s) for i in range(24) for s in (range(2) if cell["expert"]==EXPERTS[0] else range(2,4))}
            and all(r.get("student_Q2_shape")==[256,6016] and all(len(r.get(k,""))==64 for k in
                 ("student_post_Q2_sha256","output_tensor_sha256")) for r in rows), "Original own-Q2/Y SHA proof incomplete")
    return manifest,energy,previous


def preflight(protocol,hf):
    core.require("torch" not in sys.modules and "numpy" not in sys.modules
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="", "Fresh hidden-CUDA byte-only CPU process required")
    matched.precision_gate();plan=json.loads(Path(protocol).read_text());validate_plan(plan,True)
    for item in plan["repo_sources"]:matched.labels.bound(ROOT,item)
    versions={k:importlib.metadata.version(k) for k in plan["runtime_versions"]}
    core.require(versions==plan["runtime_versions"], "Frozen package versions differ")
    manifest,energy,_=verify_assets(plan,hf)
    return {"id":ID,"status":"PASS_BF_BYTES_NO_NUMPY_NO_TORCH_NO_GPU",
        "protocol_sha256":matched.labels.sha(protocol),"sources_sha256":matched.labels.state_sha(plan["repo_sources"]),
        "label_manifest_sha256":plan["label_manifest"]["sha256"],"bank_manifest_sha256":matched.BANK_SHA,
        "matched_head_RUN_sha256":plan["matched_head_RUN"]["sha256"],"cases":len(manifest["cases"]),"energy_exports":len(energy),
        "all_CUDA_devices_hidden":True,"NumPy_or_Torch_imported":"torch" in sys.modules or "numpy" in sys.modules,
        "model_calls":0,"native_calls":0,"training_updates":0,"decompositions":0}


def authorize(plan,digest,active):
    validate_plan(plan,True)
    user=active.get("user_continuation_authorization_20261027",{})
    policy=active.get("resource_scheduling_policy_20261028",{})
    core.require(user.get("status")=="ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True
        and policy.get("mode")=="DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None
        and active.get("current_session_resource_constraint_20261028",{}).get("max_simultaneously_assigned_idle_GPUs")==2,
        "Current direct session/resource authority differs")
    scopes=[s for s in active["experiment_authority"]["current_user_authorized_scopes"] if s.get("id")==ID]
    core.require(len(scopes)==1, "Unique exact BF registration required")
    scope=scopes[0]
    core.require(scope.get("protocol_sha256")==digest and scope.get("execution_status") in ("AUTHORIZED_NOT_STARTED","RUNNING_FIXED_SCOPE")
        and scope.get("resources")==RESOURCES and scope.get("limits")==exact_limits(), "Exact BF scope differs")
    return scope


def require_cpu_receipt(scope,plan,digest,hf):
    cpu=scope.get("cpu_preflight")
    core.require(isinstance(cpu,dict) and {"RUN","launch_exit"}<=set(cpu), "Independent actual CPU receipt NOT_BOUND")
    report,exited=read_bound(hf,cpu["RUN"]),read_bound(hf,cpu["launch_exit"])
    core.require(report.get("id")==ID and report.get("status")=="PASS_BF_BYTES_NO_NUMPY_NO_TORCH_NO_GPU"
        and report.get("protocol_sha256")==digest and report.get("sources_sha256")==matched.labels.state_sha(plan["repo_sources"])
        and report.get("label_manifest_sha256")==plan["label_manifest"]["sha256"] and report.get("bank_manifest_sha256")==matched.BANK_SHA
        and report.get("matched_head_RUN_sha256")==plan["matched_head_RUN"]["sha256"] and report.get("cases")==96
        and report.get("energy_exports")==2 and report.get("all_CUDA_devices_hidden") is True
        and report.get("NumPy_or_Torch_imported") is False
        and all(report.get(k)==0 for k in ("model_calls","native_calls","training_updates","decompositions")),
        "Prior BF CPU binding proof differs")
    core.require(exited.get("id")==ID and type(exited.get("child_returncode")) is int
        and exited["child_returncode"]==0 and exited.get("process_exited") is True
        and exited.get("timeout") is False and exited.get("automatic_retry") is False
        and exited.get("protocol_sha256")==digest and exited.get("run_sha256")==cpu["RUN"]["sha256"],
        "Prior BF CPU actual exit differs")
    return cpu


class Ledger(matched.Ledger):
    def __init__(self,report,persist=lambda:None,journal=lambda _:None,limits=None):
        super().__init__(report,exact_limits() if limits is None else limits,persist,journal)


def matrix_observer(torch,work):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            # The original DefaultMM head uses addmm(..., out=...). Counting
            # only the default overload would miss all672 original heads.
            name=func._schema.name
            if name in ("aten::mm","aten::addmm","aten::bmm","aten::mv"):
                value=args[1] if name=="aten::addmm" else args[0]
                name={torch.bfloat16:"BF16_matrix_products",torch.float32:"FP32_matrix_products",torch.float64:"FP64_matrix_products"}.get(value.dtype)
                core.require(name is not None,"Unregistered matrix-product dtype")
                with work(name):return func(*args,**(kwargs or {}))
            return func(*args,**(kwargs or {}))
    return Observer()


def upload(torch,value,work):
    core.require(value.device.type=="cpu","Original CPU asset required before H2D")
    with work("asset_H2D_tensor_copies"),work("asset_H2D_payload_bytes",value.numel()*value.element_size()):
        return value.to("cuda")


def save_checked(torch,path,pack,work,kind,metadata=None):
    from safetensors.torch import save_file,load_file
    # Every field owns its storage; no shared safetensors aliases.
    with work(kind+"_writes"):
        cpu={k:v.detach().cpu().contiguous().clone() for k,v in pack.items()}
        core.require(not path.exists(),"Immutable export already exists")
        save_file(cpu,str(path),metadata=metadata)
    with work(kind+"_reload_checks"):
        actual=load_file(str(path),device="cpu")
        core.require(set(actual)==set(cpu) and all(matched.same_bits(torch,actual[k],cpu[k]) for k in cpu),
                     "Actual exported/reloaded tensor bytes differ")
    return {"path":path.name,"bytes":path.stat().st_size,"sha256":matched.labels.sha(path),
        "tensors":{k:{"dtype":str(v.dtype),"shape":list(v.shape),"raw_sha256":matched.tensor_sha(torch,v)} for k,v in cpu.items()}}


def run(args):
    core.require("torch" not in sys.modules and "numpy" not in sys.modules,
                 "Bind registered BF execution before numerical imports")
    protocol,hf=args.protocol.resolve(),args.hf_root.resolve()
    plan=json.loads(protocol.read_text());digest=matched.labels.sha(protocol)
    guard=lambda:authorize(plan,digest,json.loads((ROOT/"research/ACTIVE_RESEARCH.json").read_text()))
    scope=guard();cpu=require_cpu_receipt(scope,plan,digest,hf);matched.precision_gate()
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    core.require(visible is not None and visible.strip() and len(visible.split(","))==1,"One explicitly assigned GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    try:readiness=preflight(protocol,hf)
    finally:os.environ["CUDA_VISIBLE_DEVICES"]=visible
    manifest,cells,previous=verify_assets(plan,hf)
    from worker_identity import require_worker
    require_worker()
    out=(hf/plan["output_HF_relative"]).resolve()
    core.require(out!=hf and out.is_relative_to(hf) and not out.exists(),"Fresh BF output required, no retry")
    out.mkdir(parents=True,exist_ok=False)
    report={"id":ID,"protocol_sha256":digest,"status":"RUNNING_FIXED_BF_PILOT",
        "prior_CPU_preflight":cpu,"repeated_byte_preflight":readiness,
        "operations":core.operations(),"neutral":[],"cases":[],"experts":[],"A0_cache_identity_checks":[],
        "comparison_identity":"Inherited completed A0 features; extra post-compression recovery cost is charged, not matched fresh-training cost",
        "native_identity":"M256 FFN/own dynamic Q2; original full-M geometry selected-row heads with other rows synthetic zero fill",
        "speed_claim":"NOT_MEASURED","video_claim":"NOT_MEASURED",
        "weight_activation_timing":"Static PTQ-derived A0 BF16 weights; new static BF16 factors are not requantized; deployed Q1/Q2 dynamic every call, Q2 after GELU; no middle quantizer",
        "parent_matched_source_summary":previous["source_summary"]}
    persist=lambda:write(out/"RUN.json",report)
    journal=(out/"OPERATIONS.jsonl").open("x",buffering=1)
    def event(value):journal.write(json.dumps(value,separators=(",",":"))+"\n");journal.flush()
    work=Ledger(report,journal=event);persist()
    started,monitor,torch=time.monotonic(),None,None
    def check():
        guard()
        if monitor is not None:monitor.check()
        if torch is not None and torch.cuda.is_initialized():
            core.require(torch.cuda.max_memory_reserved()<=RESOURCES["GPU_reserved_bytes"],"BF GPU reserve exceeded")
        core.require(shutil.disk_usage(out).free>=RESOURCES["free_disk_bytes"]
            and sum(p.stat().st_size for p in out.rglob("*") if p.is_file())<=RESOURCES["output_bytes"],"BF output/disk budget exceeded")
    def timeout(*_):raise TimeoutError("Fixed BF wall/signal; no automatic retry")
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,timeout)
    signal.alarm(RESOURCES["compute_seconds"])
    try:
        snapshot=out/"EXECUTED_SOURCES";snapshot.mkdir()
        for binding in plan["repo_sources"]:
            source=matched.labels.bound(ROOT,binding);dest=snapshot/binding["path"]
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        shutil.copyfile(protocol,snapshot/"protocol.json")
        import torch as torch_api
        torch=torch_api
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,report["reservation"]=reserve_before_assets(torch)
        core.require(torch.cuda.device_count()==1 and "A100" in torch.cuda.get_device_name(0),"One assigned A100 required")
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32=False
        flags={read_bound(hf,p["RUN"])["BF16_reduced_precision_reduction"] for p in manifest["parts"]}
        core.require(len(flags)==1,"Original head producers use different BF16 reduction flags")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flags.pop()
        report["BF16_reduced_precision_reduction"]=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        for name in ("HiF4_Sp","Wan2.2-I2V-A14B-W4A4"):sys.path.insert(0,str(ROOT/".runtime/code"/name))
        for name in ("HiFloat4","HiFloat4/hif4_gpu"):sys.path.insert(0,str(ROOT/".runtime/code/HiF4_Sp"/name))
        import quant.hif4_backend as backend
        from e0_b_final_block_capture import verify_native_quantizer
        with work("native_quantizer_setup"):
            report["native_quantizer"]=verify_native_quantizer(backend,plan)
            _,qtype,_=backend.get_hif4_api();core.require(qtype("hifx4").blk_size==64,"Actual original Q2 group64 required")
        from compact6016_adapter import compact_forward,error_metrics
        from e0_b_terminal_head_probe import restore_layout
        from unittest.mock import patch
        def quantize(value,force_fp32=True):
            with work("A0_Q2"):return backend.hif4_qdq_tensor(value,dim=-1,force_fp32=force_fp32)
        original_gelu=torch.nn.functional.gelu
        def gelu(*a,**kw):
            with work("A0_GELU"):return original_gelu(*a,**kw)
        with matrix_observer(torch,work):
            heads={}
            for entry in manifest["head_assets"]:
                with work("head_pack_loads"):
                    loaded=matched.load_tensors(matched.labels.bound(hf,entry["binding"]),("head_weight_runtime_KN","head_bias"))
                head={}
                for key,name,stride in (("head_weight_runtime_KN","weight",(1,5120)),("head_bias","bias",(1,))):
                    with work("head_tensor_checks"):
                        core.require(loaded[key].dtype==torch.bfloat16 and matched.tensor_sha(torch,loaded[key])==entry["tensor_sha256"][key],"Head tensor bytes differ")
                    value=upload(torch,loaded[key],work)
                    with work("GPU_original_layout_restores"):head[name]=restore_layout(torch,value,stride,"cuda")
                heads[entry["expert"]]=head
            a0,original_cpu,diagnostics={},{},{}
            for cell in cells:
                expert=cell["expert"]
                with work("A0_energy_pack_loads"):
                    loaded=matched.load_tensors(matched.labels.bound(hf,cell["export"]),matched.A0_KEYS)
                for key,value in loaded.items():
                    with work("A0_tensor_checks"):
                        spec=cell["export"]["tensors"][key]
                        core.require(str(value.dtype)==spec["dtype"] and list(value.shape)==spec["shape"]
                            and matched.tensor_sha(torch,value)==spec["sha256"],"Frozen A0 energy tensor bytes differ")
                core.require(loaded["groups"].tolist()==cell["groups"] and loaded["channels"].tolist()==
                    [g*64+j for g in cell["groups"] for j in range(64)],"Whole64 frozen support differs")
                original_cpu[expert]=loaded
                a0[expert]={key:upload(torch,loaded[key],work) for key in ("W1","W2","b1","b2","smooth2")}
                result=read_bound(hf,cell["result"])
                core.require(len(result["epoch_source_step_loss_means"])==8,"Inherited A0 recovery did not complete eight fixed epochs")
                diagnostics[expert]={(r["index"],r["step"]):r for r in result["diagnostics"]["final_folded_BF16"]}
            report["inherited_A0_offline_cost"]={"selected_energy_functions":2,
                "A0_trainable_parameters_per_parent":712704,"A0_Adam_updates_per_parent":256,
                "A0_inherited_recovery_updates":512,"BF_additional_updates":1024,
                "bank_manifest":plan["bank_manifest"],
                "selected_result_bindings":[cell["result"] for cell in cells],
                "selection_and_full_A0_offline_cost":"Already paid, inherited immutable support selection/calibration/recovery; parent protocol and part ledgers remain part of total lineage. No zero-cost or matched-total-offline-cost claim; parent cost not divided between the two child bases."}
            cases,conditioning,meta_by_step={},{},{}
            with torch.no_grad(),patch.object(torch.nn.functional,"gelu",gelu):
                for item in manifest["cases"]:
                    check();case=item["case"];expert=case["expert"];step=case["step"]
                    report["active_case"]={"case_id":case["case_id"],"stage":"OWN_A0_FEATURE_CACHE"}
                    with work("label_pack_loads"):
                        pack=matched.load_tensors(matched.labels.bound(hf,item["label"]),matched.LABEL_KEYS)
                    matched.validate_label_tensors(torch,pack,item,work)
                    binding={"path":case["operand_path"],"bytes":case["operand_bytes"],"sha256":case["operand_sha256"]}
                    with work("V1_Y_gate_pack_loads"):
                        operands=matched.load_tensors(matched.labels.bound(hf,binding),("V1","Y","gate","b2"))
                    for key,shape in (("V1",[256,5120]),("Y",[256,5120]),("gate",[1,5120]),("b2",[5120])):
                        with work("original_operand_checks"):
                            value=operands[key]
                            core.require(value.dtype==torch.bfloat16 and list(value.shape)==shape
                                and bool(torch.isfinite(value).all()),"Saved original operand shape/finite differs")
                            if key=="gate":core.require(matched.same_bits(torch,value,pack["gate"]),"Original/captured gate bytes differ")
                            if key=="b2":core.require(matched.same_bits(torch,value,original_cpu[expert]["b2"]),"Original/A0 b2 differs")
                    current={**case,"r":upload(torch,pack["r_rows"],work),"u":upload(torch,pack["u_rows"],work),
                        "head":upload(torch,pack["head_rows"],work),"gate":upload(torch,pack["gate"],work),
                        "rows_tensor":upload(torch,pack["rows"],work),"V1":upload(torch,operands["V1"],work),
                        "Y":upload(torch,operands["Y"],work),"item":item}
                    meta_by_step.setdefault(step,[]).append(item)
                    if step not in conditioning:
                        values=[]
                        for key in ("head_scale","head_shift"):
                            value=upload(torch,pack[key],work)
                            with work("GPU_original_layout_restores"):
                                values.append(restore_layout(torch,value,item["evidence"][key+"_stride"],"cuda"))
                        conditioning[step]=dict(zip(("scale","shift"),values))
                    with work("A0_own_feature_cache_calls"):
                        output,trace=compact_forward(current["V1"],a0[expert]["W1"],a0[expert]["b1"],a0[expert]["W2"],a0[expert]["b2"],a0[expert]["smooth2"],quantize,return_trace=True)
                    expected=diagnostics[expert][case["source_index"],step]
                    identity={"case_id":case["case_id"],"expert":expert,
                        "expected_post_Q2_sha256":expected["student_post_Q2_sha256"],
                        "expected_Y_sha256":expected["output_tensor_sha256"],
                        "original_A0_BF16_reduction_flag":"NOT_RECORDED_IN_ORIGINAL_PRODUCER"}
                    report["A0_cache_identity_checks"].append(identity)
                    with work("A0_original_Q2_SHA_checks"):
                        identity["actual_post_Q2_sha256"]=matched.tensor_sha(torch,trace["post_Q2"])
                        core.require(identity["actual_post_Q2_sha256"]==expected["student_post_Q2_sha256"],"Own frozen6016 Q2 differs from original final diagnostic")
                    with work("A0_original_Y_SHA_checks"):
                        identity["actual_Y_sha256"]=matched.tensor_sha(torch,output)
                        core.require(identity["actual_Y_sha256"]==expected["output_tensor_sha256"],"Frozen A0 Y differs from original final diagnostic")
                    current["feature"]=trace["post_Q2"].detach().contiguous()
                    core.require(not current["feature"].requires_grad and not torch.is_inference(current["feature"]),"Training cache must be an ordinary detached tensor, not inference-mode storage")
                    del current["V1"]
                    cases[case["case_id"]]=current
                    persist()
                for step,items in meta_by_step.items():
                    with work("conditioning_cross_source_checks"):
                        core.require(len(items)==24 and all(len({x["tensor_sha256"][key] for x in items})==1
                            for key in ("gate","head_scale","head_shift")),"Actual captured conditioning changes across sources")
                K={step:core.centered_head(torch,heads[EXPERTS[step//2]]["weight"],value["scale"],work) for step,value in conditioning.items()}
                # Complete fresh native neutrals for every original saved case.
                for case in cases.values():
                    check();report["active_case"]={"case_id":case["case_id"],"stage":"ORIGINAL_U_NEUTRAL"}
                    chosen,head_sha=matched.fullM_readout(torch,case["u"],case["rows_tensor"],heads[case["expert"]],conditioning[case["step"]]["scale"],conditioning[case["step"]]["shift"],work)
                    with work("neutral_head_byte_checks"):
                        core.require(matched.same_bits(torch,chosen,case["head"]),"Original fullM geometry neutral failed")
                    with work("neutral_cases"):
                        report["neutral"].append({"case_id":case["case_id"],"head_sha256":head_sha,"bitwise_equal":True})
                    case["statistics"]=core.q_and_variance(torch,case["u"],K[case["step"]],work)
                    persist()
                matched.neutral_gate(report["neutral"])
            for expert in EXPERTS:
                check();expert_cases=[c for c in cases.values() if c["expert"]==expert]
                steps=sorted({c["step"] for c in expert_cases});directory=out/expert;directory.mkdir()
                record={"expert":expert,"students":[],"basis_evidence":None};report["experts"].append(record)
                predictions={}
                with torch.no_grad():
                    mu,bases,evidence=core.fit_output_spaces(torch,expert_cases,K,core.P,work)
                    record["basis_evidence"]=evidence
                    geometry={"mu_FP64":mu,**{mode+".Phi_FP64":phi for mode,phi in bases.items()},
                        **{f"step{s}.K_FP64":K[s] for s in steps}}
                    record["geometry_export"]=save_checked(torch,directory/"ANALYSIS_GEOMETRY.safetensors",geometry,work,"analysis_geometry")
                def evaluate(y,case,mode,phase,ideal_q=None):
                    report["active_case"]={"case_id":case["case_id"],"basis":mode,"phase":phase}
                    with torch.no_grad():
                        u=core.native_write(torch,case["r"],y,case["gate"],work)
                        prediction,head_sha=matched.fullM_readout(torch,u,case["rows_tensor"],heads[expert],conditioning[case["step"]]["scale"],conditioning[case["step"]]["shift"],work)
                        stats=core.q_and_variance(torch,u,K[case["step"]],work)
                        metrics=core.head_error_decomposition(torch,prediction,case["head"],stats,case["statistics"],work)
                        with work("gated_Y_diagnostic_cases"):
                            metrics.update(error_metrics(y,case["Y"],case["gate"]))
                    row={"case_id":case["case_id"],"source_index":case["source_index"],"role":case["role"],"step":case["step"],"expert":expert,
                        "basis":mode,"phase":phase,"metrics":metrics,"head_sha256":head_sha,
                        "oracle_precast_q_norm_bound_scaled_SSE":ideal_q}
                    report["cases"].append(row)
                    with work("head_prediction_row_copies"):
                        predictions[mode+"."+phase+"."+case["case_id"]+".head_BF16"]=prediction.detach().cpu().contiguous()
                    check();persist()
                for mode in core.BASIS_MODES:
                    with torch.no_grad():
                        for case in expert_cases:
                            projected,ideal_q=core.oracle_projection(torch,case["Y"],bases[mode],mu,case["gate"],K[case["step"]],work)
                            if mode==core.BASIS_MODES[1]:core.require(ideal_q is not None and ideal_q<=core.P["geometry_rtol"]**2,"Continuous head-preserving oracle lost q before BF16 casts")
                            evaluate(projected.bfloat16(),case,mode,"oracle",ideal_q)
                    model=core.CoefficientAdapter(torch,bases[mode],mu,a0[expert],core.P,work)
                    student={"basis":mode,"trainable_parameters":core.count_parameters(),"epochs":[]};record["students"].append(student)
                    with torch.no_grad():
                        initial=model.merged()
                        for case in expert_cases:evaluate(model.forward(case["feature"],initial),case,mode,"initial")
                    optimizer=torch.optim.Adam(model.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                    train=core.training_cases(expert_cases)
                    for epoch in range(8):
                        losses=[]
                        for case in train:
                            check();report["active_case"]={"case_id":case["case_id"],"basis":mode,"phase":"TRAIN","epoch":epoch}
                            with work("training_batches"):
                                optimizer.zero_grad(set_to_none=True)
                                weights=model.merged();prediction=model.forward(case["feature"],weights)
                                loss=core.gated_loss(torch,prediction,case["Y"],case["gate"],work)
                                core.require(bool(torch.isfinite(loss)),"Nonfinite BF training loss")
                                with work("backward_calls"):loss.backward()
                                for parameter in model.parameters():
                                    with work("finite_gradient_checks"):
                                        core.require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),"Missing/nonfinite BF adapter gradient")
                                with work("Adam_updates"):optimizer.step()
                                losses.append(float(loss.detach()))
                        student["epochs"].append({"epoch":epoch,"source_step_equal_mean_gated_loss":sum(losses)/len(losses)})
                        persist()
                    with torch.no_grad():
                        final=model.merged()
                        for case in expert_cases:evaluate(model.forward(case["feature"],final),case,mode,"final")
                        original=original_cpu[expert]
                        exported={"W1":original["W1"],"b1":original["b1"],"smooth1":original["smooth1"],"smooth2":original["smooth2"],
                            "groups":original["groups"],"channels":original["channels"],"original_b2":original["b2"],
                            "Bc":final,"Phi":model.phi,"bc":model.bias,"mu":model.mu,
                            "training_lora_A":model.weights["A"],"training_lora_B":model.weights["B"]}
                        student["export"]=save_checked(torch,directory/(mode+".BF16_FACTORED.safetensors"),exported,work,"student_export",
                            {"id":ID,"expert":expert,"basis":mode,"format":"three BF16 GEMMs; own dynamic Q2 after GELU; no middle quantizer; no online LoRA", "initialization":"inherits completed A0 energy6016, extra Bc recovery"})
                    del model,optimizer,initial,final
                record["head_prediction_export"]=save_checked(torch,directory/"HEAD_PREDICTIONS.safetensors",predictions,work,"head_prediction")
                record["source_summary"]=core.summarize([r for r in report["cases"] if r["expert"]==expert])
                record["paired_parent_comparison"]=core.compare_source_heads(record["source_summary"],previous["source_summary"][expert])
                persist()
        work.finish();report["active_case"]=None;report["status"]="COMPLETE_BF_FROZEN6016_R512_TERMINAL_FOUR_STUDENTS"
        check()
    except BaseException as error:
        report.update(status="FAILED_FIXED_BF_NO_RETRY",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        original_error=sys.exc_info()[1];cleanup=None
        try:
            if monitor is not None:
                monitor.close();report["assigned_GPU_resources"]=monitor.report()
                core.require(not report["assigned_GPU_resources"].get("violations"),"Assigned GPU monitor violation")
        except BaseException as error:
            report.update(status="FAILED_BF_GPU_MONITOR",monitor_error={"type":type(error).__name__,"message":str(error)});cleanup=error
        report["wall_seconds"]=time.monotonic()-started
        if torch is not None and torch.cuda.is_initialized():report["peak_GPU_reserved_bytes"]=torch.cuda.max_memory_reserved()
        journal.close();report["operation_journal"]={"path":"OPERATIONS.jsonl","sha256":matched.labels.sha(out/"OPERATIONS.jsonl")}
        persist();signal.alarm(0)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if cleanup is not None and original_error is None:raise cleanup
        if cleanup is not None and original_error is not None:original_error.add_note("GPU monitor cleanup error retained in BF RUN")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("inspect","preflight","execute"));parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--hf-root",type=Path);parser.add_argument("--out",type=Path);args=parser.parse_args()
    if args.command=="inspect":
        plan=json.loads(args.protocol.read_text());validate_plan(plan)
        print(json.dumps({"id":ID,"binding_status":plan["binding_status"],"limits":exact_limits(),"operations":core.operations(),"model_calls_executed":0},indent=2))
    elif args.command=="preflight":
        core.require(args.hf_root is not None and args.out is not None and not args.out.exists(),"Fresh independent CPU output required")
        report=preflight(args.protocol,args.hf_root.resolve())
        with args.out.open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
    else:
        core.require(args.hf_root is not None,"HF root required");run(args)


if __name__=="__main__":main()
