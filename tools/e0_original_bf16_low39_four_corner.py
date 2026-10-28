"""Twelve raw-BF16 late-step flows: neutral, same-input and own-state corners.

This new scope reuses the frozen original producer's loaders and pipeline.
It never patches its globals or files. Draft/preflight are stdlib only.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import time

import e0_original_bf16_videos as native
import raw_bf16_output_projection as core

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_ORIGINAL_BF16_LOW39_R512_FOUR_CORNER_20261028"
PARENT = "config/e0_original_bf16_videos_20261028_v1.json"
PARENT_SHA = "2f5352e91a877e85c0c0e88d8c216f74f5124693161ad412b7226aff73c69122"
PARENT_RUN_SHA = "59e99d08d206aac4ffabb11453e4b6b27449d046f07d14027d5e65e9644064c9"
PARENT_EXIT_SHA = "c718ec457194d2e48deed47c9ccb18502ddb44a37355a6f4874867bd4c493b04"
RESOURCES = {"GPU": 1, "CPU_threads": 4, "GPU_reserved_max_bytes": 32 << 30,
             "host_available_min_bytes": 128 << 30, "wall_seconds": 1200,
             "outer_seconds": 1500, "new_HF_bytes": 1 << 29, "free_disk_min_bytes": 20 << 30,
             "session_max_GPUs": 2, "repository_GPU_policy": "DYNAMIC_MAX_ACTUALLY_CALLABLE", "automatic_retry": False}
SHAPE = native.VIDEO["latent_shape"]
M, D, H = 31668, 5120, 13824
STEPS = (2, 3)
OPERATIONS = (("DD2", 2), ("DD3", 3), ("SD2", 2), ("SD3", 3), ("DS3", 3), ("SS3", 3))
RUN_IDS = tuple(f"raw_low39_source{s:02d}_{op}" for s in (0, 12) for op, _ in OPERATIONS)
REQUIRED = {"tools/raw_bf16_output_projection.py", "tools/e0_original_bf16_low39_four_corner.py",
            "tests/test_original_bf16_low39_four_corner.py"}


def limits():
    result = {k: 0 for k in native.LIMITS}
    result.update(original_expert_loads=2, raw_Default_MM_loads=812, buffer_Default_MM_loads=40,
        T5_raw_Default_MM_loads=168, T5_buffer_Default_MM_loads=7,
        T5_Default_MM_calls=336, T5_checkpoint_block_buffer_loads=48,
        input_encoder_requests=2, T5_positive_prompt_encodes=2, VAE_encodes=2,
        scheduler_prepare=2, FP32_initial_noise_draws=2,
        scheduler_step_pre=12, scheduler_step_post=12, native_full_model_flows=12,
        native_blocks=480, native_FFN_calls=480, native_FFN_linears=954,
        FFN_CUDA_event_pairs=480, native_non_FFN_linears=3840,
        pre_Default_MM_calls=60, head_Default_MM_calls=12, patch_Conv3D_calls=12,
        pre_infer_calls=12, final_LayerNorm_calls=12, post_unpatchify_calls=12,
        original_checkpoint_block_buffer_loads=492, offload_buffer_resets=12)
    result.update(raw_noise_byte_controls=2, input_state_uploads=12,
        input_state_upload_bytes=12*math.prod(SHAPE)*2,
        row_index_uploads=12, row_index_upload_bytes=12*256*8, control_pack_loads=12,
        input_after_step_pre_byte_checks=12, neutral_flow_and_post_byte_checks=8,
        actual_head_conditioning_captures=12, conditioning_equal_checks=10,
        native_boundary_capture_cases=12, teacher_full_Y_references_retained=6,
        candidate_overridden_W2_dispatches=6, candidate_coefficient_GEMMs=6, candidate_output_GEMMs=6,
        raw_W2_H2D=1, raw_W2_H2D_bytes=H*D*2,
        projection_H2D=3, projection_H2D_bytes=H*512*2+D*512*2+D*2,
        FP64_raw_W2_cast=1, geometry_BF16_gain_add=2, geometry_head_build=2,
        head_gated_columns=1, head_span_QR=1, weight_covariance_build=1,
        head_complement_build=1, complement_weight_covariance=1, weight_covariance_eigh=1,
        final_basis_QR=1, basis_orthogonality_check=1, head_span_check=1,
        coefficient_projection=1, BF16_projection_export_cast=1,
        FP64_stat_matrix_products=20, FP64_q_variance_cases=12,
        native_head_decomposition_cases=8, fullM_gated_Y_diagnostic_cases=8,
        native_scheduler_rounding_cases=12, flow_four_corner_decompositions=2,
        same_input_prefix_byte_checks=6, post_error_identity_cases=2,
        projection_export_writes=1, projection_export_reload_checks=1, decompositions=3,
        flow_capture_pack_writes=12, source_conditioning_setups=2)
    return result


class Ledger:
    def __init__(self, report, journal=lambda _: None):
        self.report, self.journal = report, journal
        report.update(predebited={}, observed={}, forbidden_attempts=[])
    @contextmanager
    def work(self, key, *, rid="setup", costs=None):
        costs = {key: 1} if costs is None else costs
        if key in ("head_span_QR","weight_covariance_eigh","final_basis_QR"):
            costs={**costs,"decompositions":1}
        native.need(rid == "setup" or rid in RUN_IDS, "Unregistered finite-flow operation")
        native.need(all(k in limits() and type(n) is int and n > 0 and self.report["predebited"].get(k,0)+n <= limits()[k]
                        for k,n in costs.items()), "Finite-flow budget exceeded")
        for k,n in costs.items(): self.report["predebited"][k] = self.report["predebited"].get(k,0)+n
        self.journal({"kind":"reserved_before_call", "run_id":rid, "costs":costs})
        yield
        for k,n in costs.items(): self.report["observed"][k] = self.report["observed"].get(k,0)+n
        self.journal({"kind":"observed_return", "run_id":rid, "costs":costs})
    def call(self, rid, costs, fn, *args, **kwargs):
        with self.work("",rid=rid,costs=costs): return fn(*args, **kwargs)
    def forbidden(self, kind):
        def fail(*_a, **_kw):
            self.report["forbidden_attempts"].append(kind)
            self.journal({"kind":"forbidden_before_dispatch","operation":kind})
            raise RuntimeError("Outside fixed raw BF16 four-corner scope: "+kind)
        return fail
    def finish(self):
        native.need(not self.report["forbidden_attempts"] and all(self.report[p].get(k,0)==n
            for p in ("predebited","observed") for k,n in limits().items()), "Incomplete finite-flow budget")


def validate(plan, frozen=False):
    native.need(plan.get("id")==ID and plan.get("numeric_path")=="BF16_ORIGINAL"
        and plan.get("parameters")==core.P and plan.get("resources")==RESOURCES
        and plan.get("limits")==limits() and plan.get("run_ids")==list(RUN_IDS)
        and plan.get("parent_protocol",{}).get("sha256")==PARENT_SHA
        and plan.get("parent_RUN",{}).get("sha256")==PARENT_RUN_SHA
        and plan.get("parent_exit",{}).get("sha256")==PARENT_EXIT_SHA
        and plan.get("runtime_overlay")==native.recipe(), "Fixed raw four-corner identity differs")
    native.need([c["index"] for c in plan["cases"]]==[0,12]
        and len({c["image_sha256"] for c in plan["cases"]})==2, "Two exposed independent sources required")
    native.need(REQUIRED <= {x["path"] for x in plan["repo_sources"]}
        and len({x["path"] for x in plan["repo_sources"]})==len(plan["repo_sources"]), "Source closure incomplete")
    for c in plan["cases"]:
        native.need(set(c["controls"])=={"raw_noise_FP32","input2","input3","flow2","flow3","post2","post3"}, "Exact late-step controls required")
        for s in STEPS:
            rows=c["rows"][str(s)]
            native.need(len(rows)==len(set(rows))==256 and all(type(i) is int and 0<=i<M for i in rows), "Fixed original row mapping differs")
    path=Path(plan["output_HF_relative"])
    native.need(path.parts[:1]==("ffnds-derived",) and ".." not in path.parts, "Output escaped HF")
    if frozen: native.need(plan.get("assets_finalized") is True and plan.get("binding_status")=="BOUND_READY_FOR_REGISTRATION", "Four-corner draft NOT_BOUND")


def verify_parent(plan, hf, *, stat_check=True):
    old=native.read(native.bound(ROOT,plan["parent_protocol"]))
    run=native.read(native.bound(hf,plan["parent_RUN"])); exited=native.read(native.bound(hf,plan["parent_exit"]))
    native.validate(old,True)
    native.need(run.get("id")==old["id"] and run.get("protocol_sha256")==PARENT_SHA
        and run.get("status")=="COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING"
        and all(run["operation_ledger"][p].get(k,0)==n for p in ("predebited","observed") for k,n in native.LIMITS.items())
        and not run["operation_ledger"]["forbidden_attempts"], "Actual original BF16 parent incomplete")
    native.need(exited.get("id")==old["id"] and exited.get("protocol_sha256")==PARENT_SHA
        and exited.get("run_sha256")==PARENT_RUN_SHA and exited.get("process_exited") is True
        and type(exited.get("child_returncode")) is int and exited["child_returncode"]==0
        and exited.get("timeout") is False and exited.get("automatic_retry") is False, "Parent actual exited RUN binding differs")
    native.need(plan["raw_master_CPU_proof"]==run["CPU_preflight"]
        and all(plan[k]==old[k] for k in ("models","model_view_root","model_view_metadata","runtime_versions","local_transformers","private_worker_config_sha256")), "Raw master assets differ")
    if stat_check: native.require_CPU({"cpu_preflight":plan["raw_master_CPU_proof"]},old,PARENT_SHA,hf)
    for c in plan["cases"]:
        original=next(x for x in old["cases"] if x["index"]==c["index"])
        native.need(all(c[k]==original[k] for k in ("source_id","image_sha256","prompt_sha256","prompt","conditioning_id","image","source_group")), "Raw source conditioning differs")
        phase=next(x for x in run["phases"] if x["source_index"]==c["index"])
        wanted={"raw_noise_FP32":"raw_noise_FP32","input2":"latent_step1","input3":"latent_step2",
                "flow2":"flow_step2","flow3":"flow_step3","post2":"latent_step2","post3":"latent_step3"}
        native.need(all(c["controls"][key]==phase["states"][name] for key,name in wanted.items()), "Exact saved raw flow/post controls differ")
    return old,run


def control_bindings(plan):
    return {x["path"]:x for c in plan["cases"] for x in c["controls"].values()}


def check_state_file(hf, item):
    path=native.bound(hf,item)
    with path.open("rb") as stream:
        n=struct.unpack("<Q",stream.read(8))[0]; native.need(0<n<=1<<20,"Control header size differs")
        h=json.loads(stream.read(n)); names=set(h)-{"__metadata__"}
        native.need(names in ({"flow"},{"latents"}),"One raw state operand required")
        row=h[next(iter(names))]; dtype="F32" if item["dtype"]=="float32" else "BF16"
        size=math.prod(SHAPE)*(4 if dtype=="F32" else 2)
        native.need(row=={"dtype":dtype,"shape":SHAPE,"data_offsets":[0,size]},"Control state shape/dtype differs")
        payload=stream.read(); native.need(len(payload)==size and hashlib.sha256(payload).hexdigest()==item["raw_tensor_sha256"],"Control raw payload differs")
    return size


def cpu_identity(plan,digest):
    return {"id":ID,"protocol_sha256":digest,"sources_sha256":native.canonical(plan["repo_sources"]),
        "parent_RUN_sha256":PARENT_RUN_SHA,"raw_master_CPU_proof":plan["raw_master_CPU_proof"],
        "status":"PASS_REUSED_RAW_MASTER_STAT_AND_RAW_CONTROLS_NO_NUMERIC_IMPORTS",
        "control_files":len(control_bindings(plan)),"control_payload_bytes":sum(math.prod(SHAPE)*(4 if x["dtype"]=="float32" else 2) for x in control_bindings(plan).values()),
        "all_CUDA_devices_hidden":True,"Torch_or_NumPy_imported":False,"model_calls":0,"GPU_calls":0,"raw_master_payload_bytes_rehashed":0}


def preflight(args):
    native.need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and "torch" not in sys.modules and "numpy" not in sys.modules,"Fresh hidden-CUDA CPU worker required")
    plan=native.read(args.protocol); validate(plan,True); start=time.monotonic()
    native.need(len(os.sched_getaffinity(0))<=4,"CPU preflight affinity exceeds4")
    for b in plan["repo_sources"]: native.bound(ROOT,b)
    verify_parent(plan,args.hf_root); native.model_metadata(plan)
    native.verify_local_package(plan,{"repo":ROOT,"hf":args.hf_root},args.local_root)
    sizes=[check_state_file(args.hf_root,b) for b in control_bindings(plan).values()]
    for c in plan["cases"]: native.bound(args.hf_root,c["image"])
    result=cpu_identity(plan,native.sha(args.protocol))
    native.need(sum(sizes)==result["control_payload_bytes"] and "torch" not in sys.modules and "numpy" not in sys.modules,"CPU byte proof differs")
    result["wall_seconds"]=time.monotonic()-start
    native.need(result["wall_seconds"]<=300,"CPU proof wall300 exceeded")
    return result


def authorize(plan,digest):
    validate(plan,True); active=native.read(ROOT/"research/ACTIVE_RESEARCH.json")
    policy=active.get("resource_scheduling_policy_20261028",{})
    native.need(active.get("user_continuation_authorization_20261027",{}).get("new_scope_registration_allowed") is True
        and policy.get("mode")=="DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None
        and active.get("current_session_resource_constraint_20261028",{}).get("max_simultaneously_assigned_idle_GPUs")==2,"Current repository/session authority differs")
    scope=[x for x in active["experiment_authority"]["current_user_authorized_scopes"] if x.get("id")==ID]
    native.need(len(scope)==1 and scope[0].get("protocol_sha256")==digest
        and scope[0].get("execution_status") in ("AUTHORIZED_NOT_STARTED","RUNNING_FIXED_SCOPE")
        and scope[0].get("limits")==limits() and scope[0].get("resources")==RESOURCES
        and scope[0].get("run_ids")==list(RUN_IDS) and scope[0].get("max_gpu_cards")==1,"Unique fixed four-corner ACTIVE scope required")
    return scope[0]


def require_cpu(scope,plan,digest,hf):
    proof=scope.get("cpu_preflight"); native.need(isinstance(proof,dict) and {"RUN","launch_exit"}<=set(proof),"Actual independent CPU proof NOT_BOUND")
    run=native.read(native.bound(hf,proof["RUN"])); exit=native.read(native.bound(hf,proof["launch_exit"]))
    native.need(all(run.get(k)==v for k,v in cpu_identity(plan,digest).items()) and 0<=run.get("wall_seconds",-1)<=300,"CPU proof identity differs")
    native.need(exit.get("id")==ID and exit.get("protocol_sha256")==digest
        and exit.get("process_exited") is True and type(exit.get("child_returncode")) is int
        and exit["child_returncode"]==0 and exit.get("timeout") is False and exit.get("automatic_retry") is False
        and exit.get("run_sha256")==proof["RUN"]["sha256"],"CPU actual exit0/RUN SHA required")
    return proof


class EncoderTap(native.NativeTap):
    """Same original T5 offload and runner.init_run; no DiT/decoder is run."""
    def __init__(self,pipe,guard,case,controls):
        super().__init__(pipe,guard,"setup",None,None)
        self.case,self.controls=case,controls
    def install(self):
        torch,runner=self.guard.torch,self.runner
        for name,key in (("run_input_encoder","input_encoder_requests"),("run_vae_encoder","VAE_encodes")):
            self.counted(runner,name,key)
        for name in ("run_image_encoder","run_vae_decoder","run_vae_decoder_stream"):
            if hasattr(runner,name): self.stack.enter_context(native.patch_method(runner,name,self.guard.ledger.forbidden(name)))
        native.need(len(runner.text_encoders)==1,"One original T5 required")
        encoder=runner.text_encoders[0]; self.t5_model=t5=encoder.model
        native.need(encoder.cpu_offload and t5.cpu_offload and t5.num_layers==24
                    and len(t5.blocks_weights.offload_block_buffers)==1,"Original T5 offload differs")
        buffer=t5.blocks_weights.offload_block_buffers[0]
        for mm in native.walk(buffer):
            if type(mm).__name__=="MMWeight":
                native.need(id(mm) in self.guard.t5_loads and self.guard.t5_loads[id(mm)][0],"Unbound T5 MM")
                self.t5_mapping[id(mm)]=self.guard.t5_loads[id(mm)][2]
        native.need(set(self.t5_mapping.values())==set(native.T5_NAMES),"Seven original T5 modules required")
        old_text=runner.run_text_encoder
        def text(*a,**kw):
            native.need(not self.t5_lines and not self.t5_encoding,"Duplicate T5 prompt")
            self.t5_encoding=True
            try: return self.guard.ledger.call("setup",{"T5_positive_prompt_encodes":1},old_text,*a,**kw)
            finally: self.t5_encoding=False
        self.stack.enter_context(native.patch_method(runner,"run_text_encoder",text))
        old_load=buffer.load_state_dict
        def load(destination,block_index,adapter_block_index=None):
            native.need(self.t5_encoding and block_index==len(self.t5_blocks)==int(t5.block_idx)
                and len(self.t5_lines)==7*block_index,"T5 original offload order differs")
            self.t5_blocks.append(block_index)
            return self.guard.ledger.call("setup",{"T5_checkpoint_block_buffer_loads":1},old_load,destination,block_index,adapter_block_index)
        self.stack.enter_context(native.patch_method(buffer,"load_state_dict",load))
        scheduler=runner.scheduler
        self.counted(scheduler,"prepare","scheduler_prepare")
        original_noise=scheduler.prepare_latents
        def noise(seed,shape,dtype=None):
            native.need(seed==42 and list(shape)==SHAPE and dtype==torch.float32 and scheduler.generator is None,"Original raw setup noise branch differs")
            old_randn=torch.randn
            def randn(*a,**kw):
                native.need(self.raw_draws==0 and list(a)==SHAPE and kw.get("dtype")==torch.float32
                    and kw.get("generator") is scheduler.generator,"Unexpected setup draw")
                self.raw_draws+=1; value=old_randn(*a,**kw)
                def check():
                    native.need(torch.equal(value.detach().cpu().view(torch.uint8),self.controls["raw_noise_FP32"].view(torch.uint8)),"Setup raw noise differs from actual BF16 baseline")
                self.guard.ledger.call("setup",{"raw_noise_byte_controls":1},check)
                return value
            with native.patch_method(torch,"randn",randn):
                return self.guard.ledger.call("setup",{"FP32_initial_noise_draws":1},original_noise,seed,shape,dtype=dtype)
        self.stack.enter_context(native.patch_method(scheduler,"prepare_latents",noise))
    def finish(self):
        native.need(self.raw_draws==1 and self.t5_blocks==list(range(24))
            and self.t5_lines==[(b,n) for b in range(24) for n in native.T5_NAMES],"Original conditioning setup incomplete")


class FlowTap(native.NativeTap):
    def __init__(self,pipe,guard,rid,step,mode,rows,geometry,pack,Y_reference=None):
        super().__init__(pipe,guard,rid,None,None)
        self.actual_step,self.mode,self.rows=step,mode,rows
        self.geometry,self.pack,self.Y_reference=geometry,pack,Y_reference
        self.work=lambda key:self.guard.ledger.work(key,rid=self.rid)
        self.capture={}; self.Y_full=None; self.head=None; self.stats=None; self.diagnostics={}
    def step(self,expert=None):
        s=int(self.runner.scheduler.step_index)
        native.need(s==self.actual_step and s in STEPS and expert in (None,1),"Finite low expert/step differs")
        return s
    def mm_call(self,mm,value,meta,original):
        native.need(id(mm) in self.mapping,"MM outside registered low expert")
        role,expert,suffix=self.mapping[id(mm)]; step=self.step(expert)
        if role=="block":
            native.need(self.block is not None and self.block[0]==1,"Block linear outside active block")
            block=self.block[1]; target=(step,block,suffix)
            expected=(step,len(self.lines)//10,native.LINEAR_SUFFIXES[len(self.lines)%10])
            native.need(target==expected and target not in self.lines,"Wrong/duplicate low block linear order")
            self.lines.add(target)
            key="native_FFN_linears" if suffix.startswith("ffn.") else "native_non_FFN_linears"
            if suffix.startswith("ffn."): native.need(meta["input"][0]==M,"Original full M differs")
            if block==39 and suffix=="ffn.0": self.capture["V1"]=value.index_select(0,self.rows).clone()
            if block==39 and suffix=="ffn.2" and self.mode=="S":
                with self.guard.ledger.work("candidate_overridden_W2_dispatches",rid=self.rid):
                    result=core.projected_W2(self.guard.torch,value,self.pack,self.work)
            else: result=self.guard.ledger.call(self.rid,{key:1},original,mm,value)
            if block==39 and suffix=="ffn.2":
                self.capture["Y"]=result.index_select(0,self.rows).clone()
                if self.mode=="D":
                    self.guard.ledger.call(self.rid,{"teacher_full_Y_references_retained":1},setattr,self,"Y_full",result.detach())
                if self.Y_reference is not None:
                    self.diagnostics.update(core.gated_Y_terms(self.guard.torch,result,self.Y_reference,self.capture["gate"],self.work))
        else:
            target=(step,role,suffix)
            native.need(self.block is None and target not in self.other,"Wrong pre/head phase")
            self.other.add(target); key="pre_Default_MM_calls" if role=="pre" else "head_Default_MM_calls"
            result=self.guard.ledger.call(self.rid,{key:1},original,mm,value)
            if role=="head": self.head=result.detach()
        self.shapes.append({"step":step,"role":role,"module":suffix,**meta})
        return result
    def install(self):
        torch,runner=self.guard.torch,self.runner
        self.counted(runner.model,"infer","native_full_model_flows")
        model=runner.model.model[1]; weights,infer=model.transformer_weights,model.transformer_infer
        native.need(len(weights.blocks)==40 and not infer.has_post_adapter
            and infer.infer_dtype==infer.sensitive_layer_dtype==torch.bfloat16,"Original low forty-block branch differs")
        for name in native.PRE_NAMES: self.mapping[id(getattr(model.pre_weight,name))]=("pre",1,name)
        self.mapping[id(weights.head)]=("head",1,"head")
        for owner,name,key in ((model.pre_infer,"infer","pre_infer_calls"),(model.pre_weight.patch_embedding,"apply","patch_Conv3D_calls"),(model.post_infer,"infer","post_unpatchify_calls")):
            self.counted(owner,name,key)
        old_block,old_ffn,old_write=infer.infer_block,infer.infer_ffn,infer.post_process
        def block(*a,**kw):
            self.step(1); b=int(infer.block_idx)
            native.need(self.block is None and b==len(self.lines)//10,"Original forty-block order differs")
            self.block=(1,b)
            try:return self.guard.ledger.call(self.rid,{"native_blocks":1},old_block,*a,**kw)
            finally:self.block=None
        def ffn(phase,x,attn,shift,scale,gate=None):
            self.step(1)
            if int(infer.block_idx)==39:
                native.need(gate is not None and gate.numel()==D,"Original terminal gate differs")
                self.capture["gate"]=gate.reshape(-1).clone()
            def timed():
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                self.events.append((start,end));start.record()
                try:return old_ffn(phase,x,attn,shift,scale,gate)
                finally:end.record()
            return self.guard.ledger.call(self.rid,{"native_FFN_calls":1,"FFN_CUDA_event_pairs":1},timed)
        def writeback(x,y,gate,pre_out=None):
            if int(infer.block_idx)==39:self.capture["r"]=x.index_select(0,self.rows).clone()
            return old_write(x,y,gate,pre_out)
        self.stack.enter_context(native.patch_method(infer,"infer_block",block))
        self.stack.enter_context(native.patch_method(infer,"infer_ffn",ffn))
        self.stack.enter_context(native.patch_method(infer,"post_process",writeback))
        old_norm=weights.norm.apply; non_code=infer.infer_non_blocks.__func__.__code__
        def norm(value):
            caller=sys._getframe(1); e=caller.f_locals.get("e")
            native.need(caller.f_code is non_code and caller.f_locals.get("x") is value
                and isinstance(e,tuple) and len(e)==2 and all(x.numel()==D for x in e)
                and weights.norm.eps==core.P["eps"] and weights.norm._get_actual_weight() is None
                and weights.norm._get_actual_bias() is None,"Actual original head LN/modulation boundary differs")
            def capture_condition():
                self.capture["head_shift"],self.capture["head_scale"]=e[0].reshape(-1).clone(),e[1].reshape(-1).clone()
            self.guard.ledger.call(self.rid,{"actual_head_conditioning_captures":1},capture_condition)
            step=self.step(); values={k:self.capture[k] for k in ("gate","head_shift","head_scale")}
            if step not in self.geometry:
                native.need(self.mode=="D","Geometry must first come from neutral Dense")
                self.geometry[step]={**values,"K":core.centered_head(torch,weights.head.weight,values["head_scale"],self.work)}
            else:
                def same():native.need(all(torch.equal(v.view(torch.uint8),self.geometry[step][k].view(torch.uint8)) for k,v in values.items()),"Same-step raw conditioning changed")
                self.guard.ledger.call(self.rid,{"conditioning_equal_checks":1},same)
            self.capture["u"]=value.index_select(0,self.rows).clone()
            self.stats=core.q_variance(torch,value,self.geometry[step]["K"],self.work)
            return self.guard.ledger.call(self.rid,{"final_LayerNorm_calls":1},old_norm,value)
        self.stack.enter_context(native.patch_method(weights.norm,"apply",norm))
        manager=infer.offload_manager
        native.need(manager.offload_granularity=="block" and not manager.lazy_load and len(manager.cuda_buffers)==2,"Original double buffer offload differs")
        self.guard.ledger.call(self.rid,{"offload_buffer_resets":1},setattr,manager,"need_init_first_buffer",True)
        aux={b:native.original_rms_auxiliaries(owner,b) for b,owner in enumerate(weights.blocks)}
        for buffer in manager.cuda_buffers:
            for mm in native.walk(buffer):
                if type(mm).__name__=="MMWeight":self.mapping[id(mm)]=("block",1,".".join(mm.weight_name.split(".")[2:]).removesuffix(".weight"))
            original_load=buffer.load_state_dict
            def load(destination,block_index,adapter_block_index=None,_fn=original_load):
                self.step(1)
                def checked():
                    native.checkpoint_source_metadata(torch,destination,block_index,aux[block_index])
                    return _fn(destination,block_index,adapter_block_index)
                return self.guard.ledger.call(self.rid,{"original_checkpoint_block_buffer_loads":1},checked)
            self.stack.enter_context(native.patch_method(buffer,"load_state_dict",load))
    def finish(self):
        s=self.actual_step
        native.need(self.lines=={(s,b,k) for b in range(40) for k in native.LINEAR_SUFFIXES}
            and self.other=={(s,role,n) for role,names in (("pre",native.PRE_NAMES),("head",("head",))) for n in names}
            and set(self.capture)=={"V1","Y","r","u","gate","head_shift","head_scale"}
            and self.head is not None and self.stats is not None,"Incomplete actual forty-block finite flow")


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_controls(torch,plan,hf,ledger):
    from safetensors.torch import load_file
    cache={}
    for path,b in control_bindings(plan).items():
        def load():
            pack=load_file(str(native.bound(hf,b)),device="cpu")
            native.need(len(pack)==1,"One original saved control tensor required")
            value=next(iter(pack.values()))
            native.need(list(value.shape)==SHAPE and str(value.dtype).removeprefix("torch.")==b["dtype"]
                and tensor_sha(value)==b["raw_tensor_sha256"],"Loaded raw control tensor differs")
            return value
        cache[path]=ledger.call("setup",{"control_pack_loads":1},load)
    return {c["index"]:{k:cache[b["path"]] for k,b in c["controls"].items()} for c in plan["cases"]}


def prepare_source(pipe,guard,case,controls,hf):
    from lightx2v.utils.input_info import init_empty_input_info,update_input_info_from_dict
    from lightx2v.utils.utils import seed_all
    runner=pipe.runner
    info=init_empty_input_info("i2v",pipe.support_tasks)
    update_input_info_from_dict(info,{"seed":42,"prompt":case["prompt"],"negative_prompt":"",
        "image_path":str((hf/case["image"]["path"]).resolve()),"save_result_path":"UNUSED_NO_VIDEO"})
    seed_all(42);runner.input_info=info;runner.scheduler.generator=None
    with EncoderTap(pipe,guard,case,controls) as tap:
        def setup():
            runner.inputs=runner.run_input_encoder()
            runner.init_run()
        guard.ledger.call("setup",{"source_conditioning_setups":1},setup)
        tap.finish()
    native.need(runner.scheduler.sigmas.tolist()==[1.,.9375,.8333333134651184,.625]
        and runner.scheduler.denoising_step_list==[1000,750,500,250]
        and not runner.scheduler._uses_conditioned_latent_prefix()
        and not runner.scheduler.keep_latents_dtype_in_scheduler,"Exact original scheduler recipe differs")
    return info,runner.inputs


def single_flow(torch,pipe,guard,case,op,step,input_cpu,geometry,pack,controls,context,out,hf,
                report,persist,check,target=None,Y_target=None,prefix_reference=None):
    from safetensors.torch import save_file
    rid=f"raw_low39_source{case['index']:02d}_{op}"
    ledger=guard.ledger;work=lambda key:ledger.work(key,rid=rid)
    phase={"run_id":rid,"source_index":case["index"],"operation":op,"step":step,
        "mode":"S" if op.startswith("S") else "D","status":"STARTED_NO_RETRY",
        "prefix":"S_step2_from_Dense_step2" if op in ("DS3","SS3") else "HISTORICAL_RAW_DENSE",
        "V1_role":"RAW_PRE_W1_BF16_WITH_NO_Q1","hidden_role":"RAW_GELU_TANH_WITH_NO_Q2"}
    report["flows"].append(phase);persist()
    scheduler=pipe.runner.scheduler;pipe.runner.input_info,pipe.runner.inputs=context
    def upload():
        native.need(input_cpu.dtype==torch.bfloat16 and list(input_cpu.shape)==SHAPE,"Native BF16 flow input required")
        return input_cpu.to("cuda",non_blocking=False)
    before=ledger.call(rid,{"input_state_uploads":1,"input_state_upload_bytes":math.prod(SHAPE)*2},upload)
    scheduler.latents=before
    rows=ledger.call(rid,{"row_index_uploads":1,"row_index_upload_bytes":256*8},torch.tensor,case["rows"][str(step)],dtype=torch.int64,device="cuda")
    tap=FlowTap(pipe,guard,rid,step,phase["mode"],rows,geometry,pack,Y_target)
    started=time.perf_counter(); result=None
    try:
        ledger.call(rid,{"scheduler_step_pre":1},scheduler.step_pre,step_index=step)
        def input_check():
            native.need(scheduler.latents.dtype==torch.bfloat16 and list(scheduler.latents.shape)==SHAPE
                and tensor_sha(scheduler.latents)==tensor_sha(input_cpu),"step_pre changed imported native BF16 state")
            phase["input_after_step_pre"]={"shape":SHAPE,"dtype":"bfloat16","raw_tensor_sha256":tensor_sha(scheduler.latents)}
        ledger.call(rid,{"input_after_step_pre_byte_checks":1},input_check)
        with ledger.work("native_boundary_capture_cases",rid=rid),tap,torch.inference_mode():
            pipe.runner.model.infer(pipe.runner.inputs)
        flow=scheduler.noise_pred
        native.need(flow.dtype==torch.float32 and list(flow.shape)==SHAPE,"Actual full native FP32 flow differs")
        ledger.call(rid,{"scheduler_step_post":1},scheduler.step_post)
        post=scheduler.latents
        if op.startswith("DD"):
            for actual,key in ((flow,"flow"+str(step)),(post,"post"+str(step))):
                def same(actual=actual,key=key):
                    expected=controls[key]
                    native.need(actual.dtype==expected.dtype and torch.equal(actual.detach().cpu().view(torch.uint8),expected.view(torch.uint8)),"Dense neutral "+key+" raw bytes mismatch")
                ledger.call(rid,{"neutral_flow_and_post_byte_checks":1},same)
            phase["neutral_flow_and_post_bitwise_passed"]=True
        sigma=scheduler.sigmas[step].item(); sn=scheduler.sigmas[step+1].item() if step<3 else None
        rounding=core.scheduler_residual(torch,before,flow,post,sigma,sn,work)
        phase["scheduler_rounding"]={k:v for k,v in rounding.items() if not torch.is_tensor(v)}
        if target is not None:
            phase["head_diagnostics_vs_same_step_Dense_teacher_state"]=core.head_terms(torch,tap.head,target["head"],tap.stats,target["stats"],work)
        phase.update(tap.diagnostics)
        def save():
            path=out/(rid+".safetensors")
            payload={"flow":flow,"latents":post,"head":tap.head,"rows":rows,**tap.capture}
            cpu={k:v.detach().cpu().contiguous() for k,v in payload.items()}
            native.need(all(bool(torch.isfinite(v).all()) for v in cpu.values()),"Nonfinite actual flow/boundary capture")
            native.need(not path.exists(),"Capture overwrite forbidden")
            save_file(cpu,str(path))
            phase["capture"]={"root":"hf",**native.binding(path,hf)}
            phase["tensor_metadata"]={k:{"shape":list(v.shape),"dtype":str(v.dtype).removeprefix("torch."),
                "raw_tensor_sha256":tensor_sha(v)} for k,v in cpu.items()}
            return cpu["latents"]
        post_cpu=ledger.call(rid,{"flow_capture_pack_writes":1},save)
        torch.cuda.synchronize();tap.finish()
        if prefix_reference is not None:
            def same_prefix():
                native.need(all(torch.equal(tap.capture[k].view(torch.uint8),prefix_reference["capture"][k].view(torch.uint8))
                    for k in ("V1","r")),"Same-input D/S did not share actual native prefix/FFN input")
            ledger.call(rid,{"same_input_prefix_byte_checks":1},same_prefix)
        phase.update(status="COMPLETE_FULL40_FINITE_FLOW_NOT_VIDEO",wall_seconds=time.perf_counter()-started,
            original_MM_operand_metadata=tap.shapes,FFN_CUDA_seconds=sum(a.elapsed_time(b) for a,b in tap.events)/1000)
        result={"flow":flow,"post":post,"post_cpu":post_cpu,"head":tap.head,"stats":tap.stats,
                "Y":tap.Y_full,"capture":tap.capture,"rounding":rounding,"input":before}
        check();return result
    except BaseException as e:
        phase.update(status="FAILED_FINITE_FLOW_PRESERVED_NO_RETRY",error_type=type(e).__name__,error=str(e));raise
    finally:persist()


def export_projection(torch,pipe,geometry,out,hf,ledger):
    from safetensors.torch import load_file,save_file
    low=pipe.runner.model.model[1].transformer_weights.blocks[39].compute_phases[2].ffn_2
    A_cpu,bias=low.pin_weight,low.pin_bias
    native.need(A_cpu.dtype==bias.dtype==torch.bfloat16 and tuple(A_cpu.shape)==(H,D)
        and tuple(bias.shape)==(D,) and A_cpu.device.type==bias.device.type=="cpu","Original low W2 pinned CPU view differs")
    raw=ledger.call("setup",{"raw_W2_H2D":1,"raw_W2_H2D_bytes":H*D*2},A_cpu.to,"cuda",non_blocking=False)
    pack,evidence=core.fit_weight_basis(torch,raw,{s:geometry[s]["K"] for s in STEPS},
        {s:geometry[s]["gate"] for s in STEPS},ledger.work)
    evidence.update(raw_W2_runtime_KN_sha256=tensor_sha(A_cpu),raw_b2_sha256=tensor_sha(bias),
        source_identity="LOW_RAW_MASTER_CHECKPOINT_BLOCK39",bias="original raw b2 retained exactly; center=b2")
    expected={"B":pack["B"].cpu(),"Phi":pack["Phi"].cpu(),"b2":bias.contiguous()}
    path=out/"RAW_WEIGHT_HEAD128_PCA384_R512.safetensors"
    ledger.call("setup",{"projection_export_writes":1},save_file,expected,str(path))
    def reload():
        loaded=load_file(str(path),device="cpu")
        native.need(set(loaded)==set(expected) and all(torch.equal(loaded[k].view(torch.uint8),v.view(torch.uint8)) for k,v in expected.items()),"BF16 projection export reload differs")
        return loaded
    cpu=ledger.call("setup",{"projection_export_reload_checks":1},reload)
    del raw,pack
    deployed={}
    for k,value in cpu.items():
        deployed[k]=ledger.call("setup",{"projection_H2D":1,"projection_H2D_bytes":value.numel()*value.element_size()},value.to,"cuda",non_blocking=False)
    return deployed,{**evidence,"export":{"root":"hf",**native.binding(path,hf)},
        "tensors":{k:{"shape":list(v.shape),"dtype":"bfloat16","raw_tensor_sha256":tensor_sha(v)} for k,v in cpu.items()}}


def post_identity(torch,group,ledger):
    with ledger.work("post_error_identity_cases"):
        rho=(group[3]["rounding"]["arithmetic"]+group[3]["rounding"]["cast"]
             -group[0]["rounding"]["arithmetic"]-group[0]["rounding"]["cast"])
        ideal=(group[3]["input"].double()-group[0]["input"].double())-.625*(group[3]["flow"].double()-group[0]["flow"].double())
        actual=group[3]["post"].double()-group[0]["post"].double()
        total=float(actual.square().sum());residual=float((actual-ideal-rho).square().sum())
        native.need(residual<=1e-18*max(1.,total),"Native post error/rounding identity does not close")
        return {"post_error_squared":total,
            "input_error_squared":float((group[3]["input"].double()-group[0]["input"].double()).square().sum()),
            "native_rounding_difference_squared":float(rho.square().sum()),
            "native_rounding_signed_cross":float((2*ideal*rho).sum()),
            "post_identity_residual_squared":residual}


def run(args):
    start=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol)
    scope=authorize(plan,digest);cpu=require_cpu(scope,plan,digest,args.hf_root)
    native.need("torch" not in sys.modules and "numpy" not in sys.modules,"Fresh native worker required before numerical imports")
    verify_parent(plan,args.hf_root)
    for b in plan["repo_sources"]:native.bound(ROOT,b)
    native.model_metadata(plan)
    local,rows=native.verify_local_package(plan,{"repo":ROOT,"hf":args.hf_root},args.local_root)
    native.need({k:importlib.metadata.version(k) for k in plan["runtime_versions"]}==plan["runtime_versions"],"Native package versions differ")
    from worker_identity import require_worker,private_config_path
    require_worker();native.need(native.sha(private_config_path()) in plan["private_worker_config_sha256"],"Assigned worker identity differs")
    native.need(len(os.sched_getaffinity(0))<=4 and not any(os.environ.get(k)=="1" for k in
        ("NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")),"CPU affinity/TF32 environment differs")
    out=(args.hf_root/plan["output_HF_relative"]).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),"Fresh confined four-corner output required")
    out.mkdir(parents=True);report={"id":ID,"protocol_sha256":digest,"numeric_path":"BF16_ORIGINAL",
        "status":"STARTED_FIXED12_NO_AUTOMATIC_RETRY","CPU_preflight":cpu,"automatic_retry":False,
        "flows":[],"independent_image_clusters":2,"source_role":"PREVIOUSLY_EXPOSED_DEVELOPMENT",
        "training_updates":0,"video_generations":0,"official_quality_calls":0}
    persist=lambda:native.write(out/"RUN.json",report)
    journal=(out/"OPERATIONS.jsonl").open("x",buffering=1)
    ledger=Ledger(report.setdefault("operation_ledger",{}),lambda event:journal.write(json.dumps(event,separators=(",",":"))+"\n"))
    persist();monitor=None
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        free=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
        native.need(free>=RESOURCES["host_available_min_bytes"] and shutil.disk_usage(out).free>=RESOURCES["free_disk_min_bytes"],"Host/disk resource reserve exceeded")
        native.need(sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())<=RESOURCES["new_HF_bytes"],"HF output budget exceeded")
        native.need(time.monotonic()-start<=RESOURCES["wall_seconds"],"Four-corner wall1200 exhausted")
        if "torch" in sys.modules:
            native.need(sys.modules["torch"].cuda.max_memory_reserved()<=RESOURCES["GPU_reserved_max_bytes"],"GPU reserved limit32GiB exceeded")
    def timeout(*_):raise TimeoutError("Four-corner fixed wall1200 exhausted; no retry")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES["wall_seconds"])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder,no_hif4=CheckedPackageFinder(local,rows),native.NoHiF4()
    try:
        check();snapshot=out/"EXECUTED_SOURCES";snapshot.mkdir()
        for item in plan["repo_sources"]:
            destination=snapshot/item["path"];destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/item["path"],destination)
        shutil.copyfile(args.protocol,snapshot/"protocol.json")
        sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/".runtime/code/LightX2V")]
        sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE="BF16",SENSITIVE_LAYER_DTYPE="BF16",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,report["CUDA_reservation"]=reserve_before_assets(torch)
        native.need(torch.cuda.device_count()==1 and "A100" in torch.cuda.get_device_name(0),"One isolated A100 required")
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report["runtime_precision"]={"matmul_TF32":False,"float32_matmul_precision":"highest","bf16_reduced_precision_reduction":True}
        controls=load_controls(torch,plan,args.hf_root,ledger)
        torch.cuda.reset_peak_memory_stats()
        with native.OriginalGuard(torch,plan,ledger) as guard:
            begun=time.perf_counter();pipe=native.build_pipeline(plan)
            report["raw_model_setup_seconds"]=time.perf_counter()-begun
            contexts={c["index"]:prepare_source(pipe,guard,c,controls[c["index"]],args.hf_root) for c in plan["cases"]}
            geometry,data={},{}
            def flow(case,op,step,input_cpu,pack=None,target=None,Y_target=None,prefix_reference=None):
                value=single_flow(torch,pipe,guard,case,op,step,input_cpu,geometry,pack,controls[case["index"]],
                    contexts[case["index"]],out,args.hf_root,report,persist,check,target,Y_target,prefix_reference)
                data[case["index"],op]=value;return value
            # All four original neutral controls must pass before any fit/candidate.
            for case in plan["cases"]:
                for step in STEPS:flow(case,"DD"+str(step),step,controls[case["index"]]["input"+str(step)])
            report["all_four_neutral_flow_and_post_controls_passed"]=True;persist();check()
            pack,report["weight_space_geometry"]=export_projection(torch,pipe,geometry,out,args.hf_root,ledger)
            persist();check()
            for case in plan["cases"]:
                for step in STEPS:
                    target=data[case["index"],"DD"+str(step)]
                    flow(case,"SD"+str(step),step,controls[case["index"]]["input"+str(step)],pack,target,target["Y"],target)
            # The counterfactual teacher is paid explicitly, then used only as
            # an observer target. It never supplies candidate inference values.
            for case in plan["cases"]:
                target=data[case["index"],"DD3"]
                flow(case,"DS3",3,data[case["index"],"SD2"]["post_cpu"],None,target,target["Y"])
            for case in plan["cases"]:
                target=data[case["index"],"DD3"];same_input_teacher=data[case["index"],"DS3"]
                flow(case,"SS3",3,data[case["index"],"SD2"]["post_cpu"],pack,target,same_input_teacher["Y"],same_input_teacher)
            report["four_corner"]=[]
            for case in plan["cases"]:
                group=[data[case["index"],op] for op in ("DD3","SD3","DS3","SS3")]
                values=core.four_corner(torch,*(x["flow"] for x in group),ledger.work)
                # This is a continuous-update identity. Native FP32 arithmetic
                # and final BF16 casts are measured separately, not discarded.
                values["source_index"]=case["index"]
                values["step"]=3;values["delta_sigma"]=.625
                values.update(post_identity(torch,group,ledger))
                report["four_corner"].append(values)
            ledger.finish();check();report.update(status="COMPLETE_RAW_BF16_LOW39_R512_TWELVE_FULL_FLOWS_NO_VIDEO",
                checked_transformers_imports=finder.origins,original_loader_paths=guard.loaded,
                teacher_full_Y_retained_bytes=6*M*D*2,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        report.update(status="FAILED_RAW_FOUR_CORNER_PRESERVED_NO_RETRY",error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);monitor_error=None
        if monitor:
            try:
                monitor.close();report["assigned_GPU_resources"]=monitor.report()
                native.need(not report["assigned_GPU_resources"].get("violations"),"Final assigned GPU resource violation")
            except BaseException as e:
                monitor_error=e;report.update(status="FAILED_RAW_FOUR_CORNER_PRESERVED_NO_RETRY",resource_error_type=type(e).__name__,resource_error=str(e))
        for item in (finder,no_hif4):
            if item in sys.meta_path:sys.meta_path.remove(item)
        report["wall_seconds"]=time.monotonic()-start;persist();journal.close()
        if monitor_error is not None and sys.exc_info()[0] is None:raise monitor_error
    return report


def draft(hf):
    parent=native.read(ROOT/PARENT);folder=hf/parent["output_HF_relative"]
    run=native.read(folder/"RUN.json")
    from ffnds_experiment_plans import row_draw
    cases=[]
    for c in parent["cases"]:
        phase=next(p for p in run["phases"] if p["source_index"]==c["index"])
        mapping={"raw_noise_FP32":"raw_noise_FP32","input2":"latent_step1","input3":"latent_step2",
                 "flow2":"flow_step2","flow3":"flow_step3","post2":"latent_step2","post3":"latent_step3"}
        cases.append({**{k:c[k] for k in ("index","source_id","image_sha256","prompt_sha256","prompt","conditioning_id","image","source_group")},
            "controls":{k:phase["states"][v] for k,v in mapping.items()},
            "rows":{str(s):row_draw(M,256,base_seed=42,source_index=c["index"],step=s,block=39,state_pair=True)[0] for s in STEPS}})
    sources={b["path"] for b in parent["repo_sources"]}|REQUIRED
    result={"id":ID,"status":"DRAFT_NO_MODEL_NO_GPU","numeric_path":"BF16_ORIGINAL",
        "assets_finalized":False,"binding_status":"DECLARED_PARENT_COMPLETE_NEW_SOURCE_CPU_PROOF_PENDING",
        "parameters":core.P,"resources":RESOURCES,"limits":limits(),"run_ids":list(RUN_IDS),"cases":cases,
        "parent_protocol":native.binding(ROOT/PARENT,ROOT),"parent_RUN":native.binding(folder/"RUN.json",hf),
        "parent_exit":native.binding(folder.parent/"EXIT_RECEIPT.json",hf),"raw_master_CPU_proof":run["CPU_preflight"],
        **{k:parent[k] for k in ("models","model_view_root","model_view_metadata","runtime_overlay","runtime_versions","local_transformers","private_worker_config_sha256")},
        "repo_sources":[native.binding(ROOT/p,ROOT) for p in sorted(sources)],
        "output_HF_relative":"ffnds-derived/e0_original_bf16_low39_r512_four_corner_20261028/attempt1/worker/results",
        "basis_identity":"raw low W2 weight covariance; two actual native head/gate spans128 plus weight-PCA384; not activation-trained",
        "sequence":"all4DD neutral then basis fit then all4SD then2DS paid teacher then2SS",
        "raw_master_reuse":"actual prior exit0 raw69GB proof plus unchanged device/inode/size/mtime/ctime; no raw payload rehash",
        "row_mapping_identity":"consumer20 indices0/12, fixed ffnds row_draw source_index=index/step/block39, 256 nested audit rows; all flow/head statistics are full M",
        "online_graph":"original raw norm/modulation/W1/GELU(tanh), BF16 H@B then addmm(original_b2,coeff,Phi.T); no Q1/Q2, no teacher online",
        "cost_identity":"Full original checkpoint CPU/offload/preload cost retained, including unused original W2 prefetch. No system-speed claim.",
        "new_HF_payload_estimate_bytes":12*(math.prod(SHAPE)*6+M*64*2+4*256*D*2+3*D*2+256*8)+(H*512+D*512+D)*2,
        "teacher_full_Y_peak_references_bytes":6*M*D*2,
        "CPU_preflight_resources":{"GPU":0,"CPU_threads":4,"wall_seconds":300,"outer_seconds":360,"new_raw_master_hash_bytes":0}}
    validate(result);verify_parent(result,hf,stat_check=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("draft","inspect","preflight","run"))
    p.add_argument("--protocol",type=Path);p.add_argument("--hf-root",type=Path,required=True)
    p.add_argument("--local-root",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args();a.hf_root=a.hf_root.resolve()
    if a.command=="draft":
        native.need(a.out is not None and not a.out.exists(),"Fresh draft required")
        value=draft(a.hf_root);native.write(a.out,value)
        print(json.dumps({"id":ID,"status":value["status"],"model_calls":0}))
    elif a.command=="inspect":
        value=native.read(a.protocol);validate(value)
        print(json.dumps({"id":ID,"limits":limits(),"resources":RESOURCES,"model_calls":0},indent=2))
    elif a.command=="preflight":
        native.need(a.out is not None and a.local_root is not None,"Fresh CPU output and local package required")
        out=a.out.resolve();native.need(out.is_relative_to(a.hf_root) and not out.exists(),"Fresh confined CPU output required")
        out.mkdir(parents=True);value={"id":ID,"protocol_sha256":native.sha(a.protocol),"status":"STARTED_BYTE_PREFLIGHT_NO_RETRY"}
        native.write(out/"RUN.json",value)
        try:value=preflight(a)
        except BaseException as e:
            value.update(status="FAILED_BYTE_PREFLIGHT_PRESERVED_NO_RETRY",error_type=type(e).__name__,error=str(e));raise
        finally:native.write(out/"RUN.json",value)
        print(json.dumps({"status":value["status"],"model_calls":0}))
    else:
        native.need(a.local_root is not None,"Exact local package required")
        print(json.dumps({"status":run(a)["status"]}))


if __name__=="__main__":main()
