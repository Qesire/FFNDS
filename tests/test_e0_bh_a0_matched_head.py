"""Tiny synthetic software controls; no saved real operands/models/CUDA."""
from contextlib import contextmanager
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import e0_bh_a0_matched_head as matched


def plan_stub():
    return {"id":matched.ID,"geometry":matched.GEOMETRY,"limits":matched.exact_limits(),
        "resources":matched.RESOURCES,"modes":list(matched.MODES),"allowed_case_ids":list(matched.CASE_IDS),
        "epsilon":1e-6,"source_split":{"training":list(range(16)),"diagnostic_heldout":list(range(16,24))},
        "BH_protocol":{"sha256":matched.BH_PROTOCOL_SHA},"bank_manifest":{"sha256":matched.BANK_SHA},
        "parent_protocol":{"sha256":matched.labels.PARENT_SHA},"binding_status":"DRAFT_ASSETS_NOT_BOUND","assets_finalized":False}


def freeze_stub(plan):
    plan=copy.deepcopy(plan);plan.update(assets_finalized=True,binding_status="BOUND_READY_FOR_REGISTRATION")
    plan["repo_sources"]=[{"path":p} for p in sorted(matched.REQUIRED_SOURCES)]
    plan["repo_sources"] += [{"path":"native/hifxg_quant.so"},{"path":"native/lowbit_quant.so"}]
    return plan


def binding(root,name,value):
    raw=(json.dumps(value,sort_keys=True)+"\n").encode();(root/name).write_bytes(raw)
    return {"path":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


class BindingControls(unittest.TestCase):
    def test_import_is_byte_only(self):
        subprocess.run([sys.executable,"-c","import sys;sys.path.insert(0,'tools');import e0_bh_a0_matched_head;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"],cwd=ROOT,check=True,capture_output=True)

    def test_exact_scope_counts_and_draft_not_executable(self):
        plan=plan_stub();matched.validate_plan(plan)
        with self.assertRaisesRegex(ValueError,"NOT_BOUND"):matched.validate_plan(plan,frozen=True)
        limits=matched.exact_limits()
        self.assertEqual(limits["fullM_LN"],96+192);self.assertEqual(limits["A0_native_Q2"],192)
        self.assertEqual(limits["A0_W1_BF16_GEMM"],192);self.assertEqual(limits["BH_student_forwards"],0)
        self.assertEqual(limits["CUDA_H2D_tensor_copies"],4+96*5+4*5+96*8)
        # Enumerate actual fixed tensor geometries independently of the driver's algebra.
        N,D,H,W=256,5120,64,6016
        neutral=sum((2*N*D,2*N*H,8*N,2*D,2*D))*96
        candidate=sum((2*N*D,2*D,8*N,2*N*D,2*D,2*D,2*N*H,2*N*H))*96
        students=sum((2*W*D,2*D*W,2*W,2*D,4*W))*4
        heads=sum((2*D*H,2*H))*2
        self.assertEqual(limits["CUDA_H2D_payload_bytes"],neutral+candidate+students+heads)
        self.assertEqual(limits["GPU_original_layout_restores"],2*2+192*2)

    def test_changed_seed_grid_Q2_group_or_budget_rejected(self):
        for key in ("ids","group","budget","BH"):
            plan=plan_stub()
            if key=="ids":plan["allowed_case_ids"][0]="different"
            elif key=="group":plan["geometry"]={**plan["geometry"],"group":32}
            elif key=="budget":plan["limits"]["A0_native_Q2"]+=1
            else:plan["BH_protocol"]["sha256"]="stale"
            with self.subTest(key=key),self.assertRaises(ValueError):matched.validate_plan(plan)

    def test_precision_override_rejected(self):
        for key in ("NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
            with patch.dict("os.environ",{key:"1"}),self.assertRaisesRegex(ValueError,"TF32"):matched.precision_gate()

    def test_all_neutrals_required_before_candidates(self):
        records=[{"case_id":i,"bitwise_equal":True} for i in matched.CASE_IDS]
        matched.neutral_gate(records)
        for key in ("missing","duplicate","failed"):
            bad=copy.deepcopy(records)
            if key=="missing":bad.pop()
            elif key=="duplicate":bad[-1]=bad[0]
            else:bad[-1]["bitwise_equal"]=False
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"neutrals"):matched.neutral_gate(bad)

    def test_ledger_retains_failed_invocation(self):
        report={};events=[];ledger=matched.Ledger(report,{"call":1},journal=lambda e:events.append(copy.deepcopy(e)))
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with ledger("call"):raise RuntimeError("intentional")
        self.assertEqual(report["predebited"]["call"],1);self.assertEqual(report["invoked"]["call"],1)
        self.assertEqual(report["completed"]["call"],0)
        self.assertEqual([x["phase"] for x in events],["predebited","invoked"])
        with self.assertRaisesRegex(ValueError,"Incomplete"):ledger.finish()
        with self.assertRaisesRegex(ValueError,"Exceeded"):
            with ledger("call"):pass

    def test_exact_exited_CPU_proof_semantics_with_rebound_JSON(self):
        plan={"repo_sources":[],"label_manifest":{"sha256":"labels"},"BH_RUN":{"sha256":"bh"}}
        report={"id":matched.ID,"status":"PASS_MATCHED_HEAD_BYTES_NO_NUMPY_NO_TORCH_NO_GPU","protocol_sha256":"protocol",
            "sources_sha256":matched.labels.state_sha([]),"label_manifest_sha256":"labels","bank_manifest_sha256":matched.BANK_SHA,
            "BH_RUN_sha256":"bh","cases":96,"A0_exports":4,"all_CUDA_devices_hidden":True,"NumPy_or_Torch_imported":False,
            "model_calls":0,"native_calls":0,"training_updates":0}
        receipt={"id":matched.ID,"protocol_sha256":"protocol","child_returncode":0,"process_exited":True,"timeout":False,"automatic_retry":False}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def bound_scope(r,e):
                run_binding=binding(root,"CPU.json",r);e={**e,"run_sha256":run_binding["sha256"]} if "run_sha256" not in e else e
                return {"cpu_preflight":{"RUN":run_binding,"launch_exit":binding(root,"EXIT.json",e)}}
            scope=bound_scope(report,receipt);matched.require_cpu_receipt(scope,plan,"protocol",root)
            for key,value in (("cases",95),("NumPy_or_Torch_imported",True),("native_calls",1),("BH_RUN_sha256","different")):
                bad={**report,key:value};scope=bound_scope(bad,receipt)
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"proof differs"):
                    matched.require_cpu_receipt(scope,plan,"protocol",root)
            for key,value in (("child_returncode",1),("run_sha256","stale"),("process_exited",False)):
                scope=bound_scope(report,{**receipt,key:value})
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"actual exit"):
                    matched.require_cpu_receipt(scope,plan,"protocol",root)

    def test_session_cap_and_repository_dynamic_policy_are_independent(self):
        plan=freeze_stub(plan_stub());scope={"id":matched.ID,"protocol_sha256":"p","execution_status":"AUTHORIZED_NOT_STARTED",
            "max_gpu_cards":1,"limits":matched.exact_limits(),"resources":matched.RESOURCES}
        active={"user_continuation_authorization_20261027":{"status":"ACTIVE_CURRENT_USER_DIRECTED_SESSION","new_scope_registration_allowed":True,"max_simultaneously_assigned_idle_GPUs":99},
            "resource_scheduling_policy_20261028":{"mode":"DYNAMIC_MAX_ACTUALLY_CALLABLE","global_gpu_card_cap":None},
            "current_session_resource_constraint_20261028":{"max_simultaneously_assigned_idle_GPUs":2},
            "experiment_authority":{"current_user_authorized_scopes":[scope]}}
        self.assertEqual(matched.authorize(plan,"p",active),scope)
        active["current_session_resource_constraint_20261028"]["max_simultaneously_assigned_idle_GPUs"]=1
        with self.assertRaisesRegex(ValueError,"authority"):matched.authorize(plan,"p",active)

    def test_source_equal_ratios_and_nested_steps(self):
        rows=[]
        for i in range(24):
            for s in range(4):
                rows.append({"source_index":i,"expert":matched.EXPERTS[s//2],"step":s,
                    "arms":{arm:{"head_relative_SSE":(i+1)*(k+1)+(s%2)*2} for k,arm in enumerate(matched.MODES)}})
        report=matched.summarize(rows)
        self.assertEqual(report[matched.EXPERTS[0]]["heldout8_exposed"]["equal_source_mean"]["BH_saved"],sum(range(17,25))/8+1)
        self.assertEqual(report[matched.EXPERTS[1]]["heldout8_exposed"]["BH_lower_source_count"]["energy94"],8)
        with self.assertRaisesRegex(ValueError,"nested"):matched.summarize(rows[:-1])

    def test_fullM_metadata_rejects_wrong_stride_or_affine_norm(self):
        g={"M":9,"D":4,"outputs":2,"rows":2,"width":4,"group":2}
        item={"case":{"block":39,"rows":[1,7]},"actual_u_layout":{"shape":[9,4],"dtype":"torch.bfloat16","stride":[4,1]},
            "actual_head_operator":{"norm_eps":1e-6,"norm_weight":None,"norm_bias":None,
                "head_class":"lightx2v.common.ops.mm.mm_weight.MMWeight","norm_class":"lightx2v.common.ops.norm.layer_norm_weight.LNWeight",
                "weight_runtime_KN":{"shape":[4,2],"stride":[1,4],"dtype":"torch.bfloat16"},"bias":{"shape":[2],"stride":[1],"dtype":"torch.bfloat16"}},
            "evidence":{"head_weight_stride":[1,4],"head_bias_stride":[1],"head_scale_stride":[8,4,1],"head_shift_stride":[8,4,1]}}
        matched.validate_layout(item,g)
        bad=copy.deepcopy(item);bad["actual_u_layout"]["stride"]=[1,9]
        with self.assertRaisesRegex(ValueError,"u shape/stride"):matched.validate_layout(bad,g)
        bad=copy.deepcopy(item);bad["actual_head_operator"]["norm_weight"]="affine"
        with self.assertRaisesRegex(ValueError,"row-local"):matched.validate_layout(bad,g)


class TinyNumericalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)

    def fixture(self):
        torch=self.torch;generator=torch.Generator().manual_seed(713)
        sample=lambda *shape:torch.randn(shape,generator=generator).bfloat16()
        geometry={"M":9,"D":4,"outputs":2,"rows":2,"width":4,"group":2}
        pack={"W1":sample(4,4),"W2":sample(4,4),"b1":sample(4),"b2":sample(4),
            "smooth1":torch.full((4,),1000.),"smooth2":torch.tensor([1.,2.,4.,8.])}
        return geometry,pack,sample(2,4),sample(2,4),sample(1,4)

    def work(self):
        counts=Counter()
        @contextmanager
        def call(key,count=1):counts[key]+=count;yield
        return counts,call

    def test_A0_uses_postQ1_once_and_own_Q2_then_exact_BF16_write(self):
        torch=self.torch;g,pack,v1,residual,gate=self.fixture();calls=[];before=residual.clone();counts,work=self.work()
        def quantizer(value,force_fp32=True):
            self.assertTrue(force_fp32);calls.append(value.clone());return (value.float()*4).round().div(4).bfloat16()
        from compact6016_adapter import compact_forward
        expected,trace=compact_forward(v1,pack["W1"],pack["b1"],pack["W2"],pack["b2"],pack["smooth2"],quantizer,return_trace=True)
        calls.clear()
        y,u=matched.a0_postwrite(torch,v1,residual,gate,pack,quantizer,work,g)
        self.assertTrue(matched.same_bits(torch,y,expected));self.assertTrue(matched.same_bits(torch,calls[0],trace["pre_Q2"]))
        self.assertEqual(len(calls),1);self.assertTrue(matched.same_bits(torch,residual,before))
        oracle=residual.clone();oracle.add_(expected*gate);self.assertTrue(matched.same_bits(torch,u,oracle))
        self.assertEqual(counts,dict.fromkeys(("A0_W1_BF16_GEMM","A0_tanh_GELU","A0_FP32_Smooth2_divide","A0_native_Q2",
            "A0_W2_BF16_GEMM","BF16_gate_multiply","BF16_residual_add"),1))
        self.assertNotIn("teacher",inspect.signature(matched.a0_postwrite).parameters)

    def test_wrong_Q2_shape_fails_before_W2_or_write(self):
        torch=self.torch;g,pack,v1,r,gate=self.fixture();counts,work=self.work()
        with self.assertRaisesRegex(ValueError,"Q2 shape"):
            matched.a0_postwrite(torch,v1,r,gate,pack,lambda x,**_:x[:,:2],work,g)
        self.assertNotIn("A0_W2_BF16_GEMM",counts);self.assertNotIn("BF16_residual_add",counts)

    def test_failed_A0_byte_identity_retains_expected_actual_and_flag_boundary(self):
        torch=self.torch;value=torch.tensor([[1.,2.]],dtype=torch.bfloat16);record={};counts,work=self.work()
        with self.assertRaisesRegex(ValueError,"no flag search"):
            matched.check_a0_output(torch,value,"wrong",record,work)
        self.assertEqual(record["actual_M256_output_sha256"],matched.tensor_sha(torch,value))
        self.assertEqual(record["original_final_diagnostic_output_sha256"],"wrong")
        self.assertEqual(record["original_A0_BF16_reduction_flag"],"NOT_RECORDED_IN_ORIGINAL_PRODUCER")
        self.assertFalse(record["A0_output_bytes_match_original_diagnostic"])
        matched.check_a0_output(torch,value,matched.tensor_sha(torch,value),record,work)
        self.assertTrue(record["A0_output_bytes_match_original_diagnostic"])

    def test_fullM_replay_matches_original_operator_with_original_strides(self):
        torch=self.torch;g,_,u,_,_=self.fixture();rows=torch.tensor([1,7]);counts,work=self.work()
        w=torch.empty_strided((4,2),(1,4),dtype=torch.bfloat16);w.copy_(torch.arange(8).reshape(4,2).bfloat16()/8)
        head={"weight":w,"bias":torch.tensor([.25,-.5],dtype=torch.bfloat16)}
        scale=torch.empty_strided((1,1,4),(8,4,1),dtype=torch.bfloat16);scale.copy_(torch.tensor([.01,.2,-.1,.3]).reshape(1,1,4))
        shift=scale.clone();shift.mul_(.25)
        state=torch.zeros((9,4),dtype=torch.bfloat16);state.index_copy_(0,rows,u)
        from e0_b_terminal_head_probe import native_head
        reference,_=native_head(torch,state,w,head["bias"],scale,shift)
        result,digest=matched.fullM_readout(torch,u,rows,head,scale,shift,work,g)
        self.assertTrue(matched.same_bits(torch,result,reference.index_select(0,rows)))
        self.assertEqual(digest,matched.tensor_sha(torch,result));self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(counts,dict.fromkeys(("padded_zero_states","selected_row_writes","fullM_LN","BF16_head_gain_add",
            "BF16_head_modulation","fullM_head_GEMM","selected_u_unchanged_checks","selected_head_hashes"),1))

    def test_duplicate_or_out_of_bounds_rows_rejected_before_allocation(self):
        torch=self.torch;g,_,u,_,_=self.fixture();head={};counts,work=self.work()
        for rows in (torch.tensor([1,1]),torch.tensor([1,9])):
            with self.assertRaisesRegex(ValueError,"rows invalid"):
                matched.fullM_readout(torch,u,rows,head,None,None,work,g)
        self.assertFalse(counts)

    def test_head_metrics_are_same_target_relative_SSE_and_zero_norm_fails(self):
        torch=self.torch;counts,work=self.work();truth=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);guess=truth+1
        metrics=matched.head_metrics(torch,guess,truth,work)
        self.assertEqual(metrics["head_relative_SSE"],4/30);self.assertEqual(metrics["head_mean_row_SSE"],2)
        with self.assertRaisesRegex(ValueError,"Zero captured"):
            matched.head_metrics(torch,guess,torch.zeros_like(truth),work)


if __name__=="__main__":unittest.main()
