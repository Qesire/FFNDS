"""Synthetic tiny CPU controls only; no real tensors, model or GPU query."""
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import bf_frozen6016_output as core
import e0_bf_frozen6016_terminal as driver


def counter():
    counts=Counter()
    @contextmanager
    def work(key,amount=1):
        counts[key]+=amount
        yield
    return counts,work


def draft():
    return {"id":driver.ID,"parameters":core.P,"resources":driver.RESOURCES,"limits":driver.exact_limits(),
        "basis_modes":list(core.BASIS_MODES),"source_split":{"training":list(core.TRAIN),"diagnostic_heldout":list(core.HELD)},
        "allowed_case_ids":list(driver.matched.CASE_IDS),"bank_manifest":{"sha256":driver.matched.BANK_SHA},
        "parent_protocol":{"sha256":driver.matched.labels.PARENT_SHA},"BH_protocol":{"sha256":driver.matched.BH_PROTOCOL_SHA},
        "matched_head_protocol":{"sha256":driver.MATCHED_PROTOCOL_SHA},"matched_head_RUN":{"sha256":driver.MATCHED_RUN_SHA},
        "matched_head_exit":{"sha256":driver.MATCHED_EXIT_SHA},
        "assets_finalized":False,"binding_status":"DRAFT_ASSETS_NOT_BOUND"}


class BindingControls(unittest.TestCase):
    def test_import_no_torch_numpy_or_device_discovery(self):
        command="import sys;sys.path.insert(0,'tools');import e0_bf_frozen6016_terminal;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,"-c",command],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_scope_and_actual_resource_identity(self):
        driver.validate_plan(draft())
        with self.assertRaisesRegex(ValueError,"NOT_BOUND"):driver.validate_plan(draft(),True)
        self.assertEqual(core.count_parameters(),208896)
        self.assertEqual(core.operations()["BF_MAC_per_row"],36503552)
        self.assertEqual(core.operations()["A0_MAC_per_row"],61603840)
        limits=driver.exact_limits()
        self.assertEqual(limits["Adam_updates"],2*2*256)
        self.assertEqual(limits["fullM_head_GEMM"],96+192+384)
        self.assertEqual(limits["factored_forward_calls"],1024+384)
        self.assertEqual(limits["asset_H2D_tensor_copies"],4+10+96*7+8)
        n,d,h,w=256,5120,64,6016
        per_case=sum((2*n*d,2*n*d,2*n*h,2*d,8*n,2*n*d,2*n*d))
        a0=sum((2*w*d,2*d*w,2*w,2*d,4*w))*2
        heads=sum((2*d*h,2*h))*2
        self.assertEqual(limits["asset_H2D_payload_bytes"],heads+a0+96*per_case+4*2*2*d)
        self.assertEqual(limits["GPU_original_layout_restores"],4+8)
        self.assertEqual(driver.RESOURCES["GPU"],1)
        self.assertEqual(driver.RESOURCES["session_max_GPUs"],2)

    def test_changed_rank_count_or_parent_rejected(self):
        for key in ("rank","count","parent"):
            plan=copy.deepcopy(draft())
            if key=="rank":plan["parameters"]["rank"]=256
            elif key=="count":plan["limits"]["Adam_updates"]+=1
            else:plan["bank_manifest"]["sha256"]="wrong"
            with self.subTest(key=key),self.assertRaises(ValueError):driver.validate_plan(plan)

    def test_failed_invocation_stays_charged(self):
        report={};events=[]
        ledger=driver.Ledger(report,journal=lambda e:events.append(e),limits={"work":1})
        with self.assertRaisesRegex(RuntimeError,"failure"):
            with ledger("work"):raise RuntimeError("failure")
        self.assertEqual(report["predebited"]["work"],1);self.assertEqual(report["invoked"]["work"],1)
        self.assertEqual(report["completed"]["work"],0)
        self.assertEqual([e["phase"] for e in events],["predebited","invoked"])
        with self.assertRaisesRegex(ValueError,"Incomplete"):ledger.finish()

    def test_CPU_receipt_exact_source_protocol_and_exit_before_torch(self):
        plan={"repo_sources":[],"label_manifest":{"sha256":"labels"},"matched_head_RUN":{"sha256":"parent"}}
        report={"id":driver.ID,"status":"PASS_BF_BYTES_NO_NUMPY_NO_TORCH_NO_GPU","protocol_sha256":"protocol",
            "sources_sha256":driver.matched.labels.state_sha([]),"label_manifest_sha256":"labels",
            "bank_manifest_sha256":driver.matched.BANK_SHA,"matched_head_RUN_sha256":"parent",
            "cases":96,"energy_exports":2,"all_CUDA_devices_hidden":True,"NumPy_or_Torch_imported":False,
            "model_calls":0,"native_calls":0,"training_updates":0,"decompositions":0}
        exited={"id":driver.ID,"child_returncode":0,"process_exited":True,"timeout":False,"automatic_retry":False,"protocol_sha256":"protocol"}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def binding(name,value):
                path=root/name;path.write_text(json.dumps(value));return {"path":name,"bytes":path.stat().st_size,"sha256":driver.matched.labels.sha(path)}
            def scope(r,e):
                run=binding("CPU.json",r)
                return {"cpu_preflight":{"RUN":run,"launch_exit":binding("EXIT.json",dict(e,run_sha256=run["sha256"]))}}
            driver.require_cpu_receipt(scope(report,exited),plan,"protocol",root)
            for key,value in (("NumPy_or_Torch_imported",True),("energy_exports",4),("sources_sha256","stale"),("decompositions",1)):
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"binding proof"):
                    driver.require_cpu_receipt(scope(dict(report,**{key:value}),exited),plan,"protocol",root)
            with self.assertRaisesRegex(ValueError,"actual exit"):
                driver.require_cpu_receipt(scope(report,dict(exited,child_returncode=1)),plan,"protocol",root)

    def test_TF32_override_rejected(self):
        with patch.dict("os.environ",{"NVIDIA_TF32_OVERRIDE":"1"}),self.assertRaisesRegex(ValueError,"TF32"):
            driver.matched.precision_gate()


class TinyNumericalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)

    def fixture(self,expert_index=0):
        torch=self.torch;generator=torch.Generator().manual_seed(173+expert_index)
        p=dict(core.P,D=8,width=16,rank=6,head_outputs=2,rows=2,adapter_rank=2,updates=2)
        cases=[]
        heads={s:torch.randn((8,2),generator=generator,dtype=torch.float64) for s in (2*expert_index,2*expert_index+1)}
        gates={s:torch.randn((1,8),generator=generator).bfloat16() for s in heads}
        for index in range(24):
            for step in heads:
                y=(torch.randn((2,8),generator=generator)+index*.03).bfloat16()
                cases.append({"source_index":index,"step":step,"role":"TRAIN" if index<16 else "HELDOUT_EXPOSED",
                    "Y":y,"gate":gates[step],"feature":torch.randn((2,16),generator=generator).bfloat16()})
        a0={"W2":torch.randn((8,16),generator=generator).bfloat16(),"b2":torch.randn((8,),generator=generator).bfloat16()}
        return p,cases,heads,a0

    def test_fits_share_train_mean_and_have_same_rank_and_true_head_span(self):
        torch=self.torch;p,cases,heads,_=self.fixture();counts,work=counter()
        mu,bases,evidence=core.fit_output_spaces(torch,cases,heads,p,work)
        expected=torch.cat([c["Y"].double() for c in cases if c["source_index"]<16]).mean(0)
        self.assertTrue(torch.allclose(mu,expected,atol=1e-14,rtol=0))
        self.assertTrue(all(phi.shape==(8,6) for phi in bases.values()))
        self.assertLess(evidence[core.BASIS_MODES[1]]["head_span_relative_squared_leak"],1e-20)
        self.assertEqual(counts["output_covariance_eigh"],2)
        for case in cases:
            _,leak=core.oracle_projection(torch,case["Y"],bases[core.BASIS_MODES[1]],mu,case["gate"],heads[case["step"]])
            self.assertLess(leak,1e-20)

    def test_heldout_values_cannot_change_mean_or_basis(self):
        torch=self.torch;p,cases,heads,_=self.fixture()
        mu,bases,_=core.fit_output_spaces(torch,cases,heads,p)
        changed=[]
        for case in cases:
            other=dict(case)
            if case["source_index"]>=16:other["Y"]=torch.full_like(case["Y"],1000.)
            changed.append(other)
        other_mu,other_bases,_=core.fit_output_spaces(torch,changed,heads,p)
        self.assertTrue(torch.equal(mu,other_mu))
        self.assertTrue(all(torch.equal(bases[k],other_bases[k]) for k in bases))

    def test_source_centering_and_changed_step_gate_are_rejected(self):
        torch=self.torch;p,cases,heads,_=self.fixture()
        bad=[dict(c) for c in cases];bad[0]["gate"]=bad[0]["gate"]+1
        with self.assertRaisesRegex(ValueError,"step-constant"):core.fit_output_spaces(torch,bad,heads,p)
        with self.assertRaisesRegex(ValueError,"train16"):core.fit_output_spaces(torch,cases[2:],heads,p)

    def test_affine_factor_initialization_keeps_mu_and_inner_bias_consistent(self):
        torch=self.torch;p,_,_,a0=self.fixture();phi=torch.eye(8,dtype=torch.float64)[:,:6]
        mu=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.],dtype=torch.float64)
        model=core.CoefficientAdapter(torch,phi,mu,a0,p)
        self.assertTrue(torch.equal(model.base,a0["W2"][:6]))
        self.assertTrue(torch.equal(model.bias,(a0["b2"].double()-mu)[:6].bfloat16()))
        feature=torch.zeros((2,16),dtype=torch.bfloat16)
        y=model.forward(feature,model.merged())
        self.assertTrue(torch.equal(y[:,6:],mu.bfloat16()[None,6:].expand(2,2)))
        wrong=torch.nn.functional.linear(torch.nn.functional.linear(feature,model.base),model.phi,model.mu)
        self.assertFalse(torch.equal(y,wrong))

    def test_cached_no_grad_tensor_supports_real_Bc_backward_and_correct_mm_types(self):
        torch=self.torch;p,cases,heads,a0=self.fixture();_,bases,_=core.fit_output_spaces(torch,cases,heads,p)
        counts,work=counter()
        with torch.no_grad():feature=cases[0]["feature"].clone()
        self.assertFalse(torch.is_inference(feature))
        model=core.CoefficientAdapter(torch,bases[core.BASIS_MODES[0]],torch.zeros(8,dtype=torch.float64),a0,p)
        with driver.matrix_observer(torch,work):
            output=model.forward(feature,model.merged())
            core.gated_loss(torch,output,cases[0]["Y"],cases[0]["gate"]).backward()
        self.assertEqual(counts,{"FP32_matrix_products":3,"BF16_matrix_products":4})
        self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()))
        self.assertGreater(float(model.weights["B"].grad.abs().sum()),0)
        with torch.inference_mode():illegal=feature.clone()
        with self.assertRaisesRegex(RuntimeError,"Inference tensors"):
            model.forward(illegal,model.merged())

    def test_head_numerator_can_match_while_variance_changes_head(self):
        torch=self.torch;p=dict(core.P,D=4,head_outputs=1)
        K=torch.tensor([[1.],[-1.],[0.],[0.]],dtype=torch.float64)
        truth=torch.tensor([[1.,-1.,2.,-2.]],dtype=torch.bfloat16)
        changed=torch.tensor([[1.,-1.,0.,0.]],dtype=torch.bfloat16)
        a,b=core.q_and_variance(torch,truth,K),core.q_and_variance(torch,changed,K)
        self.assertTrue(torch.equal(a["q"],b["q"]));self.assertFalse(torch.equal(a["variance"],b["variance"]))
        target=(a["q"]/(a["variance"][:,None]+1e-6).sqrt()).bfloat16()
        predicted=(b["q"]/(b["variance"][:,None]+1e-6).sqrt()).bfloat16()
        result=core.head_error_decomposition(torch,predicted,target,b,a,p=p)
        self.assertEqual(result["direct_relative_SSE"],0)
        self.assertEqual(result["signed_cross_relative_SSE"],0)
        self.assertGreater(result["radial_relative_SSE"],0)

    def test_signed_direct_radial_cross_and_BF16_remainder_close(self):
        torch=self.torch
        true={"q":torch.tensor([[1.,2.]],dtype=torch.float64),"variance":torch.tensor([1.],dtype=torch.float64)}
        predicted={"q":torch.tensor([[2.,4.]],dtype=torch.float64),"variance":torch.tensor([4.],dtype=torch.float64)}
        target=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
        observed=target+torch.tensor([[.125,-.125]],dtype=torch.bfloat16)
        metrics=core.head_error_decomposition(torch,observed,target,predicted,true)
        self.assertLess(metrics["signed_cross_relative_SSE"],0)
        self.assertAlmostEqual(metrics["continuous_relative_SSE"],sum(metrics[k] for k in
            ("direct_relative_SSE","radial_relative_SSE","signed_cross_relative_SSE")))
        self.assertAlmostEqual(metrics["head_relative_SSE"],sum(metrics[k] for k in
            ("continuous_relative_SSE","BF16_remainder_relative_SSE","BF16_signed_remainder_cross")))

    def test_native_write_matches_BF16_multiply_then_add_and_preserves_residual(self):
        torch=self.torch;r=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16)
        y=torch.tensor([[.1,.2,.3,.4]],dtype=torch.bfloat16);gate=torch.tensor([[1.,-2.,3.,-4.]],dtype=torch.bfloat16)
        saved=r.clone();counts,work=counter();result=core.native_write(torch,r,y,gate,work)
        expected=r.clone();expected.add_(y*gate)
        self.assertTrue(torch.equal(result,expected));self.assertTrue(torch.equal(r,saved))
        self.assertEqual(counts,{"BF16_gate_multiply":1,"BF16_residual_add":1})

    def test_deployed_three_GEMM_graph_pays_own_Q2_after_GELU_on_every_call(self):
        torch=self.torch;gen=torch.Generator().manual_seed(7)
        pack={name:torch.randn(shape,generator=gen).bfloat16() for name,shape in
            (("W1",(64,8)),("b1",(64,)),("Bc",(6,64)),("bc",(6,)),("Phi",(8,6)),("mu",(8,)))}
        pack["smooth2"]=torch.linspace(.75,1.25,64);calls=[];counts,work=counter()
        def synthetic_quantizer(value,*,force_fp32):
            self.assertTrue(force_fp32);calls.append(value.clone());return value
        with torch.no_grad(),driver.matrix_observer(torch,work):
            xs=[torch.randn((2,8),generator=gen).bfloat16() for _ in range(2)]
            ys=[core.deployed_forward(torch,x,pack,synthetic_quantizer,work) for x in xs]
        self.assertEqual(counts["BF16_matrix_products"],6)
        self.assertEqual(counts["deployed_dynamic_own_Q2"],2)
        self.assertEqual(len(calls),2);self.assertFalse(torch.equal(calls[0],calls[1]))
        for x,y,q in zip(xs,ys,calls):
            expected=(torch.nn.functional.gelu(torch.nn.functional.linear(x,pack["W1"],pack["b1"]),
                approximate="tanh").float()/pack["smooth2"]).bfloat16()
            self.assertTrue(torch.equal(q,expected));self.assertEqual(q.shape,(2,64))
            expected=core.factored_forward(torch,q,pack["Bc"],pack["bc"],pack["Phi"],pack["mu"])
            self.assertTrue(torch.equal(y,expected))

    def test_export_and_actual_reload_preserve_all_tensor_bytes(self):
        torch=self.torch;counts,work=counter()
        with tempfile.TemporaryDirectory() as temporary:
            pack={"a":torch.tensor([[1.,-2.]],dtype=torch.bfloat16),"b":torch.tensor([3],dtype=torch.int64)}
            record=driver.save_checked(torch,Path(temporary)/"tiny.safetensors",pack,work,"student_export")
            self.assertEqual(record["tensors"]["a"]["shape"],[1,2])
            self.assertEqual(counts,{"student_export_writes":1,"student_export_reload_checks":1})

    def test_complete_tiny_schedule_matches_actual_forward_backward_and_head_matrix_census(self):
        """Real tiny 1024Adam/672head graph; dimensions8/16, no real data."""
        torch=self.torch;counts,work=counter()
        geometry={"M":9,"D":8,"outputs":2,"rows":2};row_ids=torch.tensor([1,6])
        zero=torch.zeros(8,dtype=torch.bfloat16)
        with driver.matrix_observer(torch,work):
            for expert in range(2):
                p,cases,heads,a0=self.fixture(expert)
                native_heads={s:{"weight":K.bfloat16(),"bias":torch.zeros(2,dtype=torch.bfloat16)} for s,K in heads.items()}
                with torch.no_grad():
                    mu,bases,_=core.fit_output_spaces(torch,cases,heads,p,work)
                    for case in cases:
                        # Synthetic frozen W1/GELU/W2 cache with two real BF16
                        # products. The native CUDA Q2 remains unexecuted.
                        feature=torch.nn.functional.gelu(torch.nn.functional.linear(case["Y"],a0["W2"].T),approximate="tanh")
                        torch.nn.functional.linear(feature,a0["W2"],a0["b2"])
                        case["feature"]=feature
                        case["u"]=case["Y"].clone()
                        driver.matched.fullM_readout(torch,case["u"],row_ids,native_heads[case["step"]],zero,zero,work,geometry)
                        case["statistics"]=core.q_and_variance(torch,case["u"],heads[case["step"]],work)
                def evaluate(y,case):
                    driver.matched.fullM_readout(torch,y,row_ids,native_heads[case["step"]],zero,zero,work,geometry)
                    core.q_and_variance(torch,y,heads[case["step"]],work)
                for mode,phi in bases.items():
                    model=core.CoefficientAdapter(torch,phi,mu,a0,p,work)
                    with torch.no_grad():
                        for case in cases:
                            y,_=core.oracle_projection(torch,case["Y"],phi,mu,case["gate"],heads[case["step"]],work)
                            evaluate(y.bfloat16(),case)
                        merged=model.merged()
                        for case in cases:evaluate(model.forward(case["feature"],merged),case)
                    optimizer=torch.optim.Adam(model.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                    original_B=model.weights["B"].detach().clone()
                    for _ in range(8):
                        for case in core.training_cases(cases,p):
                            optimizer.zero_grad(set_to_none=True)
                            y=model.forward(case["feature"],model.merged())
                            core.gated_loss(torch,y,case["Y"],case["gate"],work).backward();optimizer.step()
                    self.assertFalse(torch.equal(original_B,model.weights["B"]))
                    with torch.no_grad():
                        merged=model.merged()
                        for case in cases:evaluate(model.forward(case["feature"],merged),case)
        for key in ("BF16_matrix_products","FP32_matrix_products","FP64_matrix_products"):
            self.assertEqual(counts[key],driver.exact_limits()[key])
        self.assertEqual(counts["coefficient_merges"],1032)
        self.assertEqual(counts["gated_Y_loss_calls"],1024)
        self.assertEqual(counts["fullM_head_GEMM"],672)
        self.assertEqual(counts["output_covariance_eigh"],4)
        self.assertEqual(counts["two_step_head_QR"],2)
        self.assertEqual(counts["final_head_space_QR"],2)
        self.assertEqual(counts["oracle_projection_cases"],192)
        self.assertEqual(counts["FP64_q_variance_cases"],672)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
