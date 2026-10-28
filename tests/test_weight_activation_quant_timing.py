"""Static source-level regression: typed weight/activation quantization and native FP4 timing.

Checks repo source and machine contracts only; no Torch, network, cluster, model, GPU.
"""
from pathlib import Path
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"research/agent_protocol/WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json"

def read(path):
    return (ROOT/path).read_text(encoding="utf-8")

class StaticWeightDynamicActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.ops={x["id"]:x for x in cls.c["true_wan_path"]["operand_sequence"]}
        cls.formats={x["id"]:x for x in cls.c["formats"]}

    def test_four_distinct_operand_roles_production_times(self):
        self.assertEqual(set(self.ops),{
            "W1_STATIC_WEIGHT","A1_Q1_CURRENT_ACTIVATION",
            "W2_STATIC_WEIGHT","A2_Q2_CURRENT_ACTIVATION"})
        for k in ("W1_STATIC_WEIGHT","W2_STATIC_WEIGHT"):
            self.assertEqual(self.ops[k]["kind"],"weight")
            self.assertEqual(self.ops[k]["quantization_schedule"],"PTQ_BEFORE_INFERENCE")
        for k in ("A1_Q1_CURRENT_ACTIVATION","A2_Q2_CURRENT_ACTIVATION"):
            self.assertEqual(self.ops[k]["kind"],"activation")
            self.assertEqual(self.ops[k]["quantization_schedule"],"DYNAMIC_EVERY_FFN_CALL")
        self.assertIn("before W1",self.ops["A1_Q1_CURRENT_ACTIVATION"]["producer"])
        self.assertIn("after W1",self.ops["A2_Q2_CURRENT_ACTIVATION"]["producer"])

    def test_frozen_lightx2v_src_weight_prequant_activation_current(self):
        rt=read("upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill.py")
        for required in (
            'source = record.get("weight_qdq")',
            "self._resident_weights[module_name] = tensor",
            'x = native_hif4_activation_qdq(x, force_fp32=bool(record.get("force_fp32", True))).to(x.dtype)',
            "return F.linear(x, weight, bias)"
        ):
            with self.subTest(required=required):
                self.assertIn(required,rt)
        linear=read("upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/runtime/hifx4_linear.py")
        self.assertIn("self.weight = nn.Parameter(weight_qdq.detach().clone(), requires_grad=False)",linear)
        self.assertIn("work = self.format_backend.quantize_activation(work).to(dtype=x.dtype)",linear)
        self.assertIn("return F.linear(work, self.weight, self.bias)",linear)
        rtn=read("upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/algorithms/rtn.py")
        self.assertIn("dequantized_weight=quantized",rtn)

    def test_native_fp4_has_independent_weight_and_activation_scales(self):
        nv=self.formats["NVFP4_W4A4"]
        mx=self.formats["MXFP4_W4A4"]
        wo=self.formats["NVFP4_W4A16"]
        self.assertIn("block-16",nv["activation_quantization"])
        self.assertIn("global",nv["activation_quantization"])
        self.assertIn("E8M0",mx["activation_quantization"])
        self.assertIn("offline",mx["weight_quantization"])
        self.assertIn("NONE",wo["activation_quantization"])
        distinctions=self.c["static_dynamic_orthogonality"]
        self.assertTrue(distinctions["native_gemm_consumes_scale_factors"])
        self.assertTrue(distinctions["global_scale_static_does_not_imply_local_scale_static"])
        self.assertTrue(distinctions["Q2_activation_not_pre_W1_observation"])

    def test_old_observer_contract_references_typed_role_contract(self):
        obs=json.loads(read("research/agent_protocol/QMETA_OBSERVATION_CONTRACT_20261028.json"))
        cross=json.loads(read("research/agent_protocol/QUANT_FUNCTION_EXECUTION_FRONTIER_CROSSWALK_20261028.json"))
        rel="research/agent_protocol/WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json"
        self.assertEqual(obs["operand_quantization_timing_contract"],rel)
        self.assertEqual(cross["operand_timing_contract"],rel)
        self.assertEqual(len(obs["operand_roles"]),4)
        self.assertEqual(len(self.c["information_filtration"]),5)
        self.assertEqual(self.c["status"],"SRC_LIT_ALG_RECONSTRUCTION_NATIVE_DYNAMIC_FP4_OBSERVER_NOT_RUN")

    def test_navigational_sources_and_boundaries(self):
        for path in ["README.md","AGENTS.md",
                     ".agents/skills/ffnds-paper-discovery/SKILL.md",
                     "research/RESEARCH_NAVIGATION.md",
                     "docs/98_端到端计算图与证据注解_20261025.md"]:
            self.assertIn("WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md",read(path))
        self.assertTrue((ROOT/self.c["report"]).is_file())
        for source in self.c["primary_sources"]:
            self.assertTrue(source["url"].startswith("https://"))
        self.assertTrue(self.c["static_dynamic_orthogonality"]["swapping_HiF4_to_NVFP4_or_MXFP4_changes_group_closure_and_numeric_reference"])

if __name__=="__main__":
    unittest.main()
