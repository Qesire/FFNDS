"""Pure-static tests for original Wan source graph plus literature and mechanism crosswalk."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from graph_agent_bridge import load, validate, inspect, trace, load_crosswalk, validate_crosswalk, validate_paper_focus, inspect_paper, inspect_focus, validate_method_operation_crosswalk


class GraphAgentBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load()


    def test_full_literature_index_is_linked_to_actual_native_focus_nodes(self):
        result=validate_paper_focus(self.registry)
        self.assertEqual(result["paper_focus_primary"],98)
        self.assertEqual(result["paper_focus_supplemental"],10)
        self.assertEqual(result["paper_focus_stages"],16)
        self.assertGreaterEqual(result["paper_to_stage_links"],150)
        for paper in ("L01","L09","L14","L34","L74","L84","L89","L90","L91","L92","C01","C05","C10"):
            looked=inspect_paper(self.registry,paper)
            self.assertTrue(looked["native_stage_analogies"])
            self.assertEqual(looked["method_graph"]["source_url"],looked["paper"]["url"])
            self.assertEqual(looked["method_graph"]["review_depth"],looked["paper"]["reading_depth"])

    def test_quantization_stage_groups_distinct_papers_and_is_lit_only(self):
        focus=inspect_focus(self.registry,"T06")
        self.assertEqual(focus["focus"]["native_anchor"],"F08R")
        self.assertIn("L74",focus["primary_papers"])
        self.assertIn("C04",focus["supplemental_papers"])
        self.assertTrue(all(x["author_method"] and x["decision_time"] and
                            x["physical_or_training_action"] for x in focus["method_comparisons"]))
        stages=[p for p in self.registry["nodes"] if p["graph_section"]=="G14"]
        self.assertEqual(len(stages),16)
        self.assertTrue(all(n["evidence_type"]=="LIT" for n in stages))

    def test_registry_matches_mermaid_and_existing_sources(self):
        result = validate(self.registry)
        self.assertEqual(result["status"], "STATIC_GRAPH_BRIDGE_VALID")
        self.assertGreaterEqual(result["nodes"], 20)

    def test_actual_post_q2_input_distinct_from_gelu(self):
        q = inspect(self.registry, "F08")
        self.assertEqual(q["node"]["availability"], "after_Q2")
        self.assertTrue(any("FULL_HIDDEN" in p for p in q["node"]["evidence"]))
        h = inspect(self.registry, "F06")
        self.assertEqual(h["node"]["availability"], "after_GELU")

    def test_cost_paths_preserve_candidate_and_measured_boundaries(self):
        path = trace(self.registry, "G18", "P12")
        self.assertTrue(path["edges"])
        self.assertIn("not proof", path["warning"])
        # The G5 exact-dense SYS branch is now a valid shorter route. It must
        # not be relabeled a hypothetical approximation to satisfy this test.
        routes = {
            "P16": "conditional_approximation_cost_not_executed",
            "P17": "scoped_measured_system_cost",
            "P19": "scoped_native_local_time_excludes_preparation_and_Q1_gate_residual",
        }
        for start, relation in routes.items():
            route = trace(self.registry, start, "P12")
            self.assertIn("not proof", route["warning"])
            self.assertEqual(route["edges"], [{"from": start, "to": "P12", "relation": relation}])

    def test_g5_separates_local_function_observations_from_dense_system_speed(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        self.assertEqual(kinds["P15"], "OBS")
        self.assertEqual(kinds["P16"], "OBS")
        self.assertEqual(kinds["P17"], "SYS")
        self.assertEqual(kinds["P18"], "ALG")
        self.assertIn("G5", self.registry["graph_roles"])

    def test_full_depth_video_diagnostics_remain_open_and_policy_is_not_native_evidence(self):
        kinds={n["id"]:n["evidence_type"] for n in self.registry["nodes"]}
        # Prior open hypotheses were later grounded by completed native observations.
        self.assertEqual(kinds["P23"], "NATIVE-OBS")
        self.assertEqual(kinds["P24"], "NATIVE-OBS")
        self.assertEqual(kinds["P25"], "POLICY")
        self.assertTrue(any(e["from"]=="P23" and e["to"]=="P24"
                            for e in self.registry["edges"]))

    def test_literature_branch_is_not_native_execution(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        for i in range(1, 13):
            self.assertEqual(kinds[f"K{i:02d}"], "LIT")
        self.assertIn("G7", self.registry["graph_roles"])
        self.assertEqual(self.registry["literature_mapping_v1"]["model_execution_count"], 0)
        path = trace(self.registry, "K07", "F08R")
        self.assertTrue(path["edges"])
        self.assertIn("not proof", path["warning"])

    def test_literature_g8_preserves_paper_execution_boundaries(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        for i in range(13, 28):
            self.assertEqual(kinds[f"K{i:02d}"], "LIT")
        self.assertIn("G8", self.registry["graph_roles"])
        self.assertEqual(self.registry["literature_mapping_v2"]["gpu_executions"], 0)
        for pair in [("K13", "K15"), ("K16", "K19"), ("K20", "K23"), ("K24", "K27")]:
            result = trace(self.registry, *pair)
            self.assertTrue(result["edges"])
            self.assertIn("not proof", result["warning"])

    def test_quant_code_spec_nodes_and_hif4_coupling_are_indexed(self):
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["Q01"]["evidence_type"], "SPEC")
        self.assertEqual(nodes["Q03"]["evidence_type"], "SPEC")
        self.assertEqual(nodes["I04"]["evidence_type"], "ALG")
        self.assertEqual(nodes["Q11"]["evidence_type"], "OBS")
        self.assertEqual(nodes["I08"]["evidence_type"], "OPEN")
        self.assertTrue(any(e["from"]=="I06" and e["to"]=="I07"
                            for e in self.registry["edges"]))
        self.assertTrue(any(e["from"]=="Q32" and e["to"]=="I08"
                            for e in self.registry["edges"]))

    def test_realtime_studies_are_not_native_wan_streaming(self):
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["S13"]["evidence_type"], "LIT")
        self.assertEqual(nodes["S15"]["evidence_type"], "LIT")
        self.assertEqual(nodes["S20"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["Q33"]["evidence_type"], "LIT")
        self.assertIn("G10", self.registry["graph_roles"])

    def test_current_mechanism_crosswalk_anchors_papers_and_exact_synthesis_edges(self):
        crosswalk = load_crosswalk(self.registry)
        result = validate_crosswalk(self.registry)
        self.assertIsNotNone(crosswalk)
        self.assertEqual(result["mechanism_families"], 14)
        self.assertEqual(result["primary_papers"], 98)
        self.assertEqual(result["crossgraph_comparison_edges"], 84)
        self.assertEqual(len(self.registry["nodes"]), 217)
        self.assertEqual(len(self.registry["edges"]), 451)
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["I17"]["evidence_type"], "ALG")
        self.assertEqual(nodes["I18"]["evidence_type"], "ALG")
        self.assertEqual(nodes["U27"]["evidence_type"], "NATIVE-OBS")
        self.assertEqual(nodes["I19"]["evidence_type"], "ALG")
        self.assertEqual(nodes["U28"]["evidence_type"], "ALG")
        self.assertEqual(nodes["P30"]["evidence_type"], "NATIVE-OBS")
        self.assertTrue(all(e["edge_type"] == "SCIENTIFIC_COMPARISON_OR_CONDITIONAL"
                            for e in crosswalk["synthesis_edges"]))
        families = {f["id"]: f for f in crosswalk["families"]}
        self.assertIn("I04", families["M02"]["home"])
        self.assertIn("L74", families["M05"]["papers"])
        self.assertIn("S20", families["M10"]["home"])

    def test_function_first_output_significance_and_native_structure_are_explicit(self):
        crosswalk = load_crosswalk(self.registry)
        self.assertEqual([f["id"] for f in crosswalk["families"][:2]], ["M13", "M14"])
        families = {f["id"]: f for f in crosswalk["families"]}
        self.assertIn("U07", families["M13"]["home"])
        self.assertIn("U10", families["M14"]["home"])
        self.assertIn("L84", families["M13"]["papers"])
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["U02"]["evidence_type"], "NATIVE-OBS")
        self.assertEqual(nodes["U04"]["evidence_type"], "SRC")
        self.assertEqual(nodes["U08"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U11"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U14"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U15"]["evidence_type"], "LIT")
        self.assertTrue(any(e["from"]=="U14" and e["to"]=="I04"
                            for e in self.registry["edges"]))
        self.assertTrue(any(e["from"]=="U14" and e["to"]=="Q09"
                            for e in self.registry["edges"]))
        self.assertEqual(len(self.registry["function_first_graph_20261027"]["node_ids"]), 18)
        self.assertTrue(self.registry["function_first_graph_20261027"]["no_new_native_runs"])

    def test_four_intervention_boundaries_and_quant_coupling_math_are_registered(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cards = json.loads((root / "research/agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual({b["id"] for b in cards["boundaries"]}, {"O","D","Q","P"})
        self.assertEqual({c["id"] for c in cards["cards"]}, {"C01","C02","C03"})
        by_id = {c["id"]: c for c in cards["cards"]}
        self.assertEqual(by_id["C01"]["status"], "PROPOSED_NOT_EXECUTED")
        self.assertEqual(by_id["C03"]["status"], "PROPOSED_NOT_EXECUTED")
        self.assertEqual(by_id["C02"]["status"],
                         "LOCAL_SAVED_OPERAND_DQ_COMPLETE_SUFFIX_AND_BEHAVIOR_OPEN")
        local = by_id["C02"]["completed_local_discriminator"]
        self.assertFalse(local["quality_accepted"])
        self.assertEqual(local["graph_nodes"], ["U25", "U26", "U27"])
        observed = json.loads((root / local["result"]).read_text())
        self.assertEqual(observed["actual_budget"]["full_model_generations"], 0)
        self.assertEqual(observed["controls"]["strict_BF16_W2_base_bit_patterns_passed"], 8)
        self.assertEqual(cards["resources"]["new_cpu_gpu_model_video_executions"], 0)
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["U19"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U20"]["evidence_type"], "ALG")
        self.assertEqual(nodes["U21"]["evidence_type"], "ALG")
        self.assertEqual(nodes["U22"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U23"]["evidence_type"], "ALG")
        self.assertEqual(nodes["U24"]["evidence_type"], "ALG")
        self.assertEqual(self.registry["function_localization_methodology_20261027"]["new_comparison_edges"], 17)
        relation = {(e["from"],e["to"],e["relation"]) for e in self.registry["edges"]}
        self.assertIn(("U20","U23","G13M_fixed_post_quant_direct_term"), relation)
        self.assertIn(("U21","U23","G13M_recomputed_pre_quant_direct_complement_coupling"), relation)
        self.assertIn(("U19","U24","G13M_two_action_same_boundary_joint_output_interaction"), relation)

    def test_unregistered_discovery_needs_curator(self):
        with self.assertRaises(SystemExit):
            inspect(self.registry, "F99")

    def test_open_proposal_is_not_experimental_evidence(self):
        proposed = inspect(self.registry, "P14")
        self.assertEqual(proposed["node"]["evidence_type"], "OPEN")
        self.assertEqual(proposed["node"]["availability"], "posthoc_hypothesis_only")
        self.assertGreaterEqual(len(proposed["node"]["evidence"]), 2)
        claim_path = Path(__file__).resolve().parents[1] / proposed["node"]["evidence"][0]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        self.assertEqual(claim["review_status"], "PROPOSED")
        self.assertEqual(claim["review_refs"], [])
        self.assertFalse(claim["model_experiment_authorized"])
        candidate_route = trace(self.registry, "P06", "P14")
        self.assertEqual(len(candidate_route["edges"]), 1)
        self.assertTrue(candidate_route["edges"][0]["relation"].startswith("open_hypothesis"))


    def test_current_method_priors_and_B0_progress_are_checked(self):
        result = validate_method_operation_crosswalk(self.registry)
        self.assertEqual(result["method_operation_types"], 5)
        self.assertEqual(result["method_project_routes"], 10)
        self.assertGreaterEqual(result["method_prior_constraint_relations"], 12)
        self.assertEqual(result["method_next_agent"], "M-B")
        self.assertEqual(result["method_next_stage"], "M-B_B1_CAL24_FIXED_DICTIONARY_COMPLETE__BH_BF_STUDENT_STAGE")
        output = validate(self.registry)
        self.assertEqual(output["method_next_agent"], "M-B")
        self.assertEqual(output["method_operation_types"], 5)

    def test_current_method_source_ids_and_stage_refs_are_grounded(self):
        root = Path(__file__).resolve().parents[1]
        cross = json.loads((root / "research/agent_protocol/METHOD_OPERATION_CROSSWALK.json").read_text())
        focus = json.loads((root / "research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json").read_text())
        all_papers = {x["id"] for x in focus["paper_links"] + focus["supplemental_links"]}
        all_stages = {x["id"] for x in focus["focus_nodes"]}
        self.assertEqual(cross["schema"], "ffnds.method_operation_crosswalk.v2")
        for edge in cross["method_prior_constraint_edges"]:
            self.assertTrue(set(edge["paper_ids"]) <= all_papers)
            self.assertTrue(set(edge["focus_nodes"]) <= all_stages)
            self.assertIn(edge["relation"], cross["relation_types"])

    def test_method_progress_separates_completed_B_diagnostics_from_students(self):
        root = Path(__file__).resolve().parents[1]
        cross = json.loads((root / "research/agent_protocol/METHOD_OPERATION_CROSSWALK.json").read_text())
        progress = {x["method"]: x for x in cross["methods_progress"]}
        self.assertIn("ALL80_LOCAL_RECOVERY_COMPLETE", progress["M-A"]["status"])
        self.assertIn("B0_CAL24_LABELS_B1_FIXED_POOL_COMPLETE", progress["M-B"]["status"])
        self.assertIn("BF_OPEN", progress["M-B"]["status"])
        self.assertEqual({n["id"]: n for n in self.registry["nodes"]}["U30"]["evidence_type"], "NATIVE-OBS")
        self.assertIn("MATHEMATICAL_INPUT_OUTPUT_LIMITS_ONLY", progress["M-C"]["status"])
        self.assertEqual(cross["one_turn_next_agent"], "M-B")
        self.assertTrue((root / cross["current_method_plan"]).is_file())


    def test_research_reading_priorities_have_one_current_method_authority(self):
        root = Path(__file__).resolve().parents[1]
        nav = (root / "research/RESEARCH_NAVIGATION.md").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")
        context = (root / "research/agent_protocol/CURRENT_CONTEXT_PACKET.md").read_text(encoding="utf-8")
        skill = (root / ".agents/skills/ffnds-paper-discovery/SKILL.md").read_text(encoding="utf-8")
        copilot = (root / ".github/copilot-instructions.md").read_text(encoding="utf-8")
        method = "COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md"
        for tier in ("P0", "P1", "P2", "P3", "P4"):
            self.assertIn("## " + tier + " ·", nav)
        for entry in (nav, readme, context, skill):
            self.assertIn(method, entry)
        self.assertIn("NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md", nav)
        self.assertIn("B0", context)
        self.assertIn("98个主LID", skill)
        self.assertNotIn("89个主LID", skill)
        self.assertNotIn("89项文献、177图节点、371关系", copilot)

    def test_priority_navigation_links_resolve_in_repo(self):
        import re
        from urllib.parse import unquote
        root = Path(__file__).resolve().parents[1]
        navfile = root / "research/RESEARCH_NAVIGATION.md"
        nav = navfile.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", nav):
            rel = target.split("#")[0].split(" ")[0]
            if not rel or rel.startswith(("https://", "http://", "mailto:")):
                continue
            resolved = (navfile.parent / unquote(rel)).resolve()
            self.assertTrue(resolved.is_relative_to(root), f"Path outside repo: {target}")
            self.assertTrue(resolved.exists(), f"Broken priority link: {target}")

if __name__ == "__main__":
    unittest.main()
