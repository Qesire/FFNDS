"""CPU tests for native source cohorts and nested calibration row sampling."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_experiment_plans import (
    plan_video, plan_calibration, row_draw, sources_from_legacy, save_new, load, choose
)


def bank(sizes=(12, 12), role="CALIBRATION"):
    data = []
    for group, n in zip(("i2v_subject", "i2v_background"), sizes):
        for i in range(n):
            sid = group + ":" + str(i)
            data.append(dict(source_id=sid, native_source_ordinal=(0 if group == "i2v_subject" else 1000) + i, image_id="image:" + sid,
                             prompt_id="prompt:" + sid, conditioning_id="condition:" + sid,
                             role=role, scene_stratum=group, prompt=sid,
                             image_name=sid + ".jpg", image_path="/fake/" + sid + ".jpg",
                             exposure_history={}))
    return {"schema": "ffnds.source.bank.v1", "sources": data}


class PipelineTest(unittest.TestCase):
    def test_real_video_plan_has_distinct_noise_and_paired_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save_new(p / "bank.json", bank((2, 2), "DEVELOPMENT"))
            v = plan_video(p / "bank.json", role="DEVELOPMENT",
                           strata="i2v_subject:2,i2v_background:2", seed=42,
                           samples=5, variants=["dense", "UB1"], experiment="RUN1",
                           frames=81, fps=16, width=832, height=464,
                           scheduler="four", output=p / "plan.json")
            self.assertEqual(len(v["samples"]), 40)
            self.assertEqual(len({x["run_id"] for x in v["samples"]}), 40)
            same = [x for x in v["samples"] if x["source_id"] == "i2v_subject:0"]
            for sample_index in range(5):
                pair = [x for x in same if x["sample_index"] == sample_index]
                self.assertEqual(len(pair), 2)
                self.assertEqual({x["seed"] for x in pair}, {42 + sample_index})
                self.assertEqual(len({x["initial_noise_id"] for x in pair}), 1)
            self.assertEqual({x["status"] for x in v["samples"]}, {"PLANNED"})

    def test_cluster_first_selection_is_nested_with_shared_image_prompts(self):
        sources = bank((3, 0))["sources"]
        sources.insert(1, dict(sources[0], source_id="alternate_prompt",
                               prompt_id="alternate", conditioning_id="alternate"))
        small = choose(sources, "CALIBRATION", {"i2v_subject": 2}, 13)
        large = choose(sources, "CALIBRATION", {"i2v_subject": 3}, 13)
        self.assertEqual(len({s["image_id"] for s in small}), 2)
        self.assertEqual(len({s["image_id"] for s in large}), 3)
        self.assertTrue({s["source_id"] for s in small} <=
                        {s["source_id"] for s in large})
        self.assertEqual([s["source_id"] for s in small],
                         [s["source_id"] for s in choose(
                             sources, "CALIBRATION", {"i2v_subject": 2}, 13)])
        official = choose(sources, "CALIBRATION", {"i2v_subject": 4}, 13,
                          allow_same_image_with_distinct_prompts=True)
        self.assertEqual(len(official), 4)
        self.assertEqual(len({s["image_id"] for s in official}), 3)

    def test_quota_counts_image_clusters_and_rejects_conflicting_strata(self):
        sources = bank((1, 0))["sources"]
        sources.append(dict(sources[0], source_id="alternate_prompt"))
        with self.assertRaisesRegex(ValueError, "2 independent image clusters"):
            choose(sources, "CALIBRATION", {"i2v_subject": 2}, 13)
        sources[-1]["scene_stratum"] = "i2v_background"
        with self.assertRaisesRegex(ValueError, "incompatible scene strata"):
            choose(sources, "CALIBRATION", {"i2v_subject": 1}, 13)

    def test_rows_are_nested_deterministic_and_scoped(self):
        a, s1 = row_draw(2000, 128, base_seed=42, source_index=0,
                         step=1, block=20, state_pair=True)
        b, s2 = row_draw(2000, 256, base_seed=42, source_index=0,
                         step=1, block=20, state_pair=True)
        c, _ = row_draw(2000, 128, base_seed=42, source_index=0,
                         step=1, block=20, state_pair=True)
        self.assertEqual(s1, s2)
        self.assertEqual(a, c)
        self.assertTrue(set(a) <= set(b))

    def test_calibration_requires_all_160_actual_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save_new(p / "bank.json", bank((1, 1)))
            geometry = [{"source_id": s["source_id"], "step": st, "block": block,
                         "population_rows": 1024}
                        for s in bank((1, 1))["sources"]
                        for st in range(4) for block in range(40)]
            save_new(p / "geometry.json", geometry)
            v = plan_calibration(p / "bank.json", p / "geometry.json",
                                 strata="i2v_subject:1,i2v_background:1",
                                 rows=256, seed=42, output=p / "out.json")
            self.assertEqual(len(v["calibration"]), 320)
            self.assertEqual(v["training_rows_per_position"], 512)
            ordinal_by_source = {s["source_id"]: s["native_source_ordinal"]
                                 for s in bank((1, 1))["sources"]}
            for row in v["calibration"]:
                expected = (f"draw:42:{ordinal_by_source[row['source_id']]}:"
                            f"{row['step']}:{row['block']}")
                self.assertEqual(row["row_draw_id"], expected)
                ids, actual_seed = row_draw(row["population_rows"], row["n_rows"],
                                            base_seed=42,
                                            source_index=ordinal_by_source[row["source_id"]],
                                            step=row["step"], block=row["block"],
                                            state_pair=True)
                self.assertEqual(row["selected_row_ids"], ids)
                self.assertEqual(row["sampling_seed"], actual_seed)
            geometry.pop()
            save_new(p / "less.json", geometry)
            with self.assertRaisesRegex(ValueError, "Missing observed native geometry"):
                plan_calibration(p / "bank.json", p / "less.json",
                                 strata="i2v_subject:1,i2v_background:1",
                                 rows=256, seed=42, output=p / "unwritten.json")

    def test_teacher_student_plans_share_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save_new(p / "bank.json", bank((1, 1)))
            geometry = [{"source_id": s["source_id"], "step": st, "block": b,
                         "population_rows": 1200}
                        for s in bank((1, 1))["sources"]
                        for st in range(4) for b in range(40)]
            save_new(p / "geometry.json", geometry)
            kwargs = dict(strata="i2v_subject:1,i2v_background:1", rows=128, seed=42)
            a = plan_calibration(p / "bank.json", p / "geometry.json",
                                 output=p / "teacher.json", source_state="teacher_input", **kwargs)
            b = plan_calibration(p / "bank.json", p / "geometry.json",
                                 output=p / "student.json", source_state="student_input", **kwargs)
            self.assertEqual(a["calibration_input_state"], "teacher_input")
            self.assertEqual(b["calibration_input_state"], "student_input")
            self.assertEqual([r["selected_row_ids"] for r in a["calibration"]],
                             [r["selected_row_ids"] for r in b["calibration"]])
            self.assertTrue(all(r["source_state"] == "student_input" for r in b["calibration"]))

    def test_legacy_source_roles_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            sample = [{"source_group": "i2v_subject:0", "prompt": "A dog",
                       "image_name": "dog.jpg", "image_path": "dog.jpg", "source_index": 21,
                       "input_provenance": {"repository": "repo", "revision": "rev",
                                            "image_sha256": "historical"}}]
            save_new(p / "cal.json", sample)
            b = sources_from_legacy([f"CALIBRATION={p/'cal.json'}"])
            self.assertEqual(len(b["sources"]), 1)
            second = [{**sample[0], "source_group": "i2v_subject:1"}]
            save_new(p / "dev.json", second)
            with self.assertRaisesRegex(ValueError, "overlap data roles"):
                sources_from_legacy([f"CALIBRATION={p/'cal.json'}",
                                     f"DEVELOPMENT={p/'dev.json'}"])


if __name__ == "__main__":
    unittest.main()
