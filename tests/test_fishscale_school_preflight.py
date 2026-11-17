"""CPU-only synthetic tests; never claim current school files have been inspected."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fishscale_school_preflight import checked_file, parse_safetensors, run, sha256

class SchoolPreflightTests(unittest.TestCase):
    def fixture(self, temp):
        base = Path(temp)
        home = base / "public" / "home" / "scc" / "pb23061276"
        project = home / "projects" / "wan-quant"
        project.mkdir(parents=True)
        (project / "weights").mkdir()
        image = project / "input.png"
        video = project / "original.mp4"
        latent = project / "final_latent_bf16.safetensors"
        vae = project / "weights" / "Wan2.1_VAE.pth"
        image.write_bytes(b"real-school-input")
        video.write_bytes(b"synthetic-video-fixture")
        vae.write_bytes(b"synthetic-vae-fixture")
        h = json.dumps({"latents": {"dtype": "BF16", "shape": [16,21,58,104],
                                   "data_offsets": [0, 2]}}).encode()
        latent.write_bytes(struct.pack("<Q", len(h)) + h + b"\0\0")
        receipt = project / "native_run.json"
        receipt.write_text(json.dumps({"video": {"sha256": sha256(video)},
                                       "final_latent": {"sha256": sha256(latent)},
                                       "image": {"sha256": sha256(image)},
                                       "prompt_sha256": hashlib.sha256(b"test").hexdigest(),
                                       "seed": 42}))
        cfg = {
            "execution_domain": "USTC_SLURM_ONLY",
            "school_home": str(home),
            "school_home_documented_alias": str(base / "nonexistent_alias"),
            "school_project_root": str(project),
            "model_recipe": {"reference_geometry": {"width": 832, "height": 464,
                                                      "frames": 81, "fps": 16}},
            "expected_model_assets": [{"id": "vae", "relative_path": "weights/Wan2.1_VAE.pth",
                                       "bytes": vae.stat().st_size, "sha256": sha256(vae)}],
            "new_source_capture_spec": {}
        }
        registry = {
            "school_source_pairs": [{
                "id": "school_dense_01", "seed": 42, "prompt_sha256": hashlib.sha256(b"test").hexdigest(),
                "image": {"path": str(image), "sha256": sha256(image)},
                "video": {"path": str(video), "sha256": sha256(video)},
                "final_latent": {"path": str(latent), "sha256": sha256(latent),
                                 "tensor_name": "latents"},
                "parent_run": {"path": str(receipt), "sha256": sha256(receipt)}
            }]
        }
        config_path = project / "config.json"
        registry_path = project / "registry.json"
        config_path.write_text(json.dumps(cfg))
        registry_path.write_text(json.dumps(registry))
        args = argparse.Namespace(config=config_path, registry=registry_path, phase="decode",
                                  pair_id="school_dense_01", candidate_dir=[],
                                  source_manifest=None)
        return args, image, video, latent, receipt, project

    def test_legacy_paths_cannot_be_active(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            with self.assertRaises(ValueError):
                checked_file(project, "/home/xiang/.cache/huggingface/video.mp4")
            with self.assertRaises(ValueError):
                checked_file(project, "../../outside.mp4")

    def test_native_bf16_header(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, latent, _, _ = self.fixture(temp)
            self.assertEqual(parse_safetensors(latent)["dtype"], "BF16")
            self.assertEqual(parse_safetensors(latent)["shape"], [16,21,58,104])

    def test_strict_requires_slurm(self):
        with tempfile.TemporaryDirectory() as temp:
            args, *_ = self.fixture(temp)
            with patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
                receipt, code = run(args)
            self.assertEqual(code, 2)
            self.assertEqual(receipt["status"], "BLOCKED")
            self.assertIn("STRICT_PREFLIGHT_REQUIRES_SLURM_COMPUTE_ALLOCATION",
                          receipt["school_environment"]["errors"])

    def test_decode_passes_exact_bf16_same_run(self):
        with tempfile.TemporaryDirectory() as temp:
            args, *_ = self.fixture(temp)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-test-job"}):
                with patch("fishscale_school_preflight.video_info",
                           return_value={"width":832,"height":464,"frames":81,"fps":16}):
                    receipt, code = run(args)
            self.assertEqual(code, 0, receipt)
            self.assertEqual(receipt["status"], "VERIFIED_SCHOOL_INPUTS_READY_FOR_DECODE")

    def test_missing_parent_lineage_blocks_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _, native_run, _ = self.fixture(temp)
            native_run.write_text(json.dumps({"video": "not-binding-real-hashes"}))
            data = json.loads(args.registry.read_text())
            data["school_source_pairs"][0]["parent_run"]["sha256"] = sha256(native_run)
            args.registry.write_text(json.dumps(data))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-test-job"}):
                with patch("fishscale_school_preflight.video_info",
                           return_value={"width":832,"height":464,"frames":81,"fps":16}):
                    receipt, code = run(args)
            self.assertEqual(code, 2)
            self.assertIn("PARENT_RUN_DOES_NOT_BIND_BOTH_VIDEO_AND_LATENT",
                          receipt["school_environment"]["errors"])

    def test_fp32_observation_is_not_native_latent(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, latent, _, _ = self.fixture(temp)
            with latent.open("rb") as fd:
                length = struct.unpack("<Q",fd.read(8))[0]
                header = json.loads(fd.read(length))
            header["latents"]["dtype"] = "F32"
            raw = json.dumps(header).encode()
            latent.write_bytes(struct.pack("<Q",len(raw))+raw+b"\0\0")
            data = json.loads(args.registry.read_text())
            data["school_source_pairs"][0]["final_latent"]["sha256"] = sha256(latent)
            args.registry.write_text(json.dumps(data))
            with patch.dict(os.environ, {"SLURM_JOB_ID":"synthetic-test-job"}):
                with patch("fishscale_school_preflight.video_info",
                           return_value={"width":832,"height":464,"frames":81,"fps":16}):
                    receipt, code = run(args)
            self.assertEqual(code, 2)
            self.assertIn("NATIVE_BF16_LATENT_REQUIRED",
                          [row.get("error") for row in receipt["checks"]])

if __name__ == "__main__":
    unittest.main()
