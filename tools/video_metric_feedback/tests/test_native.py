"""Lifecycle contracts only: fixtures do not run a model or decode real video."""

from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.video_metric_feedback.matrices import sha256_file
from tools.video_metric_feedback.native import NativeRuntime, generate_batch, load_native_runtime, _recipe_contract


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value.copy())


def tensor_sha(tensor, torch):
    if not np.isfinite(tensor.value).all():
        raise RuntimeError("Nonfinite fixture tensor")
    return {"sha256": hashlib.sha256(tensor.value.tobytes()).hexdigest(),
            "shape": list(tensor.value.shape), "dtype": "fixture.float32"}


def make_runtime():
    state = SimpleNamespace(bindings=[], matrices=[], resets=0, syncs=0, anchor_checks=0)

    class Transformer:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.block_idx = 0
            self.offload_manager = SimpleNamespace(need_init_first_buffer=False)

        def infer_ffn(self, *args, **kwargs):
            raise AssertionError("Original fixture FFN should be bound during generation")

        def infer_block(self):
            return self.infer_ffn(SimpleNamespace(), None, None, None, None, gate=object())

    class Dispatcher:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.schedule = [0, 0, 1, 1]
            self.model = [SimpleNamespace(transformer_infer=Transformer(scheduler)) for _ in range(2)]

        def get_current_model_index(self):
            self.cur_model_index = self.schedule[self.scheduler.step_index]
            return self.cur_model_index

    class Matrices:
        def __init__(self, entries, sha, device):
            self.entries, self.device = entries, device
            self.closed = False
            self.releases = 0
            for entry in entries.values():
                if sha(entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Fixture matrix SHA mismatch")
            state.matrices.append(self)

        def release(self):
            self.releases += 1

        def close(self):
            self.closed = True

    class Binding:
        def __init__(self, mode, matrices, *, layers, expected_shape, execution, on_site):
            self.mode, self.matrices, self.layers = mode, matrices, layers
            self.expected_shape, self.execution = expected_shape, execution
            self.on_site = on_site
            self.owner = None
            self.dispatch = None
            self.seen = set()
            self.cache = {}
            self.counts = dict(dense=0, anchor=0, hit=0, native_ffn=0, predictions=0)
            self.uninstalled = False
            state.bindings.append(self)

        def install(self, owner):
            self.owner, self.original = owner, owner.infer_ffn
            binding = self

            def ffn(model, phase, x, attn, shift, scale, gate=None):
                step, layer = model.scheduler.step_index, model.block_idx
                if gate is None or (step, layer) in binding.seen:
                    raise RuntimeError("Invalid fixture FFN state")
                if binding.dispatch != [0, 0, 1, 1][step]:
                    raise RuntimeError("Wrong fixture dispatch")
                binding.seen.add((step, layer))
                binding.on_site(model, phase, step, "high" if binding.dispatch == 0 else "low", layer)
                binding.counts["predictions"] += 1
                if step in (0, 2):
                    binding.cache[layer] = step
                    binding.counts["anchor"] += 1
                    binding.counts["native_ffn"] += 1
                else:
                    if binding.cache.get(layer) != step - 1:
                        raise RuntimeError("Fixture anchor not rebuilt")
                    binding.counts["hit"] += 1
                return None

            owner.infer_ffn = ffn

        def finish(self):
            if self.seen != {(s, layer) for s in range(4) for layer in range(self.layers)}:
                raise RuntimeError("Incomplete fixture FFN coverage")
            return self.counts

        def uninstall(self):
            if self.owner is not None:
                self.owner.infer_ffn = self.original
                self.owner = None
            self.uninstalled = True
            self.cache.clear()
            self.matrices.release()

    def reset(scheduler, model):
        if model.scheduler is not scheduler:
            raise RuntimeError("MODEL_SCHEDULER_IDENTITY_MISMATCH")
        scheduler.generator = None
        state.resets += 1

    def synchronize():
        state.syncs += 1

    def verify_phase(phase):
        state.anchor_checks += 1

    def video_digest(path):
        if not Path(path).read_bytes().startswith(b"synthetic fixture"):
            raise RuntimeError("Missing synthetic fixture video")
        return {"frames": 81, "width": 832, "height": 464, "fps": 16.0,
                "mp4_sha256": sha256_file(path), "decoded_rgb24_sha256": "0" * 64}

    runtime = NativeRuntime(
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize), no_grad=nullcontext),
        binding_cls=Binding, matrices_cls=Matrices, transformer_cls=Transformer,
        dispatcher_cls=Dispatcher, reset_scheduler_generator=reset,
        finite_tensor_sha=tensor_sha, video_digest=video_digest,
        synthetic=True, device="fixture", verify_phase=verify_phase,
        source_identity={"kind": "synthetic_lifecycle_fixture"},
    )
    return runtime, state


class FakePipe:
    def __init__(self, runtime, layers=2):
        scheduler = SimpleNamespace(step_index=0, generator=object())
        model = runtime.dispatcher_cls(scheduler)
        self.runner = SimpleNamespace(
            scheduler=scheduler, model=model,
            config={"infer_steps": 4, "num_layers": layers, "boundary_step_index": 2, "enable_cfg": False},
        )
        self.calls = []
        self.fail_at_site = None
        self.skip_last_site = False
        self.layers = layers

    def generate(self, *, seed, image_path, prompt, negative_prompt, save_result_path):
        scheduler, model = self.runner.scheduler, self.runner.model
        if scheduler.generator is not None:
            raise AssertionError("Scheduler generator was not reset")
        if not all(m.transformer_infer.offload_manager.need_init_first_buffer for m in model.model):
            raise AssertionError("Expert prefetch buffers were not reset")
        self.calls.append({"seed": seed, "image_path": image_path, "prompt": prompt,
                           "negative_prompt": negative_prompt})
        scheduler.generator = object()
        scheduler.latents = FakeTensor([seed, seed + 1])
        counter = 0
        for step in range(4):
            scheduler.step_index = step
            model.get_current_model_index()
            infer = model.model[model.cur_model_index].transformer_infer
            for layer in range(self.layers):
                if self.skip_last_site and step == 3 and layer == self.layers - 1:
                    continue
                infer.block_idx = layer
                infer.infer_block()
                counter += 1
                if self.fail_at_site == counter:
                    raise RuntimeError("Injected generation failure")
                scheduler.latents.value += np.float32(0.25)
        for m in model.model:
            m.transformer_infer.offload_manager.need_init_first_buffer = False
        Path(save_result_path).write_bytes(f"synthetic fixture seed={seed}".encode())


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        image = self.root / "image.bin"
        image.write_bytes(b"synthetic source image")
        payload = self.root / "matrix.npy"
        np.save(payload, np.full((3, 3), 0x3F80, dtype=np.uint16))
        entries = {f"{expert}/{layer}": {"path": str(payload), "sha256": sha256_file(payload),
                                        "shape": [3, 3], "dtype": "BF16_RAW_UINT16"}
                   for expert in ("high", "low") for layer in range(2)}
        manifest = self.root / "matrix_manifest.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        self.request = {
            "schema_version": 1, "mode": "video", "candidate_id": "fixture_candidate",
            "parameters": {"high": 0.5}, "execution_fingerprint": "1" * 64,
            "evaluation_key": "2" * 64, "matrix_manifest_path": str(manifest),
            "matrix_export": {"fingerprint": "1" * 64, "matrix_manifest_sha256": sha256_file(manifest)},
            "oracle_identity": {"recipe": {"ffn_input_shape": [5, 3], "num_layers": 2}},
            "cohort": [{"source_id": "source", "image_id": "image", "seed": seed,
                        "sample_index": index, "image_path": str(image), "prompt": "Fixture scene",
                        "negative_prompt": "Fixture negative"} for index, seed in enumerate((43, 44))],
        }
        self.runtime, self.state = make_runtime()
        self.pipe = FakePipe(self.runtime)
        self.originals = (self.runtime.transformer_cls.infer_ffn,
                          self.runtime.transformer_cls.infer_block,
                          self.runtime.dispatcher_cls.get_current_model_index)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_restored(self):
        self.assertEqual(self.originals, (self.runtime.transformer_cls.infer_ffn,
                                         self.runtime.transformer_cls.infer_block,
                                         self.runtime.dispatcher_cls.get_current_model_index))
        self.assertTrue(all(m.closed for m in self.state.matrices))
        self.assertTrue(all(b.uninstalled for b in self.state.bindings))

    def test_success_fresh_cache_per_sample_and_one_matrix_set_per_candidate(self):
        result = generate_batch(self.pipe, self.request, self.root / "output", runtime=self.runtime)
        self.assertEqual(result["status"], "VIDEOS_GENERATED_METRICS_PENDING")
        self.assertTrue(result["synthetic"])
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(len(self.state.matrices), 1)
        self.assertEqual(len(self.state.bindings), 2)
        self.assertIsNot(self.state.bindings[0].cache, self.state.bindings[1].cache)
        self.assertEqual(self.state.resets, 2)
        self.assertEqual(self.state.anchor_checks, 8)
        self.assertEqual(self.state.syncs, 4)
        for row, seed in zip(result["records"], (43, 44)):
            self.assertEqual(row["initial_noise_sha256"], tensor_sha(FakeTensor([seed, seed + 1]), None)["sha256"])
            self.assertEqual(row["counts"], dict(dense=0, anchor=4, hit=4, native_ffn=4, predictions=8))
            self.assertEqual(row["dispatch"], [[0, 0], [1, 0], [2, 1], [3, 1]])
            self.assertEqual(row["site_count"], 8)
            self.assertTrue(Path(row["video_path"]).is_file())
        self.assertEqual(json.loads(Path(result["manifest_path"]).read_text()), result)
        self.assertEqual(self.pipe.calls[0]["negative_prompt"], "Fixture negative")
        self.assert_restored()

    def test_new_candidate_does_not_reuse_resident_matrices_or_anchor_caches(self):
        first = generate_batch(self.pipe, self.request, self.root / "first", runtime=self.runtime)
        self.request["candidate_id"] = "second_candidate"
        second = generate_batch(self.pipe, self.request, self.root / "second", runtime=self.runtime)
        self.assertEqual(len(self.state.matrices), 2)
        self.assertEqual(len(self.state.bindings), 4)
        self.assertEqual(len({id(b.cache) for b in self.state.bindings}), 4)
        self.assertEqual([r["initial_noise_sha256"] for r in first["records"]],
                         [r["initial_noise_sha256"] for r in second["records"]])
        self.assert_restored()

    def test_generation_failure_restores_patches_and_closes_matrices(self):
        self.pipe.fail_at_site = 2
        with self.assertRaisesRegex(RuntimeError, "Injected generation failure"):
            generate_batch(self.pipe, self.request, self.root / "failure", runtime=self.runtime)
        self.assertFalse((self.root / "failure/videos_manifest.json").exists())
        self.assertEqual(len(self.state.bindings), 1)
        self.assert_restored()

    def test_partial_install_failure_restores_all_patched_classes(self):
        original = self.runtime.binding_cls.install

        def broken_install(binding, owner):
            original(binding, owner)
            raise RuntimeError("Injected install failure")

        with patch.object(self.runtime.binding_cls, "install", broken_install):
            with self.assertRaisesRegex(RuntimeError, "Injected install failure"):
                generate_batch(self.pipe, self.request, self.root / "install_failure", runtime=self.runtime)
        self.assertEqual(self.pipe.calls, [])
        self.assert_restored()

    def test_absent_negative_prompt_uses_the_native_empty_fallback(self):
        self.request["cohort"][0].pop("negative_prompt")
        generate_batch(self.pipe, self.request, self.root / "negative_fallback", runtime=self.runtime)
        self.assertEqual(self.pipe.calls[0]["negative_prompt"], "")
        self.assertEqual(self.pipe.calls[1]["negative_prompt"], "Fixture negative")
        self.assert_restored()

    def test_output_created_while_waiting_for_lock_is_not_overwritten(self):
        output = self.root / "contended"

        class ContendedLock:
            def __enter__(self):
                path = output / "videos" / "sample_0000.mp4"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"another completed request")

            def __exit__(self, *args):
                return False

        with patch("tools.video_metric_feedback.native._LOCK", ContendedLock()):
            with self.assertRaises(FileExistsError):
                generate_batch(self.pipe, self.request, output, runtime=self.runtime)
        self.assertEqual(self.pipe.calls, [])
        self.assertEqual((output / "videos/sample_0000.mp4").read_bytes(), b"another completed request")

    def test_wrong_dispatch_fails_before_wrong_expert_forward(self):
        self.pipe.runner.model.schedule = [0, 1, 1, 1]
        with self.assertRaisesRegex(RuntimeError, "HHLL"):
            generate_batch(self.pipe, self.request, self.root / "schedule", runtime=self.runtime)
        self.assertEqual(len(self.state.bindings[0].seen), 2)
        self.assert_restored()

    def test_missing_site_is_not_a_complete_video(self):
        self.pipe.skip_last_site = True
        with self.assertRaisesRegex(RuntimeError, "Incomplete.*coverage"):
            generate_batch(self.pipe, self.request, self.root / "missing", runtime=self.runtime)
        self.assertFalse((self.root / "missing/videos_manifest.json").exists())
        self.assert_restored()

    def test_expected_initial_noise_is_checked_before_first_ffn(self):
        self.request["cohort"][0]["initial_noise_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Initial noise mismatch"):
            generate_batch(self.pipe, self.request, self.root / "noise", runtime=self.runtime)
        self.assertEqual(self.state.bindings[0].seen, set())
        self.assert_restored()

    def test_bad_manifest_and_matrix_payload_are_rejected_before_generation(self):
        self.request["matrix_export"]["matrix_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            generate_batch(self.pipe, self.request, self.root / "bad_manifest", runtime=self.runtime)
        manifest = Path(self.request["matrix_manifest_path"])
        self.request["matrix_export"]["matrix_manifest_sha256"] = sha256_file(manifest)
        np.save(self.root / "matrix.npy", np.zeros((3, 3), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Matrix SHA256"):
            generate_batch(self.pipe, self.request, self.root / "bad_payload", runtime=self.runtime)
        self.assertEqual(self.pipe.calls, [])
        self.assertEqual(self.state.matrices, [])

    def test_existing_outputs_and_pipeline_mismatch_are_rejected(self):
        generate_batch(self.pipe, self.request, self.root / "existing", runtime=self.runtime)
        with self.assertRaises(FileExistsError):
            generate_batch(self.pipe, self.request, self.root / "existing", runtime=self.runtime)
        self.pipe.runner.config["enable_cfg"] = True
        with self.assertRaisesRegex(RuntimeError, "enable_cfg"):
            generate_batch(self.pipe, self.request, self.root / "cfg", runtime=self.runtime)
        self.assertEqual(len(self.pipe.calls), 2)
        self.assert_restored()

    def test_fixed_recipe_defaults_and_conflicts(self):
        contract = _recipe_contract({})
        self.assertEqual(contract["ffn_input_shape"], [31668, 5120])
        self.assertEqual(contract["num_layers"], 40)
        self.assertTrue(contract["default_ffn_input_shape_used"])
        for recipe in ({"execution": "fused_post"}, {"infer_steps": 40}, {"video_frames": 80},
                       {"ffn_input_shape": [5, 3], "expected_shape": [6, 3]}):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                _recipe_contract(recipe)


class NativeLoaderTests(unittest.TestCase):
    def test_private_modules_support_postponed_dataclasses_and_restore_aliases(self):
        sources = {
            "runtime_guard": "from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass ScopeInfo:\n    value: int = 1\n",
            "direct": "from runtime_guard import ScopeInfo\nclass DirectFFN:\n    scope = ScopeInfo()\n",
            "binding": "from direct import DirectFFN\nclass NativeBinding:\n    direct = DirectFFN\n",
            "resident": "class ResidentMatrices:\n    pass\n",
            "legacy_helpers": "from runtime_guard import ScopeInfo\ndef reset_scheduler_generator(*args): pass\ndef finite_tensor_sha(*args): pass\ndef video_digest(*args): pass\n",
        }
        fake_modules = {}
        for name, attribute in (
            ("lightx2v.models.networks.wan.infer.transformer_infer", "WanTransformerInfer"),
            ("lightx2v.models.runners.wan.wan_distill_runner", "MultiDistillModelStruct"),
            ("lightx2v.common.ops.mm.mm_weight", "MMWeight"),
        ):
            parts = name.split(".")
            for end in range(1, len(parts) + 1):
                module_name = ".".join(parts[:end])
                if module_name not in fake_modules:
                    fake_modules[module_name] = ModuleType(module_name)
                    fake_modules[module_name].__path__ = []
            setattr(fake_modules[name], attribute, type(attribute, (), {}))
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
        fake_torch.device = lambda value: value
        fake_torch.__version__ = "synthetic-import-fixture"
        fake_modules["torch"] = fake_torch
        aliases = {name: ModuleType("preexisting_" + name) for name in sources}
        with tempfile.TemporaryDirectory() as directory:
            for name, code in sources.items():
                Path(directory, name + ".py").write_text(code)
            with patch.dict(sys.modules, {**fake_modules, **aliases}):
                runtime = load_native_runtime(directory)
                module_name = runtime.binding_cls.__module__
                self.assertIn(module_name, sys.modules)
                self.assertIs(inspect.getmodule(runtime.binding_cls), sys.modules[module_name])
                self.assertEqual(runtime.binding_cls.direct.scope.value, 1)
                for name, original in aliases.items():
                    self.assertIs(sys.modules[name], original)
                repeated = load_native_runtime(directory)
                self.assertIs(repeated.binding_cls, runtime.binding_cls)


if __name__ == "__main__":
    unittest.main()
