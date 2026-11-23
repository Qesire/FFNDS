"""Exercise the public four-step selection path without installing the school model."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ffnds_fourstep"))
import fourstep_backend


class DefaultFourStepTests(unittest.TestCase):
    def test_implicit_port_installs_checkpoint_hit(self):
        inner = SimpleNamespace(counters={})
        factory = Mock(return_value=inner)
        hit = Mock()
        with patch.dict(sys.modules, exact_port=SimpleNamespace(make_port=factory),
                        fourstep_hit=SimpleNamespace(make_hit=hit)):
            port = fourstep_backend.make_port({})
            factory.assert_called_once_with({}, "fast512")
            hit.assert_called_once_with(8192, reuse_base=True)
            self.assertIs(inner.hit_prepared, hit.return_value)
            self.assertEqual((port.counters["variant"], port.counters["tile_rows"]), ("wide8192", 8192))
            self.assertIs(fourstep_backend.make_port({}, "fast512"), inner)

    def test_default_render_reuses_port_and_explicit_none_runs_dense(self):
        class Reference:
            def render(self, source, group, method, reuse):
                return method

        module = SimpleNamespace(Experiment=Reference, load=None, save=None, equal_trace=None, digest=None)
        with patch.dict(sys.modules, generate_reference=module, torch=Mock()):
            import generate

        exp = generate.Experiment()
        port = object()
        exp.build = Mock(return_value=port)
        self.assertIs(exp.render("vbi_0392", "wide8192"), port)
        self.assertIs(exp.run(), port)
        self.assertIsNone(exp.render("vbi_0392", "dense", None))
        exp.build.assert_called_once_with()

    def test_joined_port_uses_measured_mode_and_public_receipt_id(self):
        measured = Mock(counters={"variant": "joined", "joined_y0": 80})
        measured.bind.return_value = nullcontext()
        measured.trajectory.return_value = nullcontext()
        factory = Mock(return_value=measured)
        with patch.dict(sys.modules, y0_join=SimpleNamespace(make_port=factory)):
            port = fourstep_backend.make_port({}, "wide8192_all_int8_joined")
        factory.assert_called_once_with({})
        measured.set_mode.assert_called_once_with("joined")
        with port.trajectory() as active:
            with active.bind("owner", "sha") as bound:
                self.assertIs(bound, port)
        measured.trajectory.assert_called_once_with()
        measured.bind.assert_called_once_with("owner", "sha")
        self.assertEqual(port.variant, "wide8192_all_int8_joined")
        self.assertEqual(port.counters, {"variant": port.variant, "joined_y0": 80})

    def test_joined_experiment_checks_cached_receipt_and_defaults_group(self):
        variant = fourstep_backend.JOINED_VARIANT
        class Reference:
            def __init__(self):
                self.base = {}
                self.state = {"setup": []}

            def render(self, source, group, method, reuse):
                return group, method, reuse

        module = SimpleNamespace(Experiment=Reference, load=lambda p: json.loads(p.read_bytes()),
                                 save=None, equal_trace=None, digest=None)
        with patch.dict(sys.modules, generate_reference=module, torch=Mock()):
            import generate
        port = SimpleNamespace(variant=variant)
        with TemporaryDirectory() as temp, patch.object(generate, "make_port", return_value=port) as factory:
            experiment = generate.Experiment(variant=variant)
            experiment.output = Path(temp)
            self.assertEqual(experiment.run(), (variant, port, False))
            factory.assert_called_once_with(experiment.base, variant)
            folder = experiment.output / variant
            folder.mkdir()
            for old_id in ("wide8192", "joined", variant):
                (folder / "sample.json").write_text(json.dumps({"counters": {"variant": old_id}}))
                self.assertEqual(experiment.render("sample", variant, reuse=True),
                                 (variant, port, old_id == variant))
            self.assertEqual(experiment.state["setup"][0]["mode"], variant)


if __name__ == "__main__":
    unittest.main()
