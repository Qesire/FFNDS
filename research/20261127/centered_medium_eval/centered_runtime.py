"""Thin binding to the archived job-201500 FFNDS runtime; no new kernels.

Requires the existing, runnable school runtime directory (its generate.py,
release files, LightX2V, and diagnostic import dependencies) and frozen fit.
This adapter has only been syntax/structure checked in the review workspace.
"""
import argparse
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import importlib
import inspect
import json
import os
from pathlib import Path
import sys


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class FrozenCenteredRuntime:
    """Reuse one Experiment and wide8192 port for counterbalanced evaluation."""

    def __init__(self, runtime_dir, fit_dir, binding_path=None):
        self.runtime = Path(runtime_dir).resolve()
        self.fit = Path(fit_dir).resolve()
        binding_path = Path(binding_path or Path(__file__).with_name("FROZEN_METHOD_BINDING.json")).resolve()
        self.binding = json.loads(binding_path.read_text())
        self.binding_sha256 = file_hash(binding_path)
        self.runner_sha256 = file_hash(__file__)
        self.source_manifest_sha256 = None
        for name, expected in self.binding["required_runtime_sources_sha256"].items():
            actual = file_hash(self.runtime / name)
            if actual != expected:
                raise ValueError(f"Frozen source mismatch: {name}: {actual}")
        if file_hash(self.fit / "FIT.sha256") != self.binding["fit_manifest_sha256"]:
            raise ValueError("FIT.sha256 does not identify the frozen calibration")
        # Archived code resolves attention_P_k.npy and release assets from cwd.
        os.chdir(self.runtime)
        sys.path.insert(0, str(self.runtime))
        self.torch = importlib.import_module("torch")
        self.generate = importlib.import_module("generate")
        centered = importlib.import_module("attention_mass_centered_low")
        online = importlib.import_module("attention_online_structured")
        for module in (centered, online):
            actual = Path(inspect.getfile(module)).resolve()
            if actual.parent != self.runtime:
                raise ValueError(f"Import resolved outside the selected runtime: {actual}")
        if Path(inspect.getfile(self.generate)).resolve().parent != self.runtime:
            raise ValueError("generate imported from another runtime")
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.experiment = self.generate.Experiment()
        self.attention = centered.MassCenteredLow(self.experiment, self.fit)
        self.attention.mode = "centered"  # Constructor default is raw.
        if self.attention.calibration != self.binding["calibration"]:
            raise ValueError("Loaded attention CAL differs from the frozen recipe")
        if len(self.attention.probes) != 64:
            raise ValueError("The frozen candidate requires the original P64")
        self.wide = self.experiment.build("wide8192")
        self.runtime_generate_sha256 = file_hash(inspect.getfile(self.generate))

    def register_manifest(self, path):
        """Register school-private image/prompt rows using the existing interface."""
        if self.source_manifest_sha256 is not None:
            raise RuntimeError("One frozen source manifest per runtime is supported")
        raw_manifest = Path(path).resolve().read_bytes()
        manifest = json.loads(raw_manifest)
        seen = set()
        for row in manifest["sources"]:
            sid = row["source_id"]
            if sid in seen:
                raise ValueError(f"Duplicate source in the manifest: {sid}")
            seen.add(sid)
            if not Path(row["image_path"]).is_absolute():
                raise ValueError("Manifest image_path must be absolute")
            if file_hash(row["image_path"]) != row["image_sha256"]:
                raise ValueError(f"Image fingerprint differs: {sid}")
            if sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError(f"Prompt fingerprint differs: {sid}")
            if row.get("seed", self.binding["seed"]) != self.binding["seed"]:
                raise ValueError(f"Manifest seed differs: {sid}")
            if sid in self.experiment.rows:
                existing = self.experiment.rows[sid]
                if any(existing[k] != row[k] for k in ("image_sha256", "prompt_sha256")) or existing["seed"] != self.binding["seed"]:
                    raise ValueError(f"Previously registered source has a different identity: {sid}")
            self.experiment.rows[sid] = dict(row, record_id=sid, seed=self.binding["seed"])
        self.source_manifest_sha256 = sha256(raw_manifest).hexdigest()

    @contextmanager
    def _capture_final_latent(self):
        """Use the archived scheduler hook; copy only the final quality state."""
        scheduler = self.experiment.pipe.runner.scheduler
        original_post = scheduler.step_post
        captured, seen = {}, []

        def post():
            step = int(scheduler.step_index)
            result = original_post()
            seen.append(step)
            if step == 3:
                if "latent" in captured:
                    raise RuntimeError("Final latent was produced more than once")
                captured["latent"] = scheduler.latents.detach().to("cpu", copy=True).contiguous()
            return result

        scheduler.step_post = post
        try:
            yield captured
            if seen != [0, 1, 2, 3] or "latent" not in captured:
                raise RuntimeError("Expected exactly four steps and one final latent")
        finally:
            scheduler.step_post = original_post

    def render(self, source, group, arm="centered", capture_final_latent=False):
        """Return a complete receipt; only centered is the frozen candidate.

        dense: native attention + native FFN on the same four-step model.
        wide8192: native attention + wide8192 FFN.
        raw: historical C, high OUT-ABS and low raw OUT-D.
        centered: high OUT-ABS and low mass-centered OUT-D.

        capture_final_latent is quality-only. The default adds no tensor capture
        to the original Experiment timing and trace-witness implementation.
        """
        if arm not in ("dense", "wide8192", "raw", "centered"):
            raise ValueError(arm)
        if Path(group).is_absolute() or ".." in Path(group).parts:
            raise ValueError("group must be a relative output group")
        if source not in self.experiment.rows:
            raise ValueError(f"Unregistered source: {source}")
        if self.experiment.rows[source]["seed"] != self.binding["seed"]:
            raise ValueError("Source seed differs from the frozen recipe")
        # Historical reuse only distinguishes the FFN variant; never reuse it.
        receipt = self.experiment.output / group / (source + ".json")
        if receipt.exists():
            raise FileExistsError(f"Use a new output group: {receipt}")
        self.attention.mode = "centered" if arm == "centered" else "raw"
        if self.attention.layer is not None or self.attention.anchor_shadow or self.attention.diagnostics:
            raise RuntimeError("Diagnostic attention state is active")
        bind = self.attention.bind("OUT-D", audit=None, profile=False,
                                  archive_routes=False, high_step0=False)
        context = bind if arm in ("raw", "centered") else nullcontext(({}, [], []))
        kwargs = {"reuse": False} if "reuse" in inspect.signature(self.experiment.render).parameters else {}
        capture = self._capture_final_latent() if capture_final_latent else nullcontext(None)
        with self.torch.inference_mode(), capture as final_state, context as (counts, _, _):
            row = self.experiment.render(source, group, None if arm == "dense" else self.wide, **kwargs)
        if row["source"] != source or row["experts"] != [0, 0, 1, 1]:
            raise RuntimeError("Unexpected source or expert sequence; expected HHLL")
        if any(len(row[key]) != 4 for key in ("flows", "latents", "step_ms")):
            raise RuntimeError("Expected four recorded diffusion steps")
        video = row["video"]
        expected = self.binding["geometry"]
        if any(video[key] != expected[key] for key in ("width", "height", "frames", "fps")):
            raise RuntimeError("Output geometry differs from the frozen recipe")
        if arm != "dense":
            ffn = row["counters"]
            if tuple(ffn[k] for k in ("variant", "tile_rows", "anchors", "hits")) != ("wide8192", 8192, 80, 80):
                raise RuntimeError("Unexpected FFN execution counts")
        elif "counters" in row:
            raise RuntimeError("Dense arm unexpectedly used a FFN approximation")
        if arm in ("raw", "centered"):
            keys = ("attention_sites", "anchors", "hits", "native_probe_queries")
            if tuple(counts[k] for k in keys) != (160, 80, 80, 5120):
                raise RuntimeError("Unexpected full-trajectory attention counts")
        elif counts:
            raise RuntimeError("Native-attention arm unexpectedly used the adapter")
        final_latent = None
        if capture_final_latent:
            from safetensors.torch import save_file
            path = receipt.with_name(source + ".final_latent.safetensors")
            save_file(final_state, str(path))
            tensor = final_state["latent"]
            final_latent = dict(path=str(path.resolve()), sha256=file_hash(path),
                                bytes=path.stat().st_size, shape=list(tensor.shape),
                                dtype=str(tensor.dtype), tensor_name="latent", capture_step=3)
        wrapped = dict(arm=arm, source_id=source, record=row, attention=dict(counts),
                       seed=self.binding["seed"], experts=row["experts"],
                       binding_sha256=self.binding_sha256, runner_sha256=self.runner_sha256,
                       source_manifest_sha256=self.source_manifest_sha256,
                       runtime_generate_sha256=self.runtime_generate_sha256,
                       candidate_id=self.binding["candidate_id"] if arm == "centered" else None,
                       trajectory_saved=False, diagnostic_replay=False,
                       final_latent_saved=bool(capture_final_latent), final_latent=final_latent,
                       quality_capture_in_timing_window=bool(capture_final_latent))
        # Preserve the Experiment receipt; add the complete method identity beside it.
        self.generate.save(receipt.with_name(source + ".full_method.json"), wrapped)
        return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append")
    parser.add_argument("--group", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--arm", choices=("dense", "wide8192", "raw", "centered"), default="centered")
    parser.add_argument("--capture-final-latent", action="store_true",
                        help="Quality only: save the step-3 final latent; exclude this run from formal timing")
    args = parser.parse_args()
    # Resolve the optional manifest before the archived runtime changes cwd.
    manifest = args.manifest.resolve() if args.manifest else None
    runtime = FrozenCenteredRuntime(args.runtime_dir, args.fit_dir)
    if manifest:
        runtime.register_manifest(manifest)
    for source in args.source:
        output = runtime.render(source, args.group, args.arm, capture_final_latent=args.capture_final_latent)
        print(json.dumps(dict(source_id=source, arm=args.arm,
                              binding_sha256=output["binding_sha256"])), flush=True)


if __name__ == "__main__":
    main()
